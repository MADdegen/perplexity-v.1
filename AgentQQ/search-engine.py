"""
Agent QQ — Full Perplexity Search Engine
========================================
Complete Perplexity Sonar API integration with ALL capabilities:
  * 5 models: sonar, sonar-pro, sonar-reasoning, sonar-reasoning-pro, sonar-deep-research
  * Streaming (SSE) with real-time terminal output
  * Citations  (return_citations — always on)
  * Image results (return_images)
  * Related follow-up questions (return_related_questions)
  * Domain filtering — include or exclude specific sites (search_domain_filter)
  * Recency filtering — day / week / month / year (search_recency_filter)
  * Search context size — low / medium / high (web_search_options, sonar-pro only)
  * Structured JSON output (response_format / json_schema)
  * Multi-turn conversation history
  * Cost tracking per query and per session
  * Exa + Tavily + Firecrawl + local Ollama fallback chain

Usage:
  python search-engine.py "latest Arbitrum news"
  python search-engine.py --model sonar-deep "DeFi prediction market landscape 2025"
  python search-engine.py --recency day --domain reddit.com "MAD Gambit feedback"
  python search-engine.py --images --related "Perplexity AI competitors"
  python search-engine.py --stream "explain ZK rollups"
  python search-engine.py --setup     # show API key setup guide
  python search-engine.py --keys      # check configured keys
  python search-engine.py --models    # list all models + costs

Requirements:
  pip install requests exa-py tavily-python --break-system-packages
  Env vars: PPLX_API_KEY (primary), EXA_API_KEY, TAVILY_API_KEY, FIRECRAWL_API_KEY (fallbacks)
"""

import os
import sys
import json
import time
import argparse
import re
import textwrap
from typing import Optional

# ---------------------------------------------------------------------------
# API KEYS
# ---------------------------------------------------------------------------

PPLX_KEY      = os.getenv("PPLX_API_KEY", "")
EXA_KEY       = os.getenv("EXA_API_KEY", "")
TAVILY_KEY    = os.getenv("TAVILY_API_KEY", "")
FIRECRAWL_KEY = os.getenv("FIRECRAWL_API_KEY", "")
PPLX_BASE_URL = "https://api.perplexity.ai"

# ---------------------------------------------------------------------------
# PERPLEXITY MODELS  (alias -> (api_name, description, $/1k_tok, supports_ctx_size))
# ---------------------------------------------------------------------------

PPLX_MODELS = {
    "fast":          ("sonar",                "Quick lookup, 127K ctx",                       0.001, False),
    "sonar":         ("sonar",                "Quick lookup, 127K ctx",                       0.001, False),
    "pro":           ("sonar-pro",            "Multi-step advanced, 200K ctx (default)",      0.003, True),
    "sonar-pro":     ("sonar-pro",            "Multi-step advanced, 200K ctx (default)",      0.003, True),
    "reasoning":     ("sonar-reasoning",      "CoT + web, 127K ctx — best for analysis",     0.005, False),
    "sonar-r":       ("sonar-reasoning",      "CoT + web, 127K ctx — best for analysis",     0.005, False),
    "reasoning-pro": ("sonar-reasoning-pro",  "Extended thinking + web, max accuracy",        0.008, False),
    "sonar-rp":      ("sonar-reasoning-pro",  "Extended thinking + web, max accuracy",        0.008, False),
    "deep":          ("sonar-deep-research",  "Full deep research, 30-60s, comprehensive",   0.015, False),
    "sonar-deep":    ("sonar-deep-research",  "Full deep research, 30-60s, comprehensive",   0.015, False),
}

# ---------------------------------------------------------------------------
# SESSION COST TRACKER
# ---------------------------------------------------------------------------

_session_cost    = 0.0
_session_queries = 0


def _track_cost(model_key: str, total_tokens: int) -> float:
    global _session_cost, _session_queries
    info = PPLX_MODELS.get(model_key, PPLX_MODELS["pro"])
    cost = (total_tokens / 1000.0) * info[2]
    _session_cost    += cost
    _session_queries += 1
    return cost


# ===========================================================================
# PERPLEXITY SONAR  — complete API wrapper
# ===========================================================================

def search_perplexity(
    query: str,
    *,
    model: str = "pro",
    # ── standard LLM params ─────────────────────────────────────────────────
    max_tokens: int = 2048,
    temperature: float = 0.2,
    top_p: float = 0.9,
    top_k: int = 0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 1.0,
    # ── Perplexity-specific params ──────────────────────────────────────────
    return_citations: bool = True,
    return_images: bool = False,
    return_related_questions: bool = True,
    search_domain_filter: Optional[list] = None,  # ["reddit.com", "-wikipedia.org"]
    search_recency_filter: Optional[str] = None,  # "day"|"week"|"month"|"year"
    search_context_size: str = "medium",           # "low"|"medium"|"high" (sonar-pro only)
    # ── output controls ─────────────────────────────────────────────────────
    stream: bool = False,
    response_format: Optional[dict] = None,        # {"type":"json_schema","json_schema":{...}}
    # ── conversation ────────────────────────────────────────────────────────
    conversation_history: Optional[list] = None,
    system_prompt: Optional[str] = None,
) -> dict:
    """
    Full Perplexity Sonar API call.  Every documented parameter is wired.
    Returns: answer, citations, images, related_questions, usage, cost_usd, elapsed.
    """
    if not PPLX_KEY:
        return {"error": "PPLX_API_KEY not set. Run: python search-engine.py --setup",
                "source": "perplexity"}

    import requests

    info = PPLX_MODELS.get(model, PPLX_MODELS["pro"])
    api_model          = info[0]
    supports_ctx_size  = info[3]

    default_system = (
        "You are a precise research assistant with real-time web access. "
        "Answer directly, citing sources inline as [1], [2], etc. "
        "Include only verified, current information. Flag uncertainty explicitly. "
        "Structure long answers with headers for scannability."
    )

    messages = [{"role": "system", "content": system_prompt or default_system}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": query})

    # Build body — include every Perplexity param
    body = {
        "model":                    api_model,
        "messages":                 messages,
        "max_tokens":               max_tokens,
        "temperature":              temperature,
        "top_p":                    top_p,
        "top_k":                    top_k,
        "presence_penalty":         presence_penalty,
        "frequency_penalty":        frequency_penalty,
        "return_citations":         return_citations,
        "return_images":            return_images,
        "return_related_questions": return_related_questions,
        "stream":                   stream,
    }

    if search_domain_filter:
        body["search_domain_filter"] = search_domain_filter

    if search_recency_filter in ("day", "week", "month", "year"):
        body["search_recency_filter"] = search_recency_filter

    # web_search_options only on sonar-pro
    if supports_ctx_size and search_context_size in ("low", "medium", "high"):
        body["web_search_options"] = {"search_context_size": search_context_size}

    if response_format:
        body["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {PPLX_KEY}",
        "Content-Type":  "application/json",
        "Accept":        "text/event-stream" if stream else "application/json",
    }

    t0 = time.time()

    if stream:
        return _stream_pplx(body, headers, model, t0)

    # ── non-streaming ────────────────────────────────────────────────────────
    try:
        resp    = requests.post(f"{PPLX_BASE_URL}/chat/completions",
                                headers=headers, json=body, timeout=120)
        elapsed = time.time() - t0

        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:300]}",
                    "source": f"perplexity/{api_model}"}

        data   = resp.json()
        answer = data["choices"][0]["message"].get("content", "")

        citations         = data.get("citations", [])
        images            = data.get("images", [])           # [{"image_url":…, "origin_url":…}]
        related_questions = data.get("related_questions", [])
        usage             = data.get("usage", {})
        cost              = _track_cost(model, usage.get("total_tokens", 0))

        return {
            "source":             f"perplexity/{api_model}",
            "answer":             answer,
            "citations":          citations,
            "images":             images,
            "related_questions":  related_questions,
            "model":              api_model,
            "usage": {
                "prompt_tokens":     usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens":      usage.get("total_tokens", 0),
                "citation_tokens":   usage.get("citation_tokens", 0),
            },
            "cost_usd": round(cost, 6),
            "elapsed":  round(elapsed, 2),
        }

    except Exception as e:
        return {"error": str(e), "source": f"perplexity/{api_model}",
                "elapsed": round(time.time() - t0, 2)}


def _stream_pplx(body: dict, headers: dict, model_key: str, t0: float) -> dict:
    """SSE streaming — prints tokens live, returns assembled result."""
    import requests

    full_text = ""
    citations = []
    images    = []
    related   = []
    usage     = {}

    try:
        with requests.post(f"{PPLX_BASE_URL}/chat/completions",
                           headers=headers, json=body,
                           stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                        "source": "perplexity/stream"}
            print()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk   = json.loads(payload)
                    delta   = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        print(content, end="", flush=True)
                        full_text += content
                    # metadata arrives on the final chunk
                    if "citations"         in chunk: citations = chunk["citations"]
                    if "images"            in chunk: images    = chunk["images"]
                    if "related_questions" in chunk: related   = chunk["related_questions"]
                    if "usage"             in chunk: usage     = chunk["usage"]
                except json.JSONDecodeError:
                    pass
        print()
    except Exception as e:
        return {"error": str(e), "source": "perplexity/stream"}

    elapsed = time.time() - t0
    cost    = _track_cost(model_key, usage.get("total_tokens", 0))
    return {
        "source":             f"perplexity/{body['model']} (stream)",
        "answer":             full_text,
        "citations":          citations,
        "images":             images,
        "related_questions":  related,
        "model":              body["model"],
        "usage":              usage,
        "cost_usd":           round(cost, 6),
        "elapsed":            round(elapsed, 2),
    }


# ===========================================================================
# EXA  — neural semantic search fallback
# ===========================================================================

def search_exa(
    query: str,
    *,
    num_results: int = 10,
    use_autoprompt: bool = True,
    include_domains: Optional[list] = None,
    exclude_domains: Optional[list] = None,
    start_published_date: Optional[str] = None,  # "2024-01-01"
    category: Optional[str] = None,              # "news"|"research paper"|"tweet"
    text_max_chars: int = 1500,
) -> dict:
    if not EXA_KEY:
        return {"error": "EXA_API_KEY not set", "source": "exa"}

    import requests

    t0   = time.time()
    body = {
        "query":        query,
        "numResults":   num_results,
        "useAutoprompt":use_autoprompt,
        "contents": {
            "text":       {"maxCharacters": text_max_chars},
            "highlights": {"numSentences": 3, "highlightsPerUrl": 2},
        },
    }
    if include_domains:        body["includeDomains"]      = include_domains
    if exclude_domains:        body["excludeDomains"]      = exclude_domains
    if start_published_date:   body["startPublishedDate"]  = start_published_date
    if category:               body["category"]            = category

    try:
        resp    = requests.post("https://api.exa.ai/search",
                                headers={"x-api-key": EXA_KEY, "Content-Type": "application/json"},
                                json=body, timeout=20)
        data    = resp.json()
        results = data.get("results", [])
        elapsed = time.time() - t0
        return {
            "source": "exa",
            "results": [
                {
                    "title":      r.get("title", ""),
                    "url":        r.get("url", ""),
                    "text":       r.get("text", "")[:text_max_chars],
                    "highlights": r.get("highlights", []),
                    "published":  r.get("publishedDate", ""),
                    "score":      r.get("score", 0),
                }
                for r in results
            ],
            "count":   len(results),
            "elapsed": round(elapsed, 2),
        }
    except Exception as e:
        return {"error": str(e), "source": "exa", "elapsed": round(time.time()-t0, 2)}


# ===========================================================================
# TAVILY  — agent-optimised web search fallback
# ===========================================================================

def search_tavily(
    query: str,
    *,
    search_depth: str = "advanced",   # "basic"|"advanced"
    topic: str = "general",           # "general"|"news"
    max_results: int = 10,
    include_domains: Optional[list] = None,
    exclude_domains: Optional[list] = None,
    days: Optional[int] = None,       # news only
) -> dict:
    if not TAVILY_KEY:
        return {"error": "TAVILY_API_KEY not set", "source": "tavily"}

    import requests

    t0   = time.time()
    body = {
        "api_key":            TAVILY_KEY,
        "query":              query,
        "search_depth":       search_depth,
        "topic":              topic,
        "max_results":        max_results,
        "include_answer":     True,
        "include_raw_content":False,
        "include_images":     True,
    }
    if include_domains: body["include_domains"] = include_domains
    if exclude_domains: body["exclude_domains"] = exclude_domains
    if days and topic == "news": body["days"] = days

    try:
        resp    = requests.post("https://api.tavily.com/search", json=body, timeout=25)
        data    = resp.json()
        elapsed = time.time() - t0
        return {
            "source": "tavily",
            "answer": data.get("answer", ""),
            "results": [
                {
                    "title":   r.get("title", ""),
                    "url":     r.get("url", ""),
                    "content": r.get("content", "")[:600],
                    "score":   r.get("score", 0),
                }
                for r in data.get("results", [])
            ],
            "images": [
                img.get("url", img) if isinstance(img, dict) else img
                for img in data.get("images", [])
            ],
            "count":   len(data.get("results", [])),
            "elapsed": round(elapsed, 2),
        }
    except Exception as e:
        return {"error": str(e), "source": "tavily", "elapsed": round(time.time()-t0, 2)}


# ===========================================================================
# FIRECRAWL  — full page content extraction
# ===========================================================================

def firecrawl_extract(url: str) -> dict:
    """Deep-read a URL via Firecrawl. Use after finding relevant URLs from Exa/Tavily."""
    if not FIRECRAWL_KEY:
        return {"error": "FIRECRAWL_API_KEY not set", "source": "firecrawl"}

    import requests

    t0 = time.time()
    try:
        resp    = requests.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={"Authorization": f"Bearer {FIRECRAWL_KEY}",
                     "Content-Type": "application/json"},
            json={"url": url, "formats": ["markdown"]},
            timeout=30,
        )
        data    = resp.json()
        elapsed = time.time() - t0
        return {
            "source":  "firecrawl",
            "url":     url,
            "content": data.get("data", {}).get("markdown", "")[:6000],
            "elapsed": round(elapsed, 2),
        }
    except Exception as e:
        return {"error": str(e), "source": "firecrawl"}


# ===========================================================================
# LOCAL OLLAMA SYNTHESIS  — free zero-cost fallback
# ===========================================================================

def synthesize_with_ollama(query: str, search_results: list,
                            model: str = "gemma3:4b") -> dict:
    """Synthesise Exa + Tavily results using local gemma3:4b (Agent QQ SharedExpert)."""
    import requests

    context_parts = []
    all_citations = []

    for result in search_results:
        if result.get("error"):
            continue
        if "results" in result:
            for r in result["results"][:5]:
                idx  = len(all_citations) + 1
                url  = r.get("url", "")
                all_citations.append(url)
                text = r.get("text") or r.get("content", "")
                context_parts.append(
                    f"[{idx}] {r.get('title',f'Source {idx}')} ({url})\n{text[:500]}"
                )
        if result.get("answer"):
            context_parts.append(f"[{result['source']} answer]: {result['answer'][:400]}")

    if not context_parts:
        return {"error": "No results to synthesise", "source": "ollama"}

    prompt = (
        f"Research assistant. Using ONLY these numbered sources, answer the query. "
        f"Cite every fact inline as [N]. Mark uncertainty clearly.\n\n"
        f"QUERY: {query}\n\nSOURCES:\n" +
        "\n\n".join(context_parts[:8]) +
        "\n\nProvide: direct answer, key findings with citations, caveats, Sources list."
    )

    t0 = time.time()
    try:
        resp    = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.15, "num_predict": 1200, "num_ctx": 4096}},
            timeout=180,
        )
        elapsed = time.time() - t0
        answer  = resp.json().get("response", "")
        return {
            "source":    f"ollama/{model}",
            "answer":    answer,
            "citations": [c for c in all_citations if c],
            "elapsed":   round(elapsed, 2),
        }
    except Exception as e:
        return {"error": str(e), "source": "ollama"}


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def search(
    query: str,
    *,
    model: str = "pro",
    return_images: bool = False,
    return_related_questions: bool = True,
    search_domain_filter: Optional[list] = None,
    search_recency_filter: Optional[str] = None,
    search_context_size: str = "medium",
    stream: bool = False,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    response_format: Optional[dict] = None,
    conversation_history: Optional[list] = None,
    system_prompt: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Full pipeline. Perplexity first, then Exa + Tavily + Ollama fallback.
    """
    if verbose:
        info = PPLX_MODELS.get(model, PPLX_MODELS["pro"])
        print(f"\n{'─'*62}")
        print(f"  Agent QQ Search  |  model: {info[0]}")
        print(f"  {query[:68]}{'...' if len(query)>68 else ''}")
        parts = []
        if search_recency_filter: parts.append(f"recency={search_recency_filter}")
        if search_domain_filter:  parts.append(f"domains={search_domain_filter[:3]}")
        if return_images:         parts.append("images=on")
        if parts: print(f"  {' | '.join(parts)}")
        print(f"{'─'*62}\n")

    # ── Primary: Perplexity ──────────────────────────────────────────────────
    if PPLX_KEY:
        if verbose and not stream:
            print(f"  Perplexity {model}...", end="", flush=True)
        result = search_perplexity(
            query,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            return_citations=True,
            return_images=return_images,
            return_related_questions=return_related_questions,
            search_domain_filter=search_domain_filter,
            search_recency_filter=search_recency_filter,
            search_context_size=search_context_size,
            stream=stream,
            response_format=response_format,
            conversation_history=conversation_history,
            system_prompt=system_prompt,
        )
        if not result.get("error"):
            if verbose and not stream:
                print(f" OK  ({result.get('elapsed',0)}s | "
                      f"${result.get('cost_usd',0):.5f} | "
                      f"{result.get('usage',{}).get('total_tokens',0)} tok)")
            return result
        if verbose:
            print(f" FAIL  {result['error']}")

    # ── Fallback: Exa + Tavily + Ollama ─────────────────────────────────────
    if verbose:
        print("  Fallback: Exa + Tavily + Ollama")

    inc_domains = [d for d in (search_domain_filter or []) if not d.startswith("-")]
    exc_domains = [d[1:] for d in (search_domain_filter or []) if d.startswith("-")]

    fallbacks = []
    if EXA_KEY:
        if verbose: print("  Exa...", end="", flush=True)
        r = search_exa(query, include_domains=inc_domains or None,
                        exclude_domains=exc_domains or None)
        fallbacks.append(r)
        if verbose: print(f" OK  ({r.get('elapsed',0)}s, {r.get('count',0)} results)")

    if TAVILY_KEY:
        if verbose: print("  Tavily...", end="", flush=True)
        r = search_tavily(
            query,
            topic  = "news" if search_recency_filter in ("day","week") else "general",
            days   = 7 if search_recency_filter == "week" else (1 if search_recency_filter == "day" else None),
            include_domains=inc_domains or None,
            exclude_domains=exc_domains or None,
        )
        fallbacks.append(r)
        if verbose: print(f" OK  ({r.get('elapsed',0)}s, {r.get('count',0)} results)")

    if not fallbacks:
        return {"source": "none",
                "error":  "No API keys set. Run: python search-engine.py --setup"}

    if verbose: print("  Ollama synthesis...", end="", flush=True)
    result = synthesize_with_ollama(query, fallbacks)
    if verbose: print(f" OK  ({result.get('elapsed',0)}s)")

    # Attach Tavily images if any
    for r in fallbacks:
        if r.get("images"):
            result["images"] = r["images"]
            break

    return result


# ===========================================================================
# STRUCTURED OUTPUT (JSON mode)
# ===========================================================================

def search_structured(query: str, schema: dict, model: str = "pro") -> dict:
    """
    Return a Perplexity-sourced JSON object matching your schema.

    Example:
        result = search_structured(
            "Polymarket vs Manifold: fees, volume, user count",
            schema={
                "name": "competitor_comparison",
                "schema": {
                    "type": "object",
                    "properties": {
                        "platforms": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name":   {"type": "string"},
                                    "fee":    {"type": "string"},
                                    "volume": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        )
    """
    return search_perplexity(
        query,
        model=model,
        response_format={"type": "json_schema", "json_schema": schema},
        temperature=0.0,
        return_citations=True,
        return_related_questions=False,
    )


# ===========================================================================
# AGENT QQ TOOL SPEC  (plug into agentqq-router-v2.py ToolCallEngine)
# ===========================================================================

SEARCH_TOOL_SPEC = {
    "name": "web_search",
    "description": (
        "Search the live web for current information, news, research, prices, and facts. "
        "Returns a cited answer with numbered source URLs, optional images, and follow-up questions. "
        "Use for: current events, competitor research, technical docs, market data, recent releases, "
        "crypto prices, anything that may have changed since training data cutoff."
    ),
    "parameters": {
        "query": {
            "type": "string",
            "description": "Specific search query. 'Arbitrum One gas April 2025' > 'gas fees'."
        },
        "model": {
            "type": "string",
            "enum": ["fast", "pro", "reasoning", "reasoning-pro", "deep"],
            "description": (
                "fast=quick lookup (sonar), pro=multi-step default (sonar-pro), "
                "reasoning=analysis+CoT, reasoning-pro=extended thinking, "
                "deep=full research 30-60s"
            ),
        },
        "recency": {
            "type": "string",
            "enum": ["day", "week", "month", "year"],
            "description": "Restrict to results published within this window.",
        },
        "domains": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Include or exclude domains. Prefix '-' to exclude: ['reddit.com', '-wikipedia.org']",
        },
        "return_images": {
            "type": "boolean",
            "description": "Return image results alongside the text answer.",
        },
        "context_size": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Search context depth for sonar-pro. high=more sources, higher cost.",
        },
    },
    "required": ["query"],
}


def execute_search_tool(params: dict) -> str:
    """Execute web_search tool call from Agent QQ ToolCallEngine."""
    query         = params.get("query", "")
    model         = params.get("model", "pro")
    recency       = params.get("recency")
    domains       = params.get("domains")
    ret_images    = params.get("return_images", False)
    ctx_size      = params.get("context_size", "medium")

    result = search(
        query,
        model=model,
        return_images=ret_images,
        return_related_questions=True,
        search_domain_filter=domains,
        search_recency_filter=recency,
        search_context_size=ctx_size,
        verbose=False,
    )

    if result.get("error"):
        return f"Search error: {result['error']}"

    output = result.get("answer", "")

    images = result.get("images", [])
    if images:
        output += "\n\n**Images:**\n"
        for img in images[:5]:
            iurl = img.get("image_url", img) if isinstance(img, dict) else img
            orig = img.get("origin_url", "")  if isinstance(img, dict) else ""
            output += f"- {iurl}" + (f"  (via {orig})" if orig else "") + "\n"

    citations = result.get("citations", [])
    if citations and "Sources:" not in output:
        output += "\n\n**Sources:**\n"
        for i, url in enumerate(citations[:8], 1):
            output += f"[{i}] {url}\n"

    related = result.get("related_questions", [])
    if related:
        output += "\n\n**Related:**\n"
        for q in related[:4]:
            output += f"* {q}\n"

    return output


# ===========================================================================
# DISPLAY
# ===========================================================================

def format_result(result: dict, show_images: bool = True, show_related: bool = True) -> str:
    lines   = []
    source  = result.get("source", "?")
    elapsed = result.get("elapsed", 0)
    cost    = result.get("cost_usd", 0)
    usage   = result.get("usage", {})

    meta = [f"Source: {source}", f"Time: {elapsed}s"]
    if cost:    meta.append(f"Cost: ${cost:.5f}")
    if usage.get("total_tokens"): meta.append(f"Tokens: {usage['total_tokens']:,}")
    lines.append(f"\n{'='*62}")
    lines.append("  " + " | ".join(meta))
    lines.append(f"{'='*62}\n")

    if result.get("error"):
        lines.append(f"ERROR: {result['error']}")
        return "\n".join(lines)

    answer = result.get("answer", "")
    if answer:
        lines.append(answer)

    images = result.get("images", [])
    if images and show_images:
        lines.append(f"\n{'─'*40}")
        lines.append(f"Images ({len(images)}):")
        for img in images[:6]:
            iurl = img.get("image_url", img) if isinstance(img, dict) else img
            orig = img.get("origin_url", "")  if isinstance(img, dict) else ""
            lines.append(f"  {iurl}")
            if orig: lines.append(f"    via {orig}")

    citations = result.get("citations", [])
    if citations and "Sources:" not in answer:
        lines.append(f"\n{'─'*40}")
        lines.append("Sources:")
        for i, url in enumerate(citations, 1):
            lines.append(f"  [{i}] {url}")

    related = result.get("related_questions", [])
    if related and show_related:
        lines.append(f"\n{'─'*40}")
        lines.append("Related questions:")
        for q in related[:5]:
            lines.append(f"  * {q}")

    lines.append(f"\n{'─'*40}")
    lines.append(f"Session: {_session_queries} queries | Total cost: ${_session_cost:.5f}")
    return "\n".join(lines)


# ===========================================================================
# SETUP GUIDE
# ===========================================================================

SETUP_GUIDE = """
Agent QQ Search — API Key Setup
================================
PRIMARY: Perplexity Sonar  (full capabilities, recommended)
  Sign up : https://perplexity.ai/api
  Pricing : sonar $1/M tokens | sonar-pro $3/M | sonar-deep $15/M
  Free    : $5 credit on signup
  Set     : $env:PPLX_API_KEY = "pplx-your-key"

FALLBACK 1: Exa  (neural semantic search)
  Sign up : https://exa.ai
  Free    : 1,000 searches/month
  Set     : $env:EXA_API_KEY = "your-exa-key"

FALLBACK 2: Tavily  (optimised for AI agents)
  Sign up : https://app.tavily.com
  Free    : 1,000 searches/month
  Set     : $env:TAVILY_API_KEY = "tvly-your-key"

DEEP READING: Firecrawl  (full page extraction)
  Sign up : https://firecrawl.dev
  Free    : 500 pages/month
  Set     : $env:FIRECRAWL_API_KEY = "fc-your-key"

Persist in F:\\AgentQQ\\START-V2.bat (before the python line):
  set PPLX_API_KEY=pplx-your-key-here
  set EXA_API_KEY=your-exa-key
  set TAVILY_API_KEY=tvly-your-key
  set FIRECRAWL_API_KEY=fc-your-key
"""

# ===========================================================================
# INTERACTIVE MODE
# ===========================================================================

def interactive_mode():
    print(f"\n{'='*62}")
    print("  Agent QQ Search Engine — Perplexity Sonar")
    keys = (f"PPLX={'OK' if PPLX_KEY else 'MISSING'}  "
            f"EXA={'OK' if EXA_KEY else 'missing'}  "
            f"TAVILY={'OK' if TAVILY_KEY else 'missing'}  "
            f"FIRECRAWL={'OK' if FIRECRAWL_KEY else 'missing'}")
    print(f"  {keys}")
    print(f"{'='*62}")
    print("  /fast /pro /reasoning /reasoning-pro /deep  — set model")
    print("  /recency day|week|month|year|off            — time filter")
    print("  /domain reddit.com | /domain off            — domain filter")
    print("  /images on|off  /stream  /clear  /history")
    print("  /setup  /keys  /models  quit")
    print(f"{'='*62}\n")

    state = {"model": "pro", "stream": False, "recency": None,
             "domains": None, "images": False, "related": True}
    history = []

    while True:
        try:
            raw = input("search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\nSession: {_session_queries} queries | Cost: ${_session_cost:.5f}")
            break

        if not raw: continue
        if raw.lower() in ("quit","exit","q"):
            print(f"Session: {_session_queries} queries | Cost: ${_session_cost:.5f}")
            break
        if raw == "/setup":       print(SETUP_GUIDE);  continue
        if raw == "/keys":        print(keys if 'keys' in dir() else "run --keys"); continue
        if raw == "/models":
            for k,(m,d,c,_) in PPLX_MODELS.items():
                print(f"  {k:<18} {m:<30} ${c:.3f}/1k  {d[:40]}")
            continue
        if raw == "/stream":      state["stream"] = not state["stream"]; print(f"Stream: {state['stream']}"); continue
        if raw == "/images on":   state["images"] = True;  continue
        if raw == "/images off":  state["images"] = False; continue
        if raw == "/clear":       history.clear(); print("History cleared"); continue
        if raw == "/history":     print(f"{len(history)//2} turns"); continue
        if raw.startswith("/recency "):
            v = raw.split(None,1)[1].strip()
            state["recency"] = v if v in ("day","week","month","year") else None
            print(f"Recency: {state['recency'] or 'off'}"); continue
        if raw.startswith("/domain "):
            v = raw.split(None,1)[1].strip()
            state["domains"] = None if v == "off" else [v]
            print(f"Domain filter: {state['domains']}"); continue

        # Model shortcuts
        for alias in PPLX_MODELS:
            if raw.lower().startswith(f"/{alias} "):
                state["model"] = alias
                raw = raw[len(alias)+2:]
                break

        result = search(
            raw,
            model=state["model"],
            return_images=state["images"],
            return_related_questions=state["related"],
            search_domain_filter=state["domains"],
            search_recency_filter=state["recency"],
            stream=state["stream"],
            conversation_history=history if history else None,
        )

        if not result.get("error") and not state["stream"]:
            history.append({"role": "user",      "content": raw})
            history.append({"role": "assistant",  "content": result.get("answer","")})
            if len(history) > 20: history = history[-20:]

        print(format_result(result, show_images=state["images"],
                             show_related=state["related"]))


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Agent QQ Search — Full Perplexity Sonar capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python search-engine.py "latest Arbitrum news"
          python search-engine.py --model deep "DeFi prediction market landscape"
          python search-engine.py --recency day "crypto news"
          python search-engine.py --domain reddit.com "MAD Gambit"
          python search-engine.py --no-domain wikipedia.org "blockchain"
          python search-engine.py --images --related "Perplexity competitors"
          python search-engine.py --stream --model reasoning "ZK rollups"
          python search-engine.py --context high --model pro "deep analysis"
        """),
    )
    p.add_argument("query",          nargs="?",          help="Search query")
    p.add_argument("--model","-m",   default="pro",
                   choices=list(PPLX_MODELS.keys()),    help="Model (default: pro)")
    p.add_argument("--recency","-r", choices=["day","week","month","year"])
    p.add_argument("--domain",       action="append", dest="domains",   metavar="DOMAIN")
    p.add_argument("--no-domain",    action="append", dest="exc_domains",metavar="DOMAIN")
    p.add_argument("--images",       action="store_true")
    p.add_argument("--no-related",   action="store_true")
    p.add_argument("--stream",       action="store_true")
    p.add_argument("--context",      choices=["low","medium","high"], default="medium")
    p.add_argument("--max-tokens",   type=int,   default=2048)
    p.add_argument("--temp",         type=float, default=0.2)
    p.add_argument("--json",         action="store_true")
    p.add_argument("--setup",        action="store_true")
    p.add_argument("--keys",         action="store_true")
    p.add_argument("--models",       action="store_true")
    args = p.parse_args()

    if args.setup:  print(SETUP_GUIDE); sys.exit(0)
    if args.keys:
        print(f"PPLX_API_KEY:      {'OK (set)' if PPLX_KEY else 'NOT SET  <- primary'}")
        print(f"EXA_API_KEY:       {'OK' if EXA_KEY else 'not set  (fallback 1)'}")
        print(f"TAVILY_API_KEY:    {'OK' if TAVILY_KEY else 'not set  (fallback 2)'}")
        print(f"FIRECRAWL_API_KEY: {'OK' if FIRECRAWL_KEY else 'not set  (deep reading)'}")
        sys.exit(0)
    if args.models:
        print(f"\n  {'Alias':<18} {'API model':<30} Cost/1k  Description")
        print("  " + "─"*80)
        for k,(m,d,c,sc) in PPLX_MODELS.items():
            print(f"  {k:<18} {m:<30} ${c:.3f}   {d[:42]}" + (" [ctx-size]" if sc else ""))
        sys.exit(0)

    if not args.query:
        interactive_mode()
        sys.exit(0)

    domain_filter = []
    if args.domains:     domain_filter += args.domains
    if args.exc_domains: domain_filter += [f"-{d}" for d in args.exc_domains]

    result = search(
        args.query,
        model=args.model,
        return_images=args.images,
        return_related_questions=not args.no_related,
        search_domain_filter=domain_filter or None,
        search_recency_filter=args.recency,
        search_context_size=args.context,
        stream=args.stream,
        max_tokens=args.max_tokens,
        temperature=args.temp,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    elif not args.stream:
        print(format_result(result,
                            show_images=args.images,
                            show_related=not args.no_related))
