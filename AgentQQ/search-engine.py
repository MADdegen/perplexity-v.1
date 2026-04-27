"""
Agent QQ — Search Engine Module
Perplexity Sonar API + Exa + Tavily fallback chain.
Delivers Perplexity-quality web-grounded answers with inline citations.

Usage:
    python search-engine.py "what is the latest on Arbitrum One?"
    python search-engine.py --deep "predict market trends for DeFi 2025"
    python search-engine.py --mode research "MAD Gambit prediction market competitors"

Requirements:
    pip install openai exa-py requests --break-system-packages
    Set env vars: PPLX_API_KEY, EXA_API_KEY (optional fallback), TAVILY_API_KEY (optional fallback)
"""

import os
import sys
import json
import time
import argparse
import re
from typing import Optional

# ─── CONFIG ─────────────────────────────────────────────────────────────────

PPLX_KEY     = os.getenv("PPLX_API_KEY", "")
EXA_KEY      = os.getenv("EXA_API_KEY", "")
TAVILY_KEY   = os.getenv("TAVILY_API_KEY", "")
FIRECRAWL_KEY= os.getenv("FIRECRAWL_API_KEY", "")

# Perplexity model tiers
PPLX_MODELS = {
    "fast":     "sonar",             # Fast web search, ~1-2s
    "pro":      "sonar-pro",         # Advanced, multi-step reasoning
    "reasoning":"sonar-reasoning",   # Extended reasoning + web
    "deep":     "sonar-deep-research" # Full deep research mode (~30-60s)
}

# ─── PERPLEXITY SONAR ENGINE ────────────────────────────────────────────────

def search_perplexity(query: str, mode: str = "pro", max_tokens: int = 2048) -> dict:
    """
    Core Perplexity Sonar search. Returns answer + citations.
    This is the famous Perplexity capability — web-grounded LLM with real sources.
    """
    if not PPLX_KEY:
        return {"error": "PPLX_API_KEY not set", "source": "perplexity"}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=PPLX_KEY, base_url="https://api.perplexity.ai")
    except ImportError:
        return {"error": "openai package not installed. Run: pip install openai --break-system-packages"}

    model = PPLX_MODELS.get(mode, PPLX_MODELS["pro"])

    # System prompt for max quality output — modeled after Perplexity Pro
    system = """You are a precise research assistant with real-time web access.
Always:
- Answer directly, citing sources inline with [1], [2], etc.
- Include only verified, current information
- Flag uncertainty explicitly
- Provide structured, scannable output with headers when answer is long
- End with a "Sources" section listing all cited URLs"""

    print(f"  🔍 Querying Perplexity {model}...", end="", flush=True)
    t0 = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query}
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        elapsed = time.time() - t0

        answer = response.choices[0].message.content
        citations = []

        # Extract citations from response object (Perplexity adds these)
        if hasattr(response, 'citations'):
            citations = response.citations or []
        elif hasattr(response.choices[0].message, 'citations'):
            citations = response.choices[0].message.citations or []

        # Also parse inline URLs from the answer text
        url_pattern = r'https?://[^\s\)\]>"]+'
        found_urls = re.findall(url_pattern, answer)
        if found_urls and not citations:
            citations = list(dict.fromkeys(found_urls))  # deduplicated

        print(f" ✅ ({elapsed:.1f}s)")
        return {
            "source": f"perplexity/{model}",
            "answer": answer,
            "citations": citations,
            "model": model,
            "tokens": response.usage.total_tokens if response.usage else None,
            "elapsed": elapsed
        }

    except Exception as e:
        elapsed = time.time() - t0
        print(f" ❌ ({elapsed:.1f}s)")
        return {"error": str(e), "source": f"perplexity/{model}"}


# ─── EXA SEARCH FALLBACK ────────────────────────────────────────────────────

def search_exa(query: str, num_results: int = 8) -> dict:
    """
    Exa neural search — semantic similarity, great for research and concepts.
    Returns full page text + highlights for synthesis.
    """
    if not EXA_KEY:
        return {"error": "EXA_API_KEY not set"}

    try:
        from exa_py import Exa
        exa = Exa(api_key=EXA_KEY)
    except ImportError:
        try:
            import requests
            # Fallback to raw API
            headers = {"x-api-key": EXA_KEY, "Content-Type": "application/json"}
            body = {
                "query": query,
                "numResults": num_results,
                "contents": {"text": {"maxCharacters": 1000}, "highlights": {"numSentences": 3}}
            }
            r = requests.post("https://api.exa.ai/search", headers=headers, json=body, timeout=15)
            data = r.json()
            results = data.get("results", [])
            return {
                "source": "exa",
                "results": [{"title": r.get("title"), "url": r.get("url"), "text": r.get("text", "")[:500]} for r in results],
                "count": len(results)
            }
        except Exception as e:
            return {"error": str(e), "source": "exa"}

    print(f"  🔍 Querying Exa...", end="", flush=True)
    t0 = time.time()
    try:
        results = exa.search_and_contents(
            query,
            num_results=num_results,
            use_autoprompt=True,
            text={"max_characters": 1500},
            highlights={"num_sentences": 3, "highlights_per_url": 2}
        )
        elapsed = time.time() - t0
        print(f" ✅ ({elapsed:.1f}s, {len(results.results)} results)")
        return {
            "source": "exa",
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "text": getattr(r, "text", "")[:800],
                    "highlights": getattr(r, "highlights", [])
                }
                for r in results.results
            ],
            "count": len(results.results),
            "elapsed": elapsed
        }
    except Exception as e:
        print(f" ❌")
        return {"error": str(e), "source": "exa"}


# ─── TAVILY SEARCH FALLBACK ─────────────────────────────────────────────────

def search_tavily(query: str, search_depth: str = "advanced") -> dict:
    """
    Tavily — purpose-built for AI agents, returns clean extracted content.
    Best for current events and news.
    """
    if not TAVILY_KEY:
        return {"error": "TAVILY_API_KEY not set"}

    try:
        import requests
        print(f"  🔍 Querying Tavily...", end="", flush=True)
        t0 = time.time()
        body = {
            "api_key": TAVILY_KEY,
            "query": query,
            "search_depth": search_depth,
            "max_results": 8,
            "include_answer": True,
            "include_raw_content": False
        }
        r = requests.post("https://api.tavily.com/search", json=body, timeout=20)
        data = r.json()
        elapsed = time.time() - t0
        print(f" ✅ ({elapsed:.1f}s)")
        return {
            "source": "tavily",
            "answer": data.get("answer", ""),
            "results": [
                {"title": res.get("title"), "url": res.get("url"), "content": res.get("content", "")[:500]}
                for res in data.get("results", [])
            ],
            "count": len(data.get("results", [])),
            "elapsed": elapsed
        }
    except Exception as e:
        print(f" ❌")
        return {"error": str(e), "source": "tavily"}


# ─── SYNTHESIS ENGINE (Ollama local LLM) ────────────────────────────────────

def synthesize_with_local_llm(query: str, search_results: list[dict]) -> str:
    """
    When Perplexity is unavailable, use local Ollama to synthesize
    search results from Exa/Tavily into a Perplexity-style answer.
    """
    try:
        import requests

        # Build context from search results
        context_parts = []
        citations = []
        for i, result in enumerate(search_results[:6], 1):
            if result.get("error"):
                continue
            if "results" in result:
                for j, r in enumerate(result["results"][:3], 1):
                    idx = len(citations) + 1
                    citations.append(r.get("url", ""))
                    text = r.get("text") or r.get("content", "")
                    title = r.get("title", f"Source {idx}")
                    context_parts.append(f"[{idx}] {title}\n{text[:400]}")
            elif "answer" in result and result["answer"]:
                context_parts.append(f"[Direct answer from {result['source']}]: {result['answer'][:600]}")

        context = "\n\n".join(context_parts)

        prompt = f"""You are a research assistant. Using ONLY the sources below, answer the query with inline citations [1], [2] etc.

QUERY: {query}

SOURCES:
{context}

Provide a comprehensive, well-structured answer with:
- Direct answer first
- Key findings with inline citations
- Any caveats or uncertainties
- Sources section at end"""

        print(f"  🤖 Synthesizing with local Ollama...", end="", flush=True)
        t0 = time.time()
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "gemma3:4b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 1000}
        }, timeout=120)
        elapsed = time.time() - t0
        print(f" ✅ ({elapsed:.1f}s)")

        answer = r.json().get("response", "")
        # Append citations
        if citations:
            answer += "\n\n**Sources:**\n"
            for i, url in enumerate(citations, 1):
                if url:
                    answer += f"[{i}] {url}\n"
        return answer

    except Exception as e:
        return f"Synthesis failed: {e}"


# ─── MAIN SEARCH PIPELINE ───────────────────────────────────────────────────

def search(query: str, mode: str = "pro", deep: bool = False, local_fallback: bool = True) -> dict:
    """
    Full Perplexity-class search pipeline:
    1. Try Perplexity Sonar API (best quality, cites live web)
    2. Fallback: Exa + Tavily → local Ollama synthesis
    """
    effective_mode = "deep" if deep else mode
    print(f"\n🔎 Agent QQ Search — mode: {effective_mode}")
    print(f"   Query: {query[:80]}{'...' if len(query) > 80 else ''}\n")

    # Primary: Perplexity Sonar
    if PPLX_KEY:
        result = search_perplexity(query, mode=effective_mode)
        if not result.get("error"):
            return result
        print(f"  ⚠️  Perplexity failed: {result['error']}")

    # Fallback: parallel Exa + Tavily
    print("  ↪ Falling back to Exa + Tavily + local synthesis")
    search_results = []

    if EXA_KEY:
        search_results.append(search_exa(query))
    if TAVILY_KEY:
        search_results.append(search_tavily(query))

    if not search_results:
        return {
            "source": "none",
            "error": "No search API keys configured. Set PPLX_API_KEY, EXA_API_KEY, or TAVILY_API_KEY.",
            "answer": None
        }

    if local_fallback:
        answer = synthesize_with_local_llm(query, search_results)
        citations = []
        for r in search_results:
            if "results" in r:
                citations.extend([res.get("url", "") for res in r["results"]])
        return {
            "source": "exa+tavily+ollama",
            "answer": answer,
            "citations": [c for c in citations if c][:10],
            "raw_results": search_results
        }

    return {"source": "exa+tavily", "raw_results": search_results}


# ─── AGENT QQ TOOL SPEC ─────────────────────────────────────────────────────

SEARCH_TOOL_SPEC = {
    "name": "web_search",
    "description": "Search the live web for current information, news, research, and facts. Returns cited, synthesized answers. Use for: current events, market data, competitor research, technical docs, pricing, recent releases.",
    "parameters": {
        "query": {"type": "string", "description": "Search query — be specific for better results"},
        "mode": {
            "type": "string",
            "enum": ["fast", "pro", "reasoning", "deep"],
            "description": "Search depth. fast=quick lookup, pro=default, reasoning=analysis tasks, deep=comprehensive research"
        },
        "focus": {
            "type": "string",
            "enum": ["web", "news", "academic", "finance", "code"],
            "description": "Optional domain focus for targeted search"
        }
    },
    "required": ["query"]
}


def execute_search_tool(params: dict) -> str:
    """Execute web_search tool call from Agent QQ tool engine."""
    query = params.get("query", "")
    mode = params.get("mode", "pro")
    deep = (mode == "deep")

    result = search(query, mode=mode, deep=deep)

    if result.get("error"):
        return f"Search error: {result['error']}"

    output = result.get("answer", "")
    citations = result.get("citations", [])

    if citations and "Sources:" not in output and "sources:" not in output.lower():
        output += "\n\n**Sources:**\n"
        for i, url in enumerate(citations[:8], 1):
            output += f"[{i}] {url}\n"

    return output


# ─── AGENT QQ ROUTER PATCH INSTRUCTIONS ────────────────────────────────────
"""
To add web_search to agentqq-router-v2.py:

1. Import at top:
   from search_engine import execute_search_tool, SEARCH_TOOL_SPEC

2. In AgentQQv2._register_default_tools(), add:
   self.tool_engine.register_tool(ToolSpec(
       name=SEARCH_TOOL_SPEC["name"],
       description=SEARCH_TOOL_SPEC["description"],
       parameters=SEARCH_TOOL_SPEC["parameters"],
       required=SEARCH_TOOL_SPEC["required"]
   ))

3. In AgentQQv2._execute_tool(), add case:
   elif tool_name == "web_search":
       return execute_search_tool(params)

4. Add SEARCH to TaskType enum:
   SEARCH = "search"

5. Add MODE_INSTRUCTIONS["SEARCH"]:
   "SEARCH": '''Think step by step:
   1. Identify what's being asked — fact, opinion, current event, research?
   2. Formulate a precise search query
   3. Execute web_search tool
   4. Synthesize results — distinguish facts vs inference
   5. Cite every claim with [source number]
   6. Note freshness — when was this published?'''

6. In SharedExpert routing prompt, add:
   SEARCH: live web questions, current events, prices, news, competitor research
"""


# ─── CLI ─────────────────────────────────────────────────────────────────────

def format_output(result: dict) -> str:
    """Format search result for terminal display."""
    lines = []
    source = result.get("source", "unknown")
    lines.append(f"\n{'='*60}")
    lines.append(f"Source: {source}")
    if result.get("elapsed"):
        lines.append(f"Time: {result['elapsed']:.1f}s")
    lines.append(f"{'='*60}\n")

    if result.get("error"):
        lines.append(f"❌ Error: {result['error']}")
        return "\n".join(lines)

    answer = result.get("answer", "")
    if answer:
        lines.append(answer)
    else:
        # Raw results from Exa/Tavily
        for r in result.get("raw_results", []):
            if "results" in r:
                lines.append(f"\n[{r['source'].upper()} Results]")
                for res in r["results"][:5]:
                    lines.append(f"• {res.get('title', 'No title')}")
                    lines.append(f"  {res.get('url', '')}")
                    text = res.get("text") or res.get("content", "")
                    if text:
                        lines.append(f"  {text[:200]}...")
                    lines.append("")

    citations = result.get("citations", [])
    if citations and "Sources:" not in answer:
        lines.append("\n**Sources:**")
        for i, url in enumerate(citations[:8], 1):
            lines.append(f"[{i}] {url}")

    return "\n".join(lines)


def show_setup_guide():
    print("""
╔══════════════════════════════════════════════════════╗
║       Agent QQ Search Engine — API Key Setup         ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  OPTION 1: Perplexity (BEST — exact same capability) ║
║  • Sign up: https://perplexity.ai/api                ║
║  • Plans from $20/mo (5M tokens)                     ║
║  • set PPLX_API_KEY=pplx-xxxx                        ║
║                                                      ║
║  OPTION 2: Exa (excellent neural search)             ║
║  • Sign up: https://exa.ai                           ║
║  • 1000 free searches/month                          ║
║  • set EXA_API_KEY=your-key                          ║
║                                                      ║
║  OPTION 3: Tavily (great for current events)         ║
║  • Sign up: https://app.tavily.com                   ║
║  • 1000 free searches/month                          ║
║  • set TAVILY_API_KEY=tvly-xxxx                      ║
║                                                      ║
║  OPTION 2+3: Exa + Tavily + Ollama synthesis         ║
║  (Free fallback — uses local gemma3:4b to combine)   ║
║                                                      ║
╚══════════════════════════════════════════════════════╝

Set in Windows:
  $env:PPLX_API_KEY = "pplx-your-key-here"
  $env:EXA_API_KEY  = "your-exa-key"

Or add to F:\\AgentQQ\\START-V2.bat before the python line:
  set PPLX_API_KEY=pplx-your-key-here
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent QQ Search Engine — Perplexity-class web search")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--mode", choices=["fast", "pro", "reasoning", "deep"], default="pro",
                        help="Search mode (default: pro)")
    parser.add_argument("--deep", action="store_true", help="Enable deep research mode (~30-60s)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--setup", action="store_true", help="Show API key setup guide")
    parser.add_argument("--keys", action="store_true", help="Show which API keys are configured")
    args = parser.parse_args()

    if args.setup:
        show_setup_guide()
        sys.exit(0)

    if args.keys:
        print("Configured API keys:")
        print(f"  PPLX_API_KEY:     {'✅ set' if PPLX_KEY else '❌ not set'}")
        print(f"  EXA_API_KEY:      {'✅ set' if EXA_KEY else '❌ not set'}")
        print(f"  TAVILY_API_KEY:   {'✅ set' if TAVILY_KEY else '❌ not set'}")
        print(f"  FIRECRAWL_API_KEY:{'✅ set' if FIRECRAWL_KEY else '❌ not set'}")
        sys.exit(0)

    if not args.query:
        # Interactive mode
        print("Agent QQ Search Engine — Perplexity-class web search")
        print("Commands: /deep <q>  /fast <q>  /reasoning <q>  /setup  /keys  quit\n")

        while True:
            try:
                user_input = input("search> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if user_input == "/setup":
                show_setup_guide()
                continue
            if user_input == "/keys":
                print(f"PPLX={'set' if PPLX_KEY else 'missing'}  EXA={'set' if EXA_KEY else 'missing'}  TAVILY={'set' if TAVILY_KEY else 'missing'}")
                continue

            mode = "pro"
            query = user_input
            deep = False

            for cmd, m in [("/deep ", "deep"), ("/reasoning ", "reasoning"), ("/fast ", "fast"), ("/pro ", "pro")]:
                if user_input.lower().startswith(cmd):
                    mode = m
                    query = user_input[len(cmd):]
                    deep = (m == "deep")
                    break

            result = search(query, mode=mode, deep=deep)
            print(format_output(result))

    else:
        result = search(args.query, mode=args.mode, deep=args.deep or args.mode == "deep")
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(format_output(result))
