/**
 * lib/tools/perplexity-search.ts
 *
 * Full Perplexity Sonar API tool for the perplexity-v.1 Next.js app.
 * Drop into lib/tools/ and export from lib/tools/index.ts.
 *
 * Every documented Perplexity parameter is wired:
 *   - 5 models: sonar, sonar-pro, sonar-reasoning, sonar-reasoning-pro, sonar-deep-research
 *   - return_citations      (always on)
 *   - return_images
 *   - return_related_questions
 *   - search_domain_filter  (include / exclude domains)
 *   - search_recency_filter (day | week | month | year)
 *   - web_search_options    { search_context_size: low | medium | high } (sonar-pro only)
 *   - stream                (SSE — wired into Vercel AI SDK dataStream)
 *   - response_format       (json_schema for structured output)
 */

import { tool } from 'ai';
import { z } from 'zod';
import { serverEnv } from '@/env/server';
import { UIMessageStreamWriter } from 'ai';
import { ChatMessage } from '../types';

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

const PPLX_MODELS = {
  'sonar':                { ctx: 127_000,  costPer1k: 0.001, supportsCtxSize: false },
  'sonar-pro':            { ctx: 200_000,  costPer1k: 0.003, supportsCtxSize: true  },
  'sonar-reasoning':      { ctx: 127_000,  costPer1k: 0.005, supportsCtxSize: false },
  'sonar-reasoning-pro':  { ctx: 127_000,  costPer1k: 0.008, supportsCtxSize: false },
  'sonar-deep-research':  { ctx: 128_000,  costPer1k: 0.015, supportsCtxSize: false },
} as const;

type PplxModel = keyof typeof PPLX_MODELS;

// ---------------------------------------------------------------------------
// Raw Perplexity API call (non-streaming)
// ---------------------------------------------------------------------------

interface PplxResponse {
  answer:            string;
  citations:         string[];
  images:            Array<{ image_url: string; origin_url: string }>;
  related_questions: string[];
  model:             string;
  usage: {
    prompt_tokens:     number;
    completion_tokens: number;
    total_tokens:      number;
    citation_tokens?:  number;
  };
}

async function callPerplexity(params: {
  query:                    string;
  model:                    PplxModel;
  systemPrompt?:            string;
  conversationHistory?:     Array<{ role: 'user' | 'assistant'; content: string }>;
  maxTokens?:               number;
  temperature?:             number;
  topP?:                    number;
  topK?:                    number;
  presencePenalty?:         number;
  frequencyPenalty?:        number;
  returnImages?:            boolean;
  returnRelatedQuestions?:  boolean;
  searchDomainFilter?:      string[];
  searchRecencyFilter?:     'day' | 'week' | 'month' | 'year';
  searchContextSize?:       'low' | 'medium' | 'high';
  responseFormat?:          Record<string, unknown>;
}): Promise<PplxResponse> {
  const apiKey = serverEnv.PERPLEXITY_API_KEY;
  if (!apiKey) throw new Error('PERPLEXITY_API_KEY not configured');

  const modelInfo = PPLX_MODELS[params.model];

  const defaultSystem =
    'You are a precise research assistant with real-time web access. ' +
    'Answer directly, citing sources inline as [1], [2], etc. ' +
    'Include only verified, current information. Flag uncertainty explicitly.';

  const messages: Array<{ role: string; content: string }> = [
    { role: 'system', content: params.systemPrompt ?? defaultSystem },
    ...(params.conversationHistory ?? []),
    { role: 'user', content: params.query },
  ];

  // Build body with every Perplexity-specific param
  const body: Record<string, unknown> = {
    model:                    params.model,
    messages,
    max_tokens:               params.maxTokens               ?? 2048,
    temperature:              params.temperature              ?? 0.2,
    top_p:                    params.topP                    ?? 0.9,
    top_k:                    params.topK                    ?? 0,
    presence_penalty:         params.presencePenalty          ?? 0.0,
    frequency_penalty:        params.frequencyPenalty         ?? 1.0,
    return_citations:         true,
    return_images:            params.returnImages             ?? false,
    return_related_questions: params.returnRelatedQuestions   ?? true,
    stream:                   false,
  };

  if (params.searchDomainFilter?.length) {
    body.search_domain_filter = params.searchDomainFilter;
  }
  if (params.searchRecencyFilter) {
    body.search_recency_filter = params.searchRecencyFilter;
  }
  if (modelInfo.supportsCtxSize && params.searchContextSize) {
    body.web_search_options = { search_context_size: params.searchContextSize };
  }
  if (params.responseFormat) {
    body.response_format = params.responseFormat;
  }

  const res = await fetch('https://api.perplexity.ai/chat/completions', {
    method:  'POST',
    headers: {
      Authorization:  `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Perplexity API error ${res.status}: ${err.slice(0, 200)}`);
  }

  const data = await res.json();
  const msg  = data.choices?.[0]?.message ?? {};

  return {
    answer:            msg.content ?? '',
    citations:         data.citations         ?? [],
    images:            data.images            ?? [],
    related_questions: data.related_questions ?? [],
    model:             params.model,
    usage:             data.usage             ?? {},
  };
}

// ---------------------------------------------------------------------------
// SSE streaming variant (writes directly to Vercel AI SDK dataStream)
// ---------------------------------------------------------------------------

async function streamPerplexity(params: {
  query:                   string;
  model:                   PplxModel;
  systemPrompt?:           string;
  searchDomainFilter?:     string[];
  searchRecencyFilter?:    'day' | 'week' | 'month' | 'year';
  searchContextSize?:      'low' | 'medium' | 'high';
  returnImages?:           boolean;
  returnRelatedQuestions?: boolean;
  maxTokens?:              number;
  temperature?:            number;
  dataStream?:             UIMessageStreamWriter<ChatMessage>;
}): Promise<PplxResponse> {
  const apiKey = serverEnv.PERPLEXITY_API_KEY;
  if (!apiKey) throw new Error('PERPLEXITY_API_KEY not configured');

  const modelInfo = PPLX_MODELS[params.model];

  const body: Record<string, unknown> = {
    model:                    params.model,
    messages: [
      { role: 'system', content: params.systemPrompt ??
          'You are a precise research assistant with real-time web access. Answer with inline citations [1], [2].' },
      { role: 'user', content: params.query },
    ],
    max_tokens:               params.maxTokens    ?? 2048,
    temperature:              params.temperature  ?? 0.2,
    return_citations:         true,
    return_images:            params.returnImages ?? false,
    return_related_questions: params.returnRelatedQuestions ?? true,
    stream:                   true,
  };

  if (params.searchDomainFilter?.length)  body.search_domain_filter = params.searchDomainFilter;
  if (params.searchRecencyFilter)         body.search_recency_filter = params.searchRecencyFilter;
  if (modelInfo.supportsCtxSize && params.searchContextSize)
    body.web_search_options = { search_context_size: params.searchContextSize };

  const res = await fetch('https://api.perplexity.ai/chat/completions', {
    method:  'POST',
    headers: {
      Authorization:  `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      Accept:         'text/event-stream',
    },
    body: JSON.stringify(body),
  });

  if (!res.ok || !res.body) {
    const err = await res.text();
    throw new Error(`Perplexity stream error ${res.status}: ${err.slice(0, 200)}`);
  }

  let fullText        = '';
  let citations:  string[]                                       = [];
  let images:     Array<{ image_url: string; origin_url: string }> = [];
  let related:    string[]                                       = [];
  let usage:      Record<string, number>                         = {};

  const reader  = res.body.getReader();
  const decoder = new TextDecoder();
  let   buffer  = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';                // keep incomplete last line

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const payload = line.slice(6);
      if (payload === '[DONE]') continue;
      try {
        const chunk   = JSON.parse(payload);
        const content = chunk.choices?.[0]?.delta?.content ?? '';
        if (content) {
          fullText += content;
          // Forward incremental text to the Vercel AI SDK data stream
          params.dataStream?.write({ type: 'text', value: content } as any);
        }
        if (chunk.citations)         citations = chunk.citations;
        if (chunk.images)            images    = chunk.images;
        if (chunk.related_questions) related   = chunk.related_questions;
        if (chunk.usage)             usage     = chunk.usage;
      } catch {
        // skip malformed SSE lines
      }
    }
  }

  return { answer: fullText, citations, images, related_questions: related,
           model: params.model, usage: usage as any };
}

// ---------------------------------------------------------------------------
// Vercel AI SDK tool  (drop-in for the scira web app)
// ---------------------------------------------------------------------------

export function perplexitySearchTool(
  dataStream?: UIMessageStreamWriter<ChatMessage>,
  enableStreaming = false,
) {
  return tool({
    description:
      'Search the live web using Perplexity Sonar for accurate, cited answers with ' +
      'optional images and related follow-up questions. ' +
      'Use for current events, market data, technical docs, competitor research, prices, ' +
      'and anything that may have changed since training cutoff.',

    inputSchema: z.object({
      query: z
        .string()
        .describe('Specific search query. More specific = better results.'),

      model: z
        .enum(['sonar', 'sonar-pro', 'sonar-reasoning', 'sonar-reasoning-pro', 'sonar-deep-research'])
        .optional()
        .default('sonar-pro')
        .describe(
          'sonar=fast, sonar-pro=multi-step (default), sonar-reasoning=CoT+web, ' +
          'sonar-reasoning-pro=extended thinking, sonar-deep-research=30-60s comprehensive',
        ),

      searchRecencyFilter: z
        .enum(['day', 'week', 'month', 'year'])
        .optional()
        .describe('Restrict results to those published within this time window.'),

      searchDomainFilter: z
        .array(z.string())
        .optional()
        .describe('Include or exclude domains. Prefix "-" to exclude: ["reddit.com", "-wikipedia.org"]'),

      returnImages: z
        .boolean()
        .optional()
        .default(false)
        .describe('Return image results alongside the text answer.'),

      returnRelatedQuestions: z
        .boolean()
        .optional()
        .default(true)
        .describe('Return suggested follow-up questions.'),

      searchContextSize: z
        .enum(['low', 'medium', 'high'])
        .optional()
        .default('medium')
        .describe('Context depth for sonar-pro (high = more sources, higher cost).'),

      maxTokens: z
        .number()
        .int()
        .min(256)
        .max(8192)
        .optional()
        .default(2048),
    }),

    execute: async ({
      query,
      model = 'sonar-pro',
      searchRecencyFilter,
      searchDomainFilter,
      returnImages = false,
      returnRelatedQuestions = true,
      searchContextSize = 'medium',
      maxTokens = 2048,
    }) => {
      try {
        // Write a status annotation to the data stream
        dataStream?.write({
          type:    'data',
          value:   [{ type: 'pplx-status', status: 'searching', model }],
        } as any);

        const apiParams = {
          query,
          model:                   model as PplxModel,
          returnImages,
          returnRelatedQuestions,
          searchDomainFilter,
          searchRecencyFilter,
          searchContextSize,
          maxTokens,
        };

        const result = enableStreaming
          ? await streamPerplexity({ ...apiParams, dataStream })
          : await callPerplexity(apiParams);

        // Notify data stream that Perplexity results arrived
        dataStream?.write({
          type:  'data',
          value: [{
            type:      'pplx-results',
            citations: result.citations,
            images:    result.images,
            related:   result.related_questions,
            model:     result.model,
            usage:     result.usage,
          }],
        } as any);

        return {
          answer:            result.answer,
          citations:         result.citations,
          images:            result.images,
          related_questions: result.related_questions,
          model:             result.model,
          usage:             result.usage,
          // Formatted markdown for the LLM to continue reasoning with
          formatted: formatForLLM(result),
        };

      } catch (err: any) {
        dataStream?.write({
          type:  'data',
          value: [{ type: 'pplx-error', error: err.message }],
        } as any);
        return {
          error:   err.message,
          answer:  '',
          citations:         [],
          images:            [],
          related_questions: [],
        };
      }
    },
  });
}

// ---------------------------------------------------------------------------
// Structured JSON output helper
// ---------------------------------------------------------------------------

export async function perplexityStructured<T = unknown>(
  query:      string,
  schema:     Record<string, unknown>,
  model:      PplxModel = 'sonar-pro',
): Promise<{ data: T; citations: string[] }> {
  const result = await callPerplexity({
    query,
    model,
    responseFormat: { type: 'json_schema', json_schema: schema },
    temperature:    0.0,
    returnRelatedQuestions: false,
  });

  try {
    const data = JSON.parse(result.answer) as T;
    return { data, citations: result.citations };
  } catch {
    throw new Error(`Perplexity returned invalid JSON: ${result.answer.slice(0, 200)}`);
  }
}

// ---------------------------------------------------------------------------
// Format result for LLM continued reasoning
// ---------------------------------------------------------------------------

function formatForLLM(result: PplxResponse): string {
  let md = result.answer;

  if (result.citations.length && !md.includes('Sources:')) {
    md += '\n\n**Sources:**\n';
    result.citations.forEach((url, i) => { md += `[${i + 1}] ${url}\n`; });
  }

  if (result.images.length) {
    md += '\n\n**Images:**\n';
    result.images.slice(0, 4).forEach(img => {
      md += `- ${img.image_url}${img.origin_url ? ` (via ${img.origin_url})` : ''}\n`;
    });
  }

  if (result.related_questions.length) {
    md += '\n\n**Related questions:**\n';
    result.related_questions.slice(0, 4).forEach(q => { md += `- ${q}\n`; });
  }

  return md;
}
