import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createAmazonBedrock } from '@ai-sdk/amazon-bedrock';
import { createAnthropic } from '@ai-sdk/anthropic';
import { createAzure } from '@ai-sdk/azure';
import { createCohere } from '@ai-sdk/cohere';
import { createGateway, type GatewayProviderSettings } from '@ai-sdk/gateway';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { createVertex } from '@ai-sdk/google-vertex';
import { createGroq } from '@ai-sdk/groq';
import { createHuggingFace } from '@ai-sdk/huggingface';
import { createMistral } from '@ai-sdk/mistral';
import { createOpenAI } from '@ai-sdk/openai';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { createXai } from '@ai-sdk/xai';
import { generateText, type LanguageModel } from 'ai';
import { describe, expect, it, vi } from 'vitest';

const offline = { apiKey: 'offline-fixture-key' };
const models: [string, LanguageModel][] = [
  ['bedrock', createAmazonBedrock({ region: 'us-east-1' })('offline-model')],
  ['anthropic', createAnthropic(offline)('offline-model')],
  ['azure', createAzure({ ...offline, resourceName: 'offline' })('offline-model')],
  ['cohere', createCohere(offline)('command-r-plus')],
  ['gateway', createGateway(offline)('openai/offline-model')],
  ['google', createGoogleGenerativeAI(offline)('offline-model')],
  ['vertex', createVertex({ project: 'offline', location: 'us-central1' })('offline-model')],
  ['groq', createGroq(offline)('offline-model')],
  ['huggingface', createHuggingFace(offline)('offline-model')],
  ['mistral', createMistral(offline)('offline-model')],
  ['openai', createOpenAI(offline)('offline-model')],
  [
    'openai-compatible',
    createOpenAICompatible({ ...offline, name: 'offline', baseURL: 'https://offline.invalid' })('offline-model'),
  ],
  ['xai', createXai(offline)('offline-model')],
];

function reply(content: object[], finishReason = 'stop') {
  return new Response(
    JSON.stringify({
      content,
      finishReason: { unified: finishReason, raw: finishReason },
      usage: {
        inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
        outputTokens: { total: 1, text: 1, reasoning: 0 },
      },
      warnings: [],
    }),
    { headers: { 'content-type': 'application/json' } }
  );
}

interface Consumer {
  ToolCallingEngine: {
    prototype: {
      runWithVercelAI: (
        this: { productData: Record<string, never> },
        prompt: string,
        options?: GatewayProviderSettings
      ) => Promise<string>;
    };
  };
}

const consumerURL = pathToFileURL(path.resolve(import.meta.dirname, '../../../skyyrose/build/tool-calling.js')).href;
const consumer = (await import(consumerURL)) as Consumer;
const invoke = (fetch: typeof globalThis.fetch, apiKey = 'offline-fixture-key') =>
  consumer.ToolCallingEngine.prototype.runWithVercelAI.call({ productData: {} }, 'Find the offline fixture', {
    apiKey,
    baseURL: 'https://offline.invalid/v4/ai',
    fetch,
  });

describe('coordinated AI SDK providers', () => {
  it('imports the actual ESM entrypoint under Node without running the CLI or contacting a provider', () => {
    const script =
      "globalThis.fetch = () => { throw new Error('Network forbidden during import'); }; const entry = await import(process.argv[1]); process.stdout.write(JSON.stringify(Object.keys(entry).sort()));";
    const output = execFileSync(process.execPath, ['--input-type=module', '-e', script, '--', consumerURL], {
      encoding: 'utf8',
    });
    expect(JSON.parse(output)).toEqual(
      [
        'GEMINI_FUNCTION_DECLARATIONS',
        'OPENAI_TOOLS',
        'ToolCallingEngine',
        'VALID_COLLECTIONS',
        'buildVercelTools',
        'executeTool',
      ].sort()
    );
  });

  it.each(models)('%s exposes a core-compatible provider-v4 language model', (_name, model) => {
    expect(typeof model).toBe('object');
    expect(model).toMatchObject({
      specificationVersion: 'v4',
      doGenerate: expect.any(Function),
      doStream: expect.any(Function),
    });
  });

  it('sends Gateway v4 text requests through the injected transport', async () => {
    const requests: Request[] = [];
    const fetch = vi.fn<typeof globalThis.fetch>(async (input, init) => {
      requests.push(new Request(input, init));
      return reply([{ type: 'text', text: 'Offline text' }]);
    });
    const gateway = createGateway({ ...offline, baseURL: 'https://offline.invalid/v4/ai', fetch });
    const result = await generateText({
      model: gateway('openai/offline-model'),
      prompt: 'Offline prompt',
      maxRetries: 0,
    });
    expect(result.text).toBe('Offline text');
    expect(requests).toHaveLength(1);
    expect(requests[0]!.url).toBe('https://offline.invalid/v4/ai/language-model');
    expect(requests[0]!.headers.get('ai-language-model-specification-version')).toBe('4');
    expect(await requests[0]!.json()).toMatchObject({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Offline prompt' }] }],
    });
  });

  it('runs the actual consumer tool loop and serializes the real search result', async () => {
    const requests: Request[] = [];
    const fetch = vi.fn<typeof globalThis.fetch>(async (input, init) => {
      requests.push(new Request(input, init));
      if (requests.length > 2) throw new Error('Unexpected extra generation');
      return requests.length === 1
        ? reply(
            [
              {
                type: 'tool-call',
                toolCallId: 'offline-call',
                toolName: 'search_products',
                input: JSON.stringify({ query: 'offline fixture' }),
              },
            ],
            'tool-calls'
          )
        : reply([{ type: 'text', text: 'No matching fixture' }]);
    });
    expect(await invoke(fetch)).toBe('No matching fixture');
    expect(requests).toHaveLength(2);
    expect(requests[0]!.headers.get('authorization')).toBe('Bearer offline-fixture-key');
    const first = await requests[0]!.json();
    expect(first.tools).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: 'search_products', inputSchema: expect.objectContaining({ type: 'object' }) }),
      ])
    );
    const second = await requests[1]!.json();
    expect(second.prompt).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          role: 'tool',
          content: expect.arrayContaining([
            expect.objectContaining({
              type: 'tool-result',
              toolName: 'search_products',
              toolCallId: 'offline-call',
              output: expect.objectContaining({ type: 'json', value: expect.objectContaining({ found: 0 }) }),
            }),
          ]),
        }),
      ])
    );
  });

  it('stops the actual consumer after five tool-call steps', async () => {
    let calls = 0;
    const fetch = vi.fn<typeof globalThis.fetch>(async () => {
      if (++calls > 5) throw new Error('Consumer exceeded its tool-loop limit');
      return reply(
        [
          {
            type: 'tool-call',
            toolCallId: `offline-${calls}`,
            toolName: 'search_products',
            input: '{"query":"offline"}',
          },
        ],
        'tool-calls'
      );
    });
    expect(await invoke(fetch)).toBe('[Vercel AI] No text response.');
    expect(calls).toBe(5);
  });

  it('does not mistake a direct OpenAI environment key for Gateway authorization', () => {
    const script =
      "globalThis.fetch = () => { throw new Error('Network forbidden'); }; const { ToolCallingEngine } = await import(process.argv[1]); const result = await ToolCallingEngine.prototype.runWithVercelAI.call({productData:{}}, 'Offline prompt', {fetch: globalThis.fetch}); process.stdout.write(result);";
    const output = execFileSync(process.execPath, ['--input-type=module', '-e', script, '--', consumerURL], {
      encoding: 'utf8',
      env: { ...process.env, AI_GATEWAY_API_KEY: '', OPENAI_API_KEY: 'offline-direct-key' },
    });
    expect(output).toContain('requires AI_GATEWAY_API_KEY');
  });

  it('does not make a request when Gateway authorization is explicitly absent', async () => {
    const fetch = vi.fn<typeof globalThis.fetch>(async () => {
      throw new Error('No transport allowed');
    });
    expect(await invoke(fetch, '')).toContain('requires AI_GATEWAY_API_KEY');
    expect(fetch).not.toHaveBeenCalled();
  });
});
