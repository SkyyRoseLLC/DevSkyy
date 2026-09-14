'use client';

import { useState, type FormEvent } from 'react';
import { Code2, ExternalLink, FileSearch, Loader2, ShieldCheck } from 'lucide-react';

import { TopBar } from '@/components/console/TopBar';
import { ConsoleCard } from '@/components/console/Card';
import { ApiError, api } from '@/lib/api';

const DEFAULT_SCHEMA = JSON.stringify(
  {
    type: 'object',
    properties: {
      product_summary: { type: 'string' },
      how_it_works: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            step: { type: 'string' },
            description: { type: 'string' },
            supporting_url: { type: 'string' },
          },
          required: ['step', 'description'],
        },
      },
      inputs_and_outputs: {
        type: 'object',
        properties: {
          accepted_inputs: { type: 'array', items: { type: 'string' } },
          generated_outputs: { type: 'array', items: { type: 'string' } },
        },
      },
      supported_models_and_tools: { type: 'array', items: { type: 'string' } },
      integrations: { type: 'array', items: { type: 'string' } },
      production_considerations: { type: 'array', items: { type: 'string' } },
    },
    required: ['product_summary', 'how_it_works'],
  },
  null,
  2,
);

function errorMessage(error: unknown): string {
  if (ApiError.isApiError(error)) {
    if (error.status === 401 || error.status === 403) {
      return 'This workspace requires an administrator or developer session.';
    }
    if (error.status === 429) {
      return 'The crawl budget is temporarily exhausted. Wait a moment before trying again.';
    }
    return error.message;
  }
  if (error instanceof DOMException && error.name === 'AbortError') {
    return 'The extraction took longer than three minutes. Try a smaller crawl scope.';
  }
  return error instanceof Error ? error.message : 'The extraction could not be completed.';
}

export default function WebExtractionPage() {
  const [url, setUrl] = useState('https://higgsfield.com');
  const [schemaText, setSchemaText] = useState(DEFAULT_SCHEMA);
  const [followSubdomains, setFollowSubdomains] = useState(true);
  const [maxPages, setMaxPages] = useState(25);
  const [maxDepth, setMaxDepth] = useState(2);
  const [parsePdfs, setParsePdfs] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isExtracting, setIsExtracting] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setResult(null);

    let schema: Record<string, unknown>;
    try {
      const parsed: unknown = JSON.parse(schemaText);
      if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') {
        throw new Error('The schema must be a JSON object.');
      }
      schema = parsed as Record<string, unknown>;
    } catch (parseError) {
      setError(parseError instanceof Error ? parseError.message : 'Enter valid JSON Schema.');
      return;
    }

    setIsExtracting(true);
    try {
      const response = await api.contextDev.extract({
        url: url.trim(),
        schema,
        follow_subdomains: followSubdomains,
        max_pages: maxPages,
        max_depth: maxDepth,
        parse_pdfs: parsePdfs,
      });
      setResult(response.data);
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setIsExtracting(false);
    }
  }

  return (
    <>
      <TopBar title="Web Extraction" />
      <div className="max-w-[1440px] px-9 py-8">
        <div className="mb-7 flex flex-wrap items-end justify-between gap-5">
          <div>
            <div className="mb-2 flex items-center gap-2 text-[#D4AF37]">
              <FileSearch className="h-5 w-5" />
              <span className="font-mono text-[10px] uppercase tracking-[0.2em]">Context.dev · Operator tool</span>
            </div>
            <h2 className="font-semibold text-3xl text-white" style={{ fontFamily: 'var(--font-cinzel)' }}>
              Grounded website intelligence
            </h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-[#A0A0A0]">
              Define the JSON Schema your pipeline needs, then extract only facts grounded in the crawled site.
              Results stay in this operator session; this action does not write to production data.
            </p>
          </div>
          <a
            href="https://docs.context.dev/api-reference/web-extraction/extract"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 border border-white/10 px-3 py-2 font-mono text-[10px] uppercase tracking-[0.15em] text-[#C8C8CE] transition-colors hover:bg-white/5"
          >
            Extraction reference <ExternalLink className="h-3.5 w-3.5" />
          </a>
        </div>

        <div className="grid gap-[18px] xl:grid-cols-[minmax(0,1fr)_minmax(430px,0.9fr)]">
          <ConsoleCard className="p-6">
            <form onSubmit={submit} className="space-y-6">
              <div>
                <label htmlFor="crawl-url" className="font-mono text-[10px] uppercase tracking-[0.16em] text-[#B3B3B3]">
                  Starting URL
                </label>
                <input
                  id="crawl-url"
                  required
                  type="url"
                  value={url}
                  onChange={(event) => setUrl(event.target.value)}
                  className="mt-2 h-11 w-full rounded-md border border-white/10 bg-[#0B0B0D] px-3 text-sm text-white outline-none transition focus:border-[#B76E79]"
                  placeholder="https://example.com"
                />
                <p className="mt-2 text-xs text-[#777780]">Internal addresses and non-HTTP targets are blocked before crawling.</p>
              </div>

              <div>
                <label htmlFor="json-schema" className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.16em] text-[#B3B3B3]">
                  <Code2 className="h-3.5 w-3.5" /> JSON Schema
                </label>
                <textarea
                  id="json-schema"
                  value={schemaText}
                  onChange={(event) => setSchemaText(event.target.value)}
                  spellCheck={false}
                  className="mt-2 min-h-[420px] w-full resize-y rounded-md border border-white/10 bg-[#0B0B0D] p-3 font-mono text-xs leading-5 text-[#D8D8DE] outline-none transition focus:border-[#B76E79]"
                  aria-describedby="schema-help"
                />
                <p id="schema-help" className="mt-2 text-xs text-[#777780]">
                  The schema must be a JSON Schema object (`type: object`). Required fields guide the returned shape.
                </p>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <label className="block">
                  <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-[#B3B3B3]">Maximum pages</span>
                  <input
                    type="number"
                    min={1}
                    max={25}
                    value={maxPages}
                    onChange={(event) => setMaxPages(Math.min(25, Math.max(1, Number(event.target.value) || 1)))}
                    className="mt-2 h-10 w-full rounded-md border border-white/10 bg-[#0B0B0D] px-3 font-mono text-sm text-white outline-none focus:border-[#B76E79]"
                  />
                </label>
                <label className="block">
                  <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-[#B3B3B3]">Maximum depth</span>
                  <input
                    type="number"
                    min={0}
                    max={3}
                    value={maxDepth}
                    onChange={(event) => setMaxDepth(Math.min(3, Math.max(0, Number(event.target.value) || 0)))}
                    className="mt-2 h-10 w-full rounded-md border border-white/10 bg-[#0B0B0D] px-3 font-mono text-sm text-white outline-none focus:border-[#B76E79]"
                  />
                </label>
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <label className="flex cursor-pointer items-center gap-3 rounded-md border border-white/[0.08] bg-white/[0.02] p-3 text-sm text-[#D8D8DE]">
                  <input type="checkbox" checked={followSubdomains} onChange={(event) => setFollowSubdomains(event.target.checked)} />
                  Follow subdomains
                </label>
                <label className="flex cursor-pointer items-center gap-3 rounded-md border border-white/[0.08] bg-white/[0.02] p-3 text-sm text-[#D8D8DE]">
                  <input type="checkbox" checked={parsePdfs} onChange={(event) => setParsePdfs(event.target.checked)} />
                  Parse linked PDFs
                </label>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-4 border-t border-white/[0.08] pt-5">
                <div className="flex items-center gap-2 text-xs text-[#8DD6A4]">
                  <ShieldCheck className="h-4 w-4" /> Fact-checking is enforced
                </div>
                <button
                  type="submit"
                  disabled={isExtracting}
                  className="inline-flex min-w-44 items-center justify-center gap-2 rounded-md bg-[#B76E79] px-4 py-2.5 font-mono text-xs uppercase tracking-[0.12em] text-white transition hover:bg-[#c77d88] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isExtracting ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileSearch className="h-4 w-4" />}
                  {isExtracting ? 'Extracting…' : 'Extract data'}
                </button>
              </div>
            </form>
          </ConsoleCard>

          <ConsoleCard className="min-h-[500px] overflow-hidden">
            <div className="flex items-center justify-between border-b border-white/[0.08] px-6 py-4">
              <div>
                <h3 className="font-mono text-[11px] uppercase tracking-[0.16em] text-white">Structured result</h3>
                <p className="mt-1 text-xs text-[#777780]">The response is validated against the schema you supplied.</p>
              </div>
              {result && <span className="rounded-full border border-[#5FBF7F]/30 px-2 py-1 font-mono text-[9px] uppercase tracking-[0.13em] text-[#8DD6A4]">Complete</span>}
            </div>

            {error && <div role="alert" className="m-5 rounded-md border border-[#D97C7C]/30 bg-[#D97C7C]/10 p-4 text-sm leading-6 text-[#F0AAAA]">{error}</div>}

            {isExtracting && (
              <div className="flex min-h-[420px] flex-col items-center justify-center gap-3 px-8 text-center">
                <Loader2 className="h-7 w-7 animate-spin text-[#D4AF37]" />
                <div className="font-mono text-xs uppercase tracking-[0.16em] text-[#D8D8DE]">Crawling and checking source facts</div>
                <p className="max-w-sm text-sm leading-6 text-[#777780]">Large crawls can take a few minutes. Keep this tab open while Context.dev processes the schema.</p>
              </div>
            )}

            {!isExtracting && !result && !error && (
              <div className="flex min-h-[420px] flex-col items-center justify-center px-8 text-center">
                <FileSearch className="mb-4 h-9 w-9 text-[#4A4A53]" />
                <p className="font-mono text-xs uppercase tracking-[0.14em] text-[#A0A0A0]">No extraction yet</p>
                <p className="mt-2 max-w-sm text-sm leading-6 text-[#777780]">Start with the Higgsfield workflow schema, or replace it with the exact facts your pipeline needs.</p>
              </div>
            )}

            {result && !isExtracting && (
              <pre className="max-h-[760px] overflow-auto p-6 font-mono text-xs leading-6 text-[#D8D8DE]">{JSON.stringify(result, null, 2)}</pre>
            )}
          </ConsoleCard>
        </div>
      </div>
    </>
  );
}
