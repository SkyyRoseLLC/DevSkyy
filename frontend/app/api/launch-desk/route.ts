import { run, withTrace } from '@openai/agents';
import { NextResponse, type NextRequest } from 'next/server';

import { withAuth } from '@/lib/api-auth';

import { createLaunchDeskAgent, formatLaunchPrompt } from './_lib/agent';
import {
  assessLaunchReadiness,
  buildOwnerChecklists,
  draftLaunchCopy,
  extractLaunchTasks,
} from './_lib/tools';
import { launchBriefSchema } from './_lib/types';

export const maxDuration = 120;

interface StreamEvent {
  type: 'meta' | 'tool' | 'text_delta' | 'done' | 'error';
  [key: string]: unknown;
}

function sse(event: StreamEvent): Uint8Array {
  return new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`);
}

async function handleLaunchDesk(request: NextRequest): Promise<Response> {
  if (!process.env.OPENAI_API_KEY) {
    return NextResponse.json(
      { success: false, error: 'OPENAI_API_KEY is not configured on the server.' },
      { status: 503 },
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ success: false, error: 'Request body must be JSON.' }, { status: 400 });
  }

  const parsed = launchBriefSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { success: false, error: 'Launch brief is invalid.', issues: parsed.error.flatten() },
      { status: 400 },
    );
  }

  const traceId = crypto.randomUUID();
  const signal = request.signal;

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (event: StreamEvent) => controller.enqueue(sse(event));

      send({
        type: 'meta',
        traceId,
        model: process.env.OPENAI_MODEL || 'gpt-5.6',
        message: 'Launch Desk is reading the brief.',
      });

      try {
        const runTool = <T>(name: string, execute: () => T): T => {
          send({ type: 'tool', phase: 'started', name });
          const output = execute();
          send({ type: 'tool', phase: 'completed', name });
          return output;
        };

        const tasks = runTool('extract_launch_tasks', () => extractLaunchTasks(parsed.data));
        const readiness = runTool('check_launch_readiness', () => assessLaunchReadiness(parsed.data));
        const ownerChecklists = runTool('generate_owner_checklists', () => buildOwnerChecklists(tasks));
        const channelDrafts = runTool('draft_channel_launch_copy', () =>
          draftLaunchCopy({
            productName: 'Launch candidate',
            promise: parsed.data.productBrief.split(/[.!?]/, 1)[0]?.trim() || 'A new product release',
            audience: parsed.data.audience,
          }),
        );

        await withTrace(
          'Launch Desk planning workflow',
          async () => {
            const result = await run(
              createLaunchDeskAgent(),
              formatLaunchPrompt(parsed.data, { tasks, readiness, ownerChecklists, channelDrafts }),
              { stream: true, signal, maxTurns: 2 },
            );

            for await (const event of result) {
              if (event.type === 'raw_model_stream_event') {
                const data = event.data;
                if (data.type === 'output_text_delta' && typeof data.delta === 'string') {
                  send({ type: 'text_delta', delta: data.delta });
                }
                continue;
              }
            }

            await result.completed;
            send({ type: 'done', traceId });
          },
          { groupId: traceId, metadata: { surface: 'launch-desk', route: '/api/launch-desk' } },
        );
      } catch (error) {
        console.error('Launch Desk planning failed.', { traceId, error });
        send({
          type: 'error',
          message: 'Launch Desk could not complete this plan. Retry or contact the operator with the trace ID.',
          traceId,
        });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
  });
}

export const POST = withAuth(handleLaunchDesk);
