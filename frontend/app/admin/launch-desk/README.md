# Launch Desk

Launch Desk is an authenticated launch-planning agent inside the DevSkyy operator console. It turns a product brief, audience, launch date, constraints, and available assets into a prioritized release plan, risk register, owner checklists, channel copy, and follow-up questions.

## Structure

- `page.tsx` — server-rendered route shell and metadata.
- `LaunchDeskClient.tsx` — accessible form, progressive tool status, streamed answer UI, cancellation, and error recovery.
- `../../api/launch-desk/route.ts` — authenticated SSE API route.
- `../../api/launch-desk/_lib/agent.ts` — agent instructions and prompt assembly.
- `../../api/launch-desk/_lib/tools.ts` — deterministic readiness, task, checklist, and copy tool patterns.
- `../../../tests/launch-desk-tools.test.ts` — regression tests for tool behavior.

The API runs deterministic planning tools against the server-validated request, then uses the current `@openai/agents` Responses-backed runner for streamed synthesis and SDK tracing. It does not use the deprecated Assistants API or legacy Chat Completions scaffolding. The model never supplies or rewrites the readiness inputs.

## Local setup

From `frontend/`:

```bash
npm install
cp .env.example .env.local
```

Set these server-only values in `frontend/.env.local`:

```dotenv
OPENAI_API_KEY=your_project_api_key
OPENAI_MODEL=gpt-5.6
NEXTAUTH_SECRET=your_existing_nextauth_secret
NEXTAUTH_URL=http://localhost:3000
```

Do not prefix the OpenAI key with `NEXT_PUBLIC_`. The key must only exist in the Next.js server process.

Start the app:

```bash
npm run dev
```

Sign in through `/login`, then open `/admin/launch-desk`.

## Extending the agent

1. Add a deterministic domain function in `_lib/tools.ts`.
2. Invoke it through the route's `runTool()` boundary so started/completed progress events are guaranteed.
3. Include its immutable output in `formatLaunchPrompt()` only when the synthesis agent needs it.
4. Add regression tests for missing inputs, unsafe claims, ownership, and output shape.
5. Add a user-facing progress label in `TOOL_LABELS`.

Use a handoff only when another agent needs distinct instructions, tools, or ownership. Keep deterministic validation in tools rather than asking a model to simulate a rubric.

## OpenAI references

- [Models](https://developers.openai.com/api/docs/models) — current model catalog and model-selection starting point.
- [OpenAI Agents SDK for TypeScript](https://openai.github.io/openai-agents-js/) — agents, tools, handoffs, guardrails, and tracing.
- [Agents SDK streaming](https://openai.github.io/openai-agents-js/guides/streaming/) — raw model deltas and run-item event patterns used by the API route.

## Validation checklist

### Agent behavior

- [ ] Missing launch date, constraints, or assets are marked missing; they are not silently inferred.
- [ ] The response contains a P0/P1/P2 plan, risk register, owner checklist, copy suggestions, and follow-up questions.
- [ ] Every P0 item has an owner and a concrete next action.
- [ ] Metrics, approvals, customer proof, pricing, and readiness are not fabricated.
- [ ] Trace metadata identifies the `Launch Desk planning workflow` and route.

### Frontend flow

- [ ] Product brief and audience are required and visibly labelled.
- [ ] Submit, reset, keyboard focus, stream cancellation, server error, and retry paths work.
- [ ] Tool progress is visible before or alongside streamed model text.
- [ ] Layout has no document overflow at 390, 768, and 1440 CSS pixels.
- [ ] Reduced-motion mode removes nonessential animation.

### Tool outputs

- [ ] Task extraction produces at least one P0 and cross-functional ownership.
- [ ] Readiness fails closed for missing operational evidence.
- [ ] Checklist item count equals extracted task count.
- [ ] Copy tool returns email, product update, social, and internal drafts.

### Required end-to-end stream proof

The endpoint is authenticated. Use a signed-in browser session or a locally signed NextAuth test session, then POST a realistic brief and read the SSE body to completion. Passing evidence must contain both:

```text
"type":"tool"
"type":"text_delta"
```

Health checks, a successful Next.js build, or an SSE connection without both event types do not satisfy this requirement.
