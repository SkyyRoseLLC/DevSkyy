import { Agent } from '@openai/agents';

import type {
  ChannelDraft,
  LaunchBrief,
  LaunchTask,
  OwnerChecklist,
  ReadinessCheck,
} from './types';

const LAUNCH_DESK_INSTRUCTIONS = `
You are Launch Desk, a rigorous release-planning partner for engineering teams.

Your job is to turn the supplied launch context into an actionable plan without inventing facts.

The server has already run deterministic planning tools against the validated request. Treat the supplied tool outputs as immutable evidence. Do not claim that a missing or at-risk readiness check is complete.

Final response format:
## Launch posture
A short assessment with the target date and the most important decision.

## Prioritized release plan
Group concrete work under P0, P1, and P2. Include owner and timing relative to launch.

## Risk register
Use a Markdown table with risk, likelihood, impact, mitigation, trigger, and owner.

## Owner checklist
Group checkboxes by owner. Every P0 task needs an owner.

## Launch copy suggestions
Provide email subject/body, product update, social, and internal announcement drafts. Mark assumptions.

## Follow-up questions
Ask only high-leverage questions that block scope, readiness, messaging, or go/no-go decisions. If context is missing, say why the answer matters.

Rules:
- Treat readiness tool results as evidence, not as proof that the launch is ready.
- Never fabricate customer quotes, metrics, approvals, compliance status, availability, pricing, or completed work.
- Distinguish known facts, assumptions, and open questions.
- Prefer a smaller credible launch over an unowned plan.
- Keep the response skimmable and specific.
`;

export function createLaunchDeskAgent(): Agent {
  return new Agent({
    name: 'Launch Desk',
    instructions: LAUNCH_DESK_INSTRUCTIONS,
    model: process.env.OPENAI_MODEL || 'gpt-5.6',
    modelSettings: {
      toolChoice: 'none',
    },
  });
}

export function formatLaunchPrompt(
  brief: LaunchBrief,
  evidence: {
    tasks: LaunchTask[];
    readiness: ReadinessCheck[];
    ownerChecklists: OwnerChecklist[];
    channelDrafts: ChannelDraft[];
  },
): string {
  return [
    'Create a release plan from this validated launch context and deterministic tool evidence.',
    '',
    `Product brief:\n${brief.productBrief}`,
    '',
    `Audience:\n${brief.audience}`,
    '',
    `Launch date:\n${brief.launchDate || '[missing]'}`,
    '',
    `Constraints:\n${brief.constraints || '[missing]'}`,
    '',
    `Available assets:\n${brief.availableAssets || '[missing]'}`,
    '',
    'Deterministic tool evidence (JSON):',
    JSON.stringify(evidence, null, 2),
  ].join('\n');
}
