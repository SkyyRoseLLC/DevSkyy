import type {
  ChannelDraft,
  LaunchBrief,
  LaunchTask,
  OwnerChecklist,
  ReadinessCheck,
} from './types';

function includesAny(text: string, terms: string[]): boolean {
  const normalized = text.toLowerCase();
  return terms.some((term) => normalized.includes(term));
}

export function extractLaunchTasks(input: LaunchBrief): LaunchTask[] {
  const tasks: LaunchTask[] = [
    {
      title: 'Freeze launch scope and acceptance criteria',
      owner: 'Product',
      priority: 'P0',
      rationale: 'A release cannot be validated until the shipped promise and success criteria are explicit.',
    },
    {
      title: 'Complete production readiness and rollback review',
      owner: 'Engineering',
      priority: 'P0',
      rationale: 'The launch needs tested deployment, monitoring, ownership, and recovery paths.',
    },
    {
      title: 'Validate positioning with the target audience',
      owner: 'Product Marketing',
      priority: 'P1',
      rationale: `The plan must translate the product value for ${input.audience}.`,
    },
    {
      title: 'Prepare support and incident-response coverage',
      owner: 'Support',
      priority: 'P1',
      rationale: 'Customer questions and launch-day failures need named responders and escalation paths.',
    },
    {
      title: 'Instrument launch success metrics',
      owner: 'Data',
      priority: 'P1',
      rationale: 'The team needs a shared launch dashboard and decision thresholds.',
    },
  ];

  if (includesAny(input.productBrief, ['api', 'integration', 'developer'])) {
    tasks.push({
      title: 'Publish integration documentation and runnable examples',
      owner: 'Developer Experience',
      priority: 'P1',
      rationale: 'Technical audiences need a verified path from first request to production use.',
    });
  }

  if (!input.availableAssets.trim()) {
    tasks.push({
      title: 'Create the minimum launch asset set',
      owner: 'Creative',
      priority: 'P0',
      rationale: 'No launch assets were supplied, so copy, visuals, demo media, and proof points remain unowned.',
    });
  }

  return tasks;
}

export function assessLaunchReadiness(
  input: LaunchBrief,
): ReadinessCheck[] {
  const checks: ReadinessCheck[] = [
    {
      area: 'Launch date',
      status: input.launchDate.trim() ? 'at-risk' : 'missing',
      evidence: input.launchDate.trim() || 'No target date supplied.',
      nextAction: input.launchDate.trim()
        ? 'Convert the date into freeze, go/no-go, launch, and rollback milestones.'
        : 'Name a target date or a decision window.',
    },
    {
      area: 'Audience',
      status: input.audience.length >= 20 ? 'ready' : 'at-risk',
      evidence: input.audience,
      nextAction: 'Confirm the primary buyer/user, their trigger, and their adoption barrier.',
    },
    {
      area: 'Constraints',
      status: input.constraints.trim() ? 'at-risk' : 'missing',
      evidence: input.constraints.trim() || 'No budget, compliance, staffing, or platform constraints supplied.',
      nextAction: 'Document the constraints that can move the date or reduce scope.',
    },
    {
      area: 'Launch assets',
      status: input.availableAssets.trim() ? 'at-risk' : 'missing',
      evidence: input.availableAssets.trim() || 'No existing assets supplied.',
      nextAction: 'Inventory owners, approval status, format, channel, and due date for every asset.',
    },
    {
      area: 'Operational readiness',
      status: includesAny(input.productBrief, ['monitor', 'rollback', 'support', 'on-call'])
        ? 'at-risk'
        : 'missing',
      evidence: 'Operational proof must be validated outside the launch brief.',
      nextAction: 'Run a go/no-go review covering monitoring, support, security, capacity, and rollback.',
    },
  ];

  return checks;
}

export function buildOwnerChecklists(tasks: LaunchTask[]): OwnerChecklist[] {
  const byOwner = new Map<string, string[]>();
  for (const task of tasks) {
    const current = byOwner.get(task.owner) ?? [];
    current.push(`[${task.priority}] ${task.title}`);
    byOwner.set(task.owner, current);
  }
  return [...byOwner.entries()].map(([owner, items]) => ({ owner, items }));
}

export function draftLaunchCopy(input: {
  productName: string;
  promise: string;
  audience: string;
}): ChannelDraft[] {
  return [
    {
      channel: 'email',
      copy: `Meet ${input.productName}: ${input.promise}. Built for ${input.audience}.`,
    },
    {
      channel: 'product-update',
      copy: `${input.productName} is ready to try. ${input.promise}. See what changed, how it works, and where to start.`,
    },
    {
      channel: 'social',
      copy: `${input.productName} is here — ${input.promise}.`,
    },
    {
      channel: 'internal',
      copy: `${input.productName} launch brief: ${input.promise}. Confirm owners, go/no-go evidence, support coverage, and rollback before release.`,
    },
  ];
}
