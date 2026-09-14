import { z } from 'zod';

export const launchBriefSchema = z.object({
  productBrief: z.string().trim().min(20).max(8_000),
  audience: z.string().trim().min(3).max(1_500),
  launchDate: z.string().trim().max(80).default(''),
  constraints: z.string().trim().max(3_000).default(''),
  availableAssets: z.string().trim().max(3_000).default(''),
});

export type LaunchBrief = z.infer<typeof launchBriefSchema>;

export type LaunchPriority = 'P0' | 'P1' | 'P2';

export interface LaunchTask {
  title: string;
  owner: string;
  priority: LaunchPriority;
  rationale: string;
}

export interface ReadinessCheck {
  area: string;
  status: 'ready' | 'at-risk' | 'missing';
  evidence: string;
  nextAction: string;
}

export interface OwnerChecklist {
  owner: string;
  items: string[];
}

export interface ChannelDraft {
  channel: 'email' | 'product-update' | 'social' | 'internal';
  copy: string;
}
