import { z } from 'zod';

import projectionJson from '@/data/scene-authority.json';

const SceneAuthoritySchema = z.object({
  sceneId: z.string().min(1),
  collection: z.string().min(1),
  state: z.string().min(1),
  purpose: z.string().nullable(),
  owner: z.string().min(1),
  accountable: z.string().min(1),
  attempt: z.number().int().positive(),
  gates: z.object({
    identity: z.string().min(1),
    productFidelity: z.string().min(1),
    sceneInputEligible: z.boolean(),
    paidAuthorization: z.enum(['RECORDED', 'NOT_AUTHORIZED']),
    paidExecutionReady: z.boolean(),
    runtimeWiring: z.string().min(1),
    deployment: z.string().min(1),
    promotionRequirement: z.string().min(1),
  }),
  identity: z
    .object({
      id: z.string().min(1),
      referenceAssetCount: z.number().int().nonnegative(),
      fullCandidateVerdict: z.string().min(1),
    })
    .nullable(),
  evidenceHashes: z.object({
    contract: z.string().regex(/^[a-f0-9]{64}$/),
    identityApproval: z
      .string()
      .regex(/^[a-f0-9]{64}$/)
      .nullable(),
    identityManifest: z
      .string()
      .regex(/^[a-f0-9]{64}$/)
      .nullable(),
  }),
  blockers: z.array(z.string().min(1)),
  nextAuthority: z.string().min(1),
});

const SceneAuthorityProjectionSchema = z.object({
  schema: z.literal('skyyrose.scene-authority-projection/1'),
  recordedAt: z.string().min(1),
  source: z.object({
    ledgerSha256: z.string().regex(/^[a-f0-9]{64}$/),
  }),
  approvalBoundaries: z.object({
    paidGeneration: z.string().min(1),
    runtimeWiring: z.string().min(1),
    deployment: z.string().min(1),
    promotion: z.string().min(1),
  }),
  scenes: z.array(SceneAuthoritySchema).min(1),
});

export type SceneAuthority = z.infer<typeof SceneAuthoritySchema>;
export type SceneAuthorityProjection = z.infer<typeof SceneAuthorityProjectionSchema>;

const projection = SceneAuthorityProjectionSchema.parse(projectionJson);

export function getSceneAuthorityProjection(): SceneAuthorityProjection {
  return projection;
}

export function getSceneAuthorities(): SceneAuthority[] {
  return projection.scenes;
}

export function getSceneAuthority(sceneId: string): SceneAuthority | undefined {
  const normalizedSceneId = sceneId.trim().toUpperCase();
  return projection.scenes.find(scene => scene.sceneId === normalizedSceneId);
}
