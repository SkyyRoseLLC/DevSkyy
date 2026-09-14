'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';

import type { SceneAuthority, SceneAuthorityProjection } from '@/lib/scene-authority';

interface SceneAuthorityResponse {
  success: boolean;
  data?: SceneAuthorityProjection;
  error?: string;
}

function statusTone(value: string): string {
  if (value.includes('APPROVED') || value === 'READY' || value === 'RECORDED') {
    return 'border-emerald-400/30 bg-emerald-400/10 text-emerald-300';
  }
  if (
    value.includes('FAIL') ||
    value.includes('REJECT') ||
    value.includes('BLOCKED') ||
    value.includes('NOT_AUTHORIZED')
  ) {
    return 'border-red-400/30 bg-red-400/10 text-red-300';
  }
  return 'border-amber-400/30 bg-amber-400/10 text-amber-200';
}

function StatusPill({ value, caution = false }: { value: string; caution?: boolean }) {
  return (
    <span
      className={`inline-flex rounded-full border px-2.5 py-1 font-mono text-[10px] tracking-[0.08em] ${
        caution ? 'border-amber-400/30 bg-amber-400/10 text-amber-200' : statusTone(value)
      }`}
    >
      {value.replaceAll('_', ' ')}
    </span>
  );
}

function Gate({ label, value, caution = false }: { label: string; value: string; caution?: boolean }) {
  return (
    <div className='rounded-lg border border-white/[0.07] bg-black/20 p-3'>
      <dt className='mb-2 font-mono text-[9px] uppercase tracking-[0.18em] text-[#777781]'>{label}</dt>
      <dd>
        <StatusPill value={value} caution={caution} />
      </dd>
    </div>
  );
}

function SceneCard({ scene }: { scene: SceneAuthority }) {
  return (
    <article className='dsh-card rounded-xl border border-white/[0.08] bg-[#111111] p-5'>
      <div className='flex flex-wrap items-start justify-between gap-4'>
        <div>
          <p className='font-mono text-[10px] uppercase tracking-[0.2em] text-[#B76E79]'>{scene.collection}</p>
          <h2 className='mt-1 text-xl font-semibold text-white'>{scene.sceneId}</h2>
          {scene.purpose && <p className='mt-2 max-w-3xl text-sm leading-6 text-[#9A9AA2]'>{scene.purpose}</p>}
        </div>
        <div className='text-right'>
          <p className='mb-2 font-mono text-[9px] uppercase tracking-[0.18em] text-[#777781]'>OODA record state</p>
          <StatusPill value={scene.state} caution={scene.blockers.length > 0} />
        </div>
      </div>

      <dl className='mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-3'>
        <Gate label='Model identity' value={scene.gates.identity} />
        <Gate label='Garment fidelity' value={scene.gates.productFidelity} />
        <Gate label='Scene input' value={scene.gates.sceneInputEligible ? 'ELIGIBLE' : 'BLOCKED'} />
        <Gate label='Paid approval' value={scene.gates.paidAuthorization} />
        <Gate label='Paid execution' value={scene.gates.paidExecutionReady ? 'READY' : 'BLOCKED'} />
        <Gate label='Runtime wiring' value={scene.gates.runtimeWiring} />
        <Gate label='Deployment' value={scene.gates.deployment} />
        <Gate label='Promotion requirement' value={scene.gates.promotionRequirement} caution />
      </dl>

      {scene.identity && (
        <div className='mt-4 rounded-lg border border-[#B76E79]/20 bg-[#B76E79]/[0.06] p-4'>
          <div className='flex flex-wrap items-center justify-between gap-3'>
            <div>
              <p className='font-mono text-[9px] uppercase tracking-[0.18em] text-[#B76E79]'>
                Approved identity authority
              </p>
              <p className='mt-1 text-base font-semibold text-white'>{scene.identity.id}</p>
            </div>
            <p className='text-xs text-[#9A9AA2]'>
              {scene.identity.referenceAssetCount} identity-only references · full candidate{' '}
              <span className='font-semibold text-red-300'>{scene.identity.fullCandidateVerdict}</span>
            </p>
          </div>
        </div>
      )}

      <div className='mt-5 grid gap-5 lg:grid-cols-[1fr_0.8fr]'>
        <section aria-labelledby={`${scene.sceneId}-blockers`}>
          <h3
            id={`${scene.sceneId}-blockers`}
            className='font-mono text-[10px] uppercase tracking-[0.18em] text-[#A0A0A8]'
          >
            Active blockers
          </h3>
          {scene.blockers.length > 0 ? (
            <ul className='mt-3 space-y-2'>
              {scene.blockers.map(blocker => (
                <li key={blocker} className='flex gap-2 text-sm leading-5 text-[#C1C1C7]'>
                  <span aria-hidden className='mt-2 h-1.5 w-1.5 flex-none rounded-full bg-red-400' />
                  {blocker}
                </li>
              ))}
            </ul>
          ) : (
            <p className='mt-3 text-sm text-emerald-300'>No recorded blockers.</p>
          )}
        </section>

        <section aria-labelledby={`${scene.sceneId}-next`}>
          <h3 id={`${scene.sceneId}-next`} className='font-mono text-[10px] uppercase tracking-[0.18em] text-[#A0A0A8]'>
            Next authority
          </h3>
          <p className='mt-3 text-sm leading-6 text-[#C1C1C7]'>{scene.nextAuthority}</p>
          <p className='mt-3 break-all font-mono text-[9px] leading-4 text-[#676771]'>
            Contract {scene.evidenceHashes.contract}
          </p>
        </section>
      </div>
    </article>
  );
}

export default function SceneAuthorityPage() {
  const [projection, setProjection] = useState<SceneAuthorityProjection | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadAuthority = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/scene-authority', {
        cache: 'no-store',
        headers: { Accept: 'application/json' },
      });
      const body = (await response.json()) as SceneAuthorityResponse;

      if (!response.ok || !body.success || !body.data) {
        throw new Error(body.error ?? `Scene authority request failed (${response.status})`);
      }

      setProjection(body.data);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Scene authority is unavailable');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadAuthority();
  }, [loadAuthority]);

  const summary = useMemo(() => {
    const scenes = projection?.scenes ?? [];
    return {
      total: scenes.length,
      blocked: scenes.filter(scene => scene.blockers.length > 0).length,
      identities: scenes.filter(scene => scene.gates.identity.includes('APPROVED')).length,
      executable: scenes.filter(scene => scene.gates.paidExecutionReady).length,
    };
  }, [projection]);

  return (
    <div className='mx-auto max-w-[1500px] px-6 py-8 lg:px-10'>
      <header className='border-b border-white/[0.07] pb-7'>
        <p className='font-mono text-[10px] uppercase tracking-[0.22em] text-[#B76E79]'>
          Fashion Theme Team · OODA Control
        </p>
        <div className='mt-2 flex flex-wrap items-end justify-between gap-4'>
          <div>
            <h1 className='text-3xl font-semibold tracking-[-0.02em] text-white'>Scene Authority</h1>
            <p className='mt-2 max-w-3xl text-sm leading-6 text-[#9A9AA2]'>
              Read-only release truth. Identity, product fidelity, scene input, paid execution, and runtime promotion
              remain independent gates.
            </p>
          </div>
          <button
            type='button'
            onClick={() => void loadAuthority()}
            disabled={loading}
            className='rounded-md border border-white/[0.12] px-4 py-2 font-mono text-[10px] uppercase tracking-[0.14em] text-[#D6D6DA] transition hover:border-[#B76E79]/50 hover:text-white disabled:cursor-wait disabled:opacity-50'
          >
            {loading ? 'Checking…' : 'Refresh authority'}
          </button>
        </div>
      </header>

      <section aria-label='Authority summary' className='grid gap-3 py-6 sm:grid-cols-2 xl:grid-cols-4'>
        {[
          ['Mapped scenes', summary.total],
          ['Blocked scenes', summary.blocked],
          ['Approved identities', summary.identities],
          ['Paid-executable', summary.executable],
        ].map(([label, value]) => (
          <div key={label} className='rounded-lg border border-white/[0.07] bg-[#111111] p-4'>
            <p className='font-mono text-[9px] uppercase tracking-[0.18em] text-[#777781]'>{label}</p>
            <p className='mt-2 text-2xl font-semibold text-white'>{value}</p>
          </div>
        ))}
      </section>

      <div aria-live='polite'>
        {error && (
          <div role='alert' className='rounded-xl border border-red-400/30 bg-red-400/10 p-5 text-sm text-red-200'>
            <p className='font-semibold'>Scene authority could not be loaded.</p>
            <p className='mt-1'>{error}</p>
          </div>
        )}

        {loading && !projection && (
          <div className='rounded-xl border border-white/[0.07] bg-[#111111] p-8 text-sm text-[#9A9AA2]'>
            Verifying the committed authority projection…
          </div>
        )}

        {projection && (
          <div className='space-y-4'>
            {projection.scenes.map(scene => (
              <SceneCard key={scene.sceneId} scene={scene} />
            ))}
            <p className='pb-4 text-right font-mono text-[9px] uppercase tracking-[0.14em] text-[#676771]'>
              Recorded {projection.recordedAt} · Ledger {projection.source.ledgerSha256}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
