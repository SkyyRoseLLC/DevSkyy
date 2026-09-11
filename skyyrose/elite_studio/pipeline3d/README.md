# pipeline3d — SkyyRose Product-to-3D Platform

This package owns the provider-neutral product evidence, generation workflow,
model routing, preflight and release-certification contracts. It implements a
Tripo-style production workflow with SkyyRose-owned code and open interfaces;
it does not copy a provider's private models, training data, code, or product
assets.

The original four-stage CLI remains available as a compatibility runner. New
production integrations use `SkyyRose3DPlatform` and the authenticated
`/api/v1/3d-platform` surface.

## Quick start

```bash
# Dry run (estimate only, no dispatch):
python -m skyyrose.elite_studio.pipeline3d --sku br-001 \
    --stages image-to-3d,texture,remesh,export

# Paid dispatch (STOP-AND-SHOW gate; needs TRIPO_API_KEY):
python -m skyyrose.elite_studio.pipeline3d --sku br-001 \
    --stages image-to-3d,texture,remesh,export --go
```

## Architecture

| Module | Role |
|--------|------|
| `models` | immutable data types; `Artifact` is the chaining handle (task_id + path) |
| `router` | picks an adapter per stage by capability/priority/availability + fallback |
| `executor` | runs stages in order; budget gate; telemetry; idempotent resume; chaining |
| `estimator` | one whole-job cost estimate, shown before dispatch |
| `store` | file-based stage-level idempotency (resume skips completed stages) |
| `adapters/tripo` | image-to-3D / texture / remesh via the tripo3d SDK |
| `adapters/local_export` | EXPORT stage — copies final GLB to `<output>/<sku>.glb` |
| `preflight` | resolves the canonical source image + guards against missing source |
| `platform_contracts` | hash-bound SKU evidence, quality policy and complete production stages |
| `workflow` | compiles the topology-safe, fail-closed production DAG |
| `model_registry` | routes OpenAI, local open-source models and deterministic tools |
| `specification` | OpenAI visual draft plus independent OSS visual audit |
| `deterministic_gates` | raw-GLB numeric checks independent of the exporter |
| `fidelity` | receipt-based deterministic, dual-vision and founder certification |
| `production` | canonical planning, preflight and certification facade |

Execution spine: **synchronous-within-stage** (the adapter polls to completion).
Cross-provider chaining: same provider → pass `task_id`; different provider →
hand off the downloadable `model_url`/path.

## Production boundary

The platform contracts and gates are implemented. A deployment is not ready
until the runtime preflight verifies an OpenAI key/model, an independent local
open-source VLM endpoint, the pinned TRELLIS.2 repository/model, Blender and
`@gltf-transform/cli`. Queue/worker integration continues to use the existing
`pipelines/clothing_3d` Redis runtime.

See `docs/SKYYROSE_3D_PLATFORM.md` for the full contract and current blockers.
