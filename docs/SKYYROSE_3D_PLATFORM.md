# SkyyRose Product-to-3D Platform

The canonical SkyyRose 3D platform is an owned control and fidelity plane over
open-source generation engines, OpenAI reasoning/vision, Blender authoring, and
optional external 3D providers. It reproduces a Tripo-style workflow without
copying Tripo code, model weights, training data, or branded product assets.

## Current implementation truth

| Layer | Status | Canonical code |
|---|---|---|
| Evidence and workflow contracts | Implemented | `skyyrose/elite_studio/pipeline3d/platform_contracts.py` |
| Production DAG compiler | Implemented | `skyyrose/elite_studio/pipeline3d/workflow.py` |
| OpenAI/open-source/tool registry | Implemented | `skyyrose/elite_studio/pipeline3d/model_registry.py` |
| Evidence and runtime preflight | Implemented | `skyyrose/elite_studio/pipeline3d/production.py` |
| OpenAI draft + OSS visual review | Implemented | `skyyrose/elite_studio/pipeline3d/specification.py` |
| Deterministic/vision/founder certification | Implemented | `skyyrose/elite_studio/pipeline3d/fidelity.py` |
| Authenticated platform API | Implemented | `api/v1/three_d_platform.py` |
| Resumable execution manifests | Implemented | `skyyrose/elite_studio/pipeline3d/execution.py` |
| Redis job runtime | Existing, integration required | `pipelines/clothing_3d/` |
| TRELLIS provider | Existing, runtime must be provisioned | `services/three_d/trellis/` |
| Blender rig/export authoring | Existing tools, worker integration required | `.agents/skills/3d-rigging-pipeline/` |
| Fixed-view render worker | Not yet implemented | Blocking production certification |
| Storefront publication | Existing consumers, canonical publisher required | WordPress/Three.js surfaces |

No output is called production-ready merely because these modules import. A
real run must pass preflight and return independent gate receipts bound to one
artifact hash and one evidence revision.

## Canonical workflow

```text
ingest → evidence gate → specification → preprocess → generate
→ texture → segment → complete → remesh → UV
→ riggability → rig → retarget (when requested)
→ export → fresh proof renders
→ deterministic QC + OpenAI vision QC + OSS vision QC
→ consensus → founder approval → publish
```

Rigging always follows topology-changing operations. Publishing always follows
all deterministic, dual-vision, consensus, and founder gates.

## Required configuration

```bash
# OpenAI specification author and vision judge
export OPENAI_API_KEY=...
export OPENAI_3D_PLANNER_MODEL=gpt-5.5-pro
export OPENAI_3D_VISION_MODEL=gpt-5.5-pro

# Independently hosted open-source planner/vision model via an
# OpenAI-compatible vLLM, Ollama or LocalAI endpoint
export OSS_MODEL_BASE_URL=http://127.0.0.1:8001/v1
export OSS_MODEL_API_KEY=local-not-secret
export OSS_3D_PLANNER_MODEL=<configured-local-model-id>
export OSS_3D_VISION_MODEL=<configured-local-vlm-id>

# Owned open-source 3D engine
export TRELLIS2_REPO=/opt/skyyrose/TRELLIS.2
export TRELLIS2_MODEL=microsoft/TRELLIS.2-4B
```

The application does not silently download a local LLM/VLM or model weights at
startup. Operators must provision and pin those artifacts explicitly.

## API

All endpoints require an authenticated `ADMIN` or `DEVELOPER` bearer token.

```text
GET  /api/v1/3d-platform/capabilities
POST /api/v1/3d-platform/plan
POST /api/v1/3d-platform/preflight
POST /api/v1/3d-platform/specification
POST /api/v1/3d-platform/certify
```

`/specification` is a compute-bearing endpoint: it sends approved product
images and dossier content to the configured OpenAI model and the configured
open-source vision endpoint. The open-source judge sees the same evidence and
must approve the OpenAI-authored specification. One repair is allowed; a
second rejection blocks generation.

## Certification authorities

Machine-checkable facts are never delegated to an LLM. Independent scripts
must produce receipts for source hashes, GLB structure, mesh topology, PBR
materials, real-world dimensions, required proof views, rig integrity, and
animation integrity. OpenAI and the open-source VLM independently grade visual
product identity. Founder approval is a separate hash-bound human receipt.

A skipped gate, crashed command, missing model, stale source hash, judge
disagreement, or receipt created by the artifact producer is a blocker—not a
pass.

The execution kernel also separates handler authority by construction. A
`producer` handler cannot be selected for deterministic QC, an OpenAI judge
cannot be selected for the OSS judge stage, and neither model can manufacture
the human approval stage. Every completed stage is atomically persisted to a
request-fingerprinted execution manifest for safe resume and audit.
