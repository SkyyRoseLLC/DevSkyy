# SkyyRose local ComfyUI project

This project is the governed execution surface for SkyyRose scene candidates. The canonical catalog and image sources remain in the DevSkyy repository; this directory stores workflows, hash-bound scene contracts, OODA orchestration, and append-only run evidence.

## Runtime contract

- Backend: `local` (`http://127.0.0.1:8188`)
- Project default: `defaults.where: local` in `comfy.yaml`
- Provider/API spend: forbidden unless separately authorized and the scene contract has no execution blockers
- Generated images: candidate-only until product-fidelity, physical-integration, optical-realism, independent-review, and founder-approval gates pass
- Canonical product files are uploaded from their registered repository paths; generated outputs never become source authority

## Required local models

The retained source-bound workflow uses these official Comfy-Org model files:

- `models/diffusion_models/z_image_turbo_bf16.safetensors` loaded with `weight_dtype: default`
- `models/text_encoders/qwen_3_4b_fp8_mixed.safetensors`
- `models/vae/ae.safetensors`

The shared model root is `/Users/theceo/ComfyUI-Shared/models`. Do not commit model weights.

The full BF16 diffusion plus full text-encoder pair is intentionally not used: together they forced this 16 GB unified-memory host into severe swap pressure before sampling. The retained graph pairs the native BF16 diffusion model with the smaller mixed-FP8 encoder because Apple MPS cannot execute Float8 diffusion weights.

## Apple M5 16 GB compatibility verdict

The workflow is structurally valid and source-bound, but this Z-Image model family is not a practical renderer on the current 16 GB Apple host:

- Native BF16 reaches KSampler but consumes about 19.4 GB of swap and remains at step zero.
- Loader-level Float8 is rejected because Apple MPS does not support `Float8_e4m3fn`.
- Int8 ConvRot reaches KSampler only with `PYTORCH_ENABLE_MPS_FALLBACK=1`; CPU fallback remains at step zero beyond the preview latency budget.

No candidate image was produced. The Int8 ConvRot weight and its reference template were removed after diagnosis to recover 6.2 GB; they can be downloaded again from the official template if the workflow moves to compatible hardware. Execute this retained BF16 graph on a larger-memory compatible runner, or authorize a separately budgeted provider route. Do not weaken the source bindings or promotion gates to make a run complete.

## Launch the local server

```bash
comfy --workspace /Users/theceo/ComfyUI-Installs/ComfyUI/ComfyUI launch -- \
  --enable-manager \
  --extra-model-paths-config "/Users/theceo/Library/Application Support/Comfy Desktop/instance-model-paths/inst-1788304801153.yaml" \
  --input-directory /Users/theceo/ComfyUI-Shared/input \
  --output-directory /Users/theceo/ComfyUI-Shared/output
```

This launch path does not add a telemetry feature flag. CLI tracking must remain disabled; verify it with `comfy --json env`.

## Verify before a run

```bash
comfy --json env
comfy --json project status
comfy --where local --json workflow validate --workflow workflows/lh-004-reference-pilot.api.json
comfy --where local --json run --workflow workflows/lh-004-reference-pilot.api.json --print-prompt
```

The validation receipt must report `spends_credits: false`, an empty `partner_nodes` list, and no errors before submission.

## Run and recover

```bash
comfy --where local --json run --workflow workflows/lh-004-reference-pilot.api.json
comfy --where local --json jobs watch <prompt_id>
comfy --where local --json download <prompt_id>
```

Append the prompt ID to `job_ids.txt` at submission time and the submission/QC decision to `LOG.md`. Source hashes and acceptance criteria live in `scene-contracts/`.

## Higgsfield photoshoot inside the Comfy OODA

Higgsfield Product Photoshoot is the native campaign-capture stage; it does not own source truth or promotion. Comfy owns authority validation, staging, bounded correction, integration, receipts, rejection routing, and review handoff.

Fashion Theme Team governance is embedded through `scene-contracts/scene-ooda-index.json` and the durable `scene-contracts/scene-ooda-ledger.json`. See `FASHION-THEME-TEAM-OODA.md` for role ownership and handoff boundaries. All five remaining mapped scenes have individual contracts; blocked scenes remain mapped and validated without being sent to paid generation.

The first governed packet is `BR-COMMERCE-2`:

```bash
python3 Comfy/scripts/scene_ooda.py validate \
  --manifest Comfy/scene-contracts/br-commerce-2-higgsfield-comfy-ooda.json

python3 Comfy/scripts/scene_ooda.py enhance \
  --manifest Comfy/scene-contracts/br-commerce-2-higgsfield-comfy-ooda.json
```

`enhance` uses Higgsfield's no-generation `--enhance-only` path. Paid generation is deliberately fail-closed: it requires the explicit `execute --allow-spend` action and a contract with zero source, dependency, preflight, or approval blockers.

If assertion rules change after a human/product-authority review, re-evaluate the immutable enhancement receipt without another provider request:

```bash
python3 Comfy/scripts/scene_ooda.py review-enhance \
  --manifest Comfy/scene-contracts/br-commerce-2-higgsfield-comfy-ooda.json \
  --receipt /absolute/path/to/higgsfield-enhance-receipt.json
```

A generated result remains candidate-only and can be staged locally with:

```bash
python3 Comfy/scripts/scene_ooda.py stage \
  --manifest Comfy/scene-contracts/br-commerce-2-higgsfield-comfy-ooda.json \
  --candidate /absolute/path/to/candidate.png
```

Staging uploads the candidate to local ComfyUI and writes a receipt. It does not promote, wire, deploy, or overwrite any SOT file.

## Specialized garment placement before scene generation

Scenes that lack an approved exact on-model product layer use the separate FASHN Try-On Max pilot before Higgsfield or Comfy environment work. This stage receives one approved full-body model image and one primary product image; the other physical product views remain independent review authority because Try-On Max accepts only one product image and cannot prove unseen construction.

The initial governed pilot is `LH-003-TRYON-MAX-PILOT-A1`:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 \
  Comfy/scripts/vto_ooda.py validate \
  --contract Comfy/scene-contracts/lh-003-tryon-max-vto-ooda.json
```

Validation is expected to remain `BLOCKED` until all three gates exist:

1. a founder-approved full-body `LH-MODEL-01` source and authority receipt;
2. the hash-bound product pack and registered `tryon-max` capability; and
3. an approval receipt bound to the current execution fingerprint for one four-credit candidate.

Only after those gates pass can the separately authorized command run:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 \
  Comfy/scripts/vto_ooda.py execute \
  --contract Comfy/scene-contracts/lh-003-tryon-max-vto-ooda.json \
  --allow-spend
```

The runner sends local inputs as private data URIs, requests one lossless PNG with a fixed seed, refuses automatic submission retry, and writes a contract/input/output/job/credit receipt. A successful response remains quarantined and candidate-only until an independent reviewer verifies model identity, all four zippered pockets, wearer-relative artwork, front/back construction, and absence of non-garment drift. Only that reviewed on-model layer may continue into the cinematic environment and sculpture pipeline.

The independent receipt is checked separately and cannot be authored by the candidate builder:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 \
  Comfy/scripts/vto_ooda.py review \
  --contract Comfy/scene-contracts/lh-003-tryon-max-vto-ooda.json \
  --candidate Comfy/quarantine/LH-003-VTO/LH-003-TRYON-MAX-PILOT-A1.png \
  --execution-receipt Comfy/receipts/lh-003-tryon-max-pilot-a1-execution.json \
  --review /absolute/path/to/lh-003-vto-independent-review.json
```

An independent `PASS` still reports `scene_input_authorized: false`; explicit founder product approval must be bound before the candidate becomes an on-model authority for `LH-COMMERCE-2`.
