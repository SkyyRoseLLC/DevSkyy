# SkyyRose local ComfyUI project

This project is the local, no-provider-spend execution surface for SkyyRose scene candidates. The canonical catalog and image sources remain in the DevSkyy repository; this directory stores workflows, hash-bound scene contracts, and append-only run evidence.

## Runtime contract

- Backend: `local` (`http://127.0.0.1:8188`)
- Project default: `defaults.where: local` in `comfy.yaml`
- Provider/API spend: forbidden unless separately authorized
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
