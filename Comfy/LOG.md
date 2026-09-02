# Comfy production log

Append-only execution and decision log for recoverable local scene production.

- 2026-09-01 DECISION local-only routing; partner/API spend disabled; `LH-004` selected as the first compositor-ready product pilot.
- 2026-09-01 SOURCE `lh-004-bomber-front.jpg` SHA-256 `7a00b1122a9f36e2d24221f011719776799726711b11c58f552f57e2134fb256` and `lh-004-bomber-back.png` SHA-256 `e170a556f4ee444f273c4aacd2bfce569d0c87263397826b6e757db6402888f1` bound to candidate `LH-004-COMFY-LOCAL-001`.
- 2026-09-01 DECISION candidate output is non-promotable until product-fidelity, physical-integration, optical-realism, and founder-review gates pass.
- 2026-09-01 SUBMIT `lh-004-reference-pilot` -> `10db1189-4a6c-494e-a244-7af39ca8b55c`; local Z-Image reference-latent workflow; zero partner nodes; zero provider spend.
- 2026-09-01 CANCEL `10db1189-4a6c-494e-a244-7af39ca8b55c`; BF16 diffusion plus full text encoder drove the 16 GB host to about 20 GB swap before sampling. No output produced and no provider spend.
- 2026-09-01 DECISION replace recoverable BF16/full weights with official `z_image_turbo_int8_convrot.safetensors` and `qwen_3_4b_fp8_mixed.safetensors`; reduce pilot canvas from 896x1152 to 768x1024; retain the same SOT bindings and acceptance gates.
- 2026-09-01 SUBMIT `lh-004-reference-pilot-int8` -> `ec03a00a-1ef3-45ea-9d4f-0e5aed4bd616`; local Int8/mixed-FP8 profile; zero partner nodes; zero provider spend.
- 2026-09-01 ERROR `ec03a00a-1ef3-45ea-9d4f-0e5aed4bd616`; KSampler reference latent shape was not divisible by the model 2x2 patch (`shape [1,16,71,2,57,2]`); no output and no provider spend.
- 2026-09-01 RECOVERY disable experimental auto-resize and normalize both physical references to nondistorting 1024x1024 white-padded canvases via `ResizeAndPadImage` before VAE reference encoding.
- 2026-09-01 SUBMIT `lh-004-reference-pilot-int8-shapefix` -> `e3f1f1ab-f0e6-4b64-b96b-3102ef8d2e94`; 14 local core nodes; reference canvases normalized to model patch requirements; zero provider spend.
- 2026-09-01 ERROR `e3f1f1ab-f0e6-4b64-b96b-3102ef8d2e94`; KSampler reached Int8 execution but Apple MPS lacks `aten::_int_mm`; no output and no provider spend.
- 2026-09-01 RECOVERY relaunch local ComfyUI with `PYTORCH_ENABLE_MPS_FALLBACK=1` exactly as prescribed by the PyTorch error; unsupported Int8 matrix multiplication may execute on CPU while supported operations remain on MPS.
- 2026-09-01 SUBMIT `lh-004-reference-pilot-int8-mps-fallback` -> `a757c7fa-bd89-4444-9f2d-68577c84c3b9`; same validated graph under the required Apple MPS fallback runtime; zero provider spend.
- 2026-09-01 CANCEL `a757c7fa-bd89-4444-9f2d-68577c84c3b9`; CPU fallback was active and healthy but the first 768x1024 eight-step sample exceeded the useful local-preview latency budget. No output and no provider spend.
- 2026-09-01 DECISION local route is a viability preview: 512x768 output, 512x512 nondistorted reference pads, four steps. A visually viable preview may advance to a separate higher-quality route; it cannot be promoted directly.
- 2026-09-01 SUBMIT `lh-004-reference-pilot-local-preview` -> `19432dde-d933-4a30-9f7f-33b5c116aade`; reduced local viability preview under required Apple MPS fallback; zero provider spend.
- 2026-09-01 CANCEL `19432dde-d933-4a30-9f7f-33b5c116aade`; reduced Int8 CPU-fallback preview still exceeded the useful local latency window at step zero. No output and no provider spend.
- 2026-09-01 DECISION active Mac profile uses official `z_image_turbo_bf16.safetensors` with loader `weight_dtype=fp8_e4m3fn` plus `qwen_3_4b_fp8_mixed.safetensors`; this targets lower memory without the Int8-only `_int_mm` CPU fallback.
- 2026-09-01 SUBMIT `lh-004-reference-pilot-bf16-fp8` -> `9228ff19-48a1-44d0-9564-2581875148aa`; source-bound 512x768 four-step local viability preview; 14 core nodes, zero partner nodes, zero provider spend.
- 2026-09-01 ERROR `9228ff19-48a1-44d0-9564-2581875148aa`; KSampler rejected Float8 diffusion weights because Apple MPS does not support dtype `Float8_e4m3fn`; no output and no provider spend.
- 2026-09-01 RECOVERY retain the lower-memory mixed-FP8 text encoder but load the official BF16 diffusion model at its native dtype; this is the final local compatibility attempt before classifying the 16 GB Apple runtime as hardware-blocked for this model family.
- 2026-09-01 SUBMIT `lh-004-reference-pilot-bf16-native` -> `ee4d051a-a36a-44ac-a81a-542776d3c852`; native BF16 diffusion plus mixed-FP8 text encoder at 512x768 and four steps; 14 core nodes, zero partner nodes, zero provider spend.
- 2026-09-01 CANCEL `ee4d051a-a36a-44ac-a81a-542776d3c852`; native BF16 reached KSampler but remained at step zero while host swap reached about 19.4 GB. Cancelled after 309 seconds with no output and no provider spend.
- 2026-09-01 VERDICT the source-bound graph validates with zero errors, zero warnings, zero partner nodes, and `spends_credits=false`, but Z-Image Turbo is hardware-blocked for practical rendering on this 16 GB Apple M5 runtime. No candidate exists for visual review or promotion.
- 2026-09-01 CLEANUP removed the exact recoverable `z_image_turbo_int8_convrot.safetensors` weight and its inactive reference template after MPS diagnosis, reclaiming about 6.2 GB. Native BF16, mixed-FP8 text encoder, VAE, workflow, hashes, and receipts remain installed.
- 2026-09-01 AUTHORIZATION user confirmed the Comfy Creator payment was intentional. Keep the subscription and do not attempt cancellation. This authorizes subscription ownership only; generation spend, candidate promotion, deployment, and release remain separately gated.
