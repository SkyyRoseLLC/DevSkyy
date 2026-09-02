# Comfy production log

Append-only execution and decision log for recoverable local scene production.

- 2026-09-01 DECISION local-only routing; partner/API spend disabled; `LH-004`
  selected as the first compositor-ready product pilot.
- 2026-09-01 SOURCE `lh-004-bomber-front.jpg` SHA-256
  `7a00b1122a9f36e2d24221f011719776799726711b11c58f552f57e2134fb256` and
  `lh-004-bomber-back.png` SHA-256
  `e170a556f4ee444f273c4aacd2bfce569d0c87263397826b6e757db6402888f1` bound to
  candidate `LH-004-COMFY-LOCAL-001`.
- 2026-09-01 DECISION candidate output is non-promotable until product-fidelity,
  physical-integration, optical-realism, and founder-review gates pass.
- 2026-09-01 SUBMIT `lh-004-reference-pilot` ->
  `10db1189-4a6c-494e-a244-7af39ca8b55c`; local Z-Image reference-latent
  workflow; zero partner nodes; zero provider spend.
- 2026-09-01 CANCEL `10db1189-4a6c-494e-a244-7af39ca8b55c`; BF16 diffusion plus
  full text encoder drove the 16 GB host to about 20 GB swap before sampling. No
  output produced and no provider spend.
- 2026-09-01 DECISION replace recoverable BF16/full weights with official
  `z_image_turbo_int8_convrot.safetensors` and
  `qwen_3_4b_fp8_mixed.safetensors`; reduce pilot canvas from 896x1152 to
  768x1024; retain the same SOT bindings and acceptance gates.
- 2026-09-01 SUBMIT `lh-004-reference-pilot-int8` ->
  `ec03a00a-1ef3-45ea-9d4f-0e5aed4bd616`; local Int8/mixed-FP8 profile; zero
  partner nodes; zero provider spend.
- 2026-09-01 ERROR `ec03a00a-1ef3-45ea-9d4f-0e5aed4bd616`; KSampler reference
  latent shape was not divisible by the model 2x2 patch
  (`shape [1,16,71,2,57,2]`); no output and no provider spend.
- 2026-09-01 RECOVERY disable experimental auto-resize and normalize both
  physical references to nondistorting 1024x1024 white-padded canvases via
  `ResizeAndPadImage` before VAE reference encoding.
- 2026-09-01 SUBMIT `lh-004-reference-pilot-int8-shapefix` ->
  `e3f1f1ab-f0e6-4b64-b96b-3102ef8d2e94`; 14 local core nodes; reference
  canvases normalized to model patch requirements; zero provider spend.
- 2026-09-01 ERROR `e3f1f1ab-f0e6-4b64-b96b-3102ef8d2e94`; KSampler reached Int8
  execution but Apple MPS lacks `aten::_int_mm`; no output and no provider
  spend.
- 2026-09-01 RECOVERY relaunch local ComfyUI with
  `PYTORCH_ENABLE_MPS_FALLBACK=1` exactly as prescribed by the PyTorch error;
  unsupported Int8 matrix multiplication may execute on CPU while supported
  operations remain on MPS.
- 2026-09-01 SUBMIT `lh-004-reference-pilot-int8-mps-fallback` ->
  `a757c7fa-bd89-4444-9f2d-68577c84c3b9`; same validated graph under the
  required Apple MPS fallback runtime; zero provider spend.
- 2026-09-01 CANCEL `a757c7fa-bd89-4444-9f2d-68577c84c3b9`; CPU fallback was
  active and healthy but the first 768x1024 eight-step sample exceeded the
  useful local-preview latency budget. No output and no provider spend.
- 2026-09-01 DECISION local route is a viability preview: 512x768 output,
  512x512 nondistorted reference pads, four steps. A visually viable preview may
  advance to a separate higher-quality route; it cannot be promoted directly.
- 2026-09-01 SUBMIT `lh-004-reference-pilot-local-preview` ->
  `19432dde-d933-4a30-9f7f-33b5c116aade`; reduced local viability preview under
  required Apple MPS fallback; zero provider spend.
- 2026-09-01 CANCEL `19432dde-d933-4a30-9f7f-33b5c116aade`; reduced Int8
  CPU-fallback preview still exceeded the useful local latency window at step
  zero. No output and no provider spend.
- 2026-09-01 DECISION active Mac profile uses official
  `z_image_turbo_bf16.safetensors` with loader `weight_dtype=fp8_e4m3fn` plus
  `qwen_3_4b_fp8_mixed.safetensors`; this targets lower memory without the
  Int8-only `_int_mm` CPU fallback.
- 2026-09-01 SUBMIT `lh-004-reference-pilot-bf16-fp8` ->
  `9228ff19-48a1-44d0-9564-2581875148aa`; source-bound 512x768 four-step local
  viability preview; 14 core nodes, zero partner nodes, zero provider spend.
- 2026-09-01 ERROR `9228ff19-48a1-44d0-9564-2581875148aa`; KSampler rejected
  Float8 diffusion weights because Apple MPS does not support dtype
  `Float8_e4m3fn`; no output and no provider spend.
- 2026-09-01 RECOVERY retain the lower-memory mixed-FP8 text encoder but load
  the official BF16 diffusion model at its native dtype; this is the final local
  compatibility attempt before classifying the 16 GB Apple runtime as
  hardware-blocked for this model family.
- 2026-09-01 SUBMIT `lh-004-reference-pilot-bf16-native` ->
  `ee4d051a-a36a-44ac-a81a-542776d3c852`; native BF16 diffusion plus mixed-FP8
  text encoder at 512x768 and four steps; 14 core nodes, zero partner nodes,
  zero provider spend.
- 2026-09-01 CANCEL `ee4d051a-a36a-44ac-a81a-542776d3c852`; native BF16 reached
  KSampler but remained at step zero while host swap reached about 19.4 GB.
  Cancelled after 309 seconds with no output and no provider spend.
- 2026-09-01 VERDICT the source-bound graph validates with zero errors, zero
  warnings, zero partner nodes, and `spends_credits=false`, but Z-Image Turbo is
  hardware-blocked for practical rendering on this 16 GB Apple M5 runtime. No
  candidate exists for visual review or promotion.
- 2026-09-01 CLEANUP removed the exact recoverable
  `z_image_turbo_int8_convrot.safetensors` weight and its inactive reference
  template after MPS diagnosis, reclaiming about 6.2 GB. Native BF16, mixed-FP8
  text encoder, VAE, workflow, hashes, and receipts remain installed.
- 2026-09-01 AUTHORIZATION user confirmed the Comfy Creator payment was
  intentional. Keep the subscription and do not attempt cancellation. This
  authorizes subscription ownership only; generation spend, candidate promotion,
  deployment, and release remain separately gated.
- 2026-09-01 AUTHORIZATION user explicitly requested that paid Comfy execute the
  scenes; scope limited to one 1K `SIG-COMMERCE-1` Nano Banana Pro pilot.
  Candidate promotion and deployment remain unauthorized.
- 2026-09-01 PREFLIGHT `SIG-COMMERCE-1` contract
  `26d841a43c5fe8837fbbf75d9cce923ac9549006b33570b10b23b332e85d8f76` passed with
  zero findings; live `GeminiImage2Node` availability and paid Creator account
  verified.
- 2026-09-01 SUBMIT `sig-commerce-1-nano-banana-pro-pilot` ->
  `5415c764-c03b-4c7e-bde4-b87a87ec1ba7`; local ComfyUI orchestration with one
  paid Vertex AI partner node, two hash-bound image references, 2:3 at 1K,
  explicit spend authorization, candidate-only quarantine.
- 2026-09-01 COMPLETE `5415c764-c03b-4c7e-bde4-b87a87ec1ba7` in 30.06 seconds;
  output SHA-256
  `7035f14f07eaeb5d3deeca0dcbe5a9d69adcb74fef4f3b145bdfc3cbb2561a51`, 848x1264.
  Integration improved, but semantic redraw changed protected jacket, patch,
  pose, scale, and statue pixels; quarantined as integration reference only.
- 2026-09-01 ACCOUNT UI reports 7,400 partner credits available. CLI
  `cloud status` reports a conflicting zero-credit field; successful partner
  execution establishes that the paid route is usable. Run one bounded job at a
  time and treat the account UI as balance authority.
- 2026-09-01 SOURCE cleaned the exact Signature font sculpture deterministically
  to `signature-skyyrose-font-sculpture-clean-transparent-v2.png`, SHA-256
  `4db76780d8280af2d1d7c081b424ccd385054f15778593b240cec52e34b75c3e`; no
  generative redraw, exact opaque pixels preserved, contaminated low-alpha matte
  removed.
- 2026-09-01 SUBMIT `sig-commerce-1-protected-recovery-v1` ->
  `308a7cbe-7e52-42a5-a746-e9619c6a269f`; 37 local core nodes, zero partner
  nodes, zero credit spend. Uses the mapped SIG-COMMERCE-1 environment, cleaned
  exact font sculpture, and exact SG-009/SG-007 protected look; integration
  effects are placed outside protected pixels before exact layers are composited
  last.
- 2026-09-01 REJECT `308a7cbe-7e52-42a5-a746-e9619c6a269f`; founder verdict:
  looks terrible. Visible brown subject halo, mismatched studio exposure,
  floating-foot read, and pasted wall sculpture. Quarantined as
  `SIG-COMMERCE-1-comfy-protected-recovery-v1-founder-rejected.png`; do not
  reuse as an edit or generation target.
- 2026-09-01 SUBMIT `sig-commerce-1-native-correction-v2` ->
  `5be5a2ee-3e16-4d02-a3c5-64db4eccd8f3`; one paid Nano Banana Pro node with the
  successful native-integration base, exact SG-009/SG-007 protected look, and
  cleaned exact sculpture as three independent references. Candidate-only; no
  promotion or deployment authorization.
- 2026-09-01 REJECT `5be5a2ee-3e16-4d02-a3c5-64db4eccd8f3`; visually integrated
  but fails the campaign-design bar: customer/product too small and passive,
  excess empty wall, sculpture reads as decorative wall plaque instead of a
  physically intentional brand object. Quarantined as a design-standard failure.
- 2026-09-01 DIRECTION founder supplied an external campaign image for design
  inspiration only. It is not an authority asset and must not be uploaded,
  referenced by an image-generation node, copied, or used to remap a scene.
  Extract only the professional energy: product/customer dominance, confident
  campaign subject, physically grounded collection monument, shared
  light/floor/reflection, cinematic atmosphere, and strong editorial hierarchy.
- 2026-09-01 SUBMIT `sig-commerce-1-campaign-hero-v3` ->
  `81cea340-3386-46ff-a774-1d2468ed9d74`; one paid Nano Banana Pro node, mapped
  Signature environment plus exact SG-009/SG-007 customer and cleaned Signature
  sculpture. Sculpture placement flexible but must be physically intentional.
  Candidate-only; no promotion or deployment authorization.
- 2026-09-01 REJECT `81cea340-3386-46ff-a774-1d2468ed9d74`; founder verdict:
  looks terrible. Although customer/product scale improved, the portrait
  showroom composition still lacks the required powerful customer-driven
  campaign language and the sculpture was rearranged with its copper rose
  omitted. Stop iterating this composition. No sculpture-correction v4
  submitted.
- 2026-09-02 SOURCE restored the exact founder-confirmed BR-007 four-angle
  physical authority from Git commit `2253f7206`; SHA-256
  `7142815d09c35eff5de2b9c918340a6298f64dc84b66f9c142f37a1cf9c6b130`, 1504x1344.
  Registered as supplemental cross-view authority; individual
  front/back/left/right views remain compositor inputs.
- 2026-09-02 OODA `BR-COMMERCE-2` source preflight passed: catalog, image SOT,
  seven photoshoot inputs, and three runtime-media dependencies are hash-bound.
  Paid execution remains gated; no image job submitted and no credits spent.
- 2026-09-02 REJECT Higgsfield `hero_banner` enhancement receipt
  `br-commerce-2-higgsfield-enhance-rejected-product-drift-20260902T072153Z.json`;
  enhancer moved BR-005 artwork onto the sleeve and left chest. Preserved as
  rejection evidence; never generate from it.
- 2026-09-02 PASS corrected Higgsfield `hero_banner` enhancement receipt
  `br-commerce-2-higgsfield-enhance-20260902T072421816848Z.json`; independent
  assertion review preserved BR-005 right-chest/side-body placement, exact
  BR-007 board construction, BR-004 centered graphic, 16:9 hierarchy, and
  physical font sculpture. Enhancement only; no image generation, promotion, or
  deployment.
- 2026-09-02 OODA embedded `fashion-theme-team@personal` across all five
  remaining mapped scenes. Observe routes to catalog SOT, Orient to brand
  experience plus product-fidelity image edits, Act to separately approved
  provider runtime and Comfy correction, Decide to fresh independent
  visual-commerce QA, and release evidence to theme release engineering; founder
  remains the sole promotion authority.
- 2026-09-02 VALIDATE all five scene packets PASS configuration, dependency, and
  Fashion Theme Team contract checks. Paid execution remains fail-closed:
  BR-COMMERCE-2 awaits paid preflight approval; LH-COMMERCE-2 awaits approved
  on-model/matte authority; SIG-COMMERCE-1 awaits prompt review and paid
  preflight; SIG-COMMERCE-2 awaits the correct female authority; SIG-COMMERCE-3
  awaits exact shorts-correction receipts.
- 2026-09-02 SOURCE `LH-003-SOURCE-AUTHORITY-INTAKE-20260902` bound the
  founder-supplied physical front, back, wearer-left, and wearer-right views
  byte-for-byte. Founder product truth is explicit and fail-closed: exactly two
  side pockets plus two rear pockets, with zipper closures on all four. No
  render, provider call, Comfy job, or credit spend was performed;
  `LH-COMMERCE-2` remains blocked on an approved on-model wearer and protected
  garment matte.
- 2026-09-02 DIRECTION founder supplied the 1672x941 Black Rose Bay Bridge
  monument hero (`e5ebd5a0...`) and Love Hurts rose-aisle monument hero
  (`000ec830...`) as collection-scene inspiration. Both are hash-bound to their
  matching OODA contracts for world, material, wet-floor reflection, and
  monumental-scale cues only; customer and exact product remain primary. The
  temporary screenshot is direction-only and is not a generation input. BR's
  prior paid approval stays consumed and invalid for the revised prompt; LH
  remains source-blocked. No provider call, Comfy job, or credit spend was
  performed.
- 2026-09-02 OODA added fail-closed prompt-adherence contracts to the Black Rose
  and Love Hurts scenes. Preflight now verifies collection-matched source roles
  and prompt claims; post-capture review uses weighted product, hierarchy,
  collection-world, monument, and optical-integration dimensions with a 90/100
  minimum. Review receipts bind the independent reviewer, candidate SHA-256, and
  current executable contract fingerprint. Any product, identity, pocket,
  hierarchy, monument, or native-contact hard failure quarantines the candidate
  regardless of average score, and Comfy staging rejects a missing, stale,
  below-threshold, or hard-failed review. No provider call, Comfy job, or credit
  spend was performed.
