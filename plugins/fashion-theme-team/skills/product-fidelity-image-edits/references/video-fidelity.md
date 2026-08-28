# Video fidelity

Video introduces temporal drift even when the first frame is correct. Logos, embroidery, seams, lettering, hardware, color, silhouette, and construction must be checked through time.

## Safe routes

For exact-product motion, use a protected product/model layer, tracked placement, deterministic camera movement, 2.5D parallax, or a verified 3D asset. Generate or modify only the environment, then composite the protected layer and retain its hashes.

A generative image-to-video output is review-candidate only. Bind the exact first-frame hash; optional last-frame hash; provider/model/version and request parameters; duration, fps, dimensions, seed where supported, audio policy; product region; and frame-sampling plan.

## Required temporal gates

1. Decode with a pinned toolchain and record duration, codec, fps/time base, dimensions, frame count, and audio streams.
2. Extract first, last, every scene-change frame, and at least one frame per second. Increase sampling for fast motion.
3. Track the product region. Numeric/embedding similarity is triage only; visually verify wording, color, construction, and placement.
4. Veto disappearing patches, changing letters, seam migration, panel swaps, extra anatomy, texture boil, identity drift, or unexplained product occlusion.
5. If the provider contract promises a locked first frame, compare it exactly. Reject if it changed.
6. Never promote a video frame as new product authority.

OpenAI Sora, Google Veo/Gemini video, Runway, Luma, Kling, Seedance, Higgsfield-routed models, Adobe Firefly Video, and Midjourney Video remain generative. References and remix/edit controls guide them; they do not guarantee product pixels across time.

Keep the request contract, source hashes, provider receipt, raw output, decoded metadata, contact sheet, frame findings, final hash, founder verdict, and promotion receipt. Candidate completion is not approval.
