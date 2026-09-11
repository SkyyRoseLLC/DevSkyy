# Track C — dedicated Skyy Blender optimization phase

**READY_FOR_BLENDER.** The exact accepted input and currently served model are preserved in separately verified, immutable-by-policy backups. Blender, its glTF importer and its Draco bridge are available. This status means the dedicated production phase can begin its import/preflight; it does **not** mean an optimized model has been created, visually accepted or approved for runtime replacement.

No Blender model import, optimization, decimation, retopology, rig change, texture change, animation authoring, poster replacement, software change, browser work, staging modification, paid generation or deployment occurred in this track.

## Authority and preservation

The authoritative current production input and runtime asset are the **same file**:

`wordpress-theme/skyyrose-flagship-2/assets/models/skyy-mascot.glb`

SHA256: `c571e26fb126f992f735062d628c499a7690e926ca2bbefa70c9252fc8a780fc`.

An upstream authoring `.blend` has not been identified in the available repository or existing handoff. The two backups preserve the requested source-input and runtime roles; neither is represented as a newly discovered Blender source. Historical `skyy.glb` and `skyy-child*.glb` assets must not be substituted.

Backups under `.artifacts/v2-specialized-completion-20260906/track-c/immutable/`:

- `source/skyy-mascot.glb`: accepted canonical input.
- `runtime/skyy-mascot.glb`: exact currently served model.
- `reference/`: accepted stage poster, canonical portrait references, current renderer source/minified output, stage CSS, prior exact handoff and poster reproduction helper.

All 10 backups were compared byte-for-byte by SHA256, and every original was hashed again after capture. The two model backups match each other and the original. The policy forbids editing or overwriting backups; permissions were not changed. `backup-manifest.json` records absolute source/copy paths, bytes and hashes. `preservation-verification.txt` records successful checksum checks for the originals and backups.

Before production, repeat:

```sh
shasum -a 256 -c /Users/theceo/.codex/worktrees/7116/DevSkyy/.artifacts/v2-specialized-completion-20260906/track-c/immutable-checksums.sha256
```

Any mismatch is a stop condition. Do not update the manifest to conceal a change. A changed accepted source requires a separate capture and a new contract.

## Current cost and production target

| Property | Accepted input | Dedicated phase target |
|---|---:|---:|
| Triangles | 1,930,256 | **80,000–120,000 where fidelity permits** |
| Vertices | 1,030,595 | Reduce as topology permits; no invented vertex ceiling |
| GLB size | 6,058,568 bytes | **≤2,621,440 bytes (2.5 MiB)** |
| Bones | 18 | Preserve the 18-bone identity and current compatibility |
| Materials / draw calls | 1 / 1 | Retain unless a reviewed technical reason requires otherwise |
| Geometry + texture storage | 89.20 MiB estimated lower bound | Remeasure; peak GPU/driver memory remains unknown |

The 80k–120k range is the current instruction and replaces the earlier strict-under-100k proposal. Byte count and triangle count never override visual fidelity. If the range cannot preserve the accepted appearance, return a documented exception for review rather than silently degrading the character.

Reduction priorities, in order:

1. Face and recognizable facial identity.
2. Hair silhouette and recognizable curls.
3. Hands and visible finger shapes.
4. Clothing silhouette, construction, logos and proportions.
5. Major clothing folds.

Aggressive reduction belongs only in visually safe regions identified through actual inspection. Do not choose per-region triangle allocations before examining topology and deformation.

## Exact technical contract

`track-c-contract.json` incorporates the existing exact camera, light, transform, skin, bone, material, texture, animation and poster handoff. It also binds the contract to the preserved current renderer source.

- glTF coordinates are right-handed, +Y up. The source scene root `SkyyMascotRig` has identity translation, rotation and scale. Keep mesh and armature transformations together.
- Runtime measures the initial skinned bounds, scales uniformly to `1.8 / boundsHeight`, centers X/Z and moves minimum Y to zero. The accessor-based scale estimate is approximately 1.83763; it is informational and must not replace the actual runtime calculation.
- Camera: perspective 30° FOV, aspect 220/340, near 0.1, far 50, position `(0, 1.1, 4.1)`, looking at `(0, 0.9, 0)`.
- Lights: white ambient 1.4; warm `#fff5e6` key 1.8 at `(2, 4, 3)`; cool `#e6f0ff` fill 0.6 at `(-2, 2, -1)`. Runtime uses SRGB output, alpha, antialiasing, DPR cap 1.5, and no shadows/post effects.
- Preserve the 18 joint names/hierarchy, inverse bind matrices, `JOINTS_0` / `WEIGHTS_0`, normalized skinning and armature/mesh relationship. The JSON contains every current bone transform.
- The one material uses embedded normal, base-color and roughness/metallic JPEGs, each **1024 × 1024**. Names containing “8k” do not describe their actual dimensions. Retain UV alignment, normal response, color, logos and material appearance.
- Preserve `Skyy_Idle`, `Skyy_Walk`, `Skyy_Wave`, `Skyy_Talk`, `Skyy_Joy` and `Skyy_Exit`. Current source clips are held poses; the accepted website derives motion on the same rig. Retaining these clips and the existing hierarchy is the preferred compatibility path. This phase does not require new motion authoring.
- The poster state is the derived idle clip at time zero, same facing and zero entry shift, under the camera/lights above. It appears before the 300 ms handoff and real walk. Keep the current runtime poster untouched; any later candidate poster must remain isolated until the model is accepted.

## Ordered production checklist

1. Verify all backup hashes. Create a working copy under `track-c/working/`; never open an immutable path as an output destination.
2. Start the installed Blender 5.2.0 LTS and import the working GLB using its bundled glTF importer/Draco bridge. This import has not been run during preparation. If decoding, textures, skin, hierarchy or clips fail, stop and mark the phase **BLOCKED** with the exact error.
3. Confirm 18 joints, all six clip names, one material, embedded texture dimensions, neutral/bind transforms and the measured geometry counts. Check axes and scale before applying any transform. Save a new working `.blend`; do not imply that it is an upstream original.
4. Establish an untouched reference collection and comparison renders. Reproduce the website's accepted relaxed idle, walk/turn and conversation poses using the preserved runtime recipe. A raw import T-pose is not the accepted presentation.
5. Inspect topology and create a separate candidate mesh. Start near 100k triangles as a working objective; use the 80k–120k range only where the protected face, hair, hands, silhouette and folds remain intact. Keep the reference collection untouched.
6. Reduce visually safe density first. Use topology-aware reduction or retopology where needed; avoid a blind global reduction percentage. Preserve important boundaries, UVs, normals, clothing construction and silhouette.
7. Preserve or transfer weights to the same 18-bone structure. Check normalization and deformation at shoulders, elbows, wrists, fingers, hips, knees and feet; inspect hair and clothing intersections. Reject a geometry saving that introduces visible motion defects.
8. Keep the current animation compatibility path. Validate all six actions through the existing runtime derivation. If a proposed topology/rig change breaks compatibility, document it for review rather than introducing an unapproved rig or replacement motion system.
9. Export a new candidate GLB into `track-c/candidates/`, preserving material/texture/skin/animation identity and compatible Draco compression. Confirm actual exported byte size, triangle count, joint count and material count. Do not overwrite the runtime file.
10. Load the candidate only in an isolated review fixture. Compare face close-ups and full-body poses at the actual 96–220 px display sizes and at a larger review size. Review static, walk-in, turn, idle, greeting, talking, gesture, pause and reduced-motion presentations.
11. Report silhouette overlap, alpha bounds, camera/anchor alignment and any visible differences. These measurements support independent visual review; no single numeric score automatically approves identity changes. Verify source portrait and original model comparisons side-by-side.
12. After model acceptance, render an isolated candidate poster from the same runtime camera/lights/idle state. Verify a first stable frame before crossfade with no T-pose, blank canvas, position/scale jump or lighting flash. Keep accepted runtime assets unchanged until an explicit replacement gate is passed.
13. Rerun the current focused tests and browser lifecycle/continuity checks, then profile bytes, decode, initialization tasks, first stable frame, idle/active cadence and memory boundaries. Actual physical-device validation remains necessary for a device certificate.
14. Obtain independent technical and visual review, followed by founder disposition. Deliver the candidate, working file, provenance, measured deltas and review media. No automatic software wiring, packaging, promotion, deployment or staging write is authorized by this handoff.

## Preflight result and readiness boundary

The current model hash/counts and embedded dependencies match the accepted handoff. Blender executable, glTF importer, Draco import extension and native Draco bridge are present. No external model texture dependency needs recovery. This makes the phase actionable from the canonical GLB even though an upstream `.blend` is unavailable.

**READY_FOR_BLENDER** is a handoff classification. Import parity, optimized geometry quality, deformation, final bytes, candidate poster continuity, physical-device performance and founder acceptance remain work for the dedicated phase. If any first-import check fails, the next status becomes **BLOCKED** with that concrete cause.
