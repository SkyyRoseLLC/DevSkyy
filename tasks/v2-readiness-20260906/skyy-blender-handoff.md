# Ask Skyy Blender optimization handoff

**PREPARED, NOT AUTHORED.** Blender5.2.0 LTS is installed and its version was read. This pass does not import, edit, replace or export the approved model. No upstream .blend was found in the repository search; the canonical runtime GLB is the verified input. Historical skyy.glb/skyy-child variants are not substitutes.

The adjacent JSON is the exact machine-readable manifest, including all18 bone transforms/hierarchy, skin joint indices, material channels, texture sizes, root data, camera and lighting.

## Immutable input and current cost

- Source and runtime: `wordpress-theme/skyyrose-flagship-2/assets/models/skyy-mascot.glb`
- SHA256: `c571e26fb126f992f735062d628c499a7690e926ca2bbefa70c9252fc8a780fc`
- Current GLB6,058,568bytes;1,030,595 vertices;1,930,256 triangles; one draw call/material;18 joints.
- Typed geometry76,754,012bytes; decoded textures16,777,216bytes estimated; total89.20MiB lower bound, not peak memory.
- Three embedded1024×1024JPEGs: normal68,997bytes, base color165,623bytes, roughness/metallic122,080bytes. Their names contain8k but actual images are1024px.

## Transform, rig and render contract

- glTF right-handed,+Y-up. Scene root SkyyMascotRig has identity translation/rotation/scale. Keep mesh and armature transforms together. The source physical height is not independently certified.
- Runtime computes the actual initial skinned bounds and sets uniform root scale to1.8/boundsHeight. Accessor-only estimate1.8376298711764454 is informational; actual skinned bounds remain authoritative. It then centers X/Z and puts minimumY on0. Do not blindly bake the estimate.
- Preserve JOINTS_0,WEIGHTS_0,inverse-bind matrices and all18 named joints. Bone names are listed in the JSON; runtime sanitizes punctuation for matching.
- Perspective camera30° FOV, aspect220/340, near.1/far50; position(0,1.1,4.1),lookAt(0,.9,0). Logical canvas220×340,DPRcap1.5; alpha+antialias,SRGB output, no shadows/post-processing.
- Ambientwhite1.4; warmkey#fff5e6 intensity1.8 at(2,4,3); coolfill#e6f0ff intensity.6 at(-2,2,-1).
- Preserve Skyy_Idle, Skyy_Walk, Skyy_Wave, Skyy_Talk, Skyy_Joy, Skyy_Exit names. Current GLB holds54channels/action over18bones; six held poses are expanded into actual same-rig motion by deriveRigMotion. Do not confuse these with baked walk clips. Future approved varying clips retain precedence.

## Poster state and targets

- Poster `wordpress-theme/skyyrose-flagship-2/assets/images/skyy-runtime-poster.webp`, SHA256 `714388658e58754368007853909bbee0b7c97aa797b79d23762da26f4a0766dd`.
- Stop all actions, play derivedSkyy_Idle at time0, restore facing and shift0. Use the camera/lights above. Existing poster330×510 is losslessWebP. Render stable first frame before300ms handoff and actual walk; no T-pose, changed scale or empty canvas.
- Target fewer than100,000 triangles where fidelity allows and GLB≤2,621,440bytes(2.5MiB). Keep face, curls, hands, clothing construction, logos, silhouette, UV/material identity and rig. A byte/triangle target never authorizes visual degradation.

## Production checklist

1. Copy hash-verified canonical GLB into isolated Blender session; retain immutable original.
2. Verify import hierarchy, axes, bound height, neutral bone and18-joint skin; record no missing texture warnings.
3. Save working .blend outside runtime assets; preserve camera/material baseline renders.
4. Create LOD candidates with topology-aware reduction; prioritize clothing interior/hidden surfaces, preserve curls, face, hands, logos, seams and silhouette.
5. Keep UVs, normals, material channels and skinning intact; review deformation at every joint before accepting topology.
6. Target below100k triangles and <=2.5MiB without silently accepting identity loss.
7. Retain six exact animation names and bind transforms; export Draco with one primitive/material where possible.
8. Compare static idle and walk/turn/wave/talk/joy/exit at96/220px and close-up; reject weighting, texture, logo, scale or lighting drift.
9. Validate glTF, bounds, bones, clip payloads, texture dimensions and network size; record hashes.
10. Regenerate same-camera time-zero poster only after model acceptance; compare silhouette/alpha bounds and actual first stable frame.
11. Run independent source/visual review plus current33 focused contract tests, browser states, Metal and physical mobile performance when available.
12. Provide candidate and receipts for founder review; do not replace runtime model, package, promote or deploy automatically.

The eventual candidate must be independently compared against the frozen current model and approved portrait, then founder-reviewed. This package does not authorize runtime promotion, paid generation, packaging, deployment or staging modification.
