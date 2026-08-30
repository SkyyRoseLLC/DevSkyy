# 3D product fidelity

Single-image 3D generation invents unseen geometry. A plausible mesh is not an exact replica.

## Source tiers

1. Exact authority: manufacturer CAD/pattern, verified scan/photogrammetry, CLO/Marvelous Designer file, or founder-approved measured mesh.
2. Multi-view reconstruction: synchronized front, left, right, back, top/bottom/detail views of the same physical SKU and configuration, with scale evidence.
3. Candidate-only: single flatlay, model shot, generated multi-view sheet, video frame, filename-only match, or another generated mesh.

If unseen regions contain artwork, panels, pockets, trim, hardware, embroidery, or different construction, one front view is a hard stop.

## Preflight

- Bind SKU, hashes, views, physical dimensions, units, axes, expected symmetry, garment category, and deliverables.
- Verify real alpha and protect silhouette, straps, drawstrings, cuffs, and transparent materials during background removal.
- Reject inconsistent variants across views.
- Record provider/model/version, parameters, remesh/decimation, PBR flags, texture resolution, formats, and paid-call approval.

## Geometry gate

- Load GLB/GLTF as a scene to preserve instances and multiple meshes.
- Record nodes, meshes, primitives, materials, textures, vertices, faces, components, bounds, transforms, and units.
- Check NaN/Inf, degenerate faces, non-manifold edges, winding/normals, self-intersections where available, duplicate shells, fragments, and extreme thin surfaces.
- Watertightness is required for solid/printable objects, not universally for layered garments. Declare the expected topology class.
- Compare measurements and silhouettes from front, back, sides, and three-quarter views.

## UV and material gate

- Require UV coverage and inspect overlap policy; reject missing or zero-area islands.
- Bind base color, normal, roughness, metallic, occlusion, emissive, and alpha maps by hash.
- Validate color spaces, alpha mode, double-sidedness, orientation, seams, texel density, and material assignment.
- Detect unintentionally baked lighting in base color.
- Embroidery, tackle twill, silicone, embossing, sublimation, satin, sherpa, mesh, and rib knit require distinct geometry/material treatment; flat color is not equivalent.

## Render comparison gate

- Render deterministic front, back, left, right, and three-quarter views under fixed neutral lighting and color management.
- Compare only corresponding views. A side source never proves a back.
- Use registered overlays, silhouette/landmark checks, close-up crops, and human review.
- Validate GLB/GLTF in two independent viewers. Keep master and web-decimated assets separate.

## Provider guidance

- TRELLIS.2 produces high-resolution PBR candidates and prefers alpha-masked foreground input; hidden structure remains inferred.
- Meshy 7 accepts 1–4 images and treats the first as front; use verified same-object views and cardinal thumbnails.
- Tripo v3.1 exposes image-to-model, texture, PBR, orientation, and autofix controls; all post-gates still apply.
- Hunyuan3D, SPAR3D, Stable Fast 3D, TripoSR, and InstantMesh are candidate reconstruction routes.
- Blender, CLO/Marvelous Designer, and CAD are preferred for exact measurements, panels, seams, or revision control.

Hard-veto wrong view/SKU/variant/hash, unseen regions claimed exact, logo drift hidden by one render, missing material maps, geometry reviewed only by a turntable, or direct promotion to SOT without founder approval.
