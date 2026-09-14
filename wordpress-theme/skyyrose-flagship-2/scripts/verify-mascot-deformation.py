"""Read-only evaluation of the delivered Skyy blend in background Blender.
Run with --python-exit-code 1 and pass -- /absolute/report.json.
Thresholds reject gross distortion; visual motion review is still required.
"""

import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np

root = Path(bpy.data.filepath).resolve().parent
output = (
    Path(sys.argv[sys.argv.index("--") + 1])
    if "--" in sys.argv
    else root / "mascot-verification.json"
)
rig = bpy.data.objects["SkyyCandidateRig"]
scene = bpy.context.scene
for t in rig.animation_data.nla_tracks:
    t.mute = True
report = {
    "blend_sha256": hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
    "meshes": [],
    "max_rotation_key_step_degrees": 0.0,
    "limits": [
        "Finite positions and loop closure do not prove anatomical or artistic quality.",
        "Engineering guard: maximum edge ratio <=3, p99 <=2, p01 >=0.3; rejects gross stretching/collapse, not an artistic quality certificate.",
        "Minimum mesh height does not establish planted-foot position or absence of foot sliding.",
    ],
}
for name in ["SkyyCandidateMesh", "SkyyMobileMesh"]:
    obj = bpy.data.objects[name]
    rest = np.array([v.co[:] for v in obj.data.vertices])
    edges = np.array([e.vertices[:] for e in obj.data.edges])
    region = (np.abs(rest[:, 0]) > 0.06) & (rest[:, 2] > 0.58) & (rest[:, 2] < 0.714)
    edges = edges[region[edges].all(axis=1)]
    lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    keep = lengths > 0.001
    edges = edges[keep]
    lengths = lengths[keep]
    clips = []

    def coords():
        data = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
        a = np.empty(len(data.vertices) * 3, dtype=np.float32)
        data.vertices.foreach_get("co", a)
        return a.reshape(-1, 3)

    for track in rig.animation_data.nla_tracks:
        a = track.strips[0].action
        rig.animation_data.action = a
        rig.animation_data.action_slot = a.slots[0]
        end = int(a.frame_range[1])
        max99 = 0
        maxratio = 0
        min01 = 1
        minz = 1
        maxz = -1
        finite = True
        first = None
        last = None
        for frame in np.arange(1, end + 0.1, 0.5):
            scene.frame_set(int(frame), subframe=float(frame % 1))
            p = coords()
            finite = finite and bool(np.isfinite(p).all())
            minz = min(minz, float(p[:, 2].min()))
            maxz = max(maxz, float(p[:, 2].min()))
            ratios = np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1) / lengths
            max99 = max(max99, float(np.quantile(ratios, 0.99)))
            maxratio = max(maxratio, float(ratios.max()))
            min01 = min(min01, float(np.quantile(ratios, 0.01)))
            if frame == 1:
                first = p.copy()
            if frame == end:
                last = p.copy()
        clips.append(
            {
                "name": track.name,
                "samples": 2 * end - 1,
                "finite": finite,
                "minimum_surface_range": [minz, maxz],
                "endpoint_error": float(np.linalg.norm(last - first, axis=1).max()),
                "edge_stretch_p99": max99,
                "edge_stretch_max": maxratio,
                "edge_compression_p01": min01,
            }
        )
    report["meshes"].append(
        {
            "name": name,
            "max_weight_sum_error": max(
                abs(sum(g.weight for g in v.groups) - 1) for v in obj.data.vertices
            ),
            "invalid_weight_count": sum(
                1
                for v in obj.data.vertices
                for g in v.groups
                if not np.isfinite(g.weight)
                or g.weight < 0
                or g.weight > 1.00002
                or (
                    g.weight > 0
                    and (
                        obj.vertex_groups[g.group].name not in rig.data.bones
                        or not rig.data.bones[obj.vertex_groups[g.group].name].use_deform
                    )
                )
            ),
            "clips": clips,
        }
    )
errors = []
expected_samples = {
    "Skyy_Idle": 241,
    "Skyy_Walk": 67,
    "Skyy_Wave": 169,
    "Skyy_Talk": 193,
    "Skyy_Joy": 91,
    "Skyy_Exit": 67,
}
if scene.render.fps != 30 or scene.render.fps_base != 1:
    errors.append(["scene", "unexpected animation frame rate"])

# Audit quaternion hemisphere continuity without modifying any keys.
for track in rig.animation_data.nla_tracks:
    action = track.strips[0].action
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                paths = {
                    c.data_path for c in bag.fcurves if c.data_path.endswith("rotation_quaternion")
                }
                for path in paths:
                    curves = sorted(
                        [c for c in bag.fcurves if c.data_path == path], key=lambda c: c.array_index
                    )
                    if len(curves) != 4:
                        errors.append([action.name, path, "missing quaternion channel"])
                        continue
                    values = np.array([[p.co.y for p in c.keyframe_points] for c in curves]).T
                    if np.any(np.sum(values[1:] * values[:-1], axis=1) < 0):
                        errors.append([action.name, path, "quaternion sign discontinuity"])
                    norms = np.linalg.norm(values, axis=1)
                    if len(values) < 2 or not np.isfinite(values).all() or np.any(norms < 1e-8):
                        errors.append([action.name, path, "invalid quaternion samples"])
                        continue
                    unit = values / norms[:, None]
                    step = float(
                        np.max(
                            np.degrees(
                                2
                                * np.arccos(
                                    np.clip(np.abs(np.sum(unit[1:] * unit[:-1], axis=1)), 0, 1)
                                )
                            )
                        )
                    )
                    report["max_rotation_key_step_degrees"] = max(
                        report["max_rotation_key_step_degrees"], step
                    )
                    # Calm mascot gestures must not snap through large rotations in one 30fps key interval.
                    if step > 15:
                        errors.append(
                            [action.name, path, "rotation step exceeds 15 degrees per frame"]
                        )


for mesh in report["meshes"]:
    if len(mesh["clips"]) != 6 or {c["name"] for c in mesh["clips"]} != set(expected_samples):
        errors.append([mesh["name"], "missing or duplicate required clips"])
    if (
        mesh["invalid_weight_count"]
        or not np.isfinite(mesh["max_weight_sum_error"])
        or mesh["max_weight_sum_error"] > 2e-5
    ):
        errors.append([mesh["name"], "weights are not normalized"])
    for c in mesh["clips"]:
        if c["samples"] != expected_samples.get(c["name"]):
            errors.append([mesh["name"], c["name"], "unexpected sample count"])
        if not c["finite"] or c["endpoint_error"] > 1e-4:
            errors.append([mesh["name"], c["name"], "finite/loop"])
        if (
            c["edge_stretch_max"] > 3.0
            or c["edge_stretch_p99"] > 2.0
            or c["edge_compression_p01"] < 0.3
        ):
            errors.append([mesh["name"], c["name"], "deformation guard"])
        if c["minimum_surface_range"][0] < -0.003:
            errors.append([mesh["name"], c["name"], "ground penetration"])
assert hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest() == report["blend_sha256"], (
    "Input blend changed during audit"
)
report["passed"] = not errors
report["errors"] = errors
report["expected_samples_per_mesh"] = sum(expected_samples.values())
output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
print("DEFORMATION_GATE", errors)
if errors:
    raise RuntimeError(str(errors))
