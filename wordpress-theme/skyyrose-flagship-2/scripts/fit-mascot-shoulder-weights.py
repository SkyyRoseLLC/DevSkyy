import json
from pathlib import Path

import bpy

root = Path(bpy.data.filepath).resolve().parent


def smooth(t):
    t = max(0, min(1, t))
    return t * t * (3 - 2 * t)


def span(x, knots):
    if x <= knots[0][0]:
        return [(knots[0][1], 1)]
    if x >= knots[-1][0]:
        return [(knots[-1][1], 1)]
    for (lo, a), (hi, b) in zip(knots, knots[1:]):
        if lo <= x <= hi:
            if a == b:
                return [(a, 1)]
            t = smooth((x - lo) / (hi - lo))
            return [(a, 1 - t), (b, t)]


result = {}
for name in ["SkyyCandidateMesh", "SkyyMobileMesh"]:
    obj = bpy.data.objects[name]
    ids = [v.index for v in obj.data.vertices if v.co.z > 0.575]
    for g in obj.vertex_groups:
        g.remove(ids)
    weights = {}
    for index in ids:
        x, y, z = obj.data.vertices[index].co
        ax = abs(x)
        side = "L" if x >= 0 else "R"
        knots = [
            (0.065, "chest"),
            (0.105, "shoulder_blend." + side),
            (0.15, "upper_arm." + side),
            (0.2025, "upper_arm." + side),
            (0.24, "elbow_blend." + side),
            (0.2775, "forearm." + side),
            (0.33, "forearm." + side),
            (0.36, "wrist_blend." + side),
            (0.385, "hand." + side),
            (0.402, "hand." + side),
            (0.426, "grip." + side),
            (0.451, "fingertips." + side),
        ]
        start = 0.645 + 0.03 * smooth((ax - 0.065) / 0.105)
        neck = smooth((z - start) / 0.10)
        head = smooth((z - 0.69) / 0.025)
        sleeve = smooth((z - 0.58) / 0.04)
        parts = {}

        def add(n, w):
            parts[n] = parts.get(n, 0) + w

        for n, w in span(ax, knots):
            add(n, w * sleeve * (1 - neck) * (1 - head))
        add("chest", (1 - sleeve) * (1 - neck) * (1 - head))
        add("neck", neck * (1 - head))
        add("head", head)
        for n, w in parts.items():
            if w > 1e-8:
                weights.setdefault((n, round(w, 6)), []).append(index)
    for (n, w), indices in weights.items():
        obj.vertex_groups[n].add(indices, w, "REPLACE")
    result[name] = {
        "vertices": len(ids),
        "max_weight_error": max(
            abs(sum(g.weight for g in v.groups) - 1) for v in obj.data.vertices
        ),
    }
(root / "shoulder-field-verification.json").write_text(json.dumps(result, indent=2))
