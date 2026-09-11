from __future__ import annotations

import json
import struct
from pathlib import Path

from skyyrose.elite_studio.pipeline3d.deterministic_gates import ArtifactGateRunner
from skyyrose.elite_studio.pipeline3d.fidelity import GateStatus
from skyyrose.elite_studio.pipeline3d.platform_contracts import QualityPolicy


def _write_glb(path: Path) -> None:
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "accessors": [
            {
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [-0.5, 0.0, -0.1],
                "max": [0.5, 1.0, 0.1],
            },
            {"componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "textures": [{"source": 0}],
        "images": [{"bufferView": 0, "mimeType": "image/jpeg"}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 4}],
        "buffers": [{"byteLength": 4}],
    }
    json_bytes = json.dumps(document, separators=(",", ":")).encode()
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    binary = b"test"
    length = 12 + 8 + len(json_bytes) + 8 + len(binary)
    path.write_bytes(
        struct.pack("<III", 0x46546C67, 2, length)
        + struct.pack("<II", len(json_bytes), 0x4E4F534A)
        + json_bytes
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def test_glb_runner_emits_independent_numeric_receipts(tmp_path: Path) -> None:
    glb = tmp_path / "product.glb"
    _write_glb(glb)

    receipts = ArtifactGateRunner(QualityPolicy()).verify_glb(
        glb,
        producer_id="trellis-worker",
        expected_dimensions_m=(1.0, 1.0, 0.2),
    )
    by_gate = {receipt.gate: receipt for receipt in receipts}

    assert by_gate["glb_structure"].status == GateStatus.PASS
    assert by_gate["mesh_topology"].metrics["triangles"] == 1
    assert by_gate["materials_pbr"].status == GateStatus.PASS
    assert by_gate["dimensions"].status == GateStatus.PASS
    assert by_gate["glb_structure"].producer_id != by_gate["glb_structure"].verifier_id


def test_dimensions_block_without_product_measurements(tmp_path: Path) -> None:
    glb = tmp_path / "product.glb"
    _write_glb(glb)

    receipts = ArtifactGateRunner(QualityPolicy()).verify_glb(
        glb,
        producer_id="trellis-worker",
    )
    by_gate = {receipt.gate: receipt for receipt in receipts}

    assert by_gate["dimensions"].status == GateStatus.BLOCKED
