"""Independent raw-GLB gates that emit certification receipts.

The parser reads the final binary artifact directly and imports no generator,
postprocessor or exporter code.  It is intentionally conservative: unsupported
geometry, missing measurements and incomplete PBR data block certification.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fidelity import AuthorityKind, GateReceipt, GateStatus
from .platform_contracts import QualityPolicy


class GLBValidationError(RuntimeError):
    """The final artifact is not a structurally valid GLB 2.0 file."""


@dataclass(frozen=True, slots=True)
class ParsedGLB:
    document: dict[str, Any]
    file_size: int


class ArtifactGateRunner:
    """Run independent numeric checks against a completed GLB."""

    verifier_id = "skyyrose-raw-glb-gate/v1"

    def __init__(self, policy: QualityPolicy) -> None:
        self.policy = policy

    def verify_glb(
        self,
        path: Path,
        *,
        producer_id: str,
        expected_dimensions_m: tuple[float, float, float] | None = None,
        dimension_tolerance: float = 0.05,
        source_revision: str | None = None,
    ) -> tuple[GateReceipt, ...]:
        artifact_hash = _sha256(path) if path.is_file() else None
        try:
            parsed = _parse_glb(path)
        except (OSError, GLBValidationError) as exc:
            return (
                self._receipt(
                    gate="glb_structure",
                    status=GateStatus.FAIL,
                    producer_id=producer_id,
                    path=path,
                    artifact_hash=artifact_hash,
                    source_revision=source_revision,
                    metrics={"error": str(exc)},
                ),
            )

        document = parsed.document
        structure_metrics = {
            "file_bytes": parsed.file_size,
            "scenes": len(document.get("scenes", [])),
            "nodes": len(document.get("nodes", [])),
            "meshes": len(document.get("meshes", [])),
            "accessors": len(document.get("accessors", [])),
        }
        structure_pass = (
            0 < parsed.file_size <= self.policy.max_glb_bytes
            and structure_metrics["scenes"] > 0
            and structure_metrics["nodes"] > 0
            and structure_metrics["meshes"] > 0
            and structure_metrics["accessors"] > 0
        )

        triangles, topology_errors = _triangle_count(document)
        topology_pass = not topology_errors and 0 < triangles <= self.policy.max_triangles

        materials = document.get("materials", [])
        textures = document.get("textures", [])
        images = document.get("images", [])
        pbr_count = sum(1 for material in materials if "pbrMetallicRoughness" in material)
        material_pass = bool(
            materials
            and textures
            and images
            and pbr_count == len(materials)
            and all(image.get("mimeType") in {"image/jpeg", "image/png", "image/webp"} for image in images)
        )

        actual_dimensions = _accessor_dimensions(document)
        dimension_status = GateStatus.BLOCKED
        dimension_metrics: dict[str, float | int | str | bool] = {
            "reason": "approved physical product dimensions were not provided"
        }
        if actual_dimensions is not None and expected_dimensions_m is not None:
            deltas = tuple(
                abs(actual - expected) / expected if expected > 0 else math.inf
                for actual, expected in zip(actual_dimensions, expected_dimensions_m, strict=True)
            )
            dimension_status = (
                GateStatus.PASS if max(deltas) <= dimension_tolerance else GateStatus.FAIL
            )
            dimension_metrics = {
                "actual_x_m": actual_dimensions[0],
                "actual_y_m": actual_dimensions[1],
                "actual_z_m": actual_dimensions[2],
                "expected_x_m": expected_dimensions_m[0],
                "expected_y_m": expected_dimensions_m[1],
                "expected_z_m": expected_dimensions_m[2],
                "max_relative_delta": max(deltas),
                "tolerance": dimension_tolerance,
            }
        elif actual_dimensions is None:
            dimension_metrics = {"reason": "POSITION accessors lack numeric min/max bounds"}

        return (
            self._receipt(
                gate="glb_structure",
                status=GateStatus.PASS if structure_pass else GateStatus.FAIL,
                producer_id=producer_id,
                path=path,
                artifact_hash=artifact_hash,
                source_revision=source_revision,
                metrics=structure_metrics,
            ),
            self._receipt(
                gate="mesh_topology",
                status=GateStatus.PASS if topology_pass else GateStatus.FAIL,
                producer_id=producer_id,
                path=path,
                artifact_hash=artifact_hash,
                source_revision=source_revision,
                metrics={
                    "triangles": triangles,
                    "max_triangles": self.policy.max_triangles,
                    "errors": " | ".join(topology_errors),
                },
            ),
            self._receipt(
                gate="materials_pbr",
                status=GateStatus.PASS if material_pass else GateStatus.FAIL,
                producer_id=producer_id,
                path=path,
                artifact_hash=artifact_hash,
                source_revision=source_revision,
                metrics={
                    "materials": len(materials),
                    "pbr_materials": pbr_count,
                    "textures": len(textures),
                    "images": len(images),
                },
            ),
            self._receipt(
                gate="dimensions",
                status=dimension_status,
                producer_id=producer_id,
                path=path,
                artifact_hash=artifact_hash,
                source_revision=source_revision,
                metrics=dimension_metrics,
            ),
        )

    def _receipt(
        self,
        *,
        gate: str,
        status: GateStatus,
        producer_id: str,
        path: Path,
        artifact_hash: str | None,
        source_revision: str | None,
        metrics: dict[str, float | int | str | bool],
    ) -> GateReceipt:
        return GateReceipt(
            gate=gate,
            status=status,
            authority=AuthorityKind.DETERMINISTIC,
            producer_id=producer_id,
            verifier_id=self.verifier_id,
            evidence=(str(path),),
            metrics=metrics,
            artifact_sha256=artifact_hash,
            source_revision=source_revision,
        )


def _parse_glb(path: Path) -> ParsedGLB:
    data = path.read_bytes()
    if len(data) < 20:
        raise GLBValidationError("file is too small to contain a GLB header and JSON chunk")
    magic, version, declared_length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise GLBValidationError("invalid GLB magic")
    if version != 2:
        raise GLBValidationError(f"unsupported GLB version: {version}")
    if declared_length != len(data):
        raise GLBValidationError(
            f"declared length {declared_length} does not match file size {len(data)}"
        )

    offset = 12
    document: dict[str, Any] | None = None
    while offset < len(data):
        if offset + 8 > len(data):
            raise GLBValidationError("truncated chunk header")
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        end = offset + chunk_length
        if end > len(data):
            raise GLBValidationError("chunk extends beyond declared file length")
        if chunk_type == 0x4E4F534A:
            if document is not None:
                raise GLBValidationError("multiple JSON chunks")
            try:
                decoded = json.loads(data[offset:end].decode("utf-8").rstrip(" \t\r\n\x00"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise GLBValidationError(f"invalid JSON chunk: {exc}") from exc
            if not isinstance(decoded, dict):
                raise GLBValidationError("JSON chunk root is not an object")
            document = decoded
        offset = end

    if offset != len(data):
        raise GLBValidationError("chunk table does not terminate at file boundary")
    if document is None:
        raise GLBValidationError("missing JSON chunk")
    if document.get("asset", {}).get("version") != "2.0":
        raise GLBValidationError("asset.version is not 2.0")
    return ParsedGLB(document=document, file_size=len(data))


def _triangle_count(document: dict[str, Any]) -> tuple[int, list[str]]:
    accessors = document.get("accessors", [])
    errors: list[str] = []
    total = 0
    for mesh_index, mesh in enumerate(document.get("meshes", [])):
        primitives = mesh.get("primitives", [])
        if not primitives:
            errors.append(f"mesh {mesh_index} has no primitives")
        for primitive_index, primitive in enumerate(primitives):
            if primitive.get("mode", 4) != 4:
                errors.append(f"mesh {mesh_index} primitive {primitive_index} is not TRIANGLES")
                continue
            position = primitive.get("attributes", {}).get("POSITION")
            if not isinstance(position, int) or not 0 <= position < len(accessors):
                errors.append(f"mesh {mesh_index} primitive {primitive_index} lacks POSITION")
            indices = primitive.get("indices")
            if not isinstance(indices, int) or not 0 <= indices < len(accessors):
                errors.append(f"mesh {mesh_index} primitive {primitive_index} lacks indices")
                continue
            count = accessors[indices].get("count")
            if not isinstance(count, int) or count <= 0 or count % 3:
                errors.append(f"mesh {mesh_index} primitive {primitive_index} has invalid index count")
                continue
            total += count // 3
    return total, errors


def _accessor_dimensions(document: dict[str, Any]) -> tuple[float, float, float] | None:
    minima = [math.inf, math.inf, math.inf]
    maxima = [-math.inf, -math.inf, -math.inf]
    found = False
    accessors = document.get("accessors", [])
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            index = primitive.get("attributes", {}).get("POSITION")
            if not isinstance(index, int) or not 0 <= index < len(accessors):
                continue
            accessor = accessors[index]
            low = accessor.get("min")
            high = accessor.get("max")
            if not (
                isinstance(low, list)
                and isinstance(high, list)
                and len(low) == 3
                and len(high) == 3
                and all(isinstance(value, (float, int)) for value in low + high)
            ):
                continue
            found = True
            for axis in range(3):
                minima[axis] = min(minima[axis], float(low[axis]))
                maxima[axis] = max(maxima[axis], float(high[axis]))
    if not found:
        return None
    dimensions = tuple(maxima[axis] - minima[axis] for axis in range(3))
    if any(not math.isfinite(value) or value <= 0 for value in dimensions):
        return None
    return dimensions  # type: ignore[return-value]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["ArtifactGateRunner", "GLBValidationError", "ParsedGLB"]
