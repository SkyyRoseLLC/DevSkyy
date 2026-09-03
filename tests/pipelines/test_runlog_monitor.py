"""Run-event log + live monitor: event emission, aggregation, and resilience.

No API keys, no network: a fake client + fake QC gate drive ``render_sku`` and
the assertions check the JSONL event stream and the monitor's aggregate view —
the contract the live dashboard depends on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.oai_render import config, pipeline
from scripts.oai_render.client import ImageCallResult
from scripts.oai_render.monitor import aggregate, snapshot
from scripts.oai_render.pipeline import SkuPlan
from scripts.oai_render.qc import QCVerdict
from scripts.oai_render.runlog import RunLog, load_events


def _plan() -> SkuPlan:
    return SkuPlan(
        sku="br-001",
        name="BLACK Rose Crewneck",
        collection="black-rose",
        output_slug="black-rose-crewneck",
        references=[SimpleNamespace(path=Path(__file__), label="real garment photo")],
        prompt="render it",
    )


class _FakeClient:
    def __init__(self):
        self.calls = 0

    def edit_with_metadata(self, *, prompt, image_paths, expected_input_sha256s=None):
        self.calls += 1
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        input_hashes = expected_input_sha256s or tuple(
            hashlib.sha256(str(path).encode("utf-8")).hexdigest() for path in image_paths
        )
        contract = {
            "contract_id": "skyyrose-product-image-candidate-v1",
            "authority_state": "CANDIDATE_ONLY",
            "operation": "product_image_candidate",
            "provider": "openai",
            "api_surface": "images",
            "endpoint": "/v1/images/edits",
            "requested_model": "gpt-image-2-2026-04-21",
            "prompt_sha256": prompt_sha256,
            "request_image_sha256s": list(input_hashes),
            "mask_sha256": None,
            "request_parameters": {
                "model": "gpt-image-2-2026-04-21",
                "quality": "high",
                "size": "1024x1536",
                "output_format": "png",
                "background": "auto",
                "n": 1,
            },
            "release_authority": False,
        }
        return ImageCallResult(
            data=b"png-bytes",
            request_id=f"req_runlog_{self.calls}",
            requested_model="gpt-image-2-2026-04-21",
            sdk_version="test",
            endpoint="/v1/images/edits",
            request_parameters=contract["request_parameters"],
            ordered_input_sha256s=input_hashes,
            mask_sha256=None,
            prompt_sha256=prompt_sha256,
            job_contract=contract,
            contract_sha256=hashlib.sha256(
                json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        )


class _FakeGate:
    """Scripted verdicts, one per call — exercises the retry/quarantine path."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)

    def check(self, data, exp):
        return self._verdicts.pop(0)


def test_runlog_appends_valid_jsonl_verbatim(tmp_path):
    # Size discipline belongs to emitters (QCVerdict caps analysis at construction);
    # the log layer writes fields verbatim with a timestamp.
    rl = RunLog(path=tmp_path / "run.jsonl")
    rl.emit("qc_verdict", sku="br-001", analysis="saw sherpa in the hood")
    rl.emit("accepted", sku="br-001", path="/out.png")
    events = load_events(rl.path)
    assert [e["event"] for e in events] == ["qc_verdict", "accepted"]
    assert events[0]["analysis"] == "saw sherpa in the hood"
    assert all("ts" in e for e in events)


def test_load_events_skips_torn_tail_line(tmp_path):
    p = tmp_path / "run.jsonl"
    p.write_text(json.dumps({"event": "run_start"}) + "\n" + '{"event": "attem', encoding="utf-8")
    events = load_events(p)
    assert [e["event"] for e in events] == ["run_start"]


def test_render_sku_emits_attempt_verdict_accept_sequence(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    rl = RunLog(path=tmp_path / "run.jsonl")
    gate = _FakeGate([QCVerdict(passed=True, analysis="hood sherpa visible, logo correct")])

    result = pipeline.render_sku(_plan(), _FakeClient(), gate=gate, runlog=rl)

    assert result.status == "rendered"
    events = load_events(rl.path)
    assert [e["event"] for e in events] == [
        "attempt",
        "provider_response",
        "qc_verdict",
        "accepted",
    ]
    assert events[2]["passed"] is True
    assert "sherpa" in events[2]["analysis"]
    assert ".candidate.req_runlog_1.png" in events[3]["path"]


def test_render_sku_emits_quarantine_then_exhausted(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "REJECTED_DIR", tmp_path / "rej")
    monkeypatch.setattr(config, "QC_MAX_RENDER_RETRIES", 1)
    rl = RunLog(path=tmp_path / "run.jsonl")
    fail = QCVerdict(passed=False, failure_tags=("wrong_garment",), reason="wrong hue")
    gate = _FakeGate([fail, fail])

    result = pipeline.render_sku(_plan(), _FakeClient(), gate=gate, runlog=rl)

    assert result.status == "qc_failed"
    kinds = [e["event"] for e in load_events(rl.path)]
    assert kinds == [
        "attempt",
        "provider_response",
        "qc_verdict",
        "quarantined",
        "attempt",
        "provider_response",
        "qc_verdict",
        "quarantined",
        "qc_exhausted",
    ]


def test_aggregate_builds_per_sku_dashboard_state():
    events = [
        {"event": "run_start", "ts": 1.0, "n_plans": 2, "skus": ["a", "b"], "cap_usd": 50.0},
        {"event": "attempt", "sku": "a", "attempt": 1, "name": "A", "spent_usd": 0.0},
        {
            "event": "qc_verdict",
            "sku": "a",
            "attempt": 1,
            "passed": False,
            "tags": ["branding_drift"],
            "reason": "logo wrong",
            "analysis": "saw a peony",
            "spent_usd": 0.45,
        },
        {"event": "quarantined", "sku": "a", "attempt": 1, "spent_usd": 0.45},
        {"event": "attempt", "sku": "a", "attempt": 2, "name": "A", "spent_usd": 0.45},
        {
            "event": "qc_verdict",
            "sku": "a",
            "attempt": 2,
            "passed": True,
            "tags": [],
            "reason": "pass",
            "spent_usd": 0.9,
        },
        {"event": "accepted", "sku": "a", "attempt": 2, "path": "/o/a.png", "spent_usd": 0.9},
        {"event": "attempt", "sku": "b", "attempt": 1, "name": "B", "spent_usd": 0.9},
    ]
    state = aggregate(events)
    by = {s["sku"]: s for s in state["skus"]}
    assert by["a"]["status"] == "accepted"
    assert len(by["a"]["attempts"]) == 2
    assert by["a"]["attempts"][0]["verdict"]["tags"] == ["branding_drift"]
    assert by["b"]["status"] == "rendering"
    assert state["run"]["spent_usd"] == 0.9
    assert state["run"]["progress"] == {"done": 1, "total": 2}
    assert state["run"]["finished"] is False


def test_snapshot_reports_missing_runs_dir_gracefully(tmp_path):
    state = snapshot(None, runs_dir=tmp_path)
    assert state["skus"] == []
    assert "no run-*.jsonl" in state["run"]["error"]


def test_snapshot_picks_newest_run(tmp_path):
    old = tmp_path / "run-1.jsonl"
    new = tmp_path / "run-2.jsonl"
    old.write_text(json.dumps({"event": "run_start", "skus": ["old"]}) + "\n", encoding="utf-8")
    new.write_text(json.dumps({"event": "run_start", "skus": ["new"]}) + "\n", encoding="utf-8")
    import os
    import time

    now = time.time()
    os.utime(old, (now - 100, now - 100))
    os.utime(new, (now, now))
    state = snapshot(None, runs_dir=tmp_path)
    assert state["file"] == "run-2.jsonl"
    assert state["skus"][0]["sku"] == "new"
