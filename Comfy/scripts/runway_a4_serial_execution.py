"""Execute the founder-approved A4 batch once, preserving every paid attempt."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

from Comfy.scripts.environment_plate_ooda import build_prompt_only_workflow
from skyyrose.integrations.comfy_client import (
    ComfyClient,
    _expected_environment_workflow,
    _load_runway_approval,
    runway_execution_fingerprint,
)

BASE_URL = "http://127.0.0.1:8189"
WORKTREES = Path("/Users/theceo/.codex/worktrees")
S6 = WORKTREES / "stage-06-lh-commerce-2/DevSkyy"
S7 = WORKTREES / "stage-07-br-signature-rollout/DevSkyy"
RUNWAY_NODE = Path(
    "/Users/theceo/ComfyUI-Installs/ComfyUI/ComfyUI/comfy_api_nodes/nodes_runway.py"
)
NODE_SHA = "7999cd6205e46b17f140dfe5e06ae9824a32cf62251bfb2f78ae196f52841da1"
JOBS = (
    (S6, "lh-commerce-2", "bc484ef2e6c63174030058b9a8099ad5bdc05bda06e69329383fcbca4c003d14"),
    (S7, "br-commerce-2", "6b090ea5457fcfe106270fe0f5a06b8794bb6d90a4e4238a000011c4dd48eb46"),
    (S7, "sig-commerce-1", "f76233a63d38d552993444d08d03af036cd72c1d24dd5bf83b6ff7ab4ec1a589"),
    (S7, "sig-commerce-2", "17493b37b88c9f624119a664390c3ff2fca2a66ff5c60ac0fe6172f6e082cee0"),
)
LIMITS = dict.fromkeys(
    ("protected_composition", "promotion", "storefront_wiring", "publication", "deployment"),
    False,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def history_status(prompt_id):
    if prompt_id is None:
        return None
    try:
        with urllib.request.urlopen(f"{BASE_URL}/history/{prompt_id}", timeout=10) as response:
            record = json.load(response).get(prompt_id, {})
        status = record.get("status", {})
        errors = []
        for kind, detail in status.get("messages", []):
            if kind == "execution_error":
                errors.append({key: detail.get(key) for key in (
                    "node_id", "node_type", "exception_message", "exception_type", "executed"
                )})
        return {"status_str": status.get("status_str"), "completed": status.get("completed"),
                "errors": errors}
    except Exception:
        return None


def account_balance():
    request = urllib.request.Request(
        "https://api.comfy.org/customers/balance",
        headers={"X-API-KEY": os.environ["COMFY_API_KEY"], "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        body = json.load(response)
    # Preserve provider field names; their unit is not inferred from a UI balance.
    return {key: body.get(key) for key in (
        "amount_micros", "effective_balance_micros", "pending_charges_micros", "currency"
    )}


async def main(execute):
    if digest(RUNWAY_NODE.read_bytes()) != NODE_SHA:
        raise RuntimeError("Installed Runway retry guard changed")
    prepared = []
    results = []
    async with ComfyClient(base_url=BASE_URL) as client:
        await client.inspect_runtime()
        await client._verify_partner_authentication()
        for root, slug, expected_fingerprint in JOBS:
            contract_path = root / f"Comfy/scene-contracts/{slug}-runway-environment-a4.json"
            contract = json.loads(contract_path.read_bytes())
            approval_path = root / contract["output"]["approval_path"]
            marker = root / contract["output"]["attempt_marker"]
            candidate = root / contract["output"]["candidate_path"]
            receipt_path = root / f"Comfy/receipts/{slug}-runway-environment-a4-execution.json"
            submission_path = root / f"Comfy/receipts/{slug}-runway-environment-a4-submission.json"
            if any(path.exists() for path in (marker, candidate, receipt_path, submission_path)):
                raise RuntimeError(f"{slug}: existing A4 execution evidence prohibits replay")
            payload = build_prompt_only_workflow(contract)
            if payload != _expected_environment_workflow(contract):
                raise RuntimeError(f"{slug}: builder and validator disagree")
            workflow = await client.validate_workflow(payload, expected_output_node_ids=("3",))
            approval, approval_hash = _load_runway_approval(
                contract_path=contract_path, approval_path=approval_path,
                attempt_marker=marker, workflow=workflow,
            )
            fingerprint = runway_execution_fingerprint(contract, workflow=workflow)
            if fingerprint != expected_fingerprint or workflow.paid_price_usd != 0.11:
                raise RuntimeError(f"{slug}: authorized fingerprint or price changed")
            prepared.append((root, slug, contract_path, contract, approval_path, marker,
                             candidate, receipt_path, submission_path, workflow, approval_hash))
            print(f"APPROVAL_GATE_PASS {contract['candidate_id']} {fingerprint}", flush=True)
        if not execute:
            print("PASS: all four full approval gates; zero prompt submissions", flush=True)
            return 0

        balance_before = await asyncio.to_thread(account_balance)
        for index, job in enumerate(prepared):
            if index:
                await asyncio.sleep(5)
            (root, slug, contract_path, contract, approval_path, marker, candidate,
             receipt_path, submission_path, workflow, approval_hash) = job
            if digest(RUNWAY_NODE.read_bytes()) != NODE_SHA:
                raise RuntimeError("Installed Runway retry guard changed before submission")
            prompt_id = None
            receipt = {
                "schema": "skyyrose.runway-environment-execution/1",
                "scene_id": contract["scene_id"], "candidate_id": contract["candidate_id"],
                "execution_fingerprint": JOBS[index][2], "recorded_at": now(),
                "serial_index": index + 1, "approval_receipt_path": str(approval_path),
                "approval_receipt_sha256": approval_hash, "attempt_marker_path": str(marker),
                "workflow_sha256": workflow.workflow_sha256,
                "live_node_schema_sha256": workflow.live_schema_sha256,
                "paid_price_usd": workflow.paid_price_usd, "automatic_retry": False,
                "authority_limits": LIMITS, "provider_inputs": {
                    "mode": "prompt_only", "reference_image": None, "protected_assets_sent": []
                },
            }
            try:
                submission = await client.submit_prompt(
                    workflow, paid_contract_path=contract_path,
                    paid_approval_receipt=approval_path, paid_attempt_marker=marker,
                )
                prompt_id = submission.prompt_id
                write_new(submission_path, {**receipt, "prompt_id": prompt_id,
                                           "status": "SUBMITTED", "client_id": submission.client_id})
                print(f"SUBMITTED {contract['candidate_id']} {prompt_id}", flush=True)
                execution = await client.wait_for_outputs(submission)
                if len(execution.outputs) != 1:
                    raise RuntimeError("Expected exactly one output")
                output = execution.outputs[0]
                payload = await client.download_output(execution, output)
                candidate.parent.mkdir(parents=True, exist_ok=True)
                with candidate.open("xb") as stream:
                    stream.write(payload)
                with Image.open(BytesIO(payload)) as image:
                    dimensions = image.size
                receipt.update({
                    "status": "CANDIDATE_CREATED_QUARANTINED", "history_sha256": execution.history_sha256,
                    "comfy_status": execution.status, "provider_output": {
                        "node_id": output.node_id, "filename": output.filename,
                        "subfolder": output.subfolder, "type": output.type,
                    },
                    "candidate_output": {"created": True, "path": str(candidate),
                        "sha256": digest(payload), "bytes": len(payload),
                        "width": dimensions[0], "height": dimensions[1],
                        "authority": "QUARANTINED_CANDIDATE_ONLY"},
                })
                if dimensions != (2048, 1152):
                    raise RuntimeError(f"Output dimensions mismatch: {dimensions}")
                print(f"SAVED {contract['candidate_id']} {digest(payload)}", flush=True)
            except Exception as exc:
                receipt.update({
                    "status": "STOPPED_NO_RETRY", "error": f"{type(exc).__name__}: {exc}",
                    "provider_status": await asyncio.to_thread(history_status, prompt_id),
                    "candidate_file_exists": candidate.exists(),
                })
                print(f"BATCH_STOP {contract['candidate_id']} {receipt['error']}", flush=True)
            receipt["prompt_id"] = prompt_id
            receipt["attempt_marker_sha256"] = digest(marker.read_bytes()) if marker.exists() else None
            write_new(receipt_path, receipt)
            results.append(receipt)
            if receipt["status"] != "CANDIDATE_CREATED_QUARANTINED":
                break

        try:
            balance_after = await asyncio.to_thread(account_balance)
        except Exception:
            balance_after = None
        for root, stage in ((S6, "06"), (S7, "07")):
            write_new(root / f"Comfy/receipts/stage-{stage}-runway-environment-a4-batch-execution.json", {
                "schema": "skyyrose.runway-environment-serial-batch-execution/1",
                "recorded_at": now(), "results": results,
                "unsubmitted_candidates": [item[3]["candidate_id"] for item in prepared[len(results):]],
                "maximum_combined_price_usd": 0.44, "automatic_retry": False,
                "max_concurrent_provider_tasks": 1, "stop_on_any_failure": True,
                "balance_before_raw": balance_before, "balance_after_raw": balance_after,
                "billing_note": "Raw account observations; per-prompt charges not reconciled.",
                "authority_limits": LIMITS,
            })
    print(json.dumps({"completed": sum(r["status"] == "CANDIDATE_CREATED_QUARANTINED" for r in results),
                      "processed": len(results), "authorized": 4}), flush=True)
    return 0 if len(results) == 4 and all(r["status"] == "CANDIDATE_CREATED_QUARANTINED" for r in results) else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Consume the four existing A4 approvals")
    raise SystemExit(asyncio.run(main(parser.parse_args().execute)))
