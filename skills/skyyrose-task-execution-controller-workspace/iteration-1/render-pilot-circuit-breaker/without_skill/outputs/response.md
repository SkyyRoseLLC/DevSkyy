# Controlled on-model render plan: BR-008, BR-009, BR-010

Do **not** submit a three-SKU render batch. Treat BR-008 as a single paid pilot and keep BR-009 and BR-010 locked until BR-008 passes source-fidelity review. The prior BR-008 failure—rose artwork appearing inside the plain `0`—is a hard failure, not a cosmetic preference.

## 1. Freeze the product authority before any provider call

Create a per-SKU render brief from the authoritative dossier and a versioned source packet. Each packet must include:

- SKU, intended view (`on-model/front`), and a unique task ID.
- Exact technical flat or approved product reference for that SKU.
- A reference manifest containing path, SHA-256, dimensions, and role for every input image.
- A plain-language **must match** list and **must not change** list.
- A masked editable region when any element is allowed to change; everything outside the mask is immutable.
- A visual acceptance sheet with numbered checks that a reviewer can compare against the output.

For BR-008 front, the acceptance sheet must state:

- The front reads `80`.
- The `8` is the only filled/rose-artwork digit.
- The `0` remains plain white with a black outline and contains no rose artwork.
- The red jersey sleeve construction preserves black-and-white stripe treatment.
- The front patch is visually consistent with the 3 in × 4 in jersey-patch contract.

Do not use a collage, an unrelated colorway, or an approximate garment as a source reference. If the BR-008 source packet cannot prove these details, stop before spending and request the corrected reference.

## 2. Produce exactly one BR-008 pilot candidate

Set the provider job to one image, one SKU, one front view, with automatic retries disabled. Record the provider request ID, input-source hashes, prompt hash, model/version, timestamp, and estimated cost before and after the call.

The prompt should describe only the permitted scene transformation—an on-model front presentation—and restate the protected construction details. It must explicitly prohibit changes to artwork placement, numbers, sleeve stripes, lettering, patch footprint, colorway, and garment construction.

Do not schedule BR-009 or BR-010 as follow-on jobs. A human must approve the BR-008 pilot from the saved evidence first.

## 3. Gate the pilot with deterministic checks plus visual review

Run these checks before the candidate can be marked acceptable:

| Gate | BR-008 pass condition | Failure action |
| --- | --- | --- |
| Source binding | Candidate receipt matches the frozen source-packet hashes. | Quarantine candidate; stop scope. |
| Protected-layout review | The `8` is filled; the `0` is plain; no rose artwork crosses into the `0`. | Quarantine candidate; open a placement incident. |
| Color/construction review | Red fabric, black/white sleeve-stripe construction, neck, numbers, and artwork agree with the technical flat. | Quarantine candidate; open a fidelity incident. |
| Patch review | Patch has consistent 3 in × 4 in visual proportion and approved placement. | Quarantine candidate; open a patch-contract incident. |
| On-model integrity | Garment is naturally worn without altered product geometry or hidden required details. | Quarantine candidate; open an integration incident. |

The reviewer should receive a side-by-side source/candidate board and a marked overlay for the digits, sleeves, and patch. A pass must be explicit; silence or a provider-success response is not approval.

## 4. Apply a circuit breaker to all three SKUs

The circuit breaker must enforce this state progression:

```text
BR-008 source packet frozen
        ↓
one BR-008 paid pilot
        ↓
visual pass? ── no → quarantine + incident + stop all jobs
        │
       yes
        ↓
explicit reviewer approval
        ↓
one BR-009 paid pilot
        ↓
repeat the same review before BR-010
```

An incident blocks another run for the same SKU/view/style/source fingerprint until it is resolved with: the observed failure, root cause, changed control or source, and approval to retry. Never perform an automatic paid retry after a failure. A resolved BR-008 issue does not automatically approve BR-009 or BR-010; each garment remains its own pilot.

## 5. Keep a durable task and incident record

For every task—not only render failures—append an immutable event record to a release ledger. At minimum record:

```json
{
  "task_id": "render-br-008-on-model-front-<timestamp>",
  "sku": "br-008",
  "view": "on-model/front",
  "state": "planned | submitted | candidate_received | approved | quarantined | blocked",
  "source_fingerprint": "sha256:...",
  "provider_request_id": "...",
  "model": "gpt-image-2",
  "estimated_spend_usd": 0.0,
  "acceptance_checks": ["digit-placement", "sleeve-stripes", "patch-contract"],
  "reviewer_decision": "pending | pass | fail",
  "incident_id": "optional",
  "timestamp": "ISO-8601"
}
```

On the first mismatch, create a linked incident record immediately. Include the exact mismatch (for example, `rose_artwork_inside_plain_zero`), source fingerprint, candidate path/hash, reviewer evidence, provider request ID, cost, and the blocking scope. That gives subsequent work a machine-readable reason to stop before it repeats the same spend.

## Ready-to-run condition

Only submit BR-008 when its source packet, mask, acceptance sheet, estimated budget, and reviewer are all present. BR-009 and BR-010 remain blocked until their own source packets exist and the preceding single-SKU pilot has an explicit visual approval.
