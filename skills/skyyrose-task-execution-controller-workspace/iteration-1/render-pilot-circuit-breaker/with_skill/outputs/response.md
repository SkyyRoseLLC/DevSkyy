# Controlled on-model render setup — br-008, br-009, br-010

## Release decision

**Authorized now:** one paid, front-view, on-model pilot for **br-008** only,
after the task record and preflight evidence below exist.

**Not authorized now:** br-009, br-010, any automatic retry, storefront use,
or promotion. They remain blocked until the br-008 pilot has passed visual
review and the `br-008-front-zero-fill-v1` incident is resolved by an
accountable reviewer. A provider success response is not a pass.

This order is intentional: the prior defect was specific and expensive. A
successful repair of br-008 proves that the zero-fill constraint can be
followed before spend is authorized for the other two distinct garments.

## Scope and source locks

Create three separate controlled tasks. Do not put the three SKUs in one
provider request or one task scope.

| Task | Scope | Source lock | Required visual contract |
| --- | --- | --- | --- |
| `br-008-front-on-model-pilot` | `sku=br-008`, `style=on-model`, `view=front`, `target=review-only` | `assets/products/references/br-008-design-front.jpeg`; football dossier; Black Rose cluster and NFL patch references | Red pullover football jersey; front `80`; rose artwork **only inside 8**; front `0` plain white with black outline; black-bordered white triple cuff stripes; 3 in x 4 in NFL patch at wearer's lower-left hem. |
| `br-009-front-on-model` | `sku=br-009`, `style=on-model`, `view=front`, `target=review-only` | `assets/products/references/br-009-founder-white-football-front-sot.png`; white-football patch contract; football dossier | White football jersey; front `32`; greyscale rose artwork only inside `3`; `2` plain white; black outline; black-and-white cuff bands; 3 in x 4 in NFL patch at wearer's lower-left hem. |
| `br-010-front-on-model` | `sku=br-010`, `style=on-model`, `view=front`, `target=review-only` | `assets/products/references/br-010-design-front.jpeg`; basketball dossier; Black Rose cluster and NBA patch references | Sleeveless white basketball tank; no numerals; centred `THE BAY` medallion; continuous tonal-greyscale rose fade from clean upper body to lower hem; 2 in x 2.5 in NBA patch at wearer's lower-left hem. |

Before the first provider call, calculate and attach a SHA-256 fingerprint for
every source file and the dossier/correction record used by that SKU. The
generation request must use these source paths, not a paraphrased product
description alone.

## First task record and incident

Start the br-008 pilot before submitting a provider request:

```bash
python scripts/task_control.py start \
  --kind product_render \
  --scope-json '{"sku":"br-008","style":"on-model","view":"front","target":"review-only","candidate_limit":1,"automatic_retries":0}' \
  --require source_fingerprint \
  --require prompt_hash \
  --require provider_receipt \
  --require candidate_sha256 \
  --require visual_qa \
  --require rejection_checklist
```

Record the known first-seen issue on that task before rendering, so it is a
durable constraint rather than a correction remembered only in a prompt:

```bash
python scripts/task_control.py issue --task-id TASK_ID \
  --category product_fidelity \
  --fingerprint br-008-front-zero-fill-v1 \
  --summary 'Front red football on-model candidate placed Black Rose artwork inside the plain front zero; the official design permits artwork only inside the front eight.'
```

Record the scope-specific source hashes and masked-digit preflight before the
provider call. The controller must reject the request if the `8` artwork and
the plain `0` are not independently bound by the source/mask contract.

## Pilot generation guardrails

1. Submit exactly **one** br-008 candidate. The maximum spend authorization is
   one candidate plus its applicable review cost. Automatic retry count is
   zero.
2. Use the protected masked-edit/source-bound render path. The digit `0` is a
   protected plain region: rose, vine, cloud, greyscale texture, or other
   pattern pixels inside it are a failure.
3. Keep the source image, reference pack, prompt hash, model/version, request
   ID, timestamp, candidate SHA-256, and estimated cost in the task ledger.
   Do not log API keys, raw provider payloads, or image bytes.
4. Save the candidate to review-only storage. It cannot become a product card,
   WooCommerce image, staging asset, or catalog manifest entry from this task.
5. If the provider fails, the candidate is unjudged, or visual QA detects any
   mismatch, record the incident immediately using a stable fingerprint. The
   controller blocks the same scope and all remaining batch work; it does not
   spend again.

## Required br-008 visual QA

The named reviewer must inspect the candidate at full resolution against the
front technical source and attach a signed outcome:

- `front_8_fill`: greyscale three-rose artwork is entirely inside the `8`.
- `front_0_plain`: the `0` is white with a black athletic outline and contains
  no rose, vines, cloud, tonal texture, or overflow from the `8`.
- `front_number_layout`: exactly `80`, with clean block-digit proportion and
  no duplicated or missing character.
- `cuff_construction`: each red sleeve has three white stripes, with black
  border/separator stripes rather than unframed white bands.
- `patch_scale_and_placement`: NFL patch visually reads as the 3 in x 4 in
  standard at the wearer's lower-left hem, not enlarged, moved, or replaced.
- `garment_and_pose`: authentic short-sleeve football silhouette, straight
  front pose, no placket/buttons and no unintended sleeve numerals.
- `source_match`: no invented graphics, color changes, or placement changes
  outside the on-model context.

The following outcome is a hard failure: any artwork inside the front `0`.
Log it with `br-008-front-zero-fill-v1`, attach the candidate hash and
provider receipt, quarantine the file, and leave the issue unresolved. There
is no paid retry until a reviewer records the exact correction and authorizes
a new one-candidate pilot.

## Unlock sequence

1. **br-008 only:** reviewer resolves the zero-fill incident only after the
   candidate passes every named check, and then approves this pilot in the
   task ledger.
2. **br-009:** start its own task, attach its source fingerprints, and run one
   front-view pilot. Its `3` filled / `2` plain rule is independently checked;
   br-008 approval is not evidence for br-009.
3. **br-010:** start its own task, attach its source fingerprints, and run one
   front-view pilot. Its sleeveless/no-numerals/rose-fade contract is checked
   independently.
4. Only after each SKU has a passed review-only candidate can a separate
   promotion or staging task be opened. That later task needs its own runtime
   mapping and visual evidence; it is not implied by the render task.

## Handoff state

**Outcome:** ready for a single, source-bound br-008 pilot; br-009 and br-010
are deliberately held.

**Evidence required before the next paid call:** task ID, source fingerprints,
masked-digit preflight, prompt hash, candidate limit of one, and the recorded
first-seen br-008 zero-fill issue.

**Open issue:** `br-008-front-zero-fill-v1` — rose artwork appeared inside the
plain front zero. It remains unresolved until visual QA confirms the plain
zero in a new candidate.

**Release decision:** review-only br-008 front pilot after the above record is
created. No multi-SKU request, no automated paid retry, and no storefront or
staging attachment are authorized by this setup.
