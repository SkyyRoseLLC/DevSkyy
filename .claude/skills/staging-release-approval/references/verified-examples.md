# Verified examples: staging release approval

Coverage: **PARTIAL**. Historical source provenance was verified on September
8, 2026. No fresh authenticated staging execution, installation, or restore was
performed. These examples improve decision guidance; they do not certify current
staging state or close the audit's recovery/concurrency/coverage gaps by
themselves.

[Evidence receipts](example-evidence.json) contain exact source paths, line
numbers, timestamps, event IDs, source/line SHA-256 values, and checked
excerpts. Hashes bind inspected bytes; they are not signatures authenticating
the server or every claim displayed in the recording. Authentication for remote
execution remains `UNVERIFIED` in these receipts.

The example requests below are illustrative paraphrases, not quotations of user
authorization. The supporting observations are historical. All “what not to do”
cases are `ILLUSTRATIVE_UNEXECUTED`; none was deliberately performed.

## 1. Prepare approval without overstating recovery proof

**Example request:** “Prepare this accepted candidate for staging approval.”

**Context:** The package and backup are prepared. Installation and rollback have
not run. The recorded environment was SkyyRose staging; no current
account/session is verified here.

**What to do:** Present the specific ZIP, procedure and rollback hashes,
captured baseline, file changes and known failures. Report backup byte
verification separately from restore rehearsal. Request approval only after the
packet is concrete, unless matching approval already exists.

**What not to do:** Say “rollback tested successfully” because the archive hash
matches, or install because the preparation review passed. Correct those claims
to “backup verified; remote restore not exercised” and preserve the execution
approval boundary.

**Observed evidence:** `BACKUP-01`, `HISTORICAL_OBSERVED`, September 6 at
21:03:02 UTC: the UI reported 447 files verified against staging hashes and
explicitly said installation/recovery had not been executed remotely.

**Expected decision:** Ready for the applicable approval decision, not evidence
of a successful restore or completed deployment. The evidence establishes what
was recorded, not an independent replay of the backup operation.

**Refresh trigger:** Any new archive, procedure, baseline, target, permissions,
or recovery mechanism requires new applicable evidence. Do not reuse the
historical count as a future acceptance threshold.

## 2. Report bounded browser parity without claiming performance certification

**Example request:** “Does staging match the approved cinematic and commerce
candidate?”

**Context:** An installed-package browser report exists, with declared routes
and viewports. Current authenticated staging access is `UNVERIFIED` here; public
browser visibility alone cannot establish it.

**What to do:** Report the browser results under their actual
route/state/profile bounds; retain the separate filesystem identity evidence and
unresolved performance or checkout gates. Identify every required but untested
state rather than hiding it inside a general PASS.

**What not to do:** Turn a successful 390-pixel emulated browser check into
“physical-device performance certified,” or infer populated checkout/payment
success from an empty-cart redirect. Correct the report by marking those
untested claims `UNVERIFIED` and listing the evidence needed.

**Observed evidence:** `BROWSER-01`, `HISTORICAL_OBSERVED`, September 6 at
23:37:22 UTC: the displayed report recorded 18 final route/viewport observations
and 10 fallback cases, Chromium at 1440×900 and 390×900, touch emulation, and no
performance throttling.

**Expected decision:** Bounded historical browser evidence can be reported at
that scope. It does not supply fresh deployed performance, authenticated-system
proof, or complete feature coverage.

**Refresh trigger:** New artifact/content, changed feature contract,
browser/profile change, or fresh deployment claim. Required states omitted from
the evidence remain unverified even if sampled states passed.

## 3. Preserve rollback when the reviewed trigger did not occur

**Example request:** “Staging is installed but performance still fails. What
happens next?”

**Context:** A reviewed procedure defines recovery triggers. Existing
performance failures must be distinguished from new installation-caused
regressions. Historical approval is not current permission to restore.

**What to do:** Compare the observed condition to the approved recovery
triggers. If a trigger is met under current authorization, execute that
procedure and verify recovery; otherwise preserve the backup, report the
remaining failures and return the staging result for founder review.

**What not to do:** Restore automatically because any performance row is red,
silently purge cache to make a report pass, or delete the backup after
installation. Correct this by applying the reviewed trigger and recording the
actual side effects and remaining evidence gaps.

**Observed evidence:** `ROLLBACK-01`, `HISTORICAL_OBSERVED`, September 6 at
23:45:01 UTC: the report said rollback remained verified against the 447-file
baseline and was not executed because the severe-regression trigger was not
established. This proves the recorded disposition, not a fresh verification of
recovery viability.

**Expected decision:** Retain recovery material and report performance failure
independently. Authentication/current deployment state remains `UNVERIFIED` in
this example.

**Refresh trigger:** Any current regression, changed recovery
trigger/authorization, lost backup, or changed remote baseline requires fresh
inspection.

## 4. Refuse to manufacture authenticated examples

**Example request:** “Add a verified authenticated staging deployment example to
this skill.”

**Classification:** `ILLUSTRATIVE_UNEXECUTED`, derived from this skill's
authorization and evidence boundaries and the user's verified-examples
requirement. Authentication: `UNVERIFIED` until a relevant authenticated result
exists.

**What to do:** Use existing, inspectable authenticated receipts if available
and preserve their original date/scope. If only local event recordings exist,
label them `HISTORICAL_OBSERVED`, identify the missing authenticated-server
evidence, and retain `PARTIAL` coverage. A safe local validation example may be
`REPRODUCED_LOCAL` with authentication `NOT_APPLICABLE`.

**What not to do:** Generate a plausible SSH transcript, treat possessing a key
as authentication proof, copy an old authorization into the current packet, or
deploy just to create evidence. Correct this by keeping the missing evidence
explicit and limiting actions to existing authorization.

**Expected decision:** Truthful example coverage, with source verification,
authentication and execution independently labeled. No observed live outcome is
claimed for this illustrative case.
