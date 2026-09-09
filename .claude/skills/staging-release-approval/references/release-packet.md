# Release packet and evidence receipt

Use these fields as a compact Markdown or JSON record. Fill them with measured
values; never copy historical hashes or counts as defaults. Use `UNVERIFIED`
with a reason when evidence is absent.

## Before approval

| Field            | Contents                                                                                                                                         |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Release identity | Release ID, creation time, worktree, branch, HEAD, accepted source digest and dirty-file scope                                                   |
| Target           | Staging hostname/site ID, connection identity, installation root, active theme/application                                                       |
| Archive          | Absolute path, SHA-256, byte size, root, runtime file count, manifest path/hash                                                                  |
| Change plan      | Added/replaced/unchanged/removed/excluded paths, counts, reasons                                                                                 |
| Baseline         | Capture time, complete or partial inventory scope, path/hash inventory and digest                                                                |
| Rollback         | Archive location/hash/size/count, baseline equality result, recovery-copy location, content/database backup if applicable                        |
| Procedure        | Path and SHA-256, installation steps, anticipated platform side effects, rollback triggers and commands, recovery verification, rehearsal status |
| Review           | Reviewer or self-review label, result, evidence paths/hashes, unresolved findings                                                                |
| Verification     | Routes, features, viewport/device/browser/network/cache profiles, thresholds, permitted interactions, stopping conditions                        |
| Authorization    | User approval text/reference and time; exact target/artifact/procedure/rollback binding; allowed actions and exclusions                          |
| Readiness        | Known failures, untested states, staging purpose, subsequent decisions outside scope                                                             |

Suggested approval request, after filling the packet: “Approve installing the
exact archive and procedure identified in this packet on the named staging site,
with its preserved rollback and listed verification scope?” Explain that
explicit approval is required for the staging mutation; do not ask again if
matching authorization already exists.

## After execution

Record pre-install drift result, start/end time, actual installed identity,
installer result and side effects, final identity recheck, rollback trigger
assessment, whether rollback was executed and its verification result. Retain
backups after success.

Report each evidence layer independently with status, scope/profile, artifact
path, and limits. For example, a filesystem match plus an empty-checkout
redirect can support filesystem `PASS` and bounded browser observations, while
populated checkout remains `UNVERIFIED` and performance remains `FAIL`.

Conclude with overall readiness, founder-review questions that remain, and
whether production changed. Do not claim “all approved work is deployed” unless
every claimed feature is represented in the installed manifest and relevant
content evidence.
