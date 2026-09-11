# V2 staging release — founder approval packet

**READY_FOR_FOUNDER_APPROVAL. Overall engineering readiness remains NEEDS_MORE_WORK.**

Steps 1–2 are prepared and independently reviewed. Step 3 requires founder authorization. No staging installation, configuration change, cache purge, order or payment occurred in this preparation phase.

## Why staging differs

Staging contains mixed historical source. Comparing all 608 accepted theme source files with 447 deployed files found 389 identical, 58 different and 161 missing, with no remote-only files. Both identify as version 2.4.4, so version is insufficient evidence.

Of the earlier selective card release's 46 files, 43 still match; functions.php, theme.css and theme.min.css differ. Twenty of the original 27 differing files match older revision 5de8e2f3eb40827a052996f72bfb290a95bd600a. Seven generated/metadata file histories and the intervening deployment actor/time remain unresolved. This establishes an on-disk source mismatch; cache behavior cannot explain missing PHP/media files. A current packaging classification gap was also found and reconciled in the isolated export, but is not proven to have caused the historical mismatch.

## Exact candidate

- Target: https://staging-7e48-skyyrose.wpcomstaging.com
- Artifact: skyyrose-flagship-2.zip
- Runtime files: 527
- Archive bytes: 170,023,102
- SHA256: e47588205c55e6303588a207d7ee766ad175b82f606d167198f4ec2c25b1d80d
- Isolated source tree: 5dd1902d8075e9c98bafaf5f80e3a88f2792a3de
- Accepted readiness digest: 4a23629d6b17c1d0393000cfbbe982eedf47712837dd772a66c993ad7e46bbe7

Two package builds produced identical archives. The canonical build and verification passed, including 95 JavaScript checks. Supplementary source tests passed 7/7 after correcting an export-only stale source-line expectation; the original failure is retained. No accepted runtime source bytes changed. All 608 source files remain unchanged against the phase's initial inventory.

Independent review verified every ZIP member against accepted source, all five animated hero surfaces, all nine scenes, 33 approved card originals and 99 responsive derivatives, and inclusion of native Quick View and Ask Skyy dependencies. These are artifact proofs; deployed visual and runtime parity still require step 5.

## Exact installation delta and recovery

| File action | Count |
|---|---:|
| Unchanged runtime files | 319 |
| Replaced files | 48 |
| Added runtime files | 160 |
| Existing non-runtime files retired from active theme | 80 |

The 80 retired paths are packaging exclusions: authoring, tooling, QA and superseded intermediate assets. Independent review verified final scene output is identical without them. They remain preserved in source and rollback evidence. This is one whole-theme archive installation, not selective file copying.

A read-only capture preserved the entire current staging theme: 447 files, 262,563,840-byte TAR, SHA256 1b8164b36374aadea85aa4ec7c4d3f8a18e9f8e9f34042f706bd21c090a45ac3. Every captured file matches the remote inventory. The reviewed plan prepares and verifies recovery before installation, then requires exact 527-file parity. Severe installation/feature regressions trigger whole-tree restoration and verification against all 447 original hashes. Remote identity must be rechecked immediately before installation; drift stops deployment.

Plan SHA256: ad4926a97cfea2a195de6366a4aaa61aa8b65b12f0a7dc685a69ef2c81231911.

The installation and recovery procedures have been reviewed and syntax-checked, but not executed or rehearsed remotely. Brief staging unavailability is possible. Managed cache or optimized HTML parity must be checked after installation; filesystem parity alone does not establish browser parity.

## Requested authorization and next gates

Authorize this exact ZIP for this staging target, following deployment-and-rollback.md, then verify the five hero surfaces (Home, Signature, Black Rose, Love Hurts, Kids Capsule), nine scenes, cards, native Quick View and Ask Skyy before repeating actual WordPress.com delivery/performance measurements. Shop is verified as a native archive, not an animated hero.

No production deployment, manual cache purge, platform configuration change, orders/payments, automatic optimization, model editing, content promotion or Town Line work is included. The later roadmap remains ordered and separately gated. Known performance, content and physical-device certification gaps remain open.
