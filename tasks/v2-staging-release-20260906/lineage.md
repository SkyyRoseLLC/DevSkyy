# Staging release lineage

**The observed staging theme is a mixed historical source state, not the accepted readiness candidate hidden behind caching.** A recorded selective card deployment still explains 43 of its 46 patched files. Twenty of the 27 differences in the original comparison exactly match an older source revision. The accepted later shell, commerce, scene and Skyy completion work is not present as a complete source set. The exact intervening deployment actor/time and a single complete deployed Git commit are not established.

This is a local, read-only lineage audit using root-captured remote evidence. No SSH, source edit, build or deployment was performed by this owner. `lineage.json` provides per-path hashes and matches; `.artifacts/v2-staging-release-20260906/lineage/evidence-hashes.json` binds every primary receipt used below.

## Inventory scopes must remain separate

| Comparison scope | Accepted/requested paths | Present on staging | Exact matches | Different | Missing |
|---|---:|---:|---:|---:|---:|
| Original B13 source/protected-media subset |451|322|295|27|129|
| Existing packaging classification |588|447|389|58|141|
| Full accepted theme from package source inventory |608|447|389|58|161|

All 451 original comparison rows are unchanged against the fresh full remote inventory. The complete remote snapshot has 447 regular files and no symlinks, per root's capture; there are zero remote-only paths when compared with the 608 accepted theme paths. The original 451-path digest `f2893d998356ac5c86fda86e6422adab348261f02bb00fc935b88e74c1d3ae5a` is an observed subset map, not a remote Git commit or a full theme digest. Accepted readiness digest remains `4a23629d6b17c1d0393000cfbbe982eedf47712837dd772a66c993ad7e46bbe7`.

`full-accepted-comparison.json` independently joins the package owner's source-inventory hashes to `remote-full-hashes.stdout`. The broader comparison exposes additional gaps absent from the original subset, including missing controls/global-shell CSS, and different front-page.php, footer.php, page.php and template-collection.php. Those older page/template surfaces materially constrain which accepted features can be assumed active; missing paths alone do not prove the browser requests them or currently returns404 for them.

## Recorded selective release that still matches

The primary receipt is `/Users/theceo/.codex/worktrees/3f2d/DevSkyy/.artifacts/v2-staging-card-deploy-20260905/deployment-receipt.json`. It records `DEPLOYED_VERIFIED_STAGING` at **2026-09-05T11:33:39.433382+00:00**, a 46-file card patch, candidate manifest SHA256 `12c1d69eea3bf804b0afc7455f540eb3d236dafdb0cfae57e0c8e61cfbe84aad`, 33 cards and staging-only verification. Its explicit source strategy merged approved local card components into then-current staging files while preserving newer hero/film/scene work. `candidate.sha256`, `deploy.log` and the append-only task ledger corroborate the receipt.

The fresh remote inventory still matches **43/46** candidate hashes. Three have subsequently different bytes: `assets/css/theme.css`, `assets/css/theme.min.css`, and `functions.php`. The staged `template-parts/commerce/product-card.php` still exactly matches that card-release candidate. Thus the historic selective patch is directly supported, but the staging theme cannot be described as an untouched copy of that 46-file release. No human identity is inferred from a Git author or the ledger's abstract manager name. The receipt timestamp belongs to that recorded patch; it is not assigned to subsequent modifications.

## Older application-source lineage

All 20 Git-resolved paths among the original 27 differences match revision **5de8e2f3eb40827a052996f72bfb290a95bd600a**, `chore(tokens): reconcile editor and frontend token contract`, whose recorded commit timestamp is **2026-09-05T08:32:19-07:00**. Examples include `functions.php`, both design-token files, `theme.json`, scene controller source, mascot CSS/controller/loader/renderer source, `inc/performance.php`, hero/card/PDP/QV/search templates and the mascot template. Many files are unchanged across several revisions, so a per-file match is not proof that the whole revision was deployed.

The bounded search examined 15 theme-affecting revisions. Seven paths had no exact match in that search: collection-scene-motion.min.css, collection-scene-motion.min.js, mascot-loader.min.js, skyy-3d.min.js, the translation POT, package.json and scripts/test-performance.php. They remain unresolved generated/metadata lineage. Different minified bytes may reflect a different build, but neither semantic equivalence nor a particular build toolchain is asserted without proof.

The later local history records the new shell (69249828b), responsive editorial cards (55807b574), Shop (03947e829), PDP/collection worlds (67b2cd856), Living Archive Home (fec4e9339), cinematic integration (a0af07be7) and responsive commerce/Skyy completion (aabd2bffd). Those are development provenance, not deployment receipts. Current staging source hashes align with earlier behavior rather than proving those later full surfaces were deployed. Readiness subsequently records its accepted digest at 2026-09-06T18:01:55.722403+00:00; local acceptance is not a staging-upload receipt.

As a further boundary check, the local Phase3 release manifest (`tasks/v2-phase3-20260905/release-manifest.json`) matches 353 of 379 recorded files on the fresh staging tree; 21 differ and 5 are missing. That manifest explicitly says `deployment_authorized=false` and `source_tree=null`. Its overlap demonstrates shared ancestry and cannot be converted into evidence that its entire artifact was deployed.

## The original 27 differences by component

| Component group | Different paths | Consequence bounded by source evidence |
|---|---:|---|
| Scene scheduling and composed hero |5|Four scene-motion source/build files plus composed-hero template differ; accepted lifecycle/scheduling fixes cannot be assumed deployed.|
| Typography/token contract |3|Two design-token files and theme.json differ; accepted fallback/typography delivery cannot be inferred from version2.4.4 alone.|
| Skyy |9|CSS, loader, controller, renderer pairs and mascot template differ; accepted intent gate, continuity and runtime scheduling are not the staged implementation.|
| Commerce, search and shared application policy |7|theme.js, functions.php, performance policy, card/PDP/QV/search templates differ; accepted shell and native-commerce enhancements require reconciliation.|
| Build, translation and test metadata |3|POT, package and test file differ; these do not by themselves establish a browser defect or exact deployed build.|

The complete27-path list, accepted/deployed hashes, local receipt matches and every matching Git revision are in `lineage.json` and `lineage/matches.json`. Runtime consequences remain separate from source identity. For example, B13 independently observed the older staged Skyy auto-start lifecycle; this audit does not infer it solely from a missing new file.

## The original 129 missing paths

**99** are responsive card-front derivatives:33 SKU stems at320/480/768 widths. **Two** are approved scene poster derivatives (`br-commerce-3-poster-1024w.webp` and `br-commerce-3-poster-640w.webp`). These are absent derivative paths, not proof that staging lacks all product photographs or that new imagery is authorized.

The remaining **28** are accepted feature/source files:

- Premium commerce: four CSS/JS source/build files.
- Native Quick View commerce: four CSS/JS files plus the PHP module.
- Scene handoff: four CSS/JS files.
- Search preview: four CSS/JS files.
- Shop stylesheet source/build pair.
- Visual recovery JavaScript source/build pair.
- Inter fallback font and its provenance record.
- Skyy runtime poster.
- Global-shell PHP module.
- Collection world and scroll-world templates.
- Town Line commerce template.

These counts sum exactly 129. Every path is enumerated in `lineage.json`. “Missing” means absent from this filesystem snapshot; no evidence here proves whether a path was never uploaded or was later removed. Older active templates can use different existing assets and code, so absence is not automatically an active404 or total feature absence.

## Current packaging gap is real, but its historical cause is unproven

The package owner found 608 accepted theme files but only 588 pre-existing classification entries. The 20 unclassified paths are currently missing remotely and include premium/search/handoff/Quick View assets, the Quick View PHP module, the font subset, Skyy poster and font-provenance JSON. This explains the 20-count difference between the 588-path and 608-path comparisons.

The authorized artifact-local overlay now classifies 19 as runtime and excludes the one font-provenance record from runtime packaging. `package/packaging-overlay.json` records the exact paths and reasons. This is a concrete present packaging completeness issue that must be repaired for the proposed release. **It is not proof that the old deployment omitted these files because of that classification gap**: many additions postdate the documented selective card release, and no historical upload log tying this classifier to those absences was found. This owner did not modify the overlay or accepted source.

## HTTP optimized representations are a different layer

Filesystem `sha256sum` differences and missing PHP/font/image files are source inventory facts. Jetpack Boost critical CSS, combined `/_jb_static/??...` resources and negotiated Brotli/gzip are served representations. They can affect which CSS/JS bytes a browser sees without changing theme source files. They cannot explain away the 27 different or129 missing filesystem paths.

B13 specifically verified that its browser Signature CSS `??a6acf50b00` decoded to 452031 bytes / SHA256 `80875d248f2653177fc53977ea8bca01a6407044373b5752f54440dc6de51bf5`, and browser PDP JS `??5a8acea822` decoded to 30679 bytes / SHA256 `3b271f738f88447bbdf6571fe788cc0070afc3fb64ae0b9044fbce2b448a6102`. Those current Brotli bodies matched the corresponding identity/gzip bodies. Earlier bad Brotli URLs were different (`??c9711b8a62`, `??955c29654e`); their stale representation defects cannot be assigned as the cause of the later browser observations. These assertions come from the recorded B13 report/representation comparison, not new network requests by this audit.

## Confidence and remaining unknowns

High confidence: inventory counts, no drift across the original 451 paths, 43 persistent card-release hashes, 20 exact older-source Git matches, all missing-path groups, and the current 20-file packaging classification gap. The evidence supports a retained selective card patch combined with older application lineage and subsequent changes to three patched files.

Unknown: the exact actor/time/mechanism of intervening deployments; a single complete deployed Git commit; whether currently absent files were never deployed or later deleted; the seven unmatched generated/metadata files' build provenance; and the historical mechanism behind packaging omissions. File mtimes in `remote-file-metadata.stdout` are preserved but are not reliable substitutes for deployment timestamps or actor identity.

The root release owner controls full runtime inventory, the rollback archive and any staging reconciliation. This lineage report establishes why version labels, prior local certification and optimized response URLs are insufficient evidence of accepted-source deployment; it does not authorize or perform an upload.
