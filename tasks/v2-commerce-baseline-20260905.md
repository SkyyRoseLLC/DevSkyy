# V2 commerce baseline: Phase 1 provenance checkpoint

Status: **SOURCE/STAGING PARITY GATE NOT PASSED**. Investigated September 5, 2026.
The user explicitly requires reporting before continuing when deployed source
provenance cannot be safely established. Phases 2–10 have not started.

## Implemented

- Created `codex/v2-commerce-safe-baseline-20260905` from
  `6d3e87bab9f401856e9d122b2f2ccbfa4a129b2a`.
- Verified the staging WordPress home and active stylesheet/template over SSH;
  both theme identifiers are `skyyrose-flagship-2`.
- Inventoried and SHA-256 compared every regular file in the deployed V2 theme.
- Downloaded a complete read-only theme recovery archive. All 447 archive files
  match the server manifest, with no missing or unexpected files.

## Root cause and provenance

The mismatch is present on server disk, not solely in CDN or optimized HTTP
responses. Of 234 local and 447 deployed files, 192 common files match, 39
common files differ, 216 exist only on staging, and 3 exist only locally.
Seven changed minified files differ only by a trailing LF; the other 32 are
not explained by that normalization.

Deployed `assets/js/theme.js` matches reachable commit
`41a3142def89375782208ac4bb3e5d022d96719c`. Deployed `functions.php` matches none
of 53 reachable revisions of that file; deployed `assets/css/theme.css` matches
none of 55 reachable revisions. No inspected local worktree matches those two
deployed files. This search is limited to locally reachable history and current
worktrees, not every possible remote/deleted revision.

Server-only modules include `inc/approved-card-fronts.php`,
`inc/hero-commerce-scenes.php`, scene CSS/JS, and media/contracts. Deployed
`functions.php` loads these modules and contains scene and motion behavior absent
locally. Deployed `inc/performance.php` adds a homepage-only `jquery-core` defer
strategy. That is a Phase 2 investigation lead, not a proven runtime root cause.

Observed file modification times also differ: functions.php/theme.css were
modified September 5 at 11:22:11 UTC, theme.js August 22 at 17:01:11 UTC, and
theme.min.js September 5 at 09:37:35 UTC. Timestamps do not prove authorship.
The evidence supports a diverged deployment containing work from other V2
revisions, but does not establish an exact reproducible Git release or its
deployment owner. No release manifest or .git directory was found in the
inspected deployed theme metadata search.

## Target, deployment boundary, and rollback

- Verified active staging target: `/srv/htdocs/wp-content/themes/skyyrose-flagship-2`.
- Verified runtime: WordPress 7.1 and PHP 8.4.25.
- Staging connection settings are in the existing staging-specific environment
  file in the main checkout. The generic WordPress configuration and default
  SSH alias target production and must not be substituted.
- No deployment was performed. An automatic approval hook rejected a read-only
  search of deployment scripts as a production deployment. That inspection was
  not bypassed; the existing script's V2 staging behavior is not certified here.
- Future deployment must bind an exact reviewed commit and generated artifacts
  to a manifest, re-confirm the staging home and theme path, compare the proposed
  delta, and snapshot current state before any upload. Do not use a broad
  replacement to manufacture parity or delete staging-only files.
- Recovery archive: `.artifacts/v2-commerce-baseline-20260905/staging-v2-before.tar.gz`.
  SHA-256: `3850af0ae0caa20a49e8ec4f029ef1543be4ac5ef65202b2a1a0336293f09f4a`.
  Recovery would verify that hash, unpack into a separate staging sibling,
  compare all file hashes, preserve current staging, then replace only the
  confirmed V2 theme under separately authorized deployment control. It is a
  theme-file recovery artifact, not a database backup; restoration is untested.

## Files and data changes

Only this checkpoint document is added to tracked source. Investigation evidence
is ignored under `.artifacts/v2-commerce-baseline-20260905/`, including the server
manifest, local comparison, four focused diffs, archive, and archive verification.
No PHP, JavaScript, CSS, theme assets, WordPress settings, products, plugins,
orders, customer records, or databases were changed. No deployment occurred.

## Verification and remaining risk

- Archive inventory: 447/447 hashes match; zero discrepancies.
- Active V2 theme and staging target: confirmed.
- Local branch/checkpoint: established.
- Exact deployed Git revision and complete source ownership: **unresolved**.
- Build/runtime/commerce fixes: not attempted during this phase. Earlier audit
  test evidence remains an audit of the inspected checkout, not certification
  of this different deployed artifact.

To resume, obtain the commit plus deployment receipt accounting for the deployed
PHP/CSS and staging-only modules, or explicitly authorize recovering this exact
staging snapshot into a new source baseline for review. Importing a snapshot
would create a new revision; it would not retroactively prove its provenance or
approve its media. Do not start Phase 2 until this decision resolves Phase 1.
