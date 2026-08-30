---
name: fashion-release-evidence
description: Builds candidate-bound marketplace, packaging, compatibility, security, test, rollback, and release-readiness evidence for SkyyRose themes.
---

# Fashion Release Evidence

Use after implementation and independent QA. `PASS` means ready for founder
review only; it never authorizes deployment.

## Owners and inputs

Route `theme-release-engineer` with DesignOps and independent QA verdicts. Read
acceptance gates, deliverable contract, WooCommerce coverage, candidate
manifest, package metadata, and current official platform requirements.

## Procedure

1. Freeze candidate identity from commit, patch, untracked inventory, lockfiles,
   theme metadata, and generated hashes.
2. Run source/generated parity, build, lint, type, tests, PHP/theme,
   translation/POT, RTL, license, dependency, package-content, installability,
   design-system adoption, browser, accessibility, performance, and commerce
   gates that apply.
3. Record command, tool version, environment, timestamp, artifact/hash, owner,
   reviewer, status, and applicability for every gate.
4. Produce changelog, compatibility, exclusions, prerequisites, rollback, and
   founder decision packet.
5. Issue exactly `PASS`, `FAIL`, or `BLOCKED`; skipped, stale, timed-out,
   missing, and unavailable evidence never becomes green.

## Verification

Run `scripts/report.sh`, `scripts/verify.sh`, and repository release gates.
Quarantine mixed-candidate evidence and invalidate proof after every mutation.

## Boundaries

Never deploy, merge protected branches, purchase services, upload media, or waive
security, visual, or commerce gates.
