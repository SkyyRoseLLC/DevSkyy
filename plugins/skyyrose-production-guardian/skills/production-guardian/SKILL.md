---
name: production-guardian
description: Run the DevSkyy production correction loop when code, product truth, generated artifacts, hooks, tests, or release paths need continuous hardening. Converts confirmed defects into regression locks, learns repeated patterns, applies only allowlisted deterministic repairs, independently verifies claims, and blocks push/deploy without a fresh clean receipt. Do not use it to authorize production writes or silently change product/brand truth.
---

# SkyyRose Production Guardian

Use one closed loop:

1. **Observe:** record the correction, failed command, affected surface, and evidence scope.
2. **Reproduce:** create a deterministic failing check. A test that never showed RED is not a lock.
3. **Correct:** fix the root cause, including alternate runtime paths and generated consumers.
4. **Harden:** add the narrowest regression rule that would have blocked the defect.
5. **Verify:** run focused gates, then the relevant full surface gate. SKIP or tool error is not PASS.
6. **Challenge:** use an independent verifier for high-risk, visual, commerce, security, or deployment claims.
7. **Learn:** cluster repeated failures by root cause. Promote only evidence-backed patterns into gates.
8. **Ship:** require a clean guardian receipt tied to the current commit. The receipt proves checks; it never grants deployment permission.

## Safety Contract

- Auto-fix only deterministic generated artifacts listed in `config/guardian.json`.
- Never auto-edit product appearance, prices, stock, legal copy, credentials, approvals, media verdicts, or production state.
- Never weaken a test, tolerance, hash guard, or approval requirement to obtain green.
- Never consume retired collection/lookbook outputs from a production consumer.
- Preserve the separation between record completeness and visual approval.
- External writes still require explicit user authorization immediately before execution.

## Commands

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/guardian.py" check --root "$PWD" --mode fast
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/guardian.py" check --root "$PWD" --mode ship
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/guardian.py" learn --root "$PWD"
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/guardian.py" repair --root "$PWD" --dry-run
```

Read `references/operating-model.md` when adding a new gate, changing promotion thresholds, or diagnosing a blocked ship receipt.
