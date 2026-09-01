# Operating Model

## Evidence Levels

- `repo`: source inspection only.
- `repro`: a deterministic command reproduced the behavior.
- `test`: a regression gate returned the expected result and can be proven to fail.
- `visual`: human or approved vision review against canonical physical-product evidence.
- `live`: a cache-busted probe of the deployed target.

Claims cannot exceed their evidence level.

## Learning Promotion

An observation becomes a recommendation after two matching root-cause signatures. It
becomes an enforced gate only when all conditions hold:

1. At least two confirmed occurrences exist.
2. A deterministic detector has both a positive and negative fixture.
3. False-positive scope is documented.
4. The gate fails closed when its runner errors.
5. Promotion is reviewed in a pull request.

Learned text never executes directly. It selects or proposes deterministic rules.

## Repair Classes

`safe_generated` may run only explicit commands from plugin configuration, such as
rebuilding canonical SOT projections or minified assets. `source`, `creative`,
`commerce`, `security`, `release`, and `external_write` are never automatically edited.

## Termination

Stop after three unsuccessful correction iterations for the same signature. Record the
blocker and require human input; do not widen permissions or weaken the gate.
