# One Team Distribution Contract

## Purpose

The Fashion Theme Team has one canonical editable source:

```text
/Users/theceo/plugins/fashion-theme-team
```

The DevSkyy `plugins/fashion-theme-team` directory and Codex's installed plugin
cache are distributions of that source. They exist so repository-local contracts
travel with DevSkyy and so Codex can load the installed team. They are not
independent implementation branches.

## Authority and boundaries

1. Author all new agents, skills, capabilities, charters, scripts, and verifier
   changes in the canonical source.
2. Run `bash scripts/verify.sh` in the canonical source before synchronizing a
   distribution.
3. Inspect each destination before mutation. A distribution sync is a local
   package update; it does not authorize a remote write, plugin publication,
   push, deployment, or paid provider call.
4. Use only an explicit existing target path. The sync utility rejects a target
   that is not a Fashion Theme Team package and never follows Git metadata.
5. Never use an installed cache as an authoring source. Cache changes are
   overwritten by the next installation or refresh.

## Required synchronization sequence

From the canonical source root:

```bash
python3 scripts/sync-team-distributions.py --target /path/to/distribution --check
python3 scripts/sync-team-distributions.py --target /path/to/distribution --apply --prune
python3 scripts/sync-team-distributions.py --target /path/to/distribution --check
bash /path/to/distribution/scripts/verify.sh
```

`--prune` removes only package files absent from the canonical source; it never
touches `.git` or local cache/output paths. Use it when the destination must be
exactly equal. A nonzero `--check` result means the destination is stale and may
not claim a source capability until it is synchronized.

## Candidate and release discipline

Distribution parity proves only package parity. It does not certify a theme
candidate, a product-media claim, browser QA, commerce behavior, deployment, or
external provider authority. A source mutation changes the plugin candidate;
fresh verifier evidence and the destination checks above are required before a
new capability can be described as active in that distribution.
