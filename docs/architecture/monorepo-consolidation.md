# Monorepo consolidation baseline

Generated from DevSkyy `main` tree `268e8fef8c865db1b9329b01e54ab14fd260a2ab`.

## Measured baseline

| Metric | Value |
|---|---:|
| Tracked files | 9,070 |
| Current tree bytes | 1,140,049,966 |
| Initial archival candidates | 1,130 |
| Candidate bytes | 683,989,940 |
| Exact duplicate blob groups | 604 |
| Avoidable duplicate bytes in current tree | 121,770,039 |

The candidate set is intentionally conservative: media under `assets/`,
`renders/`, `screenshots/`, `archive/`, `_prototype/`, and
`docs/design-mockups/`. Production theme assets are not automatically removable.

## Required migration gates

1. Inventory the object with original path, Git blob SHA, byte size, media type,
   source commit, proposed archive path, and schema version.
2. Classify every reference as runtime, build-time, documentation-only, or orphaned.
3. Copy to `SkyyRoseLLC/DevSkyy-Assets/archive/<original-path>`.
4. Record and verify SHA-256 for the archived object.
5. Preserve necessary optimized runtime derivatives in DevSkyy.
6. Replace source directories with locator documentation where useful.
7. Run affected builds and tests before proposing deletion.
8. Remove files only in reviewed, independently revertible batches.
9. Treat Git-history rewriting as a separate operation requiring explicit approval.

## Target architecture

- `apps/`: deployable API, storefront, and WordPress application boundaries
- `packages/`: shared Python/TypeScript/commerce libraries
- `services/`: independently deployed integrations
- `tooling/`: scripts, CI helpers, generators, and configuration
- `docs/`: active architecture and operations documentation
- private asset archive: original media, renders, source models, and historical exports

Directory moves will follow asset extraction and dependency mapping so the repository
does not combine structural churn with unverified runtime changes.
