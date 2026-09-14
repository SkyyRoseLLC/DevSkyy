# Independent delivery fixture source review

Status: **APPROVE FOR SOURCE REVIEW — verifier finding resolved; no outstanding findings in this scope.** No gateway startup, browser, Lighthouse, source mutation, customer-data operation, or deployment performed by this reviewer.

Scope established from local staged/unstaged status and newly untracked `tools/v2-runtime/delivery/`: `nginx.conf`, digest-pinned `image.txt`, `gateway.sh`, `fixture-origin.php`, `verify.cjs`, and their Node/PHP tests. No PR/CI metadata was supplied for this local scope. The code is JavaScript/PHP/shell; TypeScript is inapplicable, and no scoped ESLint installation/configuration is available.

## Finding

**RESOLVED — reject invalid checkout and ordinary-response states before writing PASS.** Independently verified new `htmlStatus()` limits checkout to 200 or an explicit supported redirect, requires a nonempty local Location, and rejects 404/500. New `ordinaryResponse()` requires Home 200, HTML MIME, gzip and Vary, absence of diagnostic hashing, and a decoded local resource census. Added regression cases cover invalid checkout status, missing/empty/remote Location, ordinary 500 and missing MIME/compression/Vary, and unwanted diagnostics. All five Node tests passed on independent rerun.

## Source assessment

- This is a real Nginx reverse proxy with standard gzip negotiation, a pinned container image, read-only filesystem, dropped capabilities, and loopback-only published port. The Host guard restricts the configured authority. It has no Lighthouse/user-agent branching, response-body replacement, `sub_filter`, response cache, or proxy store.
- The PHP adapter requires its explicit fixture constant and exact gateway Host. It changes only WordPress URL option results and specific URL API fields for the known original loopback authority. It does not rewrite arbitrary output or mutate database records. Upload physical paths are retained.
- Diagnostic requests alone enable a pass-through output buffer that hashes the same response before gzip. Normal requests do not enable that buffer. Comparing decoded HTML against that upstream hash avoids false differences from nonces across separate responses. The reviewer did not run HTTP delivery proof and therefore does not independently claim deployed-byte parity.
- The verifier confines network requests to the configured HTTP loopback origin, bounds response bytes and request time, rejects unexpected encoding, and checks static decoded bytes against disk. Eligible assets and successful HTML responses check MIME, negotiated compression and Vary. The resource census checks direct markup URLs, including dormant source/srcset attributes, but is not a complete dynamic/CSS-subresource browser census.
- The fixture installer refuses overwriting the adapter, requires the local isolation marker, and restricts installation to an existing `.artifacts` path. The launcher does not modify theme source or database settings. `stop` targets its fixed fixture container name.
- The strengthened verifier now also inspects Docker's running state, digest-pinned configured image, actual image/container identity, read-only root, and exact loopback binding. It rejects config files modified after container startup and saves a hashed `nginx -T` dump containing the intended configuration. Independently inspected these checks and the owner's saved `.artifacts/v2-delivery-20260906/parity-final.json`: it reports PASS for 18 response proofs, ordinary-request diagnostic absence, the expected digest/binding, and a configuration-dump hash that matches its saved artifact. This is an independent artifact/source check; the reviewer did not rerun HTTP or Docker commands. The adapter's source hash remains recorded, with normal behavior checked by actual response assertions.
- This fixture can support a fair synthetic delivery comparison only when both baseline and candidate use the same gateway/settings and normal request behavior. It is not evidence that a remote production host has identical compression, caching, latency, or infrastructure.

## Checks performed

- Initial `node --test tools/v2-runtime/delivery/verify.test.cjs`: **3 passed, 0 failed**; repair recheck: **5 passed, 0 failed**.
- `php tools/v2-runtime/delivery/test-fixture-origin.php guard`: **passed**.
- `php tools/v2-runtime/delivery/test-fixture-origin.php disabled`: **passed**.
- `php tools/v2-runtime/delivery/test-fixture-origin.php enabled`: **passed**.
- `sh -n tools/v2-runtime/delivery/gateway.sh`: **passed**.

These are bounded unit/guard/syntax checks plus source and saved-artifact inspection. The owner's saved wire proof was inspected as described above. Nginx startup, actual HTTP execution, cookie/transaction behavior, and performance runs remain assigned to the owning lanes.

## Direct static asset delivery follow-up

Independently reviewed the new frozen Nginx asset location, launcher mount, inventory/range helpers, and expanded verifier. **No actionable source finding.** The gateway now mounts only the expected theme `assets` directory read-only and uses Nginx's static handler for the exact public prefix. Extension allowlisting excludes PHP, JSON, Markdown, text, and extensionless paths; dot paths, ambiguous traversal encodings, directory listing, and non-GET/HEAD methods are rejected. `disable_symlinks on` protects runtime access, and the verifier also rejects symlinks in its source inventory. The URL prefix and alias terminate with matching slashes; the wider PHP theme/repository is not mounted.

Native media delivery is not transformed or sent through the PHP worker. MIME mappings cover the required video/font/WASM/GLB types; text asset gzip remains standard Nginx negotiation. HEAD and ranges are handled by Nginx without custom slicing or browser-specific conditions. HTML continues through the existing uncached guarded proxy. No DB-writing, response-body rewriting, or audit-only path was added.

The expanded verifier proves the read-only mount points to the expected real assets path, records inventory identity, validates exact prefix/suffix 206 bodies against source slices, rejects full-body 200 and corrupted range headers in unit cases, verifies 416 total size, and checks HEAD type/length/empty body. Saved `.artifacts/v2-delivery-20260906/parity-static.json` reports PASS for the 18 existing response proofs plus these static checks: a 676,790-byte source-identical WebM; 0–1023 and final-32-byte matching 206 responses; 416; required HEAD MIME checks; and private/traversal/method denials. Independently verified the saved `nginx -T` artifact hash. The owner reports seven Node tests passed; this follow-up has not rerun HTTP/browser work and does not recast the saved receipt as a new live execution.

The prior all-proxy receipt remains historical evidence. Performance comparisons must rerun both reference and candidate through this same static gateway configuration; comparing an earlier PHP-worker baseline directly against the new gateway would confound source and delivery effects.

After the visual owner closed the protected font comparison window, independently reran the expanded Node unit suite: **7 passed, 0 failed**. Shell syntax recheck also passed. No HTTP, browser, or container execution was performed.

## Owned PHP worker profile follow-up

Reviewed frozen `php-origin.sh`, `verify-php-origin.cjs`, its unit test, verifier integration, Nginx's fixed upstream change to 18309, and README restart/ownership instructions. **No actionable finding.** The launcher targets a separate loopback worker and never signals the independent 18303 process. It validates the existing local fixture/isolation files, refuses an existing ownership record, serializes launch with a lock, and requires matching PID plus process start/command identity before TERM. The profile uses explicit Xdebug-off and source-revalidating OPcache flags; it creates no data cache or DB changes.

The diagnostic is exclusive-created under a random filename, guarded by exact fixture ABSPATH, gateway Host, worker port, and a cryptographically random header token. It queries the explicitly owned worker and validates its PID, SAPI, actual Xdebug modes, OPcache state and freshness settings. Its token is not written to the receipt. Normal completion and handled failure remove the temporary MU file in `finally`; the README correctly documents that an abruptly terminated proof may leave an inert guarded file requiring cleanup before measurement. No claim of crash-proof automatic cleanup is made.

Independently ran both delivery Node suites: **8 passed, 0 failed**; worker shell syntax passed. Inspected the saved `parity-production-profile.php-origin.json`: PASS, diagnostic removed, cli-server PID recorded, no active Xdebug mode, OPcache enabled, timestamp/path validation on, revalidate frequency zero, realpath cache zero, and JIT disabled. This is a saved-evidence check, not a fresh HTTP/process execution. Binary/INI/router/launcher identities are recorded by the source-reviewed verifier.

This is a new measurement environment relative to the previous worker. Both baseline and candidate must restart, attest and measure with this same worker policy, static asset mount, gateway configuration, fixture, and browser settings. The README explicitly retains earlier receipts as historical and describes the PHP built-in server as a local fixture rather than a production deployment recommendation.

Final cleanup refinement independently read: `diagnosticCreated` becomes true only after successful exclusive creation, and `finally` unlinks only when that ownership flag is true. A preexisting filename collision is therefore preserved on creation failure. No additional finding.
