# Local delivery parity — 2026-09-06

Status: HTTP decoded-byte parity PASS; independent source recheck completed
with the prior verifier finding resolved. See `delivery-source-review.md`.
Lighthouse compression baseline/candidate measurements are owned separately by
root and are not asserted by this report. Deployed settings remain UNVERIFIED.

Implementation scope is `tools/v2-runtime/delivery/` plus a new removable MU
adapter in the existing synthetic WordPress fixture. No theme runtime source,
compiled asset, CSS policy, jQuery strategy, DB value or Lighthouse/proxy code was
changed by this task. Root is holding theme runtime at baseline
`aabd2bffdc1e322862acd05a5640c14cf00f3acf` until compressed baseline measurement.

The loopback-only Docker Nginx gateway runs on18308 with fixed upstream18303,
gzip6, Vary:Accept-Encoding and zero Nginx response caching. It preserves native
status/MIME/cookies/redirects and does not rewrite bodies. Digest is pinned to
`nginx@sha256:a8b39bd9cf0f83869a2162827a0caf6137ddf759d50a171451b335cecc87d236`.
Running container inspection confirmed read-only filesystem and exact host
binding127.0.0.1:18308. Initial startup exposed a read-only default temp-directory
error; declaring all Nginx temporary directories under/tmp fixed it. `nginx -t`
passed afterward.

The first parity run failed because native attachment URLs retained18303.
The guarded fixture adapter now maps only upload URL/baseURL through the existing
Core API; attachment identities and physical paths are unchanged. The failed
receipt remains `.artifacts/v2-delivery-20260906/parity-initial.json`.
The first successful receipt is
`.artifacts/v2-delivery-20260906/parity-upload-fixed.json` (18 responses).
The final stricter verifier and live-container attestation passed all 18 response
checks in `.artifacts/v2-delivery-20260906/parity-final.json`; its inspected
configuration is retained in `parity-final.nginx-T.txt` beside that receipt.

| Response | Identity body bytes | Gzip body bytes | Decoded-byte proof |
| --- | ---: | ---: | --- |
| theme.min.css | 54,185 | 9,918 | Disk SHA256 equals decoded gzip and identity |
| theme.min.js | 16,200 | 5,213 | Disk SHA256 equals decoded gzip and identity |
| Home | 84,021 | 15,557 | Same-response pre-encoding digest |
| Shop | 128,872 | 16,639 | Same-response pre-encoding digest |
| PDP | 107,106 | 19,605 | Gzip response decoded107,135; each response matches its own digest |
| Cart | 118,576 | 32,980 | Same-response pre-encoding digest |
| Account | 57,343 | 12,269 | Same-response pre-encoding digest |

PDP separate requests varied29bytes; this is why separately rendered HTML is
never compared for equality. Checkout's302empty-cart redirect and a native404
were preserved and checked in both encoding modes. No redirect was followed.
No orders, cart writes, emails, payment calls or outbound browser requests occurred.

The source URL census passed for all HTML resources it examines, without raw
18303 or remote resource authorities. This is an HTML census; runtime-generated
network requests remain the existing exact-origin browser runner's responsibility.
The fixture's local-isolation MU plugin still disables outbound HTTP/mail/webhooks
and payment gateways. Normal HTTP requests omit the diagnostic digest and never
enable its proof-only HTML buffering. Five Node unit tests and three PHP adapter
guard/API tests pass. Syntax checks pass. No browser/Lighthouse was run by this
agent; the exclusive window was released to root after HTTP checks completed.

Independent review found that the verifier could accept an errored checkout,
a missing redirect Location, or an uncompressed/failed ordinary Home response.
Those assertions are now fail-closed and covered by focused failure regressions.
The verifier also records actual running Docker image/start/binding and nginx-T
output, rejecting configuration changes made after container startup. This
expanded verifier's HTTP rerun passed after root's exclusive Lighthouse window
closed; its five source-level Node regression tests pass. Gateway and adapter
bytes did not change during or after that window. The final HTTP window was
released to root before its next performance comparison.

Receipt configuration SHA256:
`adeecf0a3efb71f3ac9ce19b107179a6e117a64f97c9785114adc4d0be759d67`.
Adapter SHA256:
`9d621688cdde833548810474d44ed7535a1c5dc18934710c005a0d66ef761c0a`.
Future changes require fresh receipt hashes.

Reproduction, failure handling, origin limits and serial measurement commands:
`tools/v2-runtime/delivery/README.md`. This host-owned configuration is evidence
that real compression can preserve source bytes; it does not claim the theme
alone controls WordPress.com response encoding, deployed cache policy or field CWV.

## Direct public-asset delivery addition

Current configuration now serves the allowlisted public theme asset subtree
through Nginx's native static handler. Root observed that the raw PHP static
server ignored video Range requests, returning200with the whole676,790byte
Black Rose VP9 file; assets also shared the single PHP worker with dynamic
requests. The gateway now mounts only this theme's458-file asset tree read-only,
with zero symlinks in the audited inventory, strict route/filename/extensions,
`disable_symlinks on`, no directory listing and no PHP/private-file exposure.
The fixture origin adapter, theme and original asset bytes were unchanged by
this delivery addition. Dynamic requests remain uncached and proxied as before.

New actual HTTP receipt:
`.artifacts/v2-delivery-20260906/parity-static.json` (PASS). This preserves the
previous proxy-only parity receipts as historical evidence. It passes all18
standard compressed/identity response proofs and adds:

- Complete VP9 response SHA equals approved source SHA.
- `Range: bytes=0-1023` returns206,1024bytes, `bytes 0-1023/676790`, exact slice SHA.
- Suffix `bytes=-32` returns206 with the exact final32 source bytes.
- Out-of-bounds range returns416 and `Content-Range: bytes */676790`.
- HEAD returns zero body with correct full length and MIME for WebM, WOFF2, WASM
  and GLB; WASM is `application/wasm`, GLB is `model/gltf-binary`.
- Raw/encoded/double-encoded traversal, dot files, PHP, directory listing and
  actual JSON/Markdown/text files under assets are denied. Static POST returns403.
- Live Docker mount source, read-only mode, expected config, startup and pinned
  image match. The metadata inventory digest is not presented as a content hash.

Seven Node tests now pass, including whole200instead-of-range206 rejection,
wrong range bytes/totals/encoding and a temporary symlink inventory failure.
No runtime symlink was added to the real assets tree. Gateway config checks pass.
Independent source recheck approved this addition with no outstanding findings;
the reviewer reran all seven Node tests and checked saved range/MIME/mount and
configuration evidence. No browser/Lighthouse was run here.

Current Nginx config SHA256:
`777512d22a012d4fe5d1150debee955b56c7cd950b54ba2eb898543e45fed241`.
The adapter SHA remains
`9d621688cdde833548810474d44ed7535a1c5dc18934710c005a0d66ef761c0a`.
The inspected effective config is `parity-static.nginx-T.txt` beside the receipt.

This is a new delivery configuration. Prior proxy-only compressed results must
not be presented as an identical-infrastructure baseline for this configuration.
Root must compare the relevant theme revisions through identical direct-static
settings. `V2_DELIVERY_ASSETS` allows the gateway and verifier to point at an
explicit matching source checkout; restart the gateway after changing the mount.
Changing only the fixture's PHP theme symlink would create mismatched delivery.

## Production instrumentation profile — 2026-09-06

NEW DELIVERY COMPLETION. The original PHP18303 worker was measured directly:
PHP8.5.6 cli-server already had OPcache enabled (1482 cached scripts), but Xdebug
ran in coverage mode. Its revalidation interval was 2 seconds. The new isolated
worker uses the same existing fixture/router/binary with coverage disabled,
OPcache timestamp validation on every request, path revalidation enabled and
realpath cache disabled. No page/object/customer cache, database write, source
hash weakening or theme change is part of this delivery change.

`tools/v2-runtime/delivery/php-origin.sh` owns only port18309 and records the PID
plus exact process start time/command. Stop refuses mismatched ownership. The
independent raw18303 PID9444 was explicitly checked unchanged. Nginx now proxies
dynamic requests to18309; its compression/static policies are otherwise identical.
The built-in PHP server remains a local development server. This is a reproducible
instrumentation profile comparison, not a claim of deployed hosting configuration.

Evidence:

- `.artifacts/v2-delivery-20260906/parity-production-profile.json`: PASS18 decoded
  response proofs, static native206/416/HEAD, MIME, mount/config attestation and
  source hashes; ordinary HTML compressed without proof headers.
- Associated `.php-origin.json`: same-worker PHP8.5.6, effective Xdebug modes[],
  OPcache active, validate_timestamps1, revalidate_freq0, revalidate_path1,
  realpath_cache_size0; binary/INI/router/launcher SHA256 and startup command.
- `php-origin/profile-after-restart.json`: PASS after actual owned stop/start;
  current PID3292. The random-name diagnostic was removed in finally, and no
  `v2-worker-proof-*` file remained in the fixture MU directory.
- Eight focused Node assertions/suites passed, shell syntax and diff whitespace
  checks passed. Independent source review requested; its final disposition is
  recorded separately by the reviewer.

The first `profile-initial.json` FAIL is retained: a background process launched
under the tool shell did not survive shell teardown. The launcher now uses Node
built-in detached spawning to establish its own session. A subsequent
`profile-current.json`, complete parity, and owned restart proof passed. This
historical launch failure was not used for any benchmark.

Every baseline/candidate symlink switch still requires restarting this owned PHP
worker and the asset-mounted gateway, then fresh markup/source and worker proof.
Both sides must use these identical settings; earlier raw/proxy/static results
remain historical evidence under their original delivery conditions. No
performance improvement or target closure is inferred before the parent's new
paired serial Lighthouse measurements.
