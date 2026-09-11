# Local native asset delivery repair

The final archive browser sweep exposed intermittent upstream connection 504s while the local Nginx gateway fetched native WordPress and WooCommerce assets from the PHP development server. Missing jQuery and js-cookie responses caused JavaScript errors. Native script order was unchanged. The failed browser attempts and Nginx diagnostics remain in `.artifacts/v2-completion-20260906/native-static-timeouts/` and `final-archive-visual/`.

Commit `6f76f6018` repairs this local delivery path. The gateway serves filtered public Core JS/CSS/fonts and WooCommerce assets directly from four read-only, content-addressed snapshots. The snapshot contains 2,064 files and 90,800,443 bytes, copied without transforming their contents. PHP, JSON, source maps, dotfiles, symlinks and customer uploads are excluded. Dynamic PHP, AJAX, cart, checkout and account requests retain the existing uncached upstream behavior. No theme script loading strategy, gzip setting, Lighthouse setting or production hosting configuration changed.

The launcher binds the explicit fixture to its owned PHP worker before writing a snapshot. Every copied file is hash-bound to that fixture and installed Core/WooCommerce version. Newly generated directories use explicit 0755 permissions and files use 0644; existing snapshot drift fails validation, including permission drift. Source directories are never chmodded. Independent general and TypeScript reviews closed the fixture-identity and restrictive-umask findings before activation.

Validation passed under pinned Node 22 and locked TypeScript 5.9.3: 15 focused delivery tests across the native-assets and verifier suites, scoped ESLint, shell syntax and root type checking. No tracked dependency changes were needed. Root, baseline and integrated PR gateway instances each passed 28 live HTTP response proofs. These include identity/gzip byte equality, MIME, HEAD, byte ranges, unsatisfiable ranges, denied writes/private paths and exact read-only mount identities. Receipts:

- `.artifacts/v2-completion-20260906/native-static-parity-root.json`
- `.artifacts/v2-completion-20260906/native-static-parity-baseline.json`
- `.artifacts/v2-completion-20260906/native-static-parity-pr.json`

Initial PR runtime `191c2fa61111b5653bbec4f0733d37ba5032be97` contains the reviewed theme and native-static delivery commits. Its PHP theme symlink and Nginx theme asset mount point to the same PR checkout; the owned PHP worker was restarted for each baseline/candidate switch. The independent raw worker on port 18303 was not stopped. The baseline export is `aabd2bffdc1e322862acd05a5640c14cf00f3acf`, with all 588 Git blob hashes verified.

This is a new local delivery profile. Earlier theme-only static measurements remain historical. Final baseline/candidate comparisons use the same native snapshot, fixture, Nginx configuration, PHP profile and existing exact-origin Lighthouse harness. Browser and repeated timing receipts provide separate acceptance evidence; HTTP parity alone does not establish performance, visual acceptance or production readiness.

## Final IPv4 origin binding

Preflight of the later poster-corrected PR caught a separate dynamic-origin failure before timing began. Docker DNS supplied an IPv6 address that this container could not reach; Nginx returned 502 with `Network unreachable`. The failed receipt and log remain at `final-pr-parity.json` and `final-pr-ipv6-failure.log`. A passing static-assets check did not conceal this dynamic-page failure.

Root commit `b86c4c6ce`, integrated as PR `d02d653b22e185210ce23847b12a7205abbb75dc`, resolves the fixed Docker host through one pinned-image AF_INET lookup at gateway startup. It accepts exactly one canonical private IPv4 address, supplies the sole `v2-php-origin` host override, and uses that fixed alias for PHP. No Docker IP or DNS server is hardcoded. The verifier records the actual host override and both IPv4/all-address lookups; unexpected addresses, families and overrides fail validation. Independent TypeScript review and focused tests passed before the live restart.

The final profile is `native-static-ipv4-v2`. Both baseline and candidate receive fresh owned PHP workers and the same parity sequence before measurement. `ipv4-baseline-parity.json` and `ipv4-final-pr-parity.json` bind their exact source mounts, native snapshot, PHP configuration, alias and Nginx configuration. Earlier native-static measurements remain historical. No throttle, score, film, script-order, compression or caching change was used to resolve this address-family failure.
