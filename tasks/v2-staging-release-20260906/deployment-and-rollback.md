# Exact V2 staging deployment plan — approval required

## Scope and identity

This plan prepares steps 3–6 of the founder's sequence. It is not execution authority. Steps 1–2 reconcile and package the accepted V2 source. Production, later theme optimization, Blender, product-content decisions, Town Line development and payment certification are outside this deployment.

The only target is `https://staging-7e48-skyyrose.wpcomstaging.com`, reached through the already verified staging SSH account. The active theme slug is `skyyrose-flagship-2`. Home option, siteurl, stylesheet, template, public page assignments and WooCommerce page IDs are recorded in `staging-options.stdout`. The remote runtime is WordPress7.1, PHP8.4.25, WooCommerce11.1.0 and WP-CLI2.12.0. The current theme parent is writable; unzip exists. Python3 was not found and is not a deployment dependency.

The release is the one ZIP named in `package/release-manifest.json`, with its SHA-256, 527-entry runtime file map, isolated source tree and toolchain. The final approval packet pins the exact manifest and ZIP hashes after independent review. The version label2.4.4 is descriptive only: the ZIP and file hashes establish release identity. No generic deploy script with a production default is used.

`deployment-file-plan.json` enumerates every unchanged, replaced, added and retired active-theme path. No individual source file is selected or copied during deployment. Authoring sources, obsolete candidate masters, build tools and QA artifacts excluded from the runtime package remain preserved in the full source export and exact rollback archive. Independent review must establish that exclusions do not remove active heroes, scenes, card identity or fallback behavior.

## Preconditions before any upload

1. Record the founder's explicit approval of the final ZIP SHA, staging origin, file-action manifest and this rollback scope. Approval must identify this artifact; no approval is inferred from build success or the roadmap.
2. Rehash the local ZIP and manifest against the approval record. Reject altered bytes, unsafe archive entries, symlinks, path traversal, missing runtime files or changed source/generated parity.
3. Through the staging SSH connection, reread home/siteurl/stylesheet/template and compare the complete active-theme inventory with `rollback-receipt.json`. If any of the447 current files changes or another file appears, stop before upload and refresh reconciliation/rollback evidence. Do not overwrite concurrent work.
4. Check available disk space for the ZIP, extraction and retained rollback, directory write access, and the standard installer command. No permission changes or new dependencies are authorized. Verify the existing theme remains active; no activation, demo import, product sync, content replacement or option migration is required.

## One-artifact installation

After approval and preconditions, upload only the approved ZIP to a unique staging temporary path containing the release identity. Verify its remote SHA against the approved local value before installation.

The standard operation is `wp theme install <verified-local-zip-path> --force`, executed from `/srv/htdocs` on the verified staging account. Do not add `--activate`, `--ignore-requirements` or `--insecure`. The theme is already active. WordPress.com's documented CLI supports installation from an archive; the overwrite behavior of `--force` is explicit in the upstream command reference. This is a complete runtime-theme replacement, not a patch upload.

The installer may briefly make the staging theme unavailable and may perform its normal internal update/cache housekeeping. No manual cache-purge command, CDN configuration change, optimizer toggle, maintenance-option edit or production action is part of this plan. If managed delivery remains stale after installation, report that layer separately and do not change configuration to force a pass.

Immediately rehash every installed runtime file and enumerate unexpected active-theme files. Require exact equality with the approved runtime manifest. Recheck active-theme identity and the captured page/option assignments. The install command's exit0 is not sufficient evidence of deployment parity.

## Rollback evidence and triggers

An exact read-only archive of the current447-file staging theme has already been downloaded to private local evidence storage. `rollback-receipt.json` records its path, byte count, SHA and all file hashes. It matches the complete remote inventory, rather than just the earlier451-path subset comparison. No file was written on staging to create this backup.

Before installation, retain the verified archive and make a staging-local recovery copy in a unique private temporary location, with its SHA checked. This upload is included only in the deployment authorization's rollback scope. No database backup is fabricated: this release plans no database migration, content import, activation or customer-data change.

Rollback immediately for an incomplete installation, installed hash mismatch, PHP fatal/critical route failure, loss of an approved hero/scene/card/Quick View/Ask Skyy feature, or new severe commerce/accessibility break. A pre-existing performance failure alone is not a rollback trigger; measure the correctly deployed candidate before optimization.

Recovery restores the complete captured theme tree, never a selection of remembered files. Preserve the failed tree separately for diagnosis, restore the validated archive under the original slug, and verify all447 old hashes and original options/route responses. Retain the failed candidate and logs privately; do not delete unrelated directories. If remote recovery cannot finish, stop and report the exact state rather than attempting changes to hosting configuration. Rollback is bounded to this theme; database or platform intervention needs separate authority if unforeseen behavior appears.

## Feature parity gate before performance conclusions

Use the installed file manifest plus actual browser resources and decoded optimized-bundle hashes. An old HTML/cache response cannot certify the new theme merely because its disk files are correct.

- Home, Signature, Black Rose, Love Hurts and Kids Capsule: verify the five approved animated hero surfaces, their exact poster/film sources, crop/composition, reduced-motion and no-JavaScript fallback. Home uses the approved Black Rose film. Shop remains its accepted native archive/card/Quick View surface; this release does not add an animated Shop hero.
- All three collections: all nine approved scenes, correct sequence, scene crop and artwork, current/near-scene loading, pause/resume, reverse navigation and scene→commerce handoffs.
- Shop and collection cards: approved imagery identity, responsive candidates, displayed SKU/product links and intact card layout. No imagery promotion or alternate creative substitution.
- Native Quick View: open from actual intent, verify the product-specific request, native form/variation/media dependencies, variation selection, loading/error/close/reopen handling, stale-response isolation, focus and keyboard recovery. No order or payment submission.
- Ask Skyy: lightweight static availability; untouched mobile and desktop free of GLB/Three/Draco requests; intentional activation, model/frame/animation/chat states, minimize/recall and reload lifecycle. Keep GPU/physical stable-frame limits explicit.
- Representative PDP, Cart and Checkout: route integrity, native form/media presence, correct empty-session redirects and no new fatal/missing resources. Product facts and founder media approval remain a separate content gate.

Record desktop and390px views plus reduced-motion/fallback evidence, console/network failures, resource identities and the source manifest. Keep failed feature gates explicit; do not substitute older local screenshots for staging proof.

Only after feature identity and parity are established, repeat the B13 normal/moderate profiles and relevant Lighthouse runs against the correctly deployed candidate. Separate theme, asset, WordPress/PHP, WordPress.com, CDN, network and third-party contributions. Do not apply optimization automatically within this release operation.

## Pinned execution and recovery paths

The reviewed release ZIP SHA is `e47588205c55e6303588a207d7ee766ad175b82f606d167198f4ec2c25b1d80d`. The rollback TAR SHA is `1b8164b36374aadea85aa4ec7c4d3f8a18e9f8e9f34042f706bd21c090a45ac3`. Any change invalidates this procedure and requires a refreshed approval packet.

After approval only, create `/tmp/skyyrose-v2-e47588205c55` as a new mode0700 directory on the verified staging account. Abort if it already exists. Upload the approved ZIP, the exact rollback TAR and the two verification manifests (`runtime-sha256.txt`, `rollback-sha256.txt`) there. The manifests are generated from the reviewed file maps, not from whatever files happen to be on staging after installation. No theme source is uploaded individually.

Before installation, reserve two new same-parent paths: `/srv/htdocs/wp-content/themes/.skyyrose-v2-recovery-e47588205c55` and `/srv/htdocs/wp-content/themes/.skyyrose-v2-failed-e47588205c55`. Abort if either already exists. The recovery path contains the archive's nested `skyyrose-flagship-2/` root; the failed path is reserved for the whole failed active theme. No deletion command is part of recovery.

The archive has been inspected locally:447 regular files,106 directories, all below `skyyrose-flagship-2/`, no absolute or parent-traversal paths, no symlinks, hard links or special files, and only original UID152046411. Remote TAR successfully produced this exact archive during read-only capture. Recheck the SSH UID, TAR, SHA utility and comparison utility before mutation. The upload hash check binds the locally inspected archive before extraction.

The following commands are reviewable instructions, not commands executed in this phase. Run them in Bash only on the verified staging account after its home/siteurl/theme options and approval hashes pass the preconditions above:

```bash
set -euo pipefail
cd /srv/htdocs
test "$(wp option get home)" = 'https://staging-7e48-skyyrose.wpcomstaging.com'
test "$(wp option get siteurl)" = 'https://staging-7e48-skyyrose.wpcomstaging.com'
test "$(wp option get stylesheet)" = 'skyyrose-flagship-2'
test "$(wp option get template)" = 'skyyrose-flagship-2'
test "$(id -u)" = '152046411'
printf '%s  %s\n' 'e47588205c55e6303588a207d7ee766ad175b82f606d167198f4ec2c25b1d80d' '/tmp/skyyrose-v2-e47588205c55/skyyrose-flagship-2.zip' | sha256sum -c -
printf '%s  %s\n' '1b8164b36374aadea85aa4ec7c4d3f8a18e9f8e9f34042f706bd21c090a45ac3' '/tmp/skyyrose-v2-e47588205c55/staging-theme-before.tar' | sha256sum -c -
cd /srv/htdocs/wp-content/themes/skyyrose-flagship-2
test -z "$(find . ! -type f ! -type d -print -quit)"
find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > /tmp/skyyrose-v2-e47588205c55/before.sha256
cmp /tmp/skyyrose-v2-e47588205c55/rollback-sha256.txt /tmp/skyyrose-v2-e47588205c55/before.sha256
test ! -e /srv/htdocs/wp-content/themes/.skyyrose-v2-recovery-e47588205c55
test ! -e /srv/htdocs/wp-content/themes/.skyyrose-v2-failed-e47588205c55
mkdir -m 700 /srv/htdocs/wp-content/themes/.skyyrose-v2-recovery-e47588205c55
tar --no-same-owner -xf /tmp/skyyrose-v2-e47588205c55/staging-theme-before.tar -C /srv/htdocs/wp-content/themes/.skyyrose-v2-recovery-e47588205c55
cd /srv/htdocs/wp-content/themes/.skyyrose-v2-recovery-e47588205c55/skyyrose-flagship-2
find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > /tmp/skyyrose-v2-e47588205c55/recovery.sha256
cmp /tmp/skyyrose-v2-e47588205c55/rollback-sha256.txt /tmp/skyyrose-v2-e47588205c55/recovery.sha256
cd /srv/htdocs
wp theme install /tmp/skyyrose-v2-e47588205c55/skyyrose-flagship-2.zip --force
cd /srv/htdocs/wp-content/themes/skyyrose-flagship-2
test -z "$(find . ! -type f ! -type d -print -quit)"
find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > /tmp/skyyrose-v2-e47588205c55/installed.sha256
cmp /tmp/skyyrose-v2-e47588205c55/runtime-sha256.txt /tmp/skyyrose-v2-e47588205c55/installed.sha256
```

An installer or post-install failure invokes the following exact recovery sequence rather than continuing to browser certification. The recovery tree was already extracted and hash-verified before installation. Its move and the failed-tree preservation occur on the same parent filesystem. The two moves are not claimed to be one atomic exchange; a brief staging-only gap is possible. A connection interruption between moves is recoverable by inspecting these fixed paths and completing the second move, not rerunning blindly.

```bash
set -euo pipefail
cd /srv/htdocs/wp-content/themes
test -d .skyyrose-v2-recovery-e47588205c55/skyyrose-flagship-2
test ! -e .skyyrose-v2-failed-e47588205c55
if test -e skyyrose-flagship-2; then
  mv skyyrose-flagship-2 .skyyrose-v2-failed-e47588205c55
fi
mv .skyyrose-v2-recovery-e47588205c55/skyyrose-flagship-2 skyyrose-flagship-2
cd skyyrose-flagship-2
test -z "$(find . ! -type f ! -type d -print -quit)"
find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > /tmp/skyyrose-v2-e47588205c55/restored.sha256
cmp /tmp/skyyrose-v2-e47588205c55/rollback-sha256.txt /tmp/skyyrose-v2-e47588205c55/restored.sha256
cd /srv/htdocs
test "$(wp option get home)" = 'https://staging-7e48-skyyrose.wpcomstaging.com'
test "$(wp option get stylesheet)" = 'skyyrose-flagship-2'
```

Then perform the original route/option checks and record recovery status. If the second move fails, retain the preverified recovery directory and failed directory, report the exact filesystem state, and perform no unrelated deletion, option update or inferred database repair. Installation and recovery execution logs will be new evidence; none exists yet.

## Primary platform references

- [WordPress.com SSH support](https://wordpress.com/support/ssh/): supported access and preinstalled WP-CLI.
- [WordPress.com common WP-CLI commands](https://developer.wordpress.com/docs/developer-tools/wp-cli/common-commands/): theme installation from ZIP and explicit overwrite flag.
- [WP-CLI theme install reference](https://developer.wordpress.org/cli/commands/theme/install/): archive input, force overwrite, activation and requirements flags.

These references support the proposed operation; the current target, versions, hashes, writable parent and installed tools were separately observed through read-only staging commands. No remote installation or rollback has been rehearsed yet, because those operations require step3 authorization.
