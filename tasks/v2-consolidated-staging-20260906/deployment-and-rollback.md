# Consolidated staging deployment and rollback

Authority: founder requested consolidated release and deployment to staging after viewing About and retiring the tagline globally. This authorization replaces the prior ZIP-specific preparation boundary for this new staging release; it does not authorize production.

Target: https://staging-7e48-skyyrose.wpcomstaging.com. Active slug skyyrose-flagship-2. Remote identity currently matches all527 hashes from the previous approved release, with no added/removed/changed files.

Exact new ZIP: `a4ec431146f31d036d5547137d3274a693ccd3d072be6089126593f642d14f4b`,536 runtime entries. Tenexisting runtime files changed; nine added. Existing protected media bytes remain unchanged. New About native content is inside data/editor/about-page.html, SHA9855d4d9be5c628cd0b6a38d89233508f80de8f62c88f1a69692cb58354d10ad.

Rollback theme:292ee73ab2bb494b2fe56c20ee8e871d539f658e81dadce20cac7da7912181ab,170434560bytes, matches all527 current hashes. Prior original447file rollback is preserved untouched. About9386 current-contentSHA d43ef4b28df267bafbb1611a401d6a1945ebf0efa761157e82b26e4afc18632c captured privately. Site description oldvalue captured; only the exact retiredphrase is removed, leaving Premium streetwear from Oakland, CA.

Execution: upload ZIP, rollbackTAR, exact hash manifests and reviewed migration/recovery scripts to unique0700 /tmp/skyyrose-v2-a4ec431146f3. Verify hashes and target; compare full currenttheme hashes with captured527 baseline immediately before install. Prepare recoverytree and verify527. Run wp theme install exactZIP --force withoutactivation. Verify536 exactinstalledhashes. Apply narrow Aboutcontent migration withbeforehash andafterhash checks; removeonly retiredphrase fromsite description after exactbeforevalue check. No other post/page/product metadata, catalog, mediaauthority, platformconfiguration, orders/payments orcachepurge commands.

An installer can perform its automatic normal cache housekeeping; do not addmanualpurging. Recheck targetoptions/active theme. Verifybrowserdeliversnew About, logoWebM, taglineabsence, heroes/scenes/cards/nativeQuickView/AskSkyy/commerceroutes. Record platform/performance separately. Cached stale responses are not sourceparity proof.

On install/filesystem/contentfailure: restorewhole527filetree fromverifiedrecoverycopy, preservefailedthemeunderprivatetmp, restoreexactAboutcontent/site-description value, verify527 hashes/content hash. On severe browserfeature regression, execute samerecovery. A preexistingperformancebottleneck alone does not trigger rollback. Stop on unexplained concurrentfilesystem/content/option drift.

Attempt1 recovery: exact536filesystem installation passed; WordPress rejected About update because inherited template-about.php assignment is not a registered page template. Its post-content write precedes that validation. Whole527theme and exactoldAbout/description recovery independently verified. Retry uses the unchanged ZIP in unique /tmp/skyyrose-v2-a4ec431146f3-attempt2. Apply/rollback explicitly pass empty page_template, which WordPress core treats as no template update, while guarding the existing template metadata. This preserves metadata and avoids unrelated validation. All other guards remain.
