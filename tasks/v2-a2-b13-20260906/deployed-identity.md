# Deployed identity boundary

Target: https://staging-7e48-skyyrose.wpcomstaging.com. Read-only SSH confirmed the home option matches this staging origin, stylesheet is skyyrose-flagship-2 and WordPress is7.1. The earlier attempt to pass skip-plugins/skip-themes flags was rejected by the managed CLI; retry used read-only option/version commands. No option was updated.

Observed deployment identifier: **STAGING-OBSERVED-f2893d998356**. Its deterministic observed-file-map digest is `f2893d998356ac5c86fda86e6422adab348261f02bb00fc935b88e74c1d3ae5a`. This is a hash of the successfully observed comparison inventory, not a remote Git commit or full-tree certification.

Of451 requested theme-source/protected-media paths from the accepted local inventory,322 exist on staging:295 match,27 differ,129 are missing. SHA command exit1 reflects missing paths and is retained with stdout/stderr; it is not interpreted as a completed matching deployment. The local source digest remains4a23629d6b17c1d0393000cfbbe982eedf47712837dd772a66c993ad7e46bbe7.

Notable missing accepted paths include native Quick View commerce modules, scene handoff modules, search preview modules, visual recovery modules, the same-model Skyy runtime poster and many responsive card derivatives. Different paths include mascot loader/controller/renderer, scene scheduling, card/PDP/QV templates, theme.json and functions.php. Missing paths alone do not establish runtime request failures: B13 must inspect actual emitted URLs and behaviors. Source files are not assumed identical to served minified/concatenated bundles.

The initial HTTP Home returned200, noindex/nofollow/noarchive, WordPress.com headers and platform-generated Jetpack Boost critical CSS. Real network behavior must therefore be measured on this deployment; accepted local timings are not transplanted.

B13 may inspect the authorized observed deployment once public-response/media prerequisites are sufficient. It cannot certify the accepted local candidate, and no deployment is authorized to reconcile the difference. End-of-analysis hashes must recheck the observed identity to detect concurrent deployment drift.
