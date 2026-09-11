# Completion and certification matrix

Status is **NEEDS_MORE_WORK**. PASS means the explicitly named test passed; it does not imply all requirements in that system are certified. NOT CERTIFIED is a failed release gate, not an invented observed runtime defect. All browser evidence uses a local synthetic fixture.

| Phase | Current evidence | Gate |
|---|---|---|
| A Baseline/system design | Current branch, baseline Home Lighthouse, architecture map and protected-system census | PASS for baseline; six-route matched before/after not complete |
| B Media/performance | Existing verified 33-card renditions, approved scene/poster manifests retained; native variation loading moved to intent | PARTIAL: mobile performance not acceptable |
| C Motion/material | Shared commerce/character timing, material tokens, reduced-motion mapping and reusable handoff | PARTIAL: full typography-motion adoption not completed |
| D Heroes | Source/asset preservation; Home and collection responsive/cinematic checks | NOT CERTIFIED: full independent before/after and performance acceptance absent |
| E Scroll engine | Existing shared nine-scene engine retained; nine IDs, controls, pause/play and multiple widths exercised | PARTIAL: individual visual/performance certification absent |
| F Commerce handoff | Signature pilot then Black Rose/Love Hurts; lifecycle, keyboard/hash, reduced motion/Save-Data/no-JS tests | PASS bounded handoff implementation |
| G Cards | 33 Shop cards, 25 collection cards, approved imagery and product links; purchase enhancement | PARTIAL: full stock/error/touch state matrix not separately certified |
| H PDP | Native gallery, M variation, price, quantity, Bag/Cart/Checkout journey | PARTIAL: BR-003 rejected imagery and comprehensive detail/story system incomplete |
| I Global chrome | Navigation and Bag checks at 390/768/1440; live search plus native GET | PARTIAL: complete editorial navigation-preview and transition language not certified |
| J Ask Skyy | Existing real 3D loads at 390; identity/model/software retained | NOT CERTIFIED: pose continuity, all behavior states and startup profiling incomplete |
| K Mobile | 204 responsive cases; current QV at 320/390/768/1440 | PARTIAL: device emulation, not physical-device art-direction acceptance |
| L Accessibility/failures | Axe at 390/1440 in three engines, QV/search scoped axe and failure regressions | PARTIAL: manual screen-reader and every exceptional state not certified |
| M Performance | Serial final Lighthouse and six-route desktop/mobile diagnostic table | FAIL: mobile LCP; field INP and 3D startup cost unavailable |
| N E2E | Synthetic native cart journey passed; no payment/order/account creation | PARTIAL: comprehensive commerce state matrix not certified |
| O Visual certification | Screenshots and six normal-motion recordings saved; independent source review passed | NOT CERTIFIED: no whole-system founder visual acceptance |

## Global and commerce checklist

Navigation, search, Bag, responsive and reduced-motion **PASS within recorded tests**. Save-Data **PASS for the handoff fallback**, not a blanket whole-site claim. Accessibility **PASS for automated sampled states**, manual acceptance outstanding. Product rendering, native Quick View selection, representative native purchasing, PDP gallery, representative variation/price, Cart and populated Checkout navigation **PASS**. Card hover/focus/touch/availability across every product state and full PDP detail system **NOT CERTIFIED**. Account login page renders and passes sampled reflow/axe; authenticated account flows were not tested.

## Nine scenes

| Scene | Present / approval provenance | Mobile / reduced-motion controls | CTA/product destinations | New visual approval | Performance acceptable |
|---|---|---|---|---|---|
| SIG-COMMERCE-1 | PASS, retained manifest | PASS tested | PASS route checks | NOT CERTIFIED | NOT CERTIFIED |
| SIG-COMMERCE-2 | PASS, retained manifest | PASS tested | PASS route checks | NOT CERTIFIED | NOT CERTIFIED |
| SIG-COMMERCE-3 | PASS, retained manifest | PASS tested | PASS route checks | NOT CERTIFIED | NOT CERTIFIED |
| BR-COMMERCE-1 | PASS, retained manifest | PASS tested | PASS route checks | NOT CERTIFIED | NOT CERTIFIED |
| BR-COMMERCE-2 | PASS, retained manifest | PASS tested | PASS route checks | NOT CERTIFIED | NOT CERTIFIED |
| BR-COMMERCE-3 | PASS, retained manifest | PASS tested | PASS route checks | NOT CERTIFIED | NOT CERTIFIED |
| LH-COMMERCE-1 | PASS, retained manifest | PASS tested | PASS route checks | NOT CERTIFIED | NOT CERTIFIED |
| LH-COMMERCE-2 | PASS, retained manifest | PASS tested | PASS route checks | NOT CERTIFIED | NOT CERTIFIED |
| LH-COMMERCE-3 | PASS, retained manifest | PASS tested | PASS route checks | NOT CERTIFIED | NOT CERTIFIED |

Manifest authority is unchanged; the table does not manufacture new founder approval. Exact IDs and assertions are in cinematic-integration.json.

## Ask Skyy checklist

Model **PRESENT** and real 3D activation at 390 **PASS**. Walk-in, idle, chat, gestures, pause, dismiss, focus restoration, offscreen suspension, reduced-motion presentation and complete mobile continuity are **NOT FULLY CERTIFIED in this pass**. Existing source remains intact. Deeper Blender work is permitted deferral; software continuity is still an open acceptance item.
