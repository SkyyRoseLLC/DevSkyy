# Independent final visual review — cinematic integration candidate D

Reviewer: `/root/final_visual_review`, independent of builder and source-selection author. Review date: 2026-09-05 America/Los_Angeles. Scope: local integrated candidate in worktree `19db`, base HEAD `fec4e9339cad7077bcc812876668c128eb93b6e4`, plus uncommitted integration work. This report alone is the reviewer's owned output. No runtime fixes, browser interaction, catalog changes, deployment, or founder approval were performed by this reviewer.

**Visual release verdict: REJECT — mobile performance fails the release gate; remaining full-state evidence is UNVERIFIED.**

**Observed outcome: partially-improved. `recommend_ship: false`. Founder creative review is required; this is not approval to deploy.** The recovered collection worlds and paid frames are visibly present. Source selection, integrated appearance, motion quality, commerce behavior, and performance are separate findings. The nine-source authority and a loaded video do not establish complete visual acceptance.

**Later scope steering and re-review:** the latest founder direction requires a visible on-page 3D walk-on, beyond the original dialog-only state. The new Home entrance has now been directly inspected at 390/768/1440 in six fresh walk/idle captures and 36 temporal samples extracted in memory from the three native recordings. Actual side-facing gait, movement across the page, turn and front-facing idle are visible. This closes the missing on-page movement finding for the sampled implementation, with the onset-polish caveat below. Existing opened-dialog screenshots were not reused as entrance proof. The earlier incomplete cinematic run is historical and is not counted as PASS. Fresh completed receipt `662b8c3f-6842-4ae0-b283-136cbb2e61a2` now provides bounded final integration evidence; its scope and limits appear below.

## Authority and method

Read the latest supplied cinematic shopping specification, `docs/design/fashion-design-system-team.md`, `scene-final-authority.md`, `card-frame-provenance.json`, the exact nine-scene canonical manifest, and the existing source/commerce comparison material. `.wolf/memory.md` is absent in this worktree, so its current session guidance is UNVERIFIED. Applied the Fashion Visual and Commerce QA and adversarial-verification skills. `design-qc` could not be located in available tool metadata or searched local skill/script locations; that named tool check is UNVERIFIED, not a successful execution.

Used `view_image` for direct eyes-on inspection of all nine exact approved poster references and all 27 final `D-scene-image-<id>-{390,768,1440}.png` captures. Also inspected Home, all four collection entries, Shop, and the sg-005 PDP at 390/768/1440; the opened Skyy 3D panel at 390/1440; all nine final adult-collection card captures (three collections × three widths); and A/B Home and the A mobile card reference. The root owns browser sessions and measurements; this reviewer did not interrupt those measurements. Raw JSON was inspected as recorded evidence, not misrepresented as independently executed interaction testing.

The A captures are a recovered theme reconstruction using a synthetic database, not original historical staging photography. Comparisons of A/B/D are qualitative rendered comparisons. Deterministic same-time animation screenshot diffs were not produced and remain UNVERIFIED.

## Actionable findings and first-seen record

1. **P1 performance blocker — final mobile LCP remains slow.** Directly read all five fresh `lighthouse-walkon-final-*.report.json` reports: Home mobile 68 / LCP 6541ms / CLS 0.0552 / TBT 42ms; PDP mobile 71 / LCP 5194ms / CLS 0.00482 / TBT 4ms; Shop mobile 70 / LCP 5713ms / CLS 0.000544 / TBT 0.5ms. Desktop Home is 97 / LCP 1124ms; desktop PDP is 98 / LCP 1043ms. All five accessibility scores are 100. These are final local lab measurements, not field CWV or production evidence. Mobile performance remains a release blocker; low TBT and good desktop scores do not erase it. Feature-level causal attribution and field INP remain UNVERIFIED.

2. **P2 visible motion caveat — LH-COMMERCE-1 loses the garment-led composition in the sampled film frame.** All three fresh widths show the cathedral and a clipped sleeve at the far left, while the approved poster contains two readable looks. Root identifies this as the already-approved film's walking exit. This is not evidence of an unapproved replacement or a CSS crop defect and does not authorize modifying that film. Founder review should include the complete loop and poster transition. Do not claim constant garment visibility.

3. **P2 onset polish caveat — the Home static character changes scale, pose and placement when the 3D walk begins.** The independently sampled native recordings show the smaller front-facing poster giving way to the larger side-facing moving model. Actual movement is implemented; seamless entrance continuity and Pixar-grade artistic quality are not established. Details and receipt binding follow below.

4. **Closed — mobile card scale and frame obstruction.** Fresh Shop 390 and all nine adult-collection card captures show larger, readable garment images, with a single column at 390 and two columns at 768. The 390 Shop frame is approximately 358px wide and the visible torso is roughly 100px wide, materially improving the earlier 45–55px torso. Paid frame art no longer covers the garment with a WooCommerce background. Prices, availability and native actions remain separate readable content. Longer scrolling is the tradeoff. Detailed Kids card pixels were not supplied as a separate capture; Kids coverage is the main route and recorded catalog evidence.

5. **Closed — header obstruction on the inspected main routes.** Fresh Home, Signature, Black Rose, Love Hurts, Kids, Shop and PDP captures at 390/768/1440 show Ask Skyy in the header, with collection navigation and product action regions unobscured. All 21 main captures were inspected. Earlier floating-launcher findings are historical, not current defects.

6. **Closed — initial mobile hero recognition weakness.** Fresh 390 collection captures show the shorter media area and recognizable monuments/world elements while exposing the primary Shop action earlier. The final Home hero retains readable foreground copy. No remaining header/hero overlap was visible in these captures. Source-selected scene/hero composition is not fresh product-construction certification.

7. **Evidence boundary — complete release certification is still incomplete.** All 27 scene widths now have eyes-on playing-frame review and recorded pause/play/controls checks. Recorded no-JS, save-data and failed-video fallbacks are present. A complete visual record of every loop boundary, media transition and error/focus state, deterministic animation diffs, calculated contrast across all states, authenticated account, payment recovery and completed-order inheritance is still UNVERIFIED. The local integration receipt explicitly excludes order, payment and cart mutations; it cannot certify those flows.

## Nine-scene pixel comparison

Each row represents direct inspection of its approved poster and all three final integrated playing-frame captures. Capture basenames use the lowercase scene ID.

| Exact scene | Visible comparison at 390/768/1440 | Boundary |
|---|---|---|
| SIG-COMMERCE-1 | Sherpa/beanie subject, Golden Gate horizon, wet terrace and metallic wordmark retained; garment stays readable at desktop | Small mobile subject; one film frame only |
| SIG-COMMERCE-2 | Mint looks and dress silhouette retained within terrace scene | Film frame reaches/cuts the shoes at lower edge versus full poster; no CSS/source-crop cause asserted |
| SIG-COMMERCE-3 | Three looks, bridge and gold monument retained | Full-motion anatomy/construction continuity unverified |
| BR-COMMERCE-1 | Monochrome garment, star/rose and reflected architectural space retained | Desktop portrait now fits an approximately 404×720 image, no extreme page-height enlargement |
| BR-COMMERCE-2 | Two looks, Bay Bridge, black stone and silver type retained | Mobile garment details are small; no invented SKU mismatch asserted |
| BR-COMMERCE-3 | All five lounge garments remain in the visible composition | Previously reported duplicate poster bands are absent; plain letterbox is visible and distinct from repeated imagery |
| LH-COMMERCE-1 | Cathedral identity remains | Sampled frame loses nearly all garment content; see finding 2 |
| LH-COMMERCE-2 | Shorts subject, chapel architecture, red light and floor retained | Source-selected composition is retained; physical SKU truth is not freshly re-certified by this integration review |
| LH-COMMERCE-3 | Bag subject and glass rose retained; portrait proportions are coherent | Desktop portrait approximately 540×720; bag lettering remains source/video content, not a new rendered UI label |

No additional model overlay or repeated scene layer is visible in these 27 captures. This is a pixel observation; it is not proof of the complete request graph. Product source authority is inherited from the exact reviewed manifest and receipts, not invented from appearance or filenames. There are nine scenes, three per adult collection; no Kids scene is assumed. BR-COMMERCE-3's existing Town Line label is separate from the future Pre-Order experience.

## Anti-generic and logo-off assessment

Qualitative logo-off test: discount header logos, large brand words, scene wordmarks and copy. The Bay/Golden Gate architecture, rose/star sculpture, wet black stone, restricted collection colors, cathedral glass rose, and daughter character still provide recognizable collection/world cues. Recognition is strongest at desktop and in portrait scene states; mobile framing reduces it. This is an expert visual judgment, not a blinded participant test or quantified recognition study.

No generic gradient hero, arbitrary glass panel, rounded SaaS card system, fake testimonial/metric, urgency timer or blob decoration was visible in inspected captures. Paid portal framing is treated as authorized brand material. Repeated equal-size Shop cards limit editorial rhythm; approved frames are not themselves rejected as generic. Serious accessibility and primary-journey hard fails cannot be declared absent until the remaining states are verified. No global zero-hard-fails assertion is made.

Provisional evidence-adjusted rubric uses the project's 100-point visual categories. Incomplete categories are intentionally not rounded up. The prior 79/100 is superseded by this 84/100 bounded review; this is not a completed release score.

| Category | Score | Basis |
|---|---:|---|
| Logo-independent brand recognition | 18/20 | Distinct worlds, material language and character; mobile crop weakness |
| Composition | 17/20 | Strong arrival; recovered worlds; larger mobile cards and clear header; dense repeated frame rhythm |
| Typography | 12/15 | Clear hierarchy and native purchasing text; dense mobile product names and repeated inscriptions |
| Garment protagonism | 13/15 | Unobscured cards/PDP and improved mobile scale; LH1 sampled playing-frame loss |
| Token/material discipline | 9/10 | Consistent black base, collection accents and approved materials |
| State coherence | 7/10 | Final bounded fallback/shell receipts; full critical-state visual and funnel coverage incomplete |
| Motion/responsive translation | 8/10 | All three widths and actual on-page gait; onset polish, full-loop review and mobile LCP remain open |
| **Final bounded total** | **84/100** | Below 85; all category ratios reach 70%, but performance blocker and missing release evidence prevent PASS |

## Verification coverage and limits

| Required check | Evidence actually reviewed | Status |
|---|---|---|
| Main-page 390/768/1440 | All 21 fresh Home, four collections, Shop, sg-005 PDP captures; nine adult card captures | Eyes-on final closure; no visible header/frame obstruction |
| Nine-scene 390/768/1440 | All 27 final scene-image captures, all nine poster references | Eyes-on sampled frames |
| Scene controls and motion | Final integration receipt: 27 pause/play cases, controls outside art; scene profiles: all 27 readyState 4, playing, zero page errors | Recorded bounded PASS; complete loop/error visual matrix UNVERIFIED |
| Overflow | Recorded D rows show 0 overflow for seven routes × two widths; accessibility reflow has 36 rows, 320–1920, no reported overflow | Recorded evidence; does not detect every visual occlusion |
| Keyboard/focus | Focus ring visible on Skyy question input; recorded PDP test says fit focus restored | Complete keyboard journey UNVERIFIED |
| Contrast | Recorded Lighthouse accessibility score 100; reflow JSON reports no axe violations | Complete states and calculated contrast matrix UNVERIFIED; score alone is insufficient |
| Reduced motion, touch, no-WebGL/model failure | Raw Skyy fallback JSON records static guide on save-data, no-WebGL, missing-model and missing-idle; 320 pointer fixture records a real product link | Recorded behavior; final visual matrix UNVERIFIED |
| Shop and no-JS | Recorded Shop JSON tests filtering/sort/history/empty/pagination/malformed/taxonomy with JS on and off | Recorded bounded behavior, not independent rerun |
| PDP→Bag | Recorded sg-005 size M, variation 182, quantity 1, unit/subtotal $25; no payment submitted | Fixture evidence, not inventory/payment certification |
| Full funnel | Authenticated account, payment recovery, completed order and all critical visual states not independently exercised | UNVERIFIED |
| Lighthouse/CWV | All five final walkon-final Home/PDP/Shop reports read directly | Final mobile lab gate fails; field CWV/INP UNVERIFIED |
| Screenshot diffs | A/B/D rendered comparisons, no deterministic final diff set | Qualitative comparison only; numerical diff UNVERIFIED |
| Ask Skyy | Original 3D panel at 390/1440; new Home walk/idle at 390/768/1440 plus 36 native recording samples | Actual on-page movement inspected; new motion artistic approval and complete final fallback matrix remain separate |

## Recommendation

Keep the integrated candidate available for founder comparison with the explicit unresolved findings above. Preserve the nine approved scenes, their source assets, actual WooCommerce behavior, and paid frames. Source-hash success must not be relabeled visual approval. Responsive header occlusion and the original mobile garment scale findings are closed in the final captures. Resolve measured mobile performance, complete required state evidence, and present the LH1 loop and Skyy onset to founder review before seeking a release PASS. No deployment is authorized by this report.

## Final receipt reconciliation

Fresh `cinematic-integration.json`, run `662b8c3f-6842-4ae0-b283-136cbb2e61a2`, is PASS within local navigation/media scope. It records 33 Shop products with accepted media, containment, transparent frames and keyboard Quick View; 25 core collection cards; eight preserved Town Line product links; 27 scene pause/play cases; loaded posters on save-data and failed video; a native no-JS product destination; and shell checks. These are runner results reviewed as evidence, not manual interactions by this reviewer. New shell navigation/search/bag screenshots were not independently inspected in this bounded closure, so their complete visual states remain UNVERIFIED.

Final `accessibility-reflow.json` contains 36 route/width observations from 320–1920, all HTTP 200, no reported page/HTTP errors and zero overflow. Empty violations arrays are not treated as proof that axe ran on every row; root reports 14 actual axe runs with zero violations. Header occlusion is separately closed by the 21 directly inspected 390/768/1440 captures. Named `design-qc` execution and a complete calculated contrast/state matrix remain UNVERIFIED.

Final `cinematic-integration-skyy-home-controls.json` is recorded PASS for actual Metal rendering, same-canvas dialog movement, Close/Escape focus return, offscreen suspension/resume, dismissal/reopening and one GLB fetch. Reduced motion and save-data rows record static presentation with no heavy requests; no-JS retains native Contact. These behavioral results support the observed new character but do not assert artistic approval or field performance. The nine-scene `scene-profiles-final.json` contains 27 profiles with all videos readyState 4, unpaused and zero page errors. Hashes below bind this final revision independently of the earlier overwritten captures.

## Home walk-on revision — 2026-09-05, 18:29 capture batch

Fresh files: `cinematic-integration-skyy-gait-{390,768,1440}-{walk,idle}.png`, `D-skyy-home-walkon-{390,768,1440}.webm`, and `cinematic-integration-skyy-gait.json`, all beneath `.artifacts/v2-visual-recovery-20260905/`. Six original captures were inspected at their full available sizes. Native videos are 9.32–9.8 seconds, encoded at 25 fps. For each video the reviewer independently extracted 12 time-ordered cropped samples at half-second intervals from 2 seconds, without writing or replacing an artifact. These 36 temporal samples establish visible pose/position changes and the interaction sequence; they are not a claim of continuous real-time playback review or frame-perfect smoothness.

**Bounded finding: on-page moving 3D character IMPLEMENTED AND VISIBLY OBSERVED.** At all three widths Skyy's side-facing legs change stride, her body advances and turns to front-facing idle, with the complete body in the opening viewport. Character, Pause character, Ask Skyy and Dismiss Skyy controls do not obscure the visible primary Choose Your World / Shop the House actions. The Home header places Ask Skyy in normal header space, clearing the earlier floating-launcher navigation obstruction. The paused-state recording samples show the Resume character label, followed by the chat panel and return to the Home character. Click/keyboard causality remains the runner's evidence, not a manual reviewer browser test.

**New P2 polish caveat: poster-to-3D onset discontinuity.** Before the walk, the recordings show a smaller front-facing static character. The transition introduces a larger side-facing 3D character at another horizontal position. The change of scale, pose and placement is visible in the sampled sequence. This does not negate actual gait implementation, but it weakens entrance continuity and should remain visible to the founder as candidate polish debt. No runtime repair was performed by this reviewer.

Receipt `058bba2d-10b7-451c-869a-f664ee3e9c5d` identifies `runtime-rig-upgrade-v1`, actual Apple M5 Metal rendering, maximum thigh-quaternion component changes of approximately 0.272 / 0.369 / 0.339 at 390/768/1440, 23/24/23 active frames in the sampled one-second windows and zero paused frames. The source clip entries explicitly say `varying:false`; the gait is therefore documented as **NEW derived runtime motion**, not recovered authored animation. The receipt binds the unchanged canonical GLB by SHA-256 `c571e26fb126f992f735062d628c499a7690e926ca2bbefa70c9252fc8a780fc`. The reviewer inspected the resulting pixels and recorded evidence, not the original mesh/rig internals.

This scoped finding is not Pixar-grade artistic acceptance, founder approval, a full performance PASS or approval for all commerce/critical states. The report's overall REJECT remains while the wider evidence and performance gates are open. Earlier dialog-only evidence limitations remain historical; they no longer mean the new on-page moving character is absent.

## Reviewed authority and scene snapshot hashes

Recorded 2026-09-06T01:02:30.870Z. These hashes bind the referenced source/poster and sampled scene files at review time. Any later replacement is a new evidence revision. They do not certify unobserved states.

| File | SHA-256 |
|---|---|
| `wordpress-theme/skyyrose-flagship-2/data/approved-scroll-world-scenes.json` | `20b1532710f93c98b8c8161853d44caadd3c6c4eecbcdc81ce3bc5c6d5c4fd42` |
| `tasks/v2-visual-recovery-20260905/scene-final-authority.md` | `8257934a0f11f07d3a0adac749d5f9cb066ea934598f85baa37d5783afb4dc39` |
| `tasks/v2-visual-recovery-20260905/card-frame-provenance.json` | `134d6b0ed960f04499a764b70a185cba33cec7704cf2135a2dd41c4abf09e5d3` |
| `wordpress-theme/skyyrose-flagship-2/assets/scroll-world/generated-candidates/hero-commerce-c1/sig-commerce-1-natural-font-h1-1672w.webp` | `a42deb80c50e5caecc989b2ac43982e4c2aa8e6abb2b91c7c10be1d34bfc9026` |
| `.artifacts/v2-visual-recovery-20260905/D-scene-image-sig-commerce-1-390.png` | `b5c8dcf82bd9431a9d1a77e4bdcbb408362a044e64df1414d2abdfa2d8b0848a` |
| `.artifacts/v2-visual-recovery-20260905/D-scene-image-sig-commerce-1-1440.png` | `9d2f147bafafc2176c36da2d8662729548f25e6ce0e1a3a260330d0adedfe161` |
| `wordpress-theme/skyyrose-flagship-2/assets/scroll-world/generated-candidates/hero-commerce-c1/sig-commerce-2-natural-dress-h1-1672w.webp` | `df2f79e55c5cff38dbac7efae3cd7258685cd546c4713fa413ec9b6f0640111e` |
| `.artifacts/v2-visual-recovery-20260905/D-scene-image-sig-commerce-2-390.png` | `feb72d99b15a8db53cc7d0e46da0e3da77585705099f4e550746caefcdfd4352` |
| `.artifacts/v2-visual-recovery-20260905/D-scene-image-sig-commerce-2-1440.png` | `f329c7fde6e4098188487f6c270af6bc884c98574e127d835fe1938ad29d2904` |
| `wordpress-theme/skyyrose-flagship-2/assets/scroll-world/motion/collection-scenes-k1/sig-commerce-3-motion-k1-poster.webp` | `eef20813c482773ff3217cd1f367c61bbd9c31734e45ff252ea1f510783bc452` |
| `.artifacts/v2-visual-recovery-20260905/D-scene-image-sig-commerce-3-390.png` | `286671718f48e09543d40ee2c46e2fb030b7b3ee523779ba76f69b1888b8e79b` |
| `.artifacts/v2-visual-recovery-20260905/D-scene-image-sig-commerce-3-1440.png` | `3a1589e52cc1726902904adeb5bc4abcfaf54d5e434eaced5784cf28a7ae4bce` |
| `wordpress-theme/skyyrose-flagship-2/assets/scroll-world/motion/collection-scenes-k1/br-commerce-1-archive-4-poster.webp` | `b29c1c71c8754a2c716771dcf701ec880438f423e305f6fdb097d0765a8cdc51` |
| `.artifacts/v2-visual-recovery-20260905/D-scene-image-br-commerce-1-390.png` | `2e9bbd76d8a383d66408309aa6b2852cbdfe6be88234e74b00ff792432853a26` |
| `.artifacts/v2-visual-recovery-20260905/D-scene-image-br-commerce-1-1440.png` | `08b33ed9554300d32082ec099a7e739993ef5116295ee3a6932e60aadceb42d4` |
| `wordpress-theme/skyyrose-flagship-2/assets/scroll-world/generated-candidates/hero-commerce-c1/br-commerce-2-hero-wall-statue-i3-1672w.webp` | `33d013d846261c82b14a3e015f630f3e8ef9e54a9f5e0592ce50e289fa1da338` |
| `.artifacts/v2-visual-recovery-20260905/D-scene-image-br-commerce-2-390.png` | `176fda6bb7fa869792b0875e4e971121383f0036c82b2ad5ced415441b4a1cbf` |
| `.artifacts/v2-visual-recovery-20260905/D-scene-image-br-commerce-2-1440.png` | `d050b1d6f895f07019b711457c65b464dc5d9b0d92775b7bf5763fe5a4fcd3d7` |
| `wordpress-theme/skyyrose-flagship-2/assets/scroll-world/generated-candidates/founder-commerce-scenes-v1/black-rose/br-commerce-3-five-jersey-lounge-a-black-founder-approved-v1.png` | `90aa66aa432bc9176b393dfbb7ce15d507860e47c970650337f78c4b1a480683` |
| `.artifacts/v2-visual-recovery-20260905/D-scene-image-br-commerce-3-390.png` | `a0891d27dc856020a8a19d0a88b8198165ed55471d9d51aa38f27613faaf0b77` |
| `.artifacts/v2-visual-recovery-20260905/D-scene-image-br-commerce-3-1440.png` | `168ceb024608708ef33e77b94a7d7cad90c2362273c72a9e7aa456b10931d261` |
| `wordpress-theme/skyyrose-flagship-2/assets/scroll-world/motion/collection-scenes-k1/lh-commerce-1-archive-4-poster.webp` | `6a5fbe9d202f0e47492ded17b4ea381311b41412fc3625c949874da1392d3446` |
| `.artifacts/v2-visual-recovery-20260905/D-scene-image-lh-commerce-1-390.png` | `7e86681ad01ce9476a2f7fc96141b07648a672114a37d49bdebb7bb8ad84542b` |
| `.artifacts/v2-visual-recovery-20260905/D-scene-image-lh-commerce-1-1440.png` | `ec5eca89502456fbd1a8b7dca1e491007c04e202f8aff599add2b13b4dd5f564` |
| `wordpress-theme/skyyrose-flagship-2/assets/scroll-world/generated-candidates/hero-commerce-c1/lh-commerce-2-hero-composed-c1-1672w.webp` | `bf251eba7a2366a8d133e10a7bd548f3c6d89d2137da8263a418e761bfdefa54` |
| `.artifacts/v2-visual-recovery-20260905/D-scene-image-lh-commerce-2-390.png` | `1122a3a7496fb8542ccffb6f3dacb70a96808c5d4e4d5094dcc1f89ac6c09a92` |
| `.artifacts/v2-visual-recovery-20260905/D-scene-image-lh-commerce-2-1440.png` | `e81820e26c492e1bf1e6ea16173acc16a9b396253fa97831cda0f2c667435a2e` |
| `wordpress-theme/skyyrose-flagship-2/assets/scroll-world/motion/collection-scenes-k1/lh-commerce-3-archive-4-poster.webp` | `ddff5431de7614583429a92d4c51ebab87dc0b174ff1292b997004aa7b8dfc5c` |
| `.artifacts/v2-visual-recovery-20260905/D-scene-image-lh-commerce-3-390.png` | `f2d05d1d5f40de0d5bf71bce6e1ca5a88b5dea3ecbd56039daf050d5da828b1c` |
| `.artifacts/v2-visual-recovery-20260905/D-scene-image-lh-commerce-3-1440.png` | `179a8f936faf80e565e476c3f4e272f38c251b07175584884929e9c255bf0e82` |
| `.artifacts/v2-visual-recovery-20260905/D-skyy-3d-390.png` | `e342f10e3f4be93977e54f83ca67cdbd7e7f5c6608e351e4f806f8b0ecbab1d3` |
| `.artifacts/v2-visual-recovery-20260905/D-skyy-3d-1440.png` | `62e0704f8ea70e0621817cc2bb83c75a3a8374ff8c9bb11c6c7824a58cedd91e` |

### New Home gait revision hashes

Recorded 2026-09-06T01:34:27.023Z.

| File | SHA-256 |
|---|---|
| `.artifacts/v2-visual-recovery-20260905/cinematic-integration-skyy-gait.json` | `edb3e168cb0a8c27f467f4f481711249a65e030e01650fb5d57d72dcfaadfa54` |
| `.artifacts/v2-visual-recovery-20260905/cinematic-integration-skyy-gait-390-walk.png` | `788730af2b7bed89770f04de22859f00fe4da1d0923f2a7b1350c0251be8a139` |
| `.artifacts/v2-visual-recovery-20260905/cinematic-integration-skyy-gait-390-idle.png` | `c6f926a45bcf856eac0a31131da6260d205227bb862435ae5b1c1e8d098471e6` |
| `.artifacts/v2-visual-recovery-20260905/D-skyy-home-walkon-390.webm` | `b3043f8b26fa6004d56bc3f5bf5b161f221295094826311553d755570892c05c` |
| `.artifacts/v2-visual-recovery-20260905/cinematic-integration-skyy-gait-768-walk.png` | `55215faec60dc60723ff8fb808b35bb61e7d4067ce94a0bc308cccb4fa79a49e` |
| `.artifacts/v2-visual-recovery-20260905/cinematic-integration-skyy-gait-768-idle.png` | `9d62157f9ff2371b114e83b9c37f6c745e7ad82925d42c5af0084a708e82897c` |
| `.artifacts/v2-visual-recovery-20260905/D-skyy-home-walkon-768.webm` | `391b878942937040133e8a2cdca1b8d0133a918addc3e34669baabf39907788a` |
| `.artifacts/v2-visual-recovery-20260905/cinematic-integration-skyy-gait-1440-walk.png` | `d35a3a49eaa728f1cacccee1243afffd6ba01205fb179c0bb9f52797a18810e0` |
| `.artifacts/v2-visual-recovery-20260905/cinematic-integration-skyy-gait-1440-idle.png` | `1e4b6c15bff3defebc67ee416ca91288fceba83bb6b39b6cf54e398626361ca6` |
| `.artifacts/v2-visual-recovery-20260905/D-skyy-home-walkon-1440.webm` | `3948ea665c2acb748b4b5e05566e76c82e30b2e0e8a8aa3471c73704992b8f05` |

## Final frozen revision hashes

Recorded 2026-09-06T02:34:09.850Z. Earlier hash tables are historical revisions; this table binds the fresh 21 main, nine card and 27 scene images reviewed in this closure, plus the recorded receipts and five final lab reports. Hashes alone confer no visual approval.

| Artifact relative to evidence directory | SHA-256 |
|---|---|
| `D-home-390.png` | `c9422bacdd9f308fc755079e96114333f9fb25ba645c13b1cc49f87fb02f7592` |
| `D-signature-390.png` | `a2ef51b4f0ea69c15f7f5d1a92a8b1e4b46736b39c34bcd2409187ad5f9dfd15` |
| `D-black-rose-390.png` | `c40652f2436fdc24b4ee750c87f768d9dfeef57942c2d97c4b369d0834e867f8` |
| `D-love-hurts-390.png` | `2e636eee4d681878ebbc36c7eea0e6e6180fadc587ed43357364942a562bd5bf` |
| `D-kids-390.png` | `d07e6821dc7f406890d5348236da5320e9e6f9a5259fbd42cb3a21b42f75e3da` |
| `D-pdp-390.png` | `7ddc68196bdcc7d82feda0c9e39fa7645b2dddb63bd4315fb95e161d207b9e2e` |
| `D-shop-390.png` | `dfdc9a7939163bfe0b9ff956583a8a44edff1817cd150ea145cf68ed6a591865` |
| `D-signature-cards-390.png` | `7267fdaa1d30a37ccfae2bc39fb26f3ae767681cdfbf7c7c67029b3585953331` |
| `D-black-rose-cards-390.png` | `5cd4b328f0f207631be94fb654b4fdc3b8dbc5d34fb2146623f5e41477c3fa8d` |
| `D-love-hurts-cards-390.png` | `caedee6e2e9bc384d82ebcf98227e094ccf084feb19b5b367a56c257cb68b7b8` |
| `D-scene-image-sig-commerce-1-390.png` | `9db891d9971af0078aed99815cca52cd6322fa2d964d562946c7b00fd423b5f1` |
| `D-scene-image-sig-commerce-2-390.png` | `7fb61339ba20d066349772183e4b7680ab315c494f77e1b0f5a94c5a79adc327` |
| `D-scene-image-sig-commerce-3-390.png` | `992acfcc826869924db4fd18d6ad12389ecd435c7b5013849611d8390eccf936` |
| `D-scene-image-br-commerce-1-390.png` | `1227baeb27fbe018461042acf8dbe1d1976b880bb4dab0011cbda5ca0d6db199` |
| `D-scene-image-br-commerce-2-390.png` | `db222f086a41704a77301c9a4e042a98653155edd6f7f248034e2d6eae171de8` |
| `D-scene-image-br-commerce-3-390.png` | `e052b43d5c2b643508bc34ad78ce37231b8a3166c1a2088d0e446de96d41f282` |
| `D-scene-image-lh-commerce-1-390.png` | `fbbcc0890f19b1021a19e79879d91875a91e98d3502cfbf98ec9036b88562c6f` |
| `D-scene-image-lh-commerce-2-390.png` | `c89792dedbbb2d7867f3c932769ae948eae61861800f8262d256e3aaed4cfd7e` |
| `D-scene-image-lh-commerce-3-390.png` | `bf240ae99a115fcc32ced7473130460302081add4127f79c71dec40e1370da2c` |
| `D-home-768.png` | `2ce5aea035cb64af312e0967c3c5d2711136d45340f210e58d5a57080140f600` |
| `D-signature-768.png` | `53ed0064399415fb933d820f0545bb5d6f9314c6deb4d9744c58a1420c61c9b5` |
| `D-black-rose-768.png` | `0bdf343d6cb9a72688cf43c181d6710cf9c1f67c444a72a16a73b15e19e01771` |
| `D-love-hurts-768.png` | `3a4776ad3af9eddc3b1f519b0a2382e4b14eea42d9eec2982589511fd1946e58` |
| `D-kids-768.png` | `5c6f6b157cc95e75cf6063af658be0c98e1288f6de9c0aab27433b45138e91bd` |
| `D-pdp-768.png` | `25bc78596e25b1764e07759ef745971f2478e7039251dd21b984939fd248af54` |
| `D-shop-768.png` | `263baf24e646fe82646f25bbc6d17611077333f21bef215ce176b68d01e11e49` |
| `D-signature-cards-768.png` | `24c427be6b907ac7bf8eccbd3da4ccbff11429f57cb7e743c2cf7417aa106863` |
| `D-black-rose-cards-768.png` | `d61fe68177f8885c697fdfc7953aa41b3c6da8a2e1460372e09e3446e1b73bed` |
| `D-love-hurts-cards-768.png` | `0275b0aba088de47ba75afd05957d2c6e996d5e68647fecd51baf8c098c4b333` |
| `D-scene-image-sig-commerce-1-768.png` | `5b20e4f1f690c29434fbfa29d8306e625ce05b63d0d8ccd175f2c3b107c15300` |
| `D-scene-image-sig-commerce-2-768.png` | `4cb1f850f541cab2bfba0ad90411288987287a95e205ef611496a796b8fd2a68` |
| `D-scene-image-sig-commerce-3-768.png` | `34f63434b0086d52c4f439867817fa8e965864e0222ebe81d0710db7c8194e7f` |
| `D-scene-image-br-commerce-1-768.png` | `ca1765353db8e5030bc6af38aa39125241797bf41d7d965f3ddb6d47943f1387` |
| `D-scene-image-br-commerce-2-768.png` | `419d6cf2f5e4235134e646c7e7674aefe6752c5c64791256bcdfbc8a6643fbdc` |
| `D-scene-image-br-commerce-3-768.png` | `9c5306382f480adc8a6d41dffd4387c30f42862f1dbe2be9ba8ba6de6b8c66a5` |
| `D-scene-image-lh-commerce-1-768.png` | `37b085776137a726d13adbfdd7a6479bd434afec231d961e2f7607194b493d46` |
| `D-scene-image-lh-commerce-2-768.png` | `609466bd222687916ff964ecc774908618d6bd4cec25e1e9a58e1ac071e149b6` |
| `D-scene-image-lh-commerce-3-768.png` | `9dc8a96d7ea5795deba8eb395c7cfb22cd53a98d86bb585ff98df5a4866d3dfb` |
| `D-home-1440.png` | `848b5d013f376c6f60088d5f1d9fbe20f51ab182b4b0bb3804da6919ba03ba01` |
| `D-signature-1440.png` | `0784d4f773f6ab3bae690775b3edb2ea895456c6a37a2280700c255f02e3d852` |
| `D-black-rose-1440.png` | `8ecf2feca45af0b36cad1c98af84cdb33f58af7456ec7e69abaf977fde397fa5` |
| `D-love-hurts-1440.png` | `7d03e3675b3f2a5a2f91398761d3586ab3627a2a19e8c5e6eb7f90a92590af8f` |
| `D-kids-1440.png` | `e8863cafbcfb8a5e751772d0c51d1aa76a3d9d87ea6901eac2bef1fee6881cdf` |
| `D-pdp-1440.png` | `84062b96e17e4a94fe4d2fd000ea970815d737e0deb17e5a0fbb996e68db8b99` |
| `D-shop-1440.png` | `76ede9b8909682d9ff07f7a661a12d9728f35060958b3a568351a5d89058f26e` |
| `D-signature-cards-1440.png` | `867165a39addef893feb2a7d8a2c2c9d44ab76863f8d7217b2c88b7d0de7fb44` |
| `D-black-rose-cards-1440.png` | `5f253e03c57bd13b00b492a530816ab2801b0ba6ab9fbee458cf9033b53057e8` |
| `D-love-hurts-cards-1440.png` | `60cfd7aac5833f87ce0759e7b027f30afb035b8f46fbe7e232197171e7d7a57c` |
| `D-scene-image-sig-commerce-1-1440.png` | `75019eca3934a90fc349efd309eb13624459fe94e421e44261b71d678924d982` |
| `D-scene-image-sig-commerce-2-1440.png` | `f329c7fde6e4098188487f6c270af6bc884c98574e127d835fe1938ad29d2904` |
| `D-scene-image-sig-commerce-3-1440.png` | `cc0a4a1fbddfde1b33d9befbb882e2155d4b95004b524e40d1833265757c8c67` |
| `D-scene-image-br-commerce-1-1440.png` | `28883f21d2e0f1b5b6698b7f79149461a2545db008ccaab4f49bb17f1a897393` |
| `D-scene-image-br-commerce-2-1440.png` | `036d7e424f1e460f63f85225b58540306442ad5db89a831e84178d90c39f7504` |
| `D-scene-image-br-commerce-3-1440.png` | `168ceb024608708ef33e77b94a7d7cad90c2362273c72a9e7aa456b10931d261` |
| `D-scene-image-lh-commerce-1-1440.png` | `255f14a31485c455e8fe9e44844aeab0b6c708d0484afa6f6a0c479633e07b71` |
| `D-scene-image-lh-commerce-2-1440.png` | `4163ef89af1cf80329429ae1e1358dbed58661c3f422cd00e789375acb8766cb` |
| `D-scene-image-lh-commerce-3-1440.png` | `a9d280f86a8eb1856e7719d4b786e287e2e421d5c018d3823099e46e3c2295b4` |
| `cinematic-integration.json` | `311560ad2d28617e177dc64204bc84692a6621a9b330164280b99d9ff2832ada` |
| `scene-profiles-final.json` | `3b1e10304792bda9f3394bb0f63f6bce35f16d64609f4ab812fa85766df8ecbe` |
| `accessibility-reflow.json` | `b93b03fb360c7b909385d13148affb91dcc98baa40307f6ba844ca29a363a353` |
| `cinematic-integration-skyy-home-controls.json` | `203b28d62cf4cd005238945c6845d2464f70c4b804f769599f07fa7728609390` |
| `lighthouse/lighthouse-walkon-final-home-mobile.report.json` | `6c840008d2418cd71ae6323df36202a980abc27f397877ede8c465ecc7b7914d` |
| `lighthouse/lighthouse-walkon-final-home-desktop.report.json` | `a31de1643803337176f6d4bea02b2f4a78437b2e9a0a5781ee4240780d4b749f` |
| `lighthouse/lighthouse-walkon-final-pdp-mobile.report.json` | `e6a2bcc52a9dc8971e1d0241994850115922d4d7e79db55b3c2264afcf37de63` |
| `lighthouse/lighthouse-walkon-final-pdp-desktop.report.json` | `5ea8bd1010d2b9a95d3a95a61f48345708531190a9b67f78cefb9783d7c93e1f` |
| `lighthouse/lighthouse-walkon-final-shop-mobile.report.json` | `5a254715a33eb03885cb78be36f2914ac845606120b7c64c174a2929dbb3902b` |
