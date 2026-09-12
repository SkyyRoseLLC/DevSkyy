# Skyy Home walk-on — source finding and runtime upgrade

The canonical GLB contains one 18-joint skin and six named action clips, each with 54 held channels and **zero varying channels**. The names and durations did not prove physical animation. Earlier dialog/action-label evidence establishes renderer and control behavior only. It is superseded for locomotion by the fresh hardware-rendered evidence below.

The founder's moving-character request and root's bounded implementation authorization govern this **NEW UPGRADE**. The original GLB, geometry, materials, embedded textures, skeleton and six source clip objects remain unchanged. `deriveRigMotion()` in `assets/js/skyy-3d.js` creates runtime clips only for held source actions; a future varying approved source clip keeps precedence.

The recipe uses a 1.6-second entrance containing two 0.8-second gait cycles. Thighs alternate by up to 0.38 radians, knees flex up to 0.58, ankles compensate by 0.18, and arms oppose the stride by 0.25. A small pelvis bob accompanies stepping. Shoulders and forearms use the canonical bind orientation before a 1.15-radian shoulder drop, avoiding the source's forward-held arms. The character faces its direction of travel and turns toward the visitor in the final quarter. CSS translation uses the same renderer clock as the skeleton; it cannot continue while character motion is paused or hidden. The prior dialog entrance animation is now scoped to the dialog.

Idle is a 3.2-second relaxed breathing cycle. Wave uses a restrained left-arm gesture; Talk adds a small head/hand gesture; Joy raises the shoulders briefly; Exit uses the gait. These are new derived motions, not claims about paid baked clips. There is no inverse kinematics or physical foot-contact simulation. Artistic gait acceptance remains an independent/founder review.

The existing single canvas and model move between Home's reserved stage and the native Ask Skyy dialog. The Home entrance does not open the dialog or move focus. Reduced motion and Save-Data retain the approved static character without requesting automatic model/dependencies. Pause, hidden/offscreen suspension, dismissal, late-load failure and native focus return remain supported. The camera is 4.1 units away; canonical identity, warm key and cool fill are preserved.

Fresh evidence in `.artifacts/v2-visual-recovery-20260905/`:

- `cinematic-integration-skyy-motion-recipe.json`: exact GLB SHA-256, held-channel audit, joint limits and limitations.
- `cinematic-integration-skyy-gait.json`: PASS actual served compiled-byte and GLB hash checks, real joint poses/times, Escape focus return, pause and feature-cost evidence at 390, 768 and 1440.
- `D-skyy-home-walkon-{390,768,1440}.webm`: unedited native Playwright recordings of no-click arrival, real gait, idle and conversation.
- `cinematic-integration-skyy-gait-{390,768,1440}-{walk,idle}.png`: fresh full-viewport captures.

All three widths place the full character within the opening viewport. On the measured Apple M5 / ANGLE Metal renderer, one-second active samples contained 23–24 rendered frames; paused samples contained zero. Walk samples show maximum thigh-quaternion component changes of 0.272–0.369. Observed model initialization was 1.585–2.080 seconds from the 3D controller request to readiness. These are local hardware observations, not mobile-network or low-end-device certification.

The character feature requested 7,270,194 decoded bytes, including the unchanged 6,058,568-byte GLB and local Three/Draco dependencies. Whole-page JS heap rose from about 2.83 MB before the 3D controller to 6.84–7.63 MB after interactions; this includes unrelated page allocations and is not GPU memory. The earlier SwiftShader recording was choppy and is superseded by the final Metal receipt; no smoothness claim is inferred from a software fallback benchmark.

Focused tests: 24 passed across `test-skyy-concierge.cjs` and `test-guide-focus.cjs`. The actual Three AnimationMixer test reconstructs the canonical bone hierarchy, samples real derived quaternion changes, checks relaxed wrist height, and verifies source-track serialization is unchanged. Additional regressions cover repeated transient timing and safe diagnostics after teardown. Independent source and pixel review remain separate from this implementation record. No deployment, paid generation, external chat request, order or payment was performed.

Additional Home controls evidence was executed by root using `.artifacts/v2-visual-recovery-20260905/verify-skyy-home-controls.cjs`. It separately checks same-canvas reparenting, Escape and Close focus, offscreen suspension/resume, dismissal and header recovery, one GLB fetch, fresh Home reduced-motion/Save-Data with zero heavy requests, and visible native Contact without JavaScript. These assertions **PASS** in the separate `cinematic-integration-skyy-home-controls.json` receipt (run `4ebb03ca-bbd6-4225-9d8d-ea83ab29afae`). The helper waits for the native queued dialog-close focus restoration before assessing its final state. The gait receipt alone does not certify them.
