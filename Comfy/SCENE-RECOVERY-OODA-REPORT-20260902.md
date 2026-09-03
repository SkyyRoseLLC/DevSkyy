# Scene recovery OODA report — 2026-09-02

## Executive disposition

No new scene image was generated and no credits were spent. That is the correct
result of this pass: the pipeline caught source-authority defects before they
could become another paid failure.

| Scene | Current result | Closed in this pass | Remaining hard gate |
| --- | --- | --- | --- |
| `LH-COMMERCE-2` | `PARTIALLY_RESOLVED` | Four source-RGB-preserving product-only alpha mattes created, receipt-bound, eyes-on reviewed, and uploaded to local Comfy | Exact approved on-model LH-003 garment source and wearer matte; the rejected casting garment remains prohibited |
| `SIG-COMMERCE-1` | `BLOCKED_NO_SPEND` | Green/black beanie authority conflict corrected; black SG-007 source bound; identity, pose, hierarchy, sculpture, and deny-list guards added; provider/runtime contract now passes | Corrected prompt enhancement endpoint is currently unreachable; independent enhanced-prompt PASS and candidate-bound paid approval remain absent |
| `SIG-COMMERCE-2` | `BLOCKED_SOURCE_GATE` | A distinct real-alpha SG-006 woman was discovered and classified | That exact identity/file has no founder approval and cannot replace the deny-listed woman silently |
| `SIG-COMMERCE-3` | `PARTIALLY_RESOLVED` | Existing SG-001 corrected asset, generation receipt, and zero-outside-mask verification receipt bound into the live scene packet | SG-003 mask-bound correction and independent review are still missing; full-frame regeneration remains forbidden |
| `BR-COMMERCE-2` | `BLOCKED_LOCAL_RECOVERY` | Claimed protected sources were channel-audited; exact sculpture confirmed real alpha | Cast reference and BR-004 source are opaque, while the only dual-cast alpha has a rejected product correction; no safe local final composite exists and paid retry is forbidden |

## Rendering pipeline analysis

1. **Catalog and image SOT.** The root catalog and `data/sot-images.json` are
   hash-bound before any provider or Comfy action. This caught the SG-007 color
   conflict: the intended target is the black beanie variant, not the green
   source that had been routed previously.
2. **Physical source authority.** Product claims must resolve to physical views,
   not a plausible generation. LH-003 is bound to front, back, wearer-left, and
   wearer-right photos with exactly two side and two rear zippered pockets.
3. **Identity authority.** Identity approval is distinct from garment approval.
   `LH-MODEL-01` remains approved identity-only; the garment board remains
   rejected because wearer-relative artwork changes sides.
4. **Alpha/matte truth.** File extension and an RGBA label are insufficient.
   Alpha extrema were measured. Fully opaque RGB/RGBA sources are now called
   references, not cutouts. Real-alpha inputs are still subject to visual QA.
5. **Prompt contract.** Product construction, selected variants, customer
   identity and pose, exact sculpture geometry, focal hierarchy, and prohibited
   inputs must all appear in provider-facing text. Metadata that the provider
   never receives is not protection.
6. **Provider capability.** The Higgsfield Product Photoshoot route is bound to
   the registered `gpt_image_2` candidate-only capability and its current
   runtime receipt. A reachable CLI alone is not execution approval.
7. **Prompt enhancement.** Enhancement is a no-generation preflight. The first
   SIG1 enhancement correctly failed because it described the conflicting green
   beanie. After correction, the second request reached the CLI but the
   Higgsfield enhancement endpoint was unavailable. No image request followed.
8. **Credit authorization.** A candidate-specific, one-attempt approval must be
   current for the exact executable manifest. The user request authorizes the
   workstream, but the repository's fail-closed policy still requires the
   receipt immediately before spend. No such receipt was manufactured.
9. **Candidate generation.** A provider output is always candidate-only. It
   cannot become product authority, replace protected pixels, or imply founder
   approval.
10. **Comfy staging.** Local Comfy at `127.0.0.1:8188` received the four LH-003
    product mattes plus the corrected black SG-007, protected SIG1 customer, and
    clean Signature sculpture. The upload is recorded in
    `Comfy/receipts/local-comfy-authority-upload-20260902.json`; no generation
    job was submitted. Staging an authority asset is not scene approval.
11. **Protected correction.** Product corrections require explicit masks and
    zero changed pixels outside them. SG-001 has this evidence. SG-003 does not.
12. **Independent review.** Product fidelity, identity/anatomy, focal hierarchy,
    collection world, sculpture geometry, and optical integration are separate
    dimensions. Any contracted hard fail quarantines a candidate regardless of
    average score.
13. **Founder decision.** Founder approval remains a distinct final visual gate.
    No candidate in this pass was promoted, wired, deployed, or marked approved.

## Credit-loss prevention now enforced

- Four rejected SIG1 hashes are deny-listed and cannot intersect routed sources.
- BR's one paid attempt remains consumed, with automatic retry disabled.
- SIG3 full-frame paid regeneration remains disabled.
- Product-only LH mattes are explicitly marked `on_model_authority=false`.
- The unapproved SG-006 alternative is recorded as discovery evidence only.
- Provider endpoint failure terminates the OODA loop before image generation.

## Next executable actions

1. Retry only the corrected SIG1 **enhance-only** request when the Higgsfield
   endpoint is healthy; obtain an independent PASS receipt for that exact prompt
   hash. Then, and only then, create a candidate-bound one-attempt approval.
2. Obtain founder approval or rejection for the exact discovered SG-006 alpha
   file hash; do not generate or substitute a woman meanwhile.
3. Build the SG-003 shorts reference pack and explicit mask, then run one
   localized correction with zero-outside-mask verification. Never regenerate
   the SIG3 frame.
4. Re-source BR-004 and the BR-005/BR-007 male as clean, separately reviewable
   protected alphas. Do not composite the rejected v6 correction and do not
   issue another paid BR request.
