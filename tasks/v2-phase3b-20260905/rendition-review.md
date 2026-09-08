# Phase 3B independent rendition review

Date: 2026-09-05

Verdict: **APPROVE** for the reviewed generator and runtime fallback scope. No remaining actionable critical or high defects were found after the fixes below. This review does not authorize deployment or certify browser image selection, visual rendition quality, or final package inclusion.

## Scope and method

Reviewed `tools/v2-runtime/build-card-renditions.py`, its focused regression tests, and the derivative handling added to `wordpress-theme/skyyrose-flagship-2/inc/approved-card-fronts.php`. The reviewer performed read-only source inspection and isolated temporary-fixture checks. No storefront source, approved media, browser state, or build output was changed by the reviewer. Temporary fixtures were removed automatically.

Direct metadata inspection confirmed all 33 approved input fronts are RGB, 1024 × 1536 pixels, without ICC profiles or EXIF orientation metadata. The generator's 320, 480, and 768 pixel widths therefore specify 99 derivatives at 320 × 480, 480 × 720, and 768 × 1152 pixels. Final all-source hash parity and the compiled 99-file manifest/package comparison remain integration checks owned by the parent; they are not inferred from the small test fixture.

## Resolved findings

- Existing output symlinks could previously cause a generated write to overwrite an approved source or escape the intended derivative directory. Output ancestor and leaf symlink rejection plus atomic file replacement resolve the identified path. The manifest receives equivalent handling.
- Malformed rendition lists and missing heights could previously produce PHP warnings or invalid dimensions. Array, dimension, aspect-ratio, regular-file, readability, and leaf-symlink checks now preserve original-front fallback.
- The generator entry point now has parameter and return annotations, and the Python file passes Black. Missing annotations were a maintainability/check-coverage concern, not evidence of a user-visible failure.

Source bytes are read and checked against the accepted front's exact SHA-256 and dimensions before resizing. Resizing is proportionate, uncropped, and does not upscale. Sorted iteration and JSON keys remove ordering drift. The generator does not modify approval manifests or confer approval on editorial/opening media. PHP selects derivatives only when their recorded source path and accepted-front hash match.

## Independently executed checks

Commands were run from the worktree root:

```sh
git diff -- '*.py'
ruff check tools/v2-runtime/build-card-renditions.py
black --check tools/v2-runtime/build-card-renditions.py
mypy --follow-imports=skip tools/v2-runtime/build-card-renditions.py
.artifacts/v2-source-certification-20260905/venv/bin/python tools/v2-runtime/test-card-renditions.py
```

The initial Python generator was untracked, so its contents were also inspected directly. Final Ruff, Black, and mypy checks all passed. All five Python tests passed:

1. Repeated generation and check mode produce identical fixture bytes and preserve the original source.
2. Source-hash drift is rejected before delivery output exists.
3. A rendition symlink cannot overwrite the source.
4. An output-directory symlink is rejected.
5. Check mode rejects stale derivative bytes without silently repairing them.

A separate ephemeral Python harness used `tempfile.TemporaryDirectory` outside the repository, wrote a minimal theme fixture, and invoked a fresh `php` process for each case. The PHP probe installed an error handler that throws on warnings, stubbed only the `WC_Product::get_sku()` dependency, loaded the actual reviewed helper, and inspected its JSON result. Fresh processes avoid static manifest-cache contamination between cases.

| PHP case | Result |
|---|---|
| Valid derivative record | PASS: derivative selected and original source retained |
| String instead of rendition list | PASS: original fallback, no warnings |
| Missing rendition height | PASS: original fallback, no warnings |
| Incorrect rendition height | PASS: original fallback, no warnings |
| String instead of rendition record | PASS: original fallback, no warnings |
| Source-hash mismatch | PASS: original fallback, no warnings |
| Invalid JSON manifest | PASS: original fallback, no warnings |
| Null JSON manifest | PASS: original fallback, no warnings |
| String JSON manifest | PASS: original fallback, no warnings |

The existing package writer was inspected and explicitly normalizes ZIP entry permissions to `0644`; atomic temporary-file permissions do not create an archive-serving regression.

## Integration boundaries

The parent must bind the final generated manifest and package to the runtime source, verify every original hash remains unchanged, verify all 99 expected derivative files, and measure browser `currentSrc` selection and performance. Visual quality of resized garment details remains a separate pixel review. No payment, media-promotion, staging, or production claim follows from this approval.
