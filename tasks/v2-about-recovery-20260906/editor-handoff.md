# About editor handoff — local review only

`about-editor-content.html` is a native Gutenberg document, not a static page screenshot. WordPress's bundled block-library parser/serializer accepted all 116 blocks without invalid blocks. Text, section sequence, graphic images, links and Blox source can be edited normally.

Local About displays this document through `.artifacts/v2-about-recovery-20260906/router.php`, using an in-memory query-post substitution. No `wp_update_post`, database writes or published content replacement occurred.

The theme's explicit `sr2-about-archive` group marker activates editor content. Without that marker the existing About remains available. The interpreter accepts only HTTPS YouTube watch or youtu.be links with 11-character IDs; source links remain usable without JavaScript. Theme-specific image dimensions/priority are provided by the render filter, not baked into editor-only raw HTML.

Review first: confirm restored story wording and timeline editorial claims, visual composition and collection display treatment. The Kids figure uses the founder's new supplied file, not the provisional SR mark. An eventual release must package these source changes and assets with a separately reviewed content migration. Do not apply this file to staging using the previous exact ZIP authorization.
