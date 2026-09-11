# About media recovery inventory

Status: FOUNDER_REVIEW_REQUIRED. The founder explicitly confirmed: “the blox video is the interview.” The required interview is therefore identified as Ja11W-g34Zo; public metadata works. Browser playback and source-to-poster correspondence remain separate verification gates owned by the parent task.

## Television source

- Historical title: **Founder Feature Interview**; current official YouTube title: **Corey Foster  The Skyy Rose Collection**.
- Historical outlet label: **The Blox**. The source upload belongs to **Corey Foster**, channel `https://www.youtube.com/@The-Skyy-Rose-Collection`.
- Original recovered embed: `https://www.youtube.com/embed/Ja11W-g34Zo`; privacy-enhanced historical implementation: `https://www.youtube-nocookie.com/embed/Ja11W-g34Zo?rel=0&modestbranding=1`.
- Public source: [YouTube original](https://www.youtube.com/watch?v=Ja11W-g34Zo).
- Official poster metadata: `https://i.ytimg.com/vi/Ja11W-g34Zo/hqdefault.jpg`, 480 × 360. Do not substitute the local studio portrait as a documentary still without verifying correspondence.
- Current availability: direct YouTube oEmbed HTTP 200 on this run; title/author/embed/thumbnail returned. Receipt: `.artifacts/v2-about-recovery-20260906/media/youtube-oembed.json`. Web fetch tool was throttled; direct metadata fetch succeeded. This is not playback proof.
- No third-party video was downloaded or republished.
- Historical placement: commit `3860e38cb62733116def95ddf98519d8ee203cca`, `template-about.php` Press & Media after Founder Message and before Community. Later `featured-video.php` placed the same interview directly beneath the hero. `docs/elite-web-builder-package/homepage/about.html:326` places it in Chapter IV, before press cards. Current V2 `template-parts/v2-about.php` contains the same video, link and lazy iframe in the section after the hero; inspect route/template selection separately.
- Historical context, preserved for review: “SkyyRose founder Corey Foster sits down with The Blox to discuss building a luxury streetwear brand from Oakland, the meaning behind each collection, and the vision for the future.” Source is historical theme copy, not an independently verified transcript.
- Historical self-hosted dependency: `assets/video/the-blox-interview.mp4`. The March commit message claims a 79MB LFS video, but that path is absent from that commit tree and `git log --all -- '**/the-blox-interview.mp4'` returned no entries. Local main/current theme searches found none. Mark **BROKEN / MISSING DEPENDENCY**, preserve path, use the recovered original YouTube source where valid rather than inventing/rehosting footage.
- Historical TikTok link was only the profile `https://www.tiktok.com/@skyyroseco`, not a recovered interview permalink.

**Founder confirmation resolves the interview identity:** the founder stated “the blox video is the interview,” confirming the recovered Ja11W-g34Zo source as the required interview. There is no active missing-interview or Fox 2 recovery blocker. Historical investigation note: the mockup’s Fox 2 label had no corroborating source and is inapplicable to this confirmed interview; do not carry that label into the page. Browser playback remains a separate parent-task verification gate.

## Media inventory

All recovered binaries retain exact Git bytes, with SHA-256 in `.artifacts/v2-about-recovery-20260906/media/local-media-hashes.json`. No runtime rewiring occurred in this subtask.

| Title / section | Content type and source | Current status / approval confidence | Current V2 equivalent | Action / classification |
|---|---|---|---|---|
| Interview poster | `3860e38cb:.../assets/images/press-the-blox-interview.jpg`, 187,035 bytes | Recovered; visually inspected: studio interview composition, white Love Hurts jacket. Historical use proved; documentary authenticity and subject identity not proved by filename. | `assets/sot/images/about/the-blox-premiere.webp`, 63,310 bytes; same visual composition | SUPPORTING MEDIA / HISTORICAL VERSION. Prior `founder-review-queue.json` holds WebP for founder review. Preserve; compare to official video poster before promotional use. |
| Founder Message portrait | `3860e38cb:.../assets/images/founder-portrait.jpg`, 122,041 bytes | Recovered; adult black hoodie portrait. Historic alt was “SkyyRose Founder”; no original capture/authoring proof recovered. | No exact active V2 counterpart identified | HISTORICAL VERSION / UNKNOWN. Preserve for identity/approval review. |
| Story 0 | `3860e38cb:.../assets/images/about-story-0.jpg`, 235,763 bytes | Recovered; adult drawing at desk. Historical editorial image; no factual event/capture proof recovered. | No exact active V2 counterpart identified | SUPPORTING MEDIA / HISTORICAL VERSION. Preserve; do not state this depicts a documented founding event. |
| Story 1 | `3860e38cb:.../assets/images/about-story-1.jpg`, 201,942 bytes | Recovered; rose growing from concrete, stylized editorial symbolism | No exact active V2 counterpart identified | EDITORIAL ONLY / HISTORICAL VERSION. Preserve metaphorical use and founder review. |
| Story 2 | `3860e38cb:.../assets/images/about-story-2.jpg`, 246,021 bytes | Recovered; hands holding embroidered fabric; no manufacturing provenance recovered | No exact active V2 counterpart identified | EDITORIAL ONLY / HISTORICAL VERSION. Do not convert into product construction claim. |
| Existing About hero | `assets/sot/images/about/skyy-rose-founder-hero.webp`, 55,852 bytes | Existing preserved V2 and legacy `homepage-story-founder.webp`; visually a **child alone** in white/rose outfit, 724 × 1086. Filename is misleading for adult founder. | Exact existing asset | SUPPORTING MEDIA. Preserve; do not label father/daughter pair or adult founder portrait. |
| User design reference | `/Users/theceo/Downloads/ChatGPT Image Sep 6, 2026 at 05_30_07 PM.png` | User-provided presentation reference. Does not independently prove pictured interview/outlet/family photos are authentic archival assets. | New reference only | AUTHORING / VISUAL REFERENCE. Do not cut mockup imagery into documentary assets. |
| Official YouTube thumbnail | YouTube oEmbed thumbnail URL above | Authoritative association with recovered video ID; not downloaded | Local historical poster not yet verified against it | SUPPORTING MEDIA. Prefer source-attached poster within authorized embed context. |

## Press archive verified against public sources

Four existing press records resolve publicly through the web tool; retain actual outlet/date/title and direct links. These are third-party features, not founder-authored narrative. Their publication context should remain transparent; CEO Weekly explicitly identifies its piece as third-party branded content. No entire external article was copied into this recovery report.

| Outlet | Date in historical corpus | Title / source | Action |
|---|---|---|---|
| Maxim | 2023-02-15 | [14 Game-Changing Entrepreneurs To Watch In 2023](https://www.maxim.com/partner/14-game-changing-entrepreneurs-to-watch-in-2023/) | Retain original link; page includes Corey Foster entry and daughter/brand context. |
| San Francisco Post | 2024-08-23 | [The Skyy Rose Collection: From Oakland’s Streets to Fashion Heights](https://sanfranciscopost.com/the-skyy-rose-collection-from-oaklands-streets-to-fashion-heights/) | Retain original feature link; historical corpus contains excerpts for source comparison. |
| Best of Best Review | 2024-08-20 | [The Skyy Rose Collection: Best Bay Area Clothing Line Award 2024](https://bestofbestreview.com/awards/the-skyy-rose-collection-best-bay-area-clothing-line-award-2024) | Preserve exact award title; do not turn it into independently audited award criteria. |
| CEO Weekly | 2024-10-22 | [The Unyielding Journey of a Single Father and Entrepreneur](https://ceoweekly.com/the-unyielding-journey-of-a-single-father-and-entrepreneur/) | Retain original link; specifically corroborates an appearance on The Blox, not Fox2 and not this exact clip identity. |

## Sources inspected and boundaries

Inspected current theme About templates/press partials, `docs/elite-web-builder-package/homepage/about.html`, archived WordPress About HTML files, `docs/v2-authoring/archive-20260906/about-legacy-page.fragment.txt`, `knowledge-base/seed/press-features.md`, `knowledge-base/seed/from-interview.md`, WordPress builder/Elementor/press records, media hashes/recovery queue, Git commit3860e38cb and media-path history, and main/current local theme filenames. Root owns WordPress current content/revisions and recovered remote snapshot; its findings may add media.

Historical claim warnings: `knowledge-base/seed/press-features.md` explicitly retires earlier uncorroborated claims including “Hurts is the founder’s family name,” a2019founding date, sellouts and customer counts. Preserve those as historical disputed records, not approved factual copy. Neither image filenames nor the old commit’s “real photos” prose independently prove photographic authenticity.
