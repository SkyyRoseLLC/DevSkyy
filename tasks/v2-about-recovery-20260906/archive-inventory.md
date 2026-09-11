# About archive recovery inventory

Status: **FOUNDER_REVIEW_REQUIRED**. Inspected 53 distinct saved source records and 94 worktree files. No theme edits or staging writes.

## Findings

- press-features.md explicitly retires old2019/2500customer/soldout/surname claims. Preserve as historical evidence, not publication copy.
- collection-stories.md later founder lock says Hurts is the bloodline that raised me; this does not validate old family-surname claims.
- Current V2 has removed rich V1 origin paragraphs, quote, timeline and most Oakland story.
- Screenshot says Fox2 but recovered prior templates say The Blox. These cannot be silently conflated.

This inventory records exact source text and provenance, not blanket approval of every historic string. Current founder instruction establishes preservation of prior approved work; unsupported and explicitly retired claims remain historical. WordPress page/revision ownership and current external media access are separate parallel inventories.

## High-value recovered V1 elements

### The Story / Luxury Grows from Concrete.

Source: `wordpress-theme/skyyrose-flagship/template-about.php`

Classification: **FOUNDER-APPROVED — PRESERVE**. Approval confidence: HIGH historical inclusion; exact founder authorship not independently established.

Current V2: V2 uses Built by a father. Named for his daughter. with same portrait. Action: **REFORMAT; preserve exact identity phrase and factual metadata**.

Media: assets/images/homepage-story-founder.webp. Placement/chronology: Hero.

```text
The Story
Luxury Grows from Concrete.
Skyy Rose — The Heir
$hero_meta = array(
	array(
		'key' => 'Founded',
		'val' => '2020',
	),
	array(
		'key' => 'City',
		'val' => 'Oakland, CA',
	),
	array(
		'key' => 'Audience',
		'val' => 'Gender Neutral',
	),
	array(
		'key' => 'Chapters',
		'val' => '4 Collections',
	),
);

```

### The Birth of the Rebrand — three paragraphs

Source: `wordpress-theme/skyyrose-flagship/template-about.php`

Classification: **FOUNDER-APPROVED — PRESERVE**. Approval confidence: HIGH historical inclusion; exact founder authorship not independently established.

Current V2: V2 replaces with one short paragraph; three originals absent. Action: **RESTORE VERBATIM; formatting only**.

Media: None. Placement/chronology: After featured video.

```text
$origin_paragraphs = array(
	'SkyyRose was never just a clothing brand. It was a promise &mdash; born in Deep East Oakland, forged by a single father who refused to let his circumstances write his daughter\'s story. <strong>Corey Foster</strong> named the brand after his reason for everything: his daughter, <strong>Skyy Rose</strong>.',
	'The rebrand is the full realization of that promise. Four collections, each with its own world, its own story, its own identity &mdash; unified under one crown. Oakland-born luxury streetwear that doesn\'t apologize for where it came from. It <strong>celebrates</strong> it.',
	'Fashion was always self-expression. Growing up in Oakland\'s toughest neighborhoods, what you wore said everything &mdash; who you were, where you were going, what you refused to accept. SkyyRose carries that energy into every thread, every stitch, every rose.',
);

```

### First rose / 4 AM — Corey Foster

Source: `wordpress-theme/skyyrose-flagship/template-about.php; docs/brand/collection-stories.md:150`

Classification: **FOUNDER-APPROVED — PRESERVE**. Approval confidence: HIGH documented brand narrative; founder attribution inherited, not newly verified.

Current V2: Absent. Action: **PRESERVE VERBATIM; exact text rooted in documented Signature narrative**.

Media: None. Placement/chronology: Origin pullquote between second and third paragraphs.

```text
$origin_quote      = '"I drew the first rose on a night I couldn\'t afford dinner. Broke, a baby on the way, every manufacturer I\'d worked with had scammed me. But I sat there sketching that script logo until 4 AM because something in me knew — if I could get this right, everything changes. Signature is that night made permanent."';

$origin_cite       = 'Corey Foster — Founder & CEO';

```

### The Mission

Source: `wordpress-theme/skyyrose-flagship/template-parts/about/mission.php`

Classification: **HISTORICAL VERSION**. Approval confidence: HIGH historical inclusion; exact founder authorship not independently established.

Current V2: Mission wording absent; generic collection CTA remains. Action: **FOUNDER REVIEW for all-house bloodline wording (later canon scopes bloodline to Love Hurts); preserve archive**.

Media: None. Placement/chronology: V1 About.

```text
Luxury Grows from Concrete.
Four Collections, A Bloodline, and the Heir to the Throne.
Shop the Collection
```

### Drop Calendar — 2020/2021/2023/2024/2026

Source: `wordpress-theme/skyyrose-flagship/template-about.php`

Classification: **FOUNDER-APPROVED — PRESERVE**. Approval confidence: HIGH documented historical inclusion; timeline dates separately anchored by KB; voice authorship inherited.

Current V2: Absent. Action: **RESTORE exact supported content; chronology verify against timeline.md**.

Media: None. Placement/chronology: V1 About.

```text
$timeline_milestones = array(
	array(
		'year'  => '2020',
		'event' => 'The Beginning',
		'desc'  => 'Single father in Deep East Oakland. Searching for something his daughter could wear. Nothing fit the eye. He made it himself. The brand carries her name.',
	),
	array(
		'year'  => '2021',
		'event' => 'Three Chapters Drop',
		'desc'  => 'Black Rose. Love Hurts. Signature. Three collections, one name &mdash; not a launch, a declaration.',
	),
	array(
		'year'  => '2023',
		'event' => 'National Recognition',
		'desc'  => 'Maxim names Corey Foster one of 14 game-changing entrepreneurs to watch. The work travels past the city limits.',
	),
	array(
		'year'  => '2024',
		'event' => 'The Year of Receipts',
		'desc'  => 'San Francisco Post profile. Best Bay Area Clothing Line Award. CEO Weekly cover story. The Blox interview. Independent press confirms what Oakland already knew.',
	),
	array(
		'year'  => '2026',
		'event' => 'The Kids Capsule',
		'desc'  => 'The fourth chapter. Same craftsmanship, smaller silhouettes. Passing the torch on the same terms that built the brand.',
	),
);

```

### Oakland — twelve place names

Source: `wordpress-theme/skyyrose-flagship/template-about.php`

Classification: **FOUNDER-APPROVED — PRESERVE**. Approval confidence: HIGH documented historical inclusion; timeline dates separately anchored by KB; voice authorship inherited.

Current V2: Five of twelve retained. Action: **RESTORE exact supported content; chronology verify against timeline.md**.

Media: None. Placement/chronology: V1 About.

```text
$manifesto_places = array(
	'Deep East',
	'The Hills',
	'Stone City',
	'The 100s',
	'Brookfield',
	'Sobrante Park',
	'Coliseum',
	'Real Oakland',
	'The Shows',
	'Sequoyah Highlands',
	'Lake Merritt',
	'The 510',
);

```

### The Town / two studios

Source: `wordpress-theme/skyyrose-flagship/template-about.php`

Classification: **FOUNDER-APPROVED — PRESERVE**. Approval confidence: HIGH documented historical inclusion; timeline dates separately anchored by KB; voice authorship inherited.

Current V2: Absent. Action: **RESTORE exact supported content; chronology verify against timeline.md**.

Media: None. Placement/chronology: V1 About.

```text
$community_frame = 'The Town never asked permission to be itself. Neither did he. Lake Merritt at golden hour, the 510 humming under everything, Mac Dre on the speakers like a saint &mdash; that\'s the studio. The studio is also a laptop on a folding table at the airport lounge between shifts. Both things true. Both things the brand.';

```

### Four collection narratives

Source: `wordpress-theme/skyyrose-flagship/template-parts/about/collections-grid.php`

Classification: **FOUNDER-APPROVED — PRESERVE**. Approval confidence: HIGH historical inclusion; exact founder authorship not independently established.

Current V2: All four images/routes retained; descriptions replaced with shorter lines. Action: **REFORMAT preserved descriptions; retain exact original in inventory**.

Media: Four lookbook images; 480/960 derivatives. Placement/chronology: V1 About.

```text
 array(
	'signature'    => array(
		'title' => 'Signature',
		'tag'   => 'The Origin',
		'desc'  => 'The first rose, the first script. Where SkyyRose began — gold-accented luxury streetwear, gender-neutral by default.',
		'link'  => '/collections/signature/',
		'img'   => 'assets/images/lookbook/lb-rose-hoodie-beanie-960w.webp',
	),
	'black-rose'   => array(
		'title' => 'Black Rose',
		'tag'   => 'The Refusal',
		'desc'  => 'Dark, powerful, unapologetic. Silver-on-black, gothic restraint. Streetwear armor for everyone who refused to apologize first.',
		'link'  => '/collections/black-rose/',
		'img'   => 'assets/images/lookbook/lb-black-rose-football-960w.webp',
	),
	'love-hurts'   => array(
		'title' => 'Love Hurts',
		'tag'   => 'The Grief',
		'desc'  => 'A collection named after grief, made to be worn anyway. Crimson and deep red, Beauty &amp; the Beast cadence — luxury or witchcraft, take your pick.',
		'link'  => '/collections/love-hurts/',
		'img'   => 'assets/images/lookbook/lb-love-hurts-varsity-960w.webp',
	),
	'kids-capsule' => array(
		'title' => 'Kids Capsule',
		'tag'   => 'The Heir',
		'desc'  => 'Rose gold and soft pink. The fourth chapter, smaller silhouettes, same craftsmanship. Passing the torch on the same terms that built the brand.',
		'link'  => '/collections/kids-capsule/',
		'img'   => 'assets/images/lookbook/lb-kid-black-rose-960w.webp',
	),);
```

### Four original press features

Source: `wordpress-theme/skyyrose-flagship/template-about.php; knowledge-base/seed/press-features.md`

Classification: **FOUNDER-APPROVED — PRESERVE**. Approval confidence: HIGH historical source records; external article rights not transferred by recovery.

Current V2: Four press features retained through skyyrose2_press_features. Action: **PRESERVE titles/source URLs; reconcile excerpt wording with original press source**.

Media: None. Placement/chronology: V1 About.

```text
$press_features = array(
	array(
		'src'      => 'Maxim',
		'year'     => 'Feb 2023',
		'headline' => '14 Game-Changing Entrepreneurs To Watch In 2023',
		'excerpt'  => 'Corey Foster is an innovative entrepreneur and artist who has created a truly unique clothing line. The Skyy Rose Collection blends fashion with streetwear &mdash; high quality, beautiful, unique.',
		'url'      => 'https://www.maxim.com/partner/14-game-changing-entrepreneurs-to-watch-in-2023/',
	),
	array(
		'src'      => 'San Francisco Post',
		'year'     => 'Aug 2024',
		'headline' => 'From Oakland\'s Streets to Fashion Heights',
		'excerpt'  => 'A trailblazing gender-neutral clothing brand redefining fashion in the Bay Area and beyond. Established by Corey Foster, a single father with a dream &mdash; this Oakland-based brand embodies resilience and creativity.',
		'url'      => 'https://sanfranciscopost.com/the-skyy-rose-collection-from-oaklands-streets-to-fashion-heights/',
	),
	array(
		'src'      => 'Best of Best Review',
		'year'     => 'Aug 2024',
		'headline' => 'Best Bay Area Clothing Line Award 2024',
		'excerpt'  => 'The Skyy Rose Collection has been honored with the prestigious Best Bay Area Clothing Line Award 2024 &mdash; recognized for high-end, gender-neutral clothing that transcends age and gender boundaries.',
		'url'      => 'https://bestofbestreview.com/awards/the-skyy-rose-collection-best-bay-area-clothing-line-award-2024',
	),
	array(
		'src'      => 'CEO Weekly',
		'year'     => 'Oct 2024',
		'headline' => 'The Unyielding Journey of a Single Father and Entrepreneur',
		'excerpt'  => 'Despite numerous setbacks &mdash; failed website attempts, deceitful manufacturers &mdash; this indomitable spirit refused to be quenched. SkyyRose represents hope, hard work, and the relentless spirit of never giving up.',
		'url'      => 'https://ceoweekly.com/the-unyielding-journey-of-a-single-father-and-entrepreneur/',
	),
);

```

### The Blox — Featured Video

Source: `wordpress-theme/skyyrose-flagship/template-parts/about/featured-video.php`

Classification: **FOUNDER-APPROVED — PRESERVE**. Approval confidence: HIGH historical inclusion; exact founder authorship not independently established.

Current V2: Blox retained as iframe + poster + YouTube link. Action: **PRESERVE; root/media agent checks actual access; intent-load presentation**.

Media: press-the-blox-interview.jpg / the-blox-premiere.webp / YouTube Ja11W-g34Zo. Placement/chronology: Moved from press area directly below hero in 40ba9bc6e (2026-06-05).

```text
$youtube_embed_id = 'Ja11W-g34Zo';

SkyyRose Collection — The Blox Interview
https://www.youtube-nocookie.com/embed/Ja11W-g34Zo?rel=0&modestbranding=1
```

### Optional editorial divider / customer family gallery

Source: `template-parts/about/chapter-origin.php; template-about.php`

Classification: **BROKEN / MISSING DEPENDENCY**. Approval confidence: No evidence either slot rendered in final V1.

Current V2: No V2 divider; historical customer array empty. Action: **BROKEN SOURCE for missing divider; do not invent customer photographs**.

Media: Missing about-story-1.jpg; empty photo list. Placement/chronology: V1 About.

```text
assets/images/about-story-1.jpg; customer_photos=array()
```

### Hurts is the bloodline that raised me.

Source: `docs/brand/collection-stories.md:76`

Classification: **FOUNDER-APPROVED — PRESERVE**. Approval confidence: HIGH — source explicitly records Corey locked 2026-05-23.

Current V2: Not in current About; Love Hurts narrative relationship. Action: **Preserve exact later correction; do not revive historical surname claim**.

Media: None. Placement/chronology: 2026-05-23 canon clarification; not independently evidence of past About rendering.

```text
Hurts is the bloodline that raised me.
```

## Older HTML and Elementor elements

All meaningful recovered heading/body/quote widgets are transcribed below; whitespace is normalized for reading, original bytes remain in evidence copies. These are historical/unknown, not automatic publication candidates.

### HTML-001 — Born in Oakland, Built with Love

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Born in Oakland, Built with Love

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-002 — Born in Oakland, Built with Love

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

SkyyRose isn't just a brand—it's a movement. Born from the vibrant streets of Oakland, we blend authentic street culture with luxury craftsmanship to create pieces that tell stories.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-003 — The Beginning

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The Beginning

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-004 — The Beginning

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

SkyyRose was born from a simple observation: luxury fashion had lost its soul. The streets where we grew up taught us that real style comes from authenticity, from wearing your story on your sleeve.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-005 — The Beginning

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

We set out to create something different—a brand that honors both the raw energy of street culture and the meticulous craftsmanship of luxury fashion.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-006 — The Name Behind the Brand

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The Name Behind the Brand

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-007 — The Name Behind the Brand

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The name "Love Hurts" carries deep personal meaning. Hurts is our founder's family name, and it's woven into every piece we create. It represents the beautiful pain of growth, the strength found in vulnerability, and the courage to wear your heart openly.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-008 — The Name Behind the Brand

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

SkyyRose represents aspiration—reaching for the sky while staying rooted in our origins. Together, they embody our philosophy: where love meets luxury.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-009 — Our Journey

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Our Journey

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-010 — The Seed is Planted

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The Seed is Planted

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-011 — The Seed is Planted

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

First sketches and vision development. The concept of bridging street culture with luxury fashion takes shape in Oakland.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-012 — First Collection Drops

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

First Collection Drops

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-013 — First Collection Drops

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

BLACK ROSE launches to underground acclaim. Limited pieces sell out within hours, proving the demand for authentic luxury streetwear.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-014 — LOVE HURTS Emerges

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

LOVE HURTS Emerges

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-015 — LOVE HURTS Emerges

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Our most personal collection launches, featuring the Hurts family name. The emotional resonance connects with a global audience.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-016 — SIGNATURE Completes the Vision

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

SIGNATURE Completes the Vision

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-017 — SIGNATURE Completes the Vision

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The third collection establishes SkyyRose as a complete lifestyle brand. Three distinct voices, one unified vision.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-018 — What We Stand For

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

What We Stand For

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-019 — Authenticity

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Authenticity

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-020 — Authenticity

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Every piece tells a real story. We never compromise our vision for trends or mass appeal.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-021 — Craftsmanship

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Craftsmanship

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-022 — Craftsmanship

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Premium materials, meticulous construction. Quality that lasts beyond seasons.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-023 — Community

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Community

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-024 — Community

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Built by Oakland, for the world. We never forget where we came from.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-025 — Evolution

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Evolution

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-026 — Evolution

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Always growing, never settling. Each collection pushes boundaries further.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-027 — A Message from Our Founder

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

A Message from Our Founder

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-028 — A Message from Our Founder

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

"When I started SkyyRose, I had one goal: create clothes I actually wanted to wear. Clothes that felt like armor, that made a statement without saying a word."

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-029 — A Message from Our Founder

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

"Growing up in Oakland taught me that style is survival. It's how you tell the world who you are before you speak. That's what SkyyRose is about—giving people the pieces to tell their story."

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-030 — A Message from Our Founder

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

"The LOVE HURTS collection is the most personal to me because it carries my family name. Every piece is a piece of me, shared with everyone brave enough to wear their heart openly."

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-031 — A Message from Our Founder

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Where love meets luxury. Oakland-born streetwear for those who wear their heart on their sleeve.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-032 — A Message from Our Founder

`archive/redundant/wordpress website pages/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

© 2025 SkyyRose. All rights reserved.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-033 — Born in Oakland, Built with Love

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Born in Oakland, Built with Love

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-034 — Born in Oakland, Built with Love

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

SkyyRose isn't just a brand—it's a movement. Born from the vibrant streets of Oakland, we blend authentic street culture with luxury craftsmanship to create pieces that tell stories.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-035 — The Beginning

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The Beginning

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-036 — The Beginning

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

SkyyRose was born from a simple observation: luxury fashion had lost its soul. The streets where we grew up taught us that real style comes from authenticity, from wearing your story on your sleeve.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-037 — The Beginning

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

We set out to create something different—a brand that honors both the raw energy of street culture and the meticulous craftsmanship of luxury fashion.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-038 — The Name Behind the Brand

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The Name Behind the Brand

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-039 — The Name Behind the Brand

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The name "Love Hurts" carries deep personal meaning. Hurts is our founder's family name, and it's woven into every piece we create. It represents the beautiful pain of growth, the strength found in vulnerability, and the courage to wear your heart openly.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-040 — The Name Behind the Brand

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

SkyyRose represents aspiration—reaching for the sky while staying rooted in our origins. Together, they embody our philosophy: where love meets luxury.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-041 — Our Journey

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Our Journey

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-042 — The Seed is Planted

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The Seed is Planted

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-043 — The Seed is Planted

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

First sketches and vision development. The concept of bridging street culture with luxury fashion takes shape in Oakland.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-044 — First Collection Drops

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

First Collection Drops

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-045 — First Collection Drops

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

BLACK ROSE launches to underground acclaim. Limited pieces sell out within hours, proving the demand for authentic luxury streetwear.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-046 — LOVE HURTS Emerges

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

LOVE HURTS Emerges

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-047 — LOVE HURTS Emerges

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Our most personal collection launches, featuring the Hurts family name. The emotional resonance connects with a global audience.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-048 — SIGNATURE Completes the Vision

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

SIGNATURE Completes the Vision

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-049 — SIGNATURE Completes the Vision

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The third collection establishes SkyyRose as a complete lifestyle brand. Three distinct voices, one unified vision.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-050 — What We Stand For

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

What We Stand For

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-051 — Authenticity

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Authenticity

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-052 — Authenticity

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Every piece tells a real story. We never compromise our vision for trends or mass appeal.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-053 — Craftsmanship

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Craftsmanship

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-054 — Craftsmanship

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Premium materials, meticulous construction. Quality that lasts beyond seasons.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-055 — Community

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Community

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-056 — Community

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Built by Oakland, for the world. We never forget where we came from.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-057 — Evolution

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Evolution

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-058 — Evolution

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Always growing, never settling. Each collection pushes boundaries further.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-059 — A Message from Our Founder

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

A Message from Our Founder

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-060 — A Message from Our Founder

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

"When I started SkyyRose, I had one goal: create clothes I actually wanted to wear. Clothes that felt like armor, that made a statement without saying a word."

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-061 — A Message from Our Founder

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

"Growing up in Oakland taught me that style is survival. It's how you tell the world who you are before you speak. That's what SkyyRose is about—giving people the pieces to tell their story."

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-062 — A Message from Our Founder

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

"The LOVE HURTS collection is the most personal to me because it carries my family name. Every piece is a piece of me, shared with everyone brave enough to wear their heart openly."

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-063 — A Message from Our Founder

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Where love meets luxury. Oakland-born streetwear for those who wear their heart on their sleeve.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-064 — A Message from Our Founder

`archive/wordpress-legacy/skyyrose-website/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

© 2025 SkyyRose. All rights reserved.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-065 — Untitled

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Oakland, California — Est. 2020

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-066 — Named After a Daughter. Built by a Father.

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Named After a Daughter. Built by a Father.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-067 — Named After a Daughter. Built by a Father.

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The Skyy Rose Collection is what happens when a single father from Oakland decides his daughter deserves a different story.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-068 — Named After a Daughter. Built by a Father.

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Chapter I

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-069 — From Concrete to Collection

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

From Concrete to Collection

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-070 — From Concrete to Collection

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

“You ask me this four years ago, I never would’ve thought I’d be here. I had no drive, lost it all, baby on the way, and was broke. But we knew we had to get it by any means necessary.”

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-071 — From Concrete to Collection

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

— Corey Foster, Founder & CEO

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-072 — From Concrete to Collection

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

In the heart of Oakland’s toughest neighborhoods, where opportunities are scarce and the wrong path is always the easiest one, Corey Foster made a choice. With a daughter on the way, no savings, and a community that had already claimed too many people he loved — he decided to build something.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-073 — From Concrete to Collection

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Not a hustle. Not a side project. A legacy.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-074 — From Concrete to Collection

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

He named it after the reason he couldn’t fail: his daughter, Skyy Rose. What started as a father’s promise became a brand that would redefine what luxury streetwear looks like when it comes from somewhere real.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-075 — From Concrete to Collection

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The road was brutal. Failed websites. Manufacturers who took money and delivered nothing. Balancing 3 AM feedings with business plans written on a phone screen. Zero support. Zero guarantees. But Corey had something no setback could take — a reason bigger than himself.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-076 — From Concrete to Collection

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Chapter II

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-077 — What We Stand On

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

What We Stand On

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-078 — Gender-Neutral Pioneer

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Gender-Neutral Pioneer

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-079 — Gender-Neutral Pioneer

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

One of the first Bay Area brands to design clothing that transcends gender and age. Fashion without boundaries — for anyone with taste.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-080 — Oakland Authenticity

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Oakland Authenticity

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-081 — Oakland Authenticity

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Culture created, not imported. Every piece carries the resilience of the Town — where beauty and grit coexist without apology.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-082 — Family at the Core

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Family at the Core

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-083 — Family at the Core

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Named after a daughter. Built by a father. “Hurts” is our family name. This brand isn’t a business strategy — it’s a bloodline.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-084 — Quality Over Quantity

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Quality Over Quantity

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-085 — Quality Over Quantity

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Every garment is crafted with meticulous attention to detail. This isn’t fast fashion. This is armor — designed to last and built to make a statement.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-086 — Black-Owned, Community Built

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Black-Owned, Community Built

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-087 — Black-Owned, Community Built

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

A source of pride and inspiration for Oakland. Representing cultural heritage and future potential in every stitch.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-088 — Integrity Over Shortcuts

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Integrity Over Shortcuts

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-089 — Integrity Over Shortcuts

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Where many in the community were led astray, Corey chose the harder path. Every decision reflects the values he’s teaching his daughter.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-090 — Integrity Over Shortcuts

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Chapter III

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-091 — The Journey

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The Journey

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-092 — The Promise

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The Promise

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-093 — The Promise

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

With his daughter Skyy Rose on the way, Corey commits to building a brand that would support his family and inspire his community. The Skyy Rose Collection is born from a father’s determination.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-094 — The Grind

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The Grind

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-095 — The Grind

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Multiple website failures. Scam manufacturers. Sleepless nights balancing fatherhood and business. Every setback becomes fuel. The brand takes shape through sheer persistence.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-096 — Breaking Through

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Breaking Through

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-097 — Breaking Through

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The collection launches online. Word spreads through Oakland, then the Bay Area. Three distinct collections emerge: BLACK ROSE, LOVE HURTS, and SIGNATURE — each a world of its own.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-098 — National Recognition

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

National Recognition

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-099 — National Recognition

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Featured in Maxim’s “14 Game-Changing Entrepreneurs to Watch.” Spotlighted on The Blox. The children’s collection launches in March to celebrate Skyy Rose’s birthday.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-100 — Award-Winning

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Award-Winning

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-101 — Award-Winning

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Wins Best Bay Area Clothing Line from Best of Best Review. Featured in San Francisco Post, CEO Weekly. The brand’s story resonates nationally as proof that vision and fatherhood can coexist at the highest level.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-102 — Full-Stack Luxury

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Full-Stack Luxury

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-103 — Full-Stack Luxury

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Custom marketplace platform. AI-powered operations. Three collection worlds with dedicated product experiences. SkyyRose evolves from a brand into a complete luxury ecosystem.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-104 — Full-Stack Luxury

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Chapter IV

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-105 — As Seen In

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

As Seen In

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-106 — As Seen In

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The SkyyRose story has been recognized by national and regional publications for its authenticity, innovation, and the power of its origin.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-107 — As Seen In

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Maxim

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-108 — As Seen In

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

February 2023

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-109 — “14 Game-Changing Entrepreneurs to Watch in 2023”

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

“14 Game-Changing Entrepreneurs to Watch in 2023”

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-110 — “14 Game-Changing Entrepreneurs to Watch in 2023”

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Maxim spotlighted Corey Foster alongside tech founders and multimillion-dollar CEOs as one of the year’s most compelling entrepreneurs — recognizing the SkyyRose Collection as a rising force in streetwear built on resilience and authenticity.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-111 — “14 Game-Changing Entrepreneurs to Watch in 2023”

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

CEO Weekly

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-112 — “14 Game-Changing Entrepreneurs to Watch in 2023”

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

October 2024

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-113 — “The Unyielding Journey of a Single Father and Entrepreneur”

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

“The Unyielding Journey of a Single Father and Entrepreneur”

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-114 — “The Unyielding Journey of a Single Father and Entrepreneur”

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

CEO Weekly profiled the full arc of Corey’s journey — from growing up in an environment where crime was the norm, to building a brand that embodies hope and hard work while maintaining integrity every step of the way.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-115 — “The Unyielding Journey of a Single Father and Entrepreneur”

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

San Francisco Post

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-116 — “The Unyielding Journey of a Single Father and Entrepreneur”

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

August 2024

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-117 — “From Oakland’s Streets to Fashion Heights”

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

“From Oakland’s Streets to Fashion Heights”

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-118 — “From Oakland’s Streets to Fashion Heights”

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The San Francisco Post chronicled how the brand pioneered gender-neutral fashion in the Bay Area, creating a line that transcends societal boundaries with versatile, stylish pieces accessible to all — rooted in Oakland’s cultural landscape.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-119 — “From Oakland’s Streets to Fashion Heights”

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Best of Best Review

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-120 — “From Oakland’s Streets to Fashion Heights”

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

August 2024

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-121 — “Best Bay Area Clothing Line Award 2024”

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

“Best Bay Area Clothing Line Award 2024”

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-122 — “Best Bay Area Clothing Line Award 2024”

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

SkyyRose was honored with the Best Bay Area Clothing Line award, recognizing the brand’s exceptional contribution to fashion — citing authenticity, innovation in gender-neutral design, quality craftsmanship, and community impact as deciding factors.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-123 — “Best Bay Area Clothing Line Award 2024”

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

The Mission

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-124 — Luxury Grows from Concrete

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Luxury Grows from Concrete

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### HTML-125 — Luxury Grows from Concrete

`docs/elite-web-builder-package/homepage/about.html` — HISTORICAL VERSION / founder approval UNKNOWN.

Where Bay Area authenticity meets high-fashion aesthetics. Where a father’s love becomes a brand’s foundation. Where fashion is a force for change.

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-126 — w_spinning_logo

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"shortcode": "[skyyrose_spinning_logo variant=\"gold\"]"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-127 — Luxury Grows from Concrete.

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"title": "Luxury Grows from Concrete."}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-128 — w_hero_subheading

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"editor": "<p>A love letter to Oakland. Born from The Town's heart, where BART rumbles beneath heritage and the Bay Bridge frames our ambition. A revolution in streetwear forged on Oakland streets.</p>"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-129 — Our Origin Story

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"title": "Our Origin Story"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-130 — w_origin_text

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"editor": "<p style='font-size: 18px; line-height: 1.8; margin-bottom: 24px;'>SkyyRose was born from Oakland's beating heart — where West Oakland warehouses meet Lake Merritt sunsets, where BART commuters carry dreams beneath the city's pulse, where the Bay Bridge rises through fog each morning as The Town awakens. Street culture forged in Oakland's fire meets luxury craftsmanship refined for global stages.</p><p style='font-size: 18px; line-height: 1.8; margin-bottom: 24px;'>What started as late nights in Oakland studios — surrounded by Fillmore jazz heritage, inspired by day-one supporters who believed when The Town was our only audience — has evolved into a global movement. From underground fashion shows in Jack London Square to international runways, Oakland's authenticity remains our foundation. The Bay Area doesn't follow trends; we set the standard the world chases.</p><p style='font-size: 18px; line-height: 1.8;'><strong>Every piece tells Oakland's story. Every collection honors the streets that raised us. Every stitch carries The Town's soul to the world.</strong></p>"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-131 — The Collections

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"title": "The Collections"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-132 — BLACK ROSE

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"title": "BLACK ROSE"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-133 — LOVE HURTS

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"title": "LOVE HURTS"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-134 — SIGNATURE

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"title": "SIGNATURE"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-135 — Our Commitment

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"title": "Our Commitment"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-136 — Sustainable. Ethical. Transparent.

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"title": "Sustainable. Ethical. Transparent."}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-137 — w_commit_text

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"editor": "<p style='font-size: 18px; line-height: 1.8;'>Oakland taught us luxury never comes at the expense of community or planet. The Town's values guide every stitch, every material, every decision made with Bay Area intention. From our suppliers to our day-one supporters, transparency isn't a trend — it's the Oakland way.</p>"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-138 — Press & Recognition

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"title": "Press & Recognition"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-139 — w_press_item_1

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"editor": "<div style='display: flex; align-items: flex-start; gap: 24px; padding: 32px; background: rgba(255,255,255,0.03); border-radius: 8px; border-left: 4px solid #D4AF37;'><div style='flex-shrink: 0;'><span style='display: inline-block; padding: 8px 12px; background: rgba(212, 175, 55, 0.1); border-radius: 4px; color: #D4AF37; font-family: Inter, sans-serif; font-size: 12px; font-weight: 600;'>FEBRUARY 2024</span></div><div><h4 style='color: #FFFFFF; font-family: Playfair Display, serif; font-size: 20px; margin: 0 0 8px 0;'>San Francisco Post</h4><p style='color: #CCCCCC; font-family: Inter, sans-serif; font-size: 16px; margin: 0; line-height: 1.6;'>\"Emerging Luxury Brands You Need to Know About in 2024\"</p></div></div>"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-140 — w_press_item_2

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"editor": "<div style='display: flex; align-items: flex-start; gap: 24px; padding: 32px; background: rgba(255,255,255,0.03); border-radius: 8px; border-left: 4px solid #B76E79;'><div style='flex-shrink: 0;'><span style='display: inline-block; padding: 8px 12px; background: rgba(183, 110, 121, 0.1); border-radius: 4px; color: #B76E79; font-family: Inter, sans-serif; font-size: 12px; font-weight: 600;'>APRIL 2024</span></div><div><h4 style='color: #FFFFFF; font-family: Playfair Display, serif; font-size: 20px; margin: 0 0 8px 0;'>CEO Weekly</h4><p style='color: #CCCCCC; font-family: Inter, sans-serif; font-size: 16px; margin: 0; line-height: 1.6;'>\"Women in Fashion: The SkyyRose Success Story\"</p></div></div>"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-141 — w_press_item_3

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"editor": "<div style='display: flex; align-items: flex-start; gap: 24px; padding: 32px; background: rgba(255,255,255,0.03); border-radius: 8px; border-left: 4px solid #C9A962;'><div style='flex-shrink: 0;'><span style='display: inline-block; padding: 8px 12px; background: rgba(201, 169, 98, 0.1); border-radius: 4px; color: #C9A962; font-family: Inter, sans-serif; font-size: 12px; font-weight: 600;'>JULY 2024</span></div><div><h4 style='color: #FFFFFF; font-family: Playfair Display, serif; font-size: 20px; margin: 0 0 8px 0;'>Forbes - 30 Under 30</h4><p style='color: #CCCCCC; font-family: Inter, sans-serif; font-size: 16px; margin: 0; line-height: 1.6;'>\"Meet the Founder Reimagining Luxury Fashion\"</p></div></div>"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-142 — Our Impact

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"title": "Our Impact"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-143 — w_stat_1

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"editor": "<div style='text-align: center;'><p style='font-size: 48px; font-weight: 700; color: #D4AF37; font-family: Playfair Display, serif; margin: 0 0 8px 0;'>50K+</p><p style='font-size: 14px; color: #666666; font-family: Inter, sans-serif; margin: 0; text-transform: uppercase; letter-spacing: 1px;'>Community Members</p></div>"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-144 — w_stat_2

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"editor": "<div style='text-align: center;'><p style='font-size: 48px; font-weight: 700; color: #D4AF37; font-family: Playfair Display, serif; margin: 0 0 8px 0;'>15+</p><p style='font-size: 14px; color: #666666; font-family: Inter, sans-serif; margin: 0; text-transform: uppercase; letter-spacing: 1px;'>Countries Served</p></div>"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-145 — w_stat_3

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"editor": "<div style='text-align: center;'><p style='font-size: 48px; font-weight: 700; color: #D4AF37; font-family: Playfair Display, serif; margin: 0 0 8px 0;'>1000+</p><p style='font-size: 14px; color: #666666; font-family: Inter, sans-serif; margin: 0; text-transform: uppercase; letter-spacing: 1px;'>Pieces Crafted</p></div>"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-146 — w_stat_4

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"editor": "<div style='text-align: center;'><p style='font-size: 48px; font-weight: 700; color: #D4AF37; font-family: Playfair Display, serif; margin: 0 0 8px 0;'>100%</p><p style='font-size: 14px; color: #666666; font-family: Inter, sans-serif; margin: 0; text-transform: uppercase; letter-spacing: 1px;'>Ethically Sourced</p></div>"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-147 — Ready to Join the Movement?

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"title": "Ready to Join the Movement?"}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

### ELEMENTOR-148 — w_cta_btn

`wordpress/elementor_templates/about.json` — HISTORICAL VERSION / founder approval UNKNOWN.

{"text": "Shop Now", "link": {"url": "/shop/", "is_external": false}}

Action: FOUNDER REVIEW; preserve source, do not auto-promote unsupported claims.

## Every source inspected

| Source | Kind | SHA-256 |
|---|---|---|
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/wordpress-theme/skyyrose-flagship/template-about.php` | workspace | `d29dc0b84180c2ff2dee129d3f16c21d0dcac8f0e18ad64a84be0cc8abb1f528` |
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/wordpress-theme/skyyrose-flagship-2/template-parts/v2-about.php` | workspace | `3196d809379a8ac9eee94686fd4be751e879bad00d183959814ec7a53f149238` |
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/knowledge-base/seed/timeline.md` | workspace | `bdbfc688688f324e349ede72fd412913a5eddd3d4b284642bc131c51ec1f7a87` |
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/knowledge-base/seed/press-features.md` | workspace | `388583b733339aec774a7d1975c3e03791f3a62e9abc2d3e53a8464c0ee8b3b7` |
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/knowledge-base/seed/from-interview.md` | workspace | `66572e3b71f6c0b1154b0329ad68e8fcdddd56213b1a79de192f9acd740e2a3f` |
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/docs/brand/collection-stories.md` | workspace | `93c5f0536dba2174965883f8eed787458ac853e04dbd95464b15e8c67304a70e` |
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/wordpress-theme/skyyrose-flagship/assets/css/about.css` | workspace | `147f708de45ef826d4ff93cec69c214f5f1135fd6299a7eb4cd6c5b1f51da050` |
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/wordpress-theme/skyyrose-flagship/assets/js/about.js` | workspace | `77c07f57d9c6b071a20a8ffa4911008304733960a12de450a4dc565b7228e95f` |
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/wordpress-theme/skyyrose-flagship/template-parts/about/mission.php` | workspace | `5717b8717fa0a5f26c1da1e58950342667ea70473040a70e1dbe970b8d4125aa` |
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/wordpress-theme/skyyrose-flagship/template-parts/about/chapter-origin.php` | workspace | `8bcb97bdbe2d86e20461d541d66a002fca734e5f5f46d6a0aa907842b883ecb0` |
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/wordpress-theme/skyyrose-flagship/template-parts/about/community.php` | workspace | `3ae247d38840480a1cd4fb40d545f297a1afd1570d4488715501fbe9a13a3080` |
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/wordpress-theme/skyyrose-flagship/template-parts/about/collections-grid.php` | workspace | `5ee95fce20cc4b6564bb1067ebf8bc33aa3e4be3ebda5d833e16bca454756a36` |
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/wordpress-theme/skyyrose-flagship/template-parts/about/featured-video.php` | workspace | `18fecd2361b871d26b5af43fd161edc502d79d4a9ec66bdd020475ffdf80a961` |
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/wordpress-theme/skyyrose-flagship/template-parts/about/press-section.php` | workspace | `96e8d6730b3bb40ad1a486181cbf5e5fe2a3e7e9b16c3da4b1b92c99b5987f45` |
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/wordpress-theme/skyyrose-flagship/template-parts/about/timeline.php` | workspace | `776696328c5352d8d7ac6397c2451f402c66a7a9dad7209434bea8eb335ed423` |
| `9bda4597125ab43bea907e9712eac17d91826f28:wordpress-theme/skyyrose-flagship/template-about.php` | git | `d29dc0b84180c2ff2dee129d3f16c21d0dcac8f0e18ad64a84be0cc8abb1f528` |
| `b6acc56f84e9e90cab736d8694b34a15911d8f88:wordpress-theme/skyyrose-flagship/template-about.php` | git | `7d20f6909a14815bf122ce6e9688e55ea6a5bfb36046580f5b84f589c35f8ae7` |
| `1c6ad34286336cb9cf0f8e4eacbc983595b14c04:wordpress-theme/skyyrose-flagship/template-about.php` | git | `bbd56059b0c69c4bc5d44a5049cf4fff8f514ed651b639935649ce3df7c564aa` |
| `40ba9bc6e8bd0edcc60473ee1b1e080226edcdf0:wordpress-theme/skyyrose-flagship/template-about.php` | git | `a56543548d81508e14ef3d62eca5925e77621ed87ccea1da8c24af4cad945ec2` |
| `de495b11c3bccb4bd57f81f4122498d50e9e9e9b:wordpress-theme/skyyrose-flagship/template-about.php` | git | `dc4ab9ebc98bfb8dd489c6a6b9ccc3f0388b50d71f5d5264fd6d9604f3d755dd` |
| `13e9ef721dd6caf3cfa49623b8414bfde997f6e4:wordpress-theme/skyyrose-flagship/template-about.php` | git | `369a5938b875f34f390fce06233cda03379c59de82c6edd92eac5a691555f6f1` |
| `0cafe0728c630443c486a267c36a202afb1a9fba:wordpress-theme/skyyrose-flagship/template-about.php` | git | `c8716c40b7fc61ee3d768465848706b9c362f428a58caad014968f72604f93cb` |
| `4919b750a7519df3d972bc151ec8bfb32afe76cc:wordpress-theme/skyyrose-flagship/template-about.php` | git | `0ca00f64b34cd9d08d2a91a07950d0af408dc331215b3c01a207b9c6f91e228c` |
| `5dd55c1875f751b77d74e1720dba5a4e4d1986fb:wordpress-theme/skyyrose-flagship/template-about.php` | git | `eec1fc7e20fdd48ebbac0f1fac25046794495c3a5b143bfa9a4f44358524e7c1` |
| `76b26d349355936563105b8b5415a1a137be3bde:wordpress-theme/skyyrose-flagship/template-about.php` | git | `9dedf26e4a024e1b8c769fe298bc14d6fafeba95c96c526e86568edfcd2d898b` |
| `c6e60fdbc84bdd4ceac6198d8bbf3b7477724e9b:wordpress-theme/skyyrose-flagship/template-about.php` | git | `93ea872e60df326c05c7a88d8ba90c4e6c44dcb736e89c653815d5548cd98e28` |
| `19c52997b671f036e1dbce4bcf5b829294025595:wordpress-theme/skyyrose-flagship/template-about.php` | git | `c97125726f8e16438686be13a30b76c32ed1cfeededc21e75fe14cf4520721ef` |
| `c4d00b98ac3b00292a93bb1944d7163eadd82c7a:wordpress-theme/skyyrose-flagship/template-about.php` | git | `b3dc7a90583c3bc6aa08baf00903b15ec7595f8423a6190a00e75e11df7708e8` |
| `116d53c2d269cad6780ea7d2a31f47a73e701629:wordpress-theme/skyyrose-flagship/template-about.php` | git | `ec9c590580c5aec7777a2d0171d01e78e11222aff96b2ac1743555e491083302` |
| `01ee10376507262fc829c5f7b021acfe6e0e5b11:wordpress-theme/skyyrose-flagship/template-about.php` | git | `c2b350d05862b8f567dfdb1ecf0a13a92ce85d8a5e143c2efdcb63fd4c674ed1` |
| `8ad0df3139a212809733167b5ff10d2fab038680:wordpress-theme/skyyrose-flagship/template-about.php` | git | `38fc68fbe4ecfd972b11de763f6d115bc5246f6441b869d6750e139f66d2edc0` |
| `611f3e4debe788bbbf38d0a6af291265565d755f:wordpress-theme/skyyrose-flagship/template-about.php` | git | `9cb06ea0c209cc1c136159751978953c9c4e3459e921d89d9b97cec0816c6330` |
| `3cf9e029e789b6c78d451f3da91cd50bce918fd5:wordpress-theme/skyyrose-flagship/template-about.php` | git | `85b43c12ffd5647f6a340781377d9e13a4703632e0665c98863dfcf314d8e9c5` |
| `38a873c15124c3f4a2a970883b42c37143e79bd5:wordpress-theme/skyyrose-flagship/template-about.php` | git | `99fb922d3bf09bab5984a408ace191d16dd16e08a49c0ffaa3be65213e8def06` |
| `9a747bbcf3b3e9b6655555b27c4af4ad0cf9b966:wordpress-theme/skyyrose-flagship/template-about.php` | git | `939bba351386733c0cadacea9ed7d0adc78a11caa593dae175684f7b76aefcb1` |
| `9124644bae2bffdda2ae48f051ff4ed67a512e0d:wordpress-theme/skyyrose-flagship/template-about.php` | git | `67d76e65e6b95f64f9fd974fcdc41d8b9b99021d858ea024fbd9bb41b78fd8fa` |
| `3860e38cb62733116def95ddf98519d8ee203cca:wordpress-theme/skyyrose-flagship/template-about.php` | git | `5d69a3901cc0c1b1fc2c2833fefb0be5ec6cab9786870135099f86fd258087c2` |
| `31da8c9a43e2c6108d08b42454ab9d1b50a9020c:wordpress-theme/skyyrose-flagship/template-about.php` | git | `601e1c9f56722cba03c80dc1b150ac2ab39332f4d84dff56d349ac97236a6ee2` |
| `6b668f578dae3c468b2b88b28e0c0b0050140e80:wordpress-theme/skyyrose-flagship/template-about.php` | git | `50d1741a88f85d6e15b185f6ab8dc7a903b24952e1b7d4de2f9b6249bbb5066a` |
| `6c854f35e8f42e9ab89e28dfaddca3eedaa99785:wordpress-theme/skyyrose-flagship/template-about.php` | git | `f5f21cf067b8422085480f375730ebefaf466420bc24ec91a129a3fbdeb16502` |
| `c1dd81429abd1f3ede0a92e3179d41b8d9e74742:wordpress-theme/skyyrose-flagship/template-about.php` | git | `dbb9ad7c4d57b7a3fec2339c6daf80c658a43a5855834afc7d7167d4da81bcf5` |
| `2385b83699455462eb1ff5cb5ba177724dec4de5:wordpress-theme/skyyrose-flagship/template-about.php` | git | `7d0a84b75af200c5b758bf3671e84071859f8a7b9127e7e9611d3a698893c3b3` |
| `0b50f16c393e2760eba1aa6ce94bfd764b2b9683:wordpress-theme/skyyrose-flagship/template-about.php` | git | `4a25975d241b659095ee36a07f5a0be269d1766e730e8303d5c1453ce5fbfd7b` |
| `d8a918f4a11a262d84cad78769f96bd175b2d145:wordpress-theme/skyyrose-flagship/template-about.php` | git | `204e82ebbf3aea91b60f67028a5a7db953c2ae2dccab352b6a7e6ee947126500` |
| `f2f01bbcffd2ad1e1974ecd839b8a350241e5a8a:wordpress-theme/skyyrose-flagship-2/template-parts/v2-about.php` | git | `3196d809379a8ac9eee94686fd4be751e879bad00d183959814ec7a53f149238` |
| `862cbafe278f380b4cbbddfd5b7b788992749010:wordpress-theme/skyyrose-flagship-2/template-parts/v2-about.php` | git | `ae83814d57b455283f03d162bb4bdccfcd39e8bc152a767ec47465613657afb7` |
| `91c0f9f891465abcd6cc0af37dccfe5fa8314c90:wordpress-theme/skyyrose-flagship-2/template-parts/v2-about.php` | git | `4fa379838d900c5056c5420d15c8ceba9eb6aeb6b4188521623b8b6ed3a55363` |
| `1ef16b011e57089540a080e67a1417576d922afd:wordpress-theme/skyyrose-flagship/template-about.php` | git | `0ca00f64b34cd9d08d2a91a07950d0af408dc331215b3c01a207b9c6f91e228c` |
| `447-file rollback TAR::skyyrose-flagship-2/template-parts/v2-about.php` | rollback | `3196d809379a8ac9eee94686fd4be751e879bad00d183959814ec7a53f149238` |
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/archive/redundant/wordpress website pages/about.html` | workspace | `ebaed72c6ac9e213d527f16d6bf02458136b8b9b801499434dfd039e29e92213` |
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/archive/wordpress-legacy/skyyrose-website/about.html` | workspace | `d1a7bdebd9afadd19d2fd46a9fd243289d6cfae8fcca96812c3c1e301b058470` |
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/docs/elite-web-builder-package/homepage/about.html` | workspace | `a8280e109719dbf8516c09b4d91dcfec02afef8a48911f7bb791450fd58c7083` |
| `/Users/theceo/.codex/worktrees/7116/DevSkyy/wordpress/elementor_templates/about.json` | workspace | `9d2fdfd175c402d911ed875e9d0616ce8abece60f2d07df758e7fc797c89dd3d` |

Worktree full path/hash manifest: `.artifacts/v2-about-recovery-20260906/archive/worktree-manifest.json`.
All unique Git entrypoint versions are retained with exact full text and long-copy literal line inventory in `source-index.json`; historical JavaScript/CSS and current V1 partials are retained byte-for-byte.

Current-local screenshots recovered by filename at 390/768/1440 in `.artifacts/v2-cinematic-finalization-20260906/responsive/completion-chromium-about-{width}.png`. These document current V2, not legacy V1. No old browser screenshot found within the narrowly checked About filenames.

## Source identity and historical edit ownership

The current V2 About partial and captured 447-file rollback About partial are byte-identical: `3196d809379a8ac9eee94686fd4be751e879bad00d183959814ec7a53f149238`. The rich V1 content omission predates the latest authorized staging install. The first preserved August 14 V2 source already contains the condensed narrative; subsequent V2 changes add press excerpts and responsive images.

Historical March 30 source `8ad0df3139` exposes `about_youtube_id` through WordPress theme mods with default `Ja11W-g34Zo`. The March 1 source `3860e38cb6` references `assets/video/the-blox-interview.mp4` and `assets/images/press-the-blox-interview.jpg`. Actual remote theme-mod values and media availability remain under separate read-only investigation. No Fox2/KTVU string was found in the captured About entrypoint Git sources; the supplied visual reference must not be treated as evidence that The Blox is a Fox2 interview.
