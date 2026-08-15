# V2 typography and artwork provenance contract

This contract keeps page typography legally and semantically separate from
collection lockups. A font file may be used as a page font only when its
license and source are recorded; a SkyyRose graphic or custom script remains
approved artwork and is never presented as an open-source font.

## Page typography

The self-hosted files below contain SIL Open Font License metadata in their
font names. They are the approved page-font layer for V2 and may be rebuilt
from their named upstream projects without changing collection identity.

| Role | Family | Local file | Upstream provenance | License gate |
|---|---|---|---|---|
| Display | Archivo | `assets/sot/fonts/archivo-latin.woff2` | [Omnibus-Type/Archivo](https://github.com/Omnibus-Type/Archivo) | SIL OFL 1.1; preserve the reserved-font-name notice and license text |
| Body | Hanken Grotesk | `assets/sot/fonts/hanken-grotesk-latin.woff2` | [marcologous/hanken-grotesk](https://github.com/marcologous/hanken-grotesk) | SIL OFL 1.1; preserve upstream attribution |
| Utility | Anton | `assets/sot/fonts/anton-latin.woff2` | [googlefonts/AntonFont](https://github.com/googlefonts/AntonFont) | SIL OFL 1.1; preserve the reserved-font-name notice |
| Ceremonial metadata | Cinzel | `assets/sot/fonts/cinzel-latin.woff2` | [NDISCOVER/Cinzel](https://github.com/NDISCOVER/Cinzel) | SIL OFL 1.1; preserve upstream attribution |
| Editorial script accent | Grand Hotel | `assets/sot/fonts/grand-hotel-latin.woff2` | Astigmatic / Grand Hotel | SIL OFL 1.1; preserve the reserved-font-name notice |
| UI fallback | Inter | `assets/sot/fonts/inter-latin.woff2` | [rsms/inter](https://github.com/rsms/inter) | SIL OFL 1.1; preserve upstream attribution |
| Signature accent | Pinyon Script | `assets/sot/fonts/pinyon-script-latin.woff2` | [SorkinType/Pinyon](https://github.com/SorkinType/Pinyon) | SIL OFL 1.1; verify the distributed license file when replacing bytes |

The exact license text must ship with any redistributed replacement bytes.
The font files in the theme are a build projection; the upstream project and
license are the authority, not the filename.

## Collection artwork boundary

These files are not treated as open-source page typography:

- `skyyrose-black-rose-script-latin.woff2`
- `skyyrose-love-hurts-graffiti-latin.woff2`

Their internal family names identify them as SkyyRose artwork, but no license
or public upstream source is embedded in the files. They are therefore
reserved for founder-approved collection lockups, monument scenes, and other
art-directed artifacts. They must not be used for generic headings, exported
as a downloadable font, or substituted for the page type roles above until a
rights record exists.

## Collection mapping

| Collection | Page-type system | Approved artwork layer |
|---|---|---|
| Signature | Archivo / Hanken Grotesk / Anton / Cinzel | Founder-approved Signature lockup or rose-gold graphic; Pinyon/Grand Hotel only when the source record says lockup artwork |
| Black Rose | Archivo / Hanken Grotesk / Anton / Cinzel | Black Rose star graphic and founder-approved Black Rose script lockup |
| Love Hurts | Archivo / Hanken Grotesk / Anton / Cinzel | Love Hurts star graphic and founder-approved graffiti lockup |
| Kids Capsule | Archivo / Hanken Grotesk / Anton / Cinzel | Full-color heir/mascot and throne artwork; no monochrome statue substitute |
| Jersey Series (Black Rose release) | Archivo / Hanken Grotesk / Anton / Cinzel | Jersey-specific marks and product imagery; retain the dedicated release treatment without inheriting core Black Rose artwork |

The CSS and `theme.json` mappings must continue to use the page-type system.
Collection-specific scripts and graphics belong in image/artifact slots so
they remain accessible, rights-reviewable, and replaceable without changing
the reading hierarchy.

## Verification gate

Before a font byte is changed, record its family, upstream commit/tag, license
file checksum, reserved-font-name obligations, and the candidate asset hash in
the SOT. Run `npm run verify` and the candidate media/rights gate. A missing
license, unknown custom-face provenance, or unbound lockup is `BLOCKED`, not an
implicit approval.
