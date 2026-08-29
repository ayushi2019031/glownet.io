# Glownet — design brief

Implementation spec for the visual redesign of glownet.io (Jekyll + Minima).
Follow this exactly. Where it is silent, prefer the quietest option.

---

## 1. Concept

The site is called Glownet. The identity is **light**: a lamp switched on over a
desk, illuminating how AI systems work. Everything derives from that — the amber
palette, the lamp mark, the warm paper ground, the glyph tiles.

This is the only rule that matters when making a judgment call: **if a choice
would look at home on any other engineer's blog, it is wrong.**

Do not introduce: monospace nav links, lowercase-only navigation, a
name-plus-job-title hero, terminal styling, gradient meshes, or card grids with
drop shadows.

---

## 2. Tokens

Declare on `:root`. Nothing below this line may hardcode a colour.

### Light (default)

| Token | Value | Role |
|---|---|---|
| `--paper` | `#FAEEDA` | page ground |
| `--paper-raised` | `#FDF7EC` | article surface, code blocks |
| `--ink` | `#412402` | headings, primary text |
| `--ink-body` | `#633806` | body copy |
| `--ink-soft` | `#854F0B` | metadata, nav, captions |
| `--filament` | `#EF9F27` | rules, mark, active states |
| `--tile` | `#FAC775` | glyph tiles, primary button |
| `--ember` | `#F5C4B3` | tag pills |
| `--ember-ink` | `#4A1B0C` | text on tag pills |

### Dark (`@media (prefers-color-scheme: dark)`)

A lamp in a dark room. The amber stays; the ground drops away.

| Token | Value |
|---|---|
| `--paper` | `#211405` |
| `--paper-raised` | `#2E1D08` |
| `--ink` | `#FAEEDA` |
| `--ink-body` | `#E8D3AE` |
| `--ink-soft` | `#C79A55` |
| `--filament` | `#EF9F27` |
| `--tile` | `#5A3608` |
| `--ember` | `#4A1B0C` |
| `--ember-ink` | `#F5C4B3` |

Only the variable block changes between modes. If you find yourself writing a
dark-mode rule that is not a variable reassignment, the light styles are wrong.

---

## 3. Typography

Two faces, loaded from Google Fonts:

- **Fraunces** (variable serif) — display. Headings, post titles, the wordmark.
  Weight 400–500 only. Use `font-optical-sizing: auto`. Set `--wonk: 1` and
  `--soft: 30` on display headings via `font-variation-settings` so it keeps its
  personality; do not use it at default settings, which reads like any serif.
- **Karla** — body, nav, metadata, UI. Weights 400 and 500.

Scale (rem, 16px root):

| Element | Size | Face | Tracking |
|---|---|---|---|
| Hero thesis | 2.0 | Fraunces 500 | −0.02em |
| Page h1 | 1.9 | Fraunces 500 | −0.02em |
| h2 | 1.35 | Fraunces 500 | −0.015em |
| h3 | 1.1 | Fraunces 500 | −0.01em |
| Post title (list) | 1.1 | Fraunces 500 | −0.01em |
| Body | 1.0625 | Karla 400 | normal |
| Metadata / nav | 0.85 | Karla 400 | normal |
| Tag pill | 0.72 | Karla 400 | 0.01em |

Body line-height `1.7`. Prose measure capped at `68ch`. Headings `1.2`.

---

## 4. Layout

Content column `min(46rem, 100% − 3rem)`, centred. Two exceptions: the hero may
run to `52rem`, and figures inside posts may break out to `52rem`.

### Masthead

Lamp mark (`#g-lamp-mark` from the glyph sprite, 22px) + "Glownet" in Fraunces,
left. Nav right: `writing` · `papers` · `about` in Karla sentence case, colour
`--ink-soft`, active item gets a 1.5px `--filament` bottom border. Separated from
the page by a 1px `--filament` rule.

### Hero (homepage only)

Two columns at ≥720px, stacked below. Left: thesis headline, one paragraph of
intro, a row of pill links. Right: the `#g-lamp-hero` illustration at 150px.

Copy — use verbatim:

> **Making the inside of AI systems visible.**
>
> I'm Ayushi — I build multi-agent systems on Azure at Microsoft, then take them
> apart here so you can see how they work.

Pills: "Résumé" filled (`--tile` bg, `--ink` text), the rest outlined (1px
`--filament`, `--ink-soft` text). `border-radius: 20px`, padding `5px 13px`.

### Post list

Each entry is a flex row: a 60px rounded-10px `--tile` square holding the post's
glyph at 34px, then title / description / tags. `gap: 16px`, `1.5rem` between
entries. No card borders, no shadows — the tile is the only visual anchor.

Tags are pills: `--ember` background, `--ember-ink` text, `border-radius: 20px`,
`gap: 6px`, wrapping.

**Bug to fix while you are in here:** tags currently render with no separator
(`Azure AI FoundryAgentic AIAzureCI/CD`). Each tag needs its own element inside a
flex container.

---

## 5. Glyph system

Instead of per-post illustration, there is a fixed set of abstract marks. Each
post declares one in front matter:

```yaml
---
title: Evaluating AI agents on Azure with AI Foundry
glyph: agents
---
```

Available: `agents`, `cuda`, `paper`, `generative`, `cloud`, `auth`. Default to
`paper` when unset.

The sprite lives at `_includes/glyphs.svg` and is inlined once in the default
layout (hidden, `width:0;height:0;position:absolute`) so the symbols inherit page
CSS variables. Reference with `<use href="#g-{{ post.glyph }}">`.

Do not add glyphs beyond the six unless a new topic genuinely recurs. The point
of a fixed set is that publishing costs one front-matter line.

---

## 6. Content fixes (do these too)

1. `<title>` currently contains the entire site description. Set it to
   `Glownet — Ayushi Jain` on the homepage, `{{ page.title }} · Glownet`
   elsewhere. Same for `og:title`.
2. Add `Ayushi Jain` to `og:site_name` and the meta description so the site is
   findable by name.
3. The About page skills list ends mid-sentence: `Git • VS Code • Jupyter •`
4. About page lists `AZ-201`. The correct code is **AZ-204** — her own
   credential link confirms it.
5. About page certification year says 2024; the résumé says 2023. Reconcile.
6. Email address must match the résumé exactly. Pick one and use it in the
   footer, the About page Connect row, and the closing line.
7. About page ordering: intro → current work → selected writing → credentials
   (Microsoft, IIIT-Delhi, Stanford, Azure certs, ACM publication) → personal
   note → books. Move Class 10 / Class 12 / JEE / NTSE below the fold or cut
   them; they do not help an international audience.
8. The JEE Advanced entry claims "top 0.3 percentile" beside AIR 11404. Those
   numbers do not reconcile. Correct or remove.

---

## 7. Quality floor

- Responsive to 360px. The hero stacks, the glyph tile shrinks to 48px.
- Visible keyboard focus: `outline: 2px solid var(--filament); outline-offset: 2px`.
- `@media (prefers-reduced-motion: reduce)` disables all transitions.
- Body text contrast ≥ 4.5:1 in both modes. Verify `--ink-body` on `--paper`.
- No layout shift from font loading: `font-display: swap` and a system fallback
  stack on both faces.

---

## 8. Files

| Path | Action |
|---|---|
| `_sass/glownet.scss` | new — the stylesheet below |
| `assets/main.scss` *or* `assets/css/style.scss` | import Minima, then `glownet` |
| `_includes/glyphs.svg` | new — sprite |
| `_layouts/default.html` | inline the sprite once, after `<body>` |
| `_includes/header.html` | override Minima's — masthead per §4 |
| `index.html` | hero + restyled post loop |
| `_config.yml` | title, description, email |

Check which of the two `assets/` paths exists before creating one; Minima 2.x
uses `main.scss`, Minima 3.x uses `css/style.scss`.
