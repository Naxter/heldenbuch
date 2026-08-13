# StoryTime

Make personalised picture books for your own child — with their photo or
without — and get a file a print shop will accept.

The hard part of an AI picture book is not writing it. It is that the child
looks like a different child on page seven. StoryTime solves that by getting
the character right **once**, then pointing every page at that one drawing.

```bash
python -m storytime serve
```

---

## How it works

Four steps. Steps 1 and 2 you do once per child; after that a new book is a
sentence away.

**1. Der Held.** Upload two to four photos, or just describe the character.
A vision model writes down what an illustrator needs — hair, eyes, skin tone,
build, an outfit — and an image model draws that as a **character sheet**: the
same character from four angles on a plain background. You get three versions
and pick one.

The photos stay on your machine. They are sent once, for this step. No page
illustration ever sees a photo; they see the drawing.

**2. Der Look.** Pick a style from eight presets, or describe your own wish
("wie ein alter Scherenschnitt") and it gets translated into something an image
model actually responds to. Either way it immediately renders *your character
in that style*, so you judge the real combination. When you like it, the
character sheet is redrawn in that style — identity and look locked into one
reference image.

**3. Die Geschichte.** One line is enough: *"Claudio verliert seinen Gummistiefel
im Matsch."* Leave it blank and it invents the idea. It writes the whole story
at the right reading level for the age you choose, splits it into pages, and
writes both the text and the illustration instruction for each page. Every page
is editable, and every page has a "neu" button.

Pick several languages and each is **written**, not translated — same story,
same page breaks, same beat, but the rhythm works in each language. The
pictures are shared, so a second language costs a few cents of text.

**4. Das Buch.** It draws every page from the locked reference, then checks
each one against it and flags any page where the character drifted. One click
redraws just that page. Then it exports a PDF.

---

## What comes out

| Format | For |
|---|---|
| **Druckerei — quadratisch 21,6 cm** | The standard children's book trim. 3,175 mm bleed on all sides, text inside the 12,7 mm safety margin, 300 dpi, no crop marks, one PDF — what Lulu, Gelato and most print-on-demand shops ask for. Cover exported separately as JPG. |
| **Druckerei — Kinderbuch 15,6 × 14,8 cm** | epubli's children's format. Smaller and cheaper, prints single copies. |
| **Zuhause drucken — A4 quer** | Picture left, text right, no bleed. Any normal printer. |
| **Zum Vorlesen am Bildschirm** | Landscape, small file. |

One PDF per language. Print formats are padded to a multiple of four pages,
which is how books are actually bound.

**Render quality** matters for print. *Entwurf* draws at 1024 px — fast and
cheap, right for while the story is still moving. *Druckqualität* draws at
2624 px, which is exactly 300 dpi across a 21,6 cm page with its bleed. The
export warns you if the pictures are too small for the format you picked.

Text is never drawn by the image model. Image models produce convincing-looking
gibberish instead of letters, in every language. It is typeset afterwards with
a real font (Georgia by default — it has Cyrillic, so Russian sets properly).

---

## Setup

Python 3.11+.

```bash
pip install -e .
```

Copy `.env.example` to `.env` and add an OpenAI key — that alone runs
everything. Optional: `GEMINI_API_KEY` or `BFL_API_KEY` for other image models,
`ANTHROPIC_API_KEY` for a different writer.

```bash
python -m storytime doctor
```

Core dependencies are numpy, Pillow and PyYAML. The web app and the OpenAI and
FLUX backends use only the standard library. Gemini needs
`pip install -e ".[gemini]"`.

**Drawing locally, for free:** the `comfy` backend sends jobs to your own
ComfyUI (FLUX.2 klein runs on a 12 GB card). No key, no cost — and the photos
never leave the machine, not even for the character sheet. It appears as a
Bilddienst automatically whenever ComfyUI is running. One-time setup: export
your working workflow in API format and drop in placeholders — instructions at
the top of `src/storytime/backends/comfy.py`.

Everything lives in plain folders under `library/`:

```
library/heroes/<id>/    hero.json, photos/, sheet_*.png
library/styles/<id>/    style.json, preview_*.png
library/books/<id>/     book.json, pages/, export/
```

A book is one directory. Back it up by copying it. `library/` is gitignored —
it contains photos of a real child.

---

## Das Labor

The app has a second half at `/benchmark`: the harness this was built on. It
answers "which image model keeps a character most consistent, and which way of
conditioning it works best" by generating the same book through several
backends and three strategies, then scoring every page.

<details>
<summary>Details</summary>

**Backends** — `openai` (gpt-image-2), `gemini` (Nano Banana),
`bfl` (FLUX.2), and `stub` (offline, no key, no cost).

**Strategies** — the independent variable:

| strategy | what the model gets |
|---|---|
| `text_only` | a written description, no reference image |
| `sheet_ref` | the character sheet as an image, and no description |
| `sheet_plus_prev` | the sheet **and** the previous page, chained |

The reference arms deliberately leave the description out. Once you pass a
reference image, repeating identity details in text makes the two compete and
the model splits the difference. The app uses `sheet_plus_prev`.

**Scoring** — free colour metrics (palette similarity over saturated pixels,
smoothed circularly across hue so a one-bin shift is not read as a total
mismatch), plus a vision model that scores identity, attributes and style 1–5
and lists concrete discrepancies. Optional DINOv2 similarity with
`pip install -e ".[embed]"`.

Prefer a judge that is not also under test — the CLI warns when it is.

```bash
python -m storytime run --backends stub --no-judge
```

```bash
python -m storytime run --backends gemini bfl --judge openai
```

Also: `generate`, `score`, `report`, `prompts`, and `--limit N` to keep costs
down. Generation is resumable — existing images are reused, so an interrupted
run picks up where it stopped, and deleting one page's file redraws only that
page.

`benchmark.yaml` is the whole experiment. Its scenes each stress a different
failure mode: back view, night lighting, a second character, extreme wide shot,
eyes closed.

</details>

---

## Rechtliches, falls das Haus verlässt

Purely AI-generated images get no copyright protection under German law —
protection needs a genuine human creative contribution. The EU AI Act's
labelling duty for AI-generated content applies from 2 August 2026. And
uploading a photo of a child to a foreign API is a GDPR question the moment
other people's children are involved. None of that affects a book you make for
your own child; all of it affects anything you sell or publish.
