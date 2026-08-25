# Heldenbuch

![heldenbuch — personalised children's picture books](docs/heldenbuch-banner.png)

[![ci](https://github.com/Naxter/heldenbuch/actions/workflows/ci.yml/badge.svg)](https://github.com/Naxter/heldenbuch/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Make personalised picture books for your own child — with their photo or
without — and get a file a print shop will accept.

The hard part of an AI picture book is not writing it. It is that the child
looks like a different child on page seven. Heldenbuch solves that by getting
the character right **once**, then pointing every page at that one drawing.

## Try it without an API key

```bash
pip install -e .
python -m heldenbuch demo
python -m heldenbuch serve --library demo_library
```

That builds a finished eight-page book — hero, style, cast, pages, export —
and opens it at `http://127.0.0.1:8765`. It writes to `demo_library/` and
never touches your own books. The pictures are offline placeholders drawn by
the `stub` backend, so this shows the shape of a book rather than the quality
of an image model; everything else is the real thing. `--fresh` rebuilds it,
and deleting `demo_library/` removes it.

For a real book you need one API key — see [Setup](#setup).

The interface speaks German or English; switch it any time under the gear icon.
Books themselves can be written in any of the supported languages, independent
of the interface.

---

## How it works

Four steps. Steps 1 and 2 you do once per child; after that a new book is a
sentence away. The full pipeline, including how consistency is actually
achieved, is in **[docs/how-it-works.md](docs/how-it-works.md)**.

**1. The hero.** Upload two to four photos, or just describe the character.
A vision model writes down what an illustrator needs — hair, eyes, skin tone,
build, an outfit — and an image model draws that as a **character sheet**: the
same character from four angles on a plain background. You get three versions
and pick one.

**2. The look.** Pick a style from the presets, or describe your own wish
("like an old paper cut-out") and it gets translated into something an image
model actually responds to. Either way it immediately renders *your character
in that style*, so you judge the real combination. When you like it, the
character sheet is redrawn in that style — identity and look locked into one
reference image.

**3. The story.** One line is enough: *"Claudio loses his rubber boot in the
mud."* Leave it blank and it invents the idea. It writes the whole story
at the right reading level for the age you choose, splits it into pages, and
writes both the text and the illustration instruction for each page. Every page
is editable, and every page can be redrawn on its own.

Pick several languages and each is **written**, not translated — same story,
same page breaks, same beat, but the rhythm works in each language. The
pictures are shared, so a second language costs a few cents of text.

**4. The book.** It draws every page from the locked reference — each page is
conditioned on a single cropped figure from the sheet, which is what keeps the
character from drifting or duplicating — then checks every page against the
reference and flags any that drifted. One click redraws just that page. Then
it exports a PDF.

Text is never drawn by the image model. Image models produce convincing-looking
gibberish instead of letters, in every language. It is typeset afterwards with
a real font (Georgia by default — it has Cyrillic, so Russian sets properly).

---

## What comes out

| Format | For |
|---|---|
| **Print shop — square, 21.6 cm** | The standard children's book trim. 3.175 mm bleed on all sides, text inside the 12.7 mm safety margin, 300 dpi, no crop marks, one PDF — what Lulu, Gelato and most print-on-demand shops ask for. Cover exported separately as JPG. |
| **Print shop — children's book, 15.6 × 14.8 cm** | epubli's children's format. Smaller and cheaper, prints single copies. |
| **Home printing — A4 landscape** | Picture left, text right, no bleed. Any normal printer. |
| **Reading on screen** | Landscape, small file. |

One PDF per language. Print formats are padded to a multiple of four pages,
which is how books are actually bound.

**Render quality** matters for print. *Draft* draws at 1024 px — fast and
cheap, right for while the story is still moving. *Print quality* draws at
2624 px, which is exactly 300 dpi across a 21.6 cm page with its bleed. The
export measures the real pixels of every page and refuses a print export whose
effective resolution is too low, instead of letting the PDF claim 300 dpi.

With the optional `print` extra installed (`pip install -e ".[print]"`),
exports also carry an embedded sRGB profile, so a press knows what the colours
are supposed to mean instead of guessing.

---

## Setup

Python 3.11+.

```bash
pip install -e .
```

Copy `.env.example` to `.env` and add an OpenAI key — that alone runs
everything: story, pictures, narration and the consistency check. Optional:
`GEMINI_API_KEY` or `BFL_API_KEY` for other image models, `ANTHROPIC_API_KEY`
for a second text provider so one company is not grading its own drawings.

```bash
python -m heldenbuch doctor
```

tells you which backends are ready and which keys are missing.

Core dependencies are numpy, Pillow and PyYAML — three, on purpose. The web
app and the OpenAI and FLUX backends use only the standard library. Gemini
needs `pip install -e ".[gemini]"`.

**Drawing locally, for free:** the `comfy` backend sends jobs to your own
ComfyUI (FLUX.2 klein runs on a 12 GB card). No key, no per-image cost. It
appears as an image service automatically whenever ComfyUI is running. One-time
setup: export your working workflow in API format and drop in placeholders —
instructions at the top of `src/heldenbuch/backends/comfy.py`.

Everything lives in plain folders under `library/`:

```
library/heroes/<id>/    hero.json, photos/, sheet_*.png
library/styles/<id>/    style.json, preview_*.png
library/books/<id>/     book.json, pages/, export/
```

A book is one directory. Back it up by copying it. `library/` is gitignored —
it contains photos of a real child.

---

## What it costs, and where photos go

Every button that spends money shows a euro estimate before you click it, and
every provider response's usage is written to a per-book ledger you can open.
The `stub` backend draws labelled placeholder images for free, so you can walk
the whole app — story, pages, export — without a key or a cent.

Photos are stored only on your machine, but creating a character sheet does
send them out, and it is worth being precise about when:

- once to the **text provider**, which writes the illustrator's description;
- once **per character sheet drawn** to the **image provider** — three
  variants by default, and again each time you ask for more.

Depending on your configuration those are two different companies. After the
sheet exists, no photo is sent again: every page illustration is conditioned
on the drawing, never on a photo. If the `comfy` backend draws the sheets,
that drawing happens entirely on your own machine; only the written
description step still uses a text provider — or upload no photos and type
the description yourself, and nothing personal leaves the house at all.

---

## The lab

The app has a second half at `/benchmark`: the harness this project was built
on. It answers "which image model keeps a character most consistent, and which
way of conditioning it works best" by generating the same book through several
backends and strategies, then scoring every page.

```bash
python -m heldenbuch run --backends stub --no-judge     # free smoke test
python -m heldenbuch run --backends gemini bfl --judge openai
```

Backends, strategies, scoring and the CLI are documented in
**[docs/benchmark.md](docs/benchmark.md)**.

---

## Developing

The code map, how to run the tests, and how to add an image backend, a style
or a print format are in **[docs/development.md](docs/development.md)**.
Ground rules for contributions are in **[CONTRIBUTING.md](CONTRIBUTING.md)**.

```bash
pip install -e ".[dev]"
python -m pytest
```

The tests call no APIs and need no keys.

---

## Security and scope

This is a single-user tool that runs on your own machine. It binds to
`127.0.0.1` and **has no authentication of any kind** — anyone who can reach
the port can spend your API credit. Do not put it on a public interface, and
do not port-forward it. It refuses requests whose `Host` header is not
localhost and rejects cross-origin writes, which stops a web page you visit
from driving it behind your back; that is the whole of the threat model.

Uploaded photos live in `library/` in the clear, and API keys live in `.env`
in the clear. Both are gitignored. Anything you can read on that machine, a
program running as you can read too.

Costs are real and are yours: every button that spends shows an estimate
first, and a per-book ledger records what each call actually cost, but nothing
here can stop a provider charging you. Set a budget cap on a render if that
matters to you.

## Legal, before anything leaves the house

Purely AI-generated images get no copyright protection under German law —
protection needs a genuine human creative contribution. The EU AI Act's
labelling duty for AI-generated content applies from 2 August 2026. And
uploading a photo of a child to a foreign API is a GDPR question the moment
other people's children are involved. None of that affects a book you make for
your own child; all of it affects anything you sell or publish.

## License

MIT — see [LICENSE](LICENSE).
