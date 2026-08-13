# The benchmark

Heldenbuch began as a character-consistency benchmark, and the harness
survives as the app's quality lab. It answers two questions:

1. Which image model keeps a character most consistent across the pages of
   one book?
2. Which way of conditioning the model — words, a reference image, a chained
   previous page — works best?

It lives in the app at `/benchmark` and on the command line.

## Backends

| backend | model family | key |
|---|---|---|
| `openai` | gpt-image-2 | `OPENAI_API_KEY` |
| `gemini` | Gemini image models ("Nano Banana") | `GEMINI_API_KEY` |
| `bfl` | FLUX.2 / FLUX.1 Kontext | `BFL_API_KEY` |
| `comfy` | whatever your local ComfyUI runs | none (local server) |
| `stub` | offline placeholders | none |

A stub run exercises the whole harness for free. Treat its output as a
plumbing test, not a result — the images are synthetic and say nothing about
any real model's ranking.

## Strategies

The independent variable: what the model is given besides the scene
description.

| strategy | what the model gets |
|---|---|
| `text_only` | a written character description, no reference image |
| `sheet_ref` | the character sheet as an image, and no description |
| `sheet_plus_prev` | the sheet **and** the previous page, chained |

The reference arms deliberately leave the written description out. Once you
pass a reference image, repeating identity details in text makes the two
compete, and the model splits the difference.

The app itself conditions pages on a single figure cropped from the sheet —
closest to `sheet_ref`, with the crop described in
[how-it-works.md](how-it-works.md). It does not chain pages.

## Scoring

- **Colour metrics** (free, `metrics/cheap.py`): palette similarity over
  saturated pixels, smoothed circularly across hue so a one-bin shift is not
  read as a total mismatch.
- **Vision judge** (`metrics/judge.py`): scores identity, attributes and
  style 1–5 and lists concrete discrepancies. Providers: `openai`,
  `anthropic`, `gemini`.
- **Embedding similarity** (optional, `metrics/embed.py`): DINOv2, via
  `pip install -e ".[embed]"` — pulls in torch, which is large.

Prefer a judge that is not also under test; a model grading its own images
is not neutral, and the CLI warns when that happens.

## Running it

```bash
python -m heldenbuch run --backends stub --no-judge        # free end-to-end
python -m heldenbuch run --backends gemini bfl --judge openai
python -m heldenbuch run --limit 2 --backends openai       # cheap smoke test
```

`run` is `generate` + `score` + `report`; each stage also exists as its own
command, and `score`/`report` default to the most recent run:

```bash
python -m heldenbuch generate --backends bfl --tag my-experiment
python -m heldenbuch score --judge anthropic
python -m heldenbuch report
```

Useful flags:

- `--limit N` — only the first N scenes, for keeping costs down.
- `--model backend=model` — pin a model, e.g. `--model gemini=gemini-3.1-flash-image`.
- `--shared-sheet PATH` — one character sheet for every backend, instead of
  each drawing its own. Without it, sheet quality and page quality are
  confounded.
- `--workers N` — parallel requests for unchained strategies.
- `--judge / --judge-model / --no-judge / --embed` — scoring control.
- `prompts` — print every prompt without calling anything.

Generation is resumable: existing images are reused, an interrupted run picks
up where it stopped, and deleting one page's file redraws only that page.

## The spec

`benchmark.yaml` is the whole experiment: scenes, backends, strategies,
judge. Each scene stresses a different failure mode — back view, night
lighting, a second character, extreme wide shot, eyes closed. The spec is
snapshotted into every run directory, so a run stays reproducible after the
spec moves on.

Runs land in `runs/<timestamp>/` with every image, a `pages.json` of records
and scores, and a `report.html` with contact sheets.
