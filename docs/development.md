# Development

## Getting a working setup

```bash
pip install -e ".[dev]"
python -m pytest
```

The tests call no APIs, need no keys and touch no network. If they pass, your
environment works.

To run the app without spending anything, pick **Stub (kostenlos)** as the
Bilddienst in the UI, or on the CLI:

```bash
python -m heldenbuch run --backends stub --no-judge
```

The stub backend draws labelled placeholder images offline. Every job path —
hero sheets, page renders, exports — runs end to end with it, so almost all
development never needs a key.

Two practical notes:

- **Restart `serve` after editing Python.** A running server keeps its
  imported modules; you will otherwise chase tracebacks pointing at lines
  that cannot raise them.
- After renaming or moving the package, re-run `pip install -e .` — the
  editable install keeps pointing at the old path.

## Code map

```
src/heldenbuch/
  cli.py            command line: serve, doctor, prompts, run, generate, score, report
  config.py         .env loading, default paths
  llm.py            provider-agnostic text/vision calls (complete_json)
  pricing.py        price tables and the spend ledger
  imageutil.py      shared image helpers (subject masks etc.)

  backends/         image generation, one file per provider
    base.py         the Backend interface: retries, atomic writes, error wording
    openai.py       gpt-image-2 (stdlib only)
    gemini.py       Gemini image models, incl. batch mode (needs [gemini] extra)
    bfl.py          FLUX via Black Forest Labs (stdlib only)
    comfy.py        your own local ComfyUI; setup notes at the top of the file
    stub.py         offline placeholders, free

  book/             the picture-book pipeline
    models.py       Book, Page, Hero, Style, CastMember; atomic JSON persistence
    library.py      the library/ folder: ids, paths, delete guards
    hero.py         photos/description -> character sheet
    look.py         style presets, custom-style translation, styled sheet
    author.py       story, pages, per-page illustration briefs, cast
    cast.py         cast member reference sheets (characters, places, props)
    solo.py         cutting one figure out of a sheet for page conditioning
    illustrate.py   rendering, the quality gate, seam measurement, retries
    layout.py       typesetting and PDF export; print presets
    preflight.py    export checks (resolution, text fit)
    handoff.py      the sheet that goes to the print shop
    narrate.py      audio narration
    scout.py        compares backends on a sample page

  metrics/          scoring for the benchmark
    cheap.py        colour metrics, no API
    judge.py        vision-model judge
    embed.py        optional DINOv2 similarity (needs [embed] extra)

  pipeline.py       benchmark run layout and execution
  prompts.py        benchmark prompt construction
  report.py         benchmark contact sheets and report.html
  types.py          shared dataclasses (GenRequest, OutputSpec, ...)

  web/              the app
    server.py       stdlib http.server; routing, host/origin checks
    bookapi.py      JSON endpoints for the book UI
    bookjobs.py     background job workers for the book UI
    benchjobs.py    background job workers for /benchmark
    jobs.py         the job queue the two share
    static/app.html the whole frontend: one file, no build step
```

There is no web framework and no frontend build. The server is
`http.server` with a dispatch table; the UI is a single HTML file with
inline JavaScript. Long-running work goes through `jobs.py`: the client
starts a job, polls its status, and the worker logs progress lines the
drawer displays.

## Adding an image backend

1. Create `backends/yourprovider.py` with a class extending `Backend`
   (`backends/base.py`). You implement two things:
   - `default_model` — a property returning the model id used when none is
     pinned;
   - `_generate(req) -> tuple[bytes, str]` — take a `GenRequest` (prompt,
     reference image paths, an `OutputSpec` with aspect ratio, size and
     quality), return the image bytes and a short human-readable cost note.
   Set `name` (the short id used on the CLI and in paths) and
   `max_references` (how many reference images one call accepts). Raise
   `BackendError` for failures a retry will not fix; raise anything else for
   transient trouble and the base class retries with backoff. Do not write
   files yourself — the base class writes atomically.
2. Register it in `backends/__init__.py`: add the name to `BACKEND_NAMES`,
   its environment variable to `REQUIRED_KEY` (or `None`), and a branch to
   `get_backend`. Imports stay inside the branch so a missing SDK only breaks
   the backend that needs it.
3. Add its prices to `pricing.py`, or the euro estimates and the ledger will
   be wrong. If you cannot verify the price against the provider's rate
   card, say so in a comment.
4. Prefer the standard library. `openai.py` and `bfl.py` show that a
   full backend fits in a couple hundred lines of stdlib; a new runtime
   dependency needs a reason.

`doctor` should recognise your backend automatically once it is registered —
check with `python -m heldenbuch doctor`.

## Adding a style preset

`PRESETS` in `book/look.py`. Write the description the way the existing ones
are written: concrete pigments, line quality, lighting — not art-history
labels, which image models respond to poorly. Describe **style only**; a
location or a time of day in a style description overrides every page's
brief.

## Adding a print format

`PRESETS` in `book/layout.py`, a `PrintPreset` with trim size, bleed and
safety margin. Take the numbers from the print provider's spec sheet, not
from another preset. Anything with bleed is subject to the measured-dpi hard
error in `preflight.py`.

## Conventions

- **Three runtime dependencies** (numpy, Pillow, PyYAML) is a decision, not
  an accident. Argue for a fourth in the pull request, not in the code.
- Every write to a JSON file or an image goes through temp-file-plus-rename.
  A crash must never tear a file that a resume would then trust.
- Anything that costs money shows an estimate before it runs and writes its
  actual usage to the ledger afterwards.
- User-facing strings in the app are German; code, comments and commit
  messages are English.
- Comments state constraints and reasons, not what the next line does.
- Errors shown in the UI are written for a parent whose page just failed,
  not for the developer — see `explain_provider_error` in `backends/base.py`
  for the tone.
