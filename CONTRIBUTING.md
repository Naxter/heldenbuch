# Contributing

Glad you are here. The short version: keep it small, keep it honest about
money, and never let a child's photo near the repository.

## Setup

```bash
pip install -e ".[dev]"
python -m pytest
```

The tests call no APIs and need no keys. [docs/development.md](docs/development.md)
has the code map and walkthroughs for the common extensions: a new image
backend, a style preset, a print format.

## Ground rules

- **Tests stay offline.** No test may call a paid API or need a key. Use the
  `stub` backend; that is what it is for.
- **Money is visible.** Anything that can spend shows an estimate before it
  runs and records actual usage in the ledger. If you add a model, add its
  price to `pricing.py` — verified against the provider's rate card, or
  marked unverified.
- **Dependencies are a decision.** The runtime needs exactly numpy, Pillow
  and PyYAML; backends stick to the standard library where possible. Make
  the case for a new dependency in the pull request description.
- **Privacy claims must be exact.** The README states precisely when photos
  leave the machine. If your change alters that — a new call that sees
  photos, a new provider in the path — update the claim in the same pull
  request.
- **Writes are atomic.** JSON and images go through temp-file-plus-rename.
  A crash must never leave a truncated file a resume would trust.

## Style

- Python 3.11+, type hints on public functions.
- Comments state constraints and reasons, not what the next line does.
- UI strings are German; code, comments and commits are English.
- Commit messages: a short subject line, at most a couple of lines of body.
- No emoji in code, commits or logs.

## Pull requests

One change per pull request, with tests where behaviour changed. If it
touches layout or export, include a before/after render made with the stub
backend — it is free and reviewers can reproduce it.
