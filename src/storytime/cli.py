"""Command line interface.

    storytime serve               open the local control panel in a browser
    storytime doctor              what is installed and which keys are set
    storytime prompts             print the prompts, call nothing
    storytime run                 generate + score + report
    storytime generate            generate images only
    storytime score               score an existing run
    storytime report              rebuild the report from pages.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .backends import BACKEND_NAMES, REQUIRED_KEY
from .config import DEFAULT_LIBRARY, DEFAULT_RUNS_DIR, DEFAULT_SPEC, load_dotenv, load_spec
from .metrics import cheap, embed, judge
from .pipeline import (
    RunLayout,
    latest_run,
    load_pages,
    load_run_spec,
    new_run,
    run_experiment,
    save_pages,
    snapshot_spec,
)
from .prompts import scene_prompt, sheet_prompt
from .report import build_all, print_summary
from .types import STRATEGIES


def _parse_models(pairs: list[str] | None) -> dict[str, str]:
    models: dict[str, str] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--model expects backend=model, got {pair!r}")
        backend, _, model = pair.partition("=")
        models[backend.strip()] = model.strip()
    return models


def _resolve_layout(args) -> RunLayout:
    if args.run:
        layout = RunLayout(Path(args.run))
        if not layout.pages_json.is_file():
            raise SystemExit(f"{layout.pages_json} not found -- is that a run directory?")
        return layout
    return latest_run(Path(args.runs_dir))


# --------------------------------------------------------------------------- commands


def cmd_doctor(args) -> int:
    load_dotenv()
    print(f"storytime {__version__}   python {sys.version.split()[0]}")
    print(f"spec:      {args.spec}  {'(found)' if Path(args.spec).is_file() else '(MISSING)'}")
    print(f"runs dir:  {args.runs_dir}")
    print("\nbackends:")
    for name in BACKEND_NAMES:
        key = REQUIRED_KEY[name]
        if name == "comfy":
            from .backends.comfy import _base_url, _workflow_path, is_available

            if not is_available():
                status = f"NOT ready -- no ComfyUI server at {_base_url()} (optional, local)"
            elif not _workflow_path().is_file():
                status = f"NOT ready -- ComfyUI answers, but {_workflow_path()} is missing"
            else:
                status = f"ready (ComfyUI at {_base_url()})"
        elif key is None:
            status = "ready (no key needed)"
        elif os.environ.get(key, "").strip():
            status = f"ready ({key} set)"
        else:
            status = f"NOT ready -- {key} is missing"
        print(f"  {name:<8} {status}")

    print("\noptional packages:")
    for module, purpose in (
        ("google.genai", "gemini backend and gemini judge"),
        ("torch", "embedding metric"),
        ("transformers", "embedding metric"),
    ):
        try:
            __import__(module)
            print(f"  {module:<20} installed   ({purpose})")
        except ImportError:
            print(f"  {module:<20} missing     ({purpose})")
    return 0


def cmd_prompts(args) -> int:
    spec = load_spec(Path(args.spec))
    print("=" * 72)
    print("CHARACTER SHEET PROMPT")
    print("=" * 72)
    print(sheet_prompt(spec))
    scene = spec.scenes[0]
    for strategy in spec.experiment.strategies:
        for has_previous in (False, True) if strategy == "sheet_plus_prev" else (False,):
            print("\n" + "=" * 72)
            label = strategy + (" (with previous page)" if has_previous else "")
            print(f"SCENE PROMPT -- {label} -- scene {scene.id}")
            print("=" * 72)
            print(scene_prompt(spec, scene, strategy, has_previous))
    return 0


def cmd_generate(args) -> int:
    spec = load_spec(Path(args.spec))
    backends = args.backends or spec.experiment.backends
    strategies = args.strategies or spec.experiment.strategies

    load_dotenv()
    for name in backends:
        key = REQUIRED_KEY.get(name)
        if key and not os.environ.get(key, "").strip():
            print(f"warning: backend '{name}' needs {key}, which is not set -- it will be skipped\n")

    layout = new_run(spec, Path(args.runs_dir), tag=args.tag)
    snapshot_spec(layout, Path(args.spec))
    args.layout = layout  # so `run` can hand it straight to score/report
    print(f"run: {layout.root}\n")

    records = run_experiment(
        spec,
        layout,
        backends=backends,
        strategies=strategies,
        models=_parse_models(args.model),
        limit=args.limit,
        shared_sheet=Path(args.shared_sheet) if args.shared_sheet else None,
        workers=args.workers,
    )
    print(f"{len(records)} pages recorded in {layout.pages_json}")
    return 0


def cmd_score(args, layout: RunLayout | None = None) -> int:
    layout = layout or _resolve_layout(args)
    spec = load_run_spec(layout, Path(args.spec))
    records = load_pages(layout)
    print(f"scoring {len(records)} pages in {layout.root}")

    print("  colour metrics")
    cheap.score_run(records, layout.root)

    if args.embed:
        embed.score_run(records, layout.root)

    provider = "none" if args.no_judge else (args.judge or spec.judge.provider)
    if provider != "none":
        tested = {r.backend for r in records}
        if provider in tested:
            print(f"  note: the judge ({provider}) is also one of the backends under test. "
                  "Its scores for its own images are not neutral.")
        # The spec's model belongs to the spec's provider. If the provider was
        # overridden on the command line, fall back to that provider's default.
        model = args.judge_model
        if model is None and provider == spec.judge.provider:
            model = spec.judge.model
        judge.score_run(records, layout.root, spec, provider, model)

    save_pages(layout, records)
    print()
    print_summary(records)
    return 0


def cmd_report(args, layout: RunLayout | None = None) -> int:
    layout = layout or _resolve_layout(args)
    spec = load_run_spec(layout, Path(args.spec))
    records = load_pages(layout)
    print(f"building report for {layout.root}")
    path = build_all(layout, records, spec.run_name)
    print(f"\nopen: {path}")
    return 0


def cmd_serve(args) -> int:
    from .web import serve

    load_dotenv()
    serve(
        spec_path=Path(args.spec),
        runs_dir=Path(args.runs_dir),
        library_dir=Path(args.library),
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )
    return 0


def cmd_run(args) -> int:
    code = cmd_generate(args)
    if code:
        return code
    layout = getattr(args, "layout", None) or latest_run(Path(args.runs_dir))
    code = cmd_score(args, layout)
    if code:
        return code
    return cmd_report(args, layout)


# --------------------------------------------------------------------------- parser


def _default_port() -> int:
    """Port 8765, unless the environment names one.

    Anything that starts the app for you -- a container, a supervisor, an
    editor's run configuration -- hands the port over in `PORT`. Reading it
    means those do not each need their own command line.
    """
    raw = (os.environ.get("PORT") or "").strip()
    if raw.isdigit() and 0 <= int(raw) <= 65535:
        return int(raw)
    return 8765


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="storytime",
        description="Benchmark character consistency across image backends.",
    )
    parser.add_argument("--version", action="version", version=f"storytime {__version__}")
    parser.add_argument("--spec", default=str(DEFAULT_SPEC), help="benchmark spec YAML")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR), help="where runs are written")

    sub = parser.add_subparsers(dest="command", required=True)

    def add_generation_args(p):
        p.add_argument("--backends", nargs="+", choices=BACKEND_NAMES,
                       help="override the backends listed in the spec")
        p.add_argument("--strategies", nargs="+", choices=STRATEGIES,
                       help="override the strategies listed in the spec")
        p.add_argument("--model", action="append", metavar="BACKEND=MODEL",
                       help="pin a model, e.g. --model gemini=gemini-3.1-flash-image")
        p.add_argument("--limit", type=int, help="only the first N scenes (cheap smoke test)")
        p.add_argument("--workers", type=int, default=1,
                       help="parallel requests for unchained strategies (default 1)")
        p.add_argument("--shared-sheet", metavar="PATH",
                       help="use one character sheet for every backend instead of "
                            "letting each generate its own")
        p.add_argument("--tag", help="name the run directory instead of using a timestamp")

    def add_scoring_args(p):
        p.add_argument("--judge", choices=("openai", "anthropic", "gemini", "none"),
                       help="override the judge provider from the spec")
        p.add_argument("--judge-model", help="override the judge model")
        p.add_argument("--no-judge", action="store_true", help="colour metrics only")
        p.add_argument("--embed", action="store_true",
                       help="also compute DINOv2 similarity (needs storytime[embed])")

    def add_run_selector(p):
        p.add_argument("--run", metavar="DIR", help="run directory (default: the most recent)")

    p = sub.add_parser("serve", help="open the app in a browser")
    p.add_argument("--library", default=str(DEFAULT_LIBRARY),
                   help="where heroes, styles and books are stored")
    p.add_argument("--port", type=int, default=_default_port(),
                   help="port (default 8765, or $PORT if set; 0 picks a free one)")
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address; the panel has no authentication, so keep it on localhost")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("doctor", help="check keys and installed packages")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("prompts", help="print the prompts without calling any API")
    p.set_defaults(func=cmd_prompts)

    p = sub.add_parser("generate", help="generate images only")
    add_generation_args(p)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("score", help="score an existing run")
    add_run_selector(p)
    add_scoring_args(p)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("report", help="rebuild contact sheets and report.html")
    add_run_selector(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("run", help="generate, score and report in one go")
    add_generation_args(p)
    add_scoring_args(p)
    p.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # `run` needs the run-selector default that only score/report declare.
    if not hasattr(args, "run"):
        args.run = None
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
