"""Job workers for the benchmark panel.

Thin wrappers around the same pipeline the CLI drives -- nothing here knows
about HTTP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import load_spec
from ..metrics import cheap, embed, judge
from ..pipeline import (
    RunLayout,
    latest_run,
    load_pages,
    load_run_spec,
    new_run,
    run_experiment,
    save_pages,
    snapshot_spec,
)
from ..report import build_all, print_summary
from .jobs import Job


class BenchmarkJobs:
    def __init__(self, spec_path: Path, runs_dir: Path) -> None:
        self.spec_path = spec_path
        self.runs_dir = runs_dir

    def workers(self) -> dict[str, Any]:
        return {
            "bench_run": self._run,
            "bench_generate": self._generate_only,
            "bench_score": self._score_only,
            "bench_report": self._report_only,
        }

    # ------------------------------------------------------------- entry points

    def _run(self, job: Job, log) -> None:
        layout = self._generate(job, log)
        self._score(job, layout, log)
        self._report(job, layout, log)

    def _generate_only(self, job: Job, log) -> None:
        self._generate(job, log)

    def _score_only(self, job: Job, log) -> None:
        self._score(job, self._select_run(job), log)

    def _report_only(self, job: Job, log) -> None:
        self._report(job, self._select_run(job), log)

    # ------------------------------------------------------------------ stages

    def _generate(self, job: Job, log) -> RunLayout:
        params = job.params
        spec = load_spec(self.spec_path)
        layout = new_run(spec, self.runs_dir, tag=params.get("tag") or None)
        snapshot_spec(layout, self.spec_path)
        job.result["run_dir"] = layout.root.name
        log(f"run: {layout.root.name}")
        log("")

        run_experiment(
            spec,
            layout,
            backends=params.get("backends") or spec.experiment.backends,
            strategies=params.get("strategies") or spec.experiment.strategies,
            limit=params.get("limit") or None,
            workers=int(params.get("workers") or 1),
            log=log,
            should_stop=lambda: job.cancelled,
        )
        return layout

    def _select_run(self, job: Job) -> RunLayout:
        name = job.params.get("run")
        if name:
            # Same guard as the /api/runs routes: the name comes from the
            # client, and scoring writes into the folder it names.
            root = (self.runs_dir / name).resolve()
            if not root.is_relative_to(self.runs_dir.resolve()):
                raise FileNotFoundError(f"run {name!r} escapes the runs folder")
            layout = RunLayout(root)
            if not layout.pages_json.is_file():
                raise FileNotFoundError(f"run {name!r} has no pages.json")
        else:
            layout = latest_run(self.runs_dir)
        job.result["run_dir"] = layout.root.name
        return layout

    def _score(self, job: Job, layout: RunLayout, log) -> None:
        params = job.params
        spec = load_run_spec(layout, self.spec_path)
        records = load_pages(layout)
        log(f"scoring {len(records)} pages")

        log("  colour metrics")
        cheap.score_run(records, layout.root)
        if params.get("embed"):
            embed.score_run(records, layout.root, log=log)

        provider = params.get("judge") or spec.judge.provider
        if provider != "none":
            if provider in {r.backend for r in records}:
                log(f"  note: the judge ({provider}) is also a backend under test — "
                    "its scores for its own images are not neutral")
            model = params.get("judge_model") or (
                spec.judge.model if provider == spec.judge.provider else None
            )
            judge.score_run(records, layout.root, spec, provider, model,
                            log=log, should_stop=lambda: job.cancelled)
        else:
            log("  judge: disabled")

        save_pages(layout, records)
        log("")
        print_summary(records, log=log)

    def _report(self, job: Job, layout: RunLayout, log) -> None:
        log("building report")
        spec = load_run_spec(layout, self.spec_path)
        build_all(layout, load_pages(layout), spec.run_name, log=log)
