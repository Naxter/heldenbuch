"""Running the experiment: character sheet first, then every page.

Layout of a run directory:

    runs/<run_name>-<timestamp>/
      spec.yaml                    the exact spec this run used
      <backend>/
        sheet.png                  the reference every page is measured against
        <strategy>/
          page_01.png ...
      pages.json                   one record per page, scores appended later
"""

from __future__ import annotations

import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from .backends import Backend, BackendError, get_backend
from .prompts import scene_prompt, sheet_prompt
from .types import BenchmarkSpec, GenRequest, PageRecord, Scene, write_json


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @property
    def pages_json(self) -> Path:
        return self.root / "pages.json"

    @property
    def report_html(self) -> Path:
        return self.root / "report.html"

    def sheet(self, backend: str) -> Path:
        return self.root / backend / "sheet.png"

    def page(self, backend: str, strategy: str, scene_id: str) -> Path:
        return self.root / backend / strategy / f"page_{scene_id}.png"


def new_run(spec: BenchmarkSpec, runs_dir: Path, tag: str | None = None) -> RunLayout:
    stamp = tag or time.strftime("%Y%m%d-%H%M%S")
    layout = RunLayout(runs_dir / f"{spec.run_name}-{stamp}")
    layout.root.mkdir(parents=True, exist_ok=True)
    return layout


def latest_run(runs_dir: Path) -> RunLayout:
    candidates = [p for p in runs_dir.glob("*") if (p / "pages.json").is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"no finished run found under {runs_dir}. Run `heldenbuch run` first."
        )
    return RunLayout(max(candidates, key=lambda p: p.stat().st_mtime))


def snapshot_spec(layout: RunLayout, spec_path: Path) -> None:
    """Copy the spec into the run so results stay interpretable later."""
    if spec_path.is_file():
        shutil.copy2(spec_path, layout.root / "spec.yaml")


def ensure_sheet(
    backend: Backend,
    spec: BenchmarkSpec,
    layout: RunLayout,
    shared_sheet: Path | None = None,
    log=print,
) -> Path:
    """Generate (or reuse) the character sheet this backend will be measured against."""
    target = layout.sheet(backend.name)
    if target.is_file():
        log(f"  sheet: reusing {target.name}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    if shared_sheet is not None:
        # One reference for every backend -- use this when the question is
        # "who reproduces *this* character best", not "who is self-consistent".
        shutil.copy2(shared_sheet, target)
        log(f"  sheet: copied shared sheet from {shared_sheet}")
        return target

    log("  sheet: generating")
    result = backend.generate(
        GenRequest(prompt=sheet_prompt(spec), output=spec.output, kind="sheet"), target
    )
    log(f"  sheet: done in {result.latency_s}s ({result.cost_note})")
    return target


def _page_references(
    strategy: str, sheet: Path, previous: Path | None
) -> tuple[list[Path], list[str]]:
    if strategy == "text_only":
        return [], []
    if strategy == "sheet_plus_prev" and previous is not None:
        return [sheet, previous], ["character sheet", "previous page"]
    return [sheet], ["character sheet"]


def generate_pages(
    backend: Backend,
    spec: BenchmarkSpec,
    strategy: str,
    sheet: Path,
    layout: RunLayout,
    scenes: list[Scene],
    workers: int = 1,
    log=print,
    should_stop: Callable[[], bool] | None = None,
) -> list[PageRecord]:
    """Generate every page for one (backend, strategy) pair."""
    chained = strategy == "sheet_plus_prev"
    stop = should_stop or (lambda: False)

    def build(scene: Scene, previous: Path | None) -> PageRecord:
        references, roles = _page_references(strategy, sheet, previous)
        prompt = scene_prompt(spec, scene, strategy, has_previous=previous is not None)
        target = layout.page(backend.name, strategy, scene.id)
        record = PageRecord(
            scene_id=scene.id,
            backend=backend.name,
            strategy=strategy,
            image_path=str(target.relative_to(layout.root)).replace("\\", "/"),
            prompt=prompt,
            stress=scene.stress,
        )
        if target.is_file():
            log(f"    page {scene.id}: reusing existing image")
            return record
        if stop():
            record.error = "cancelled"
            return record
        try:
            result = backend.generate(
                GenRequest(
                    prompt=prompt,
                    reference_images=references,
                    output=spec.output,
                    reference_roles=roles,
                ),
                target,
            )
            log(f"    page {scene.id}: {result.latency_s}s  {result.cost_note}")
        except (BackendError, Exception) as exc:  # keep going; one bad page is data too
            record.error = f"{type(exc).__name__}: {exc}"
            log(f"    page {scene.id}: FAILED -- {record.error}")
        return record

    if chained or workers <= 1:
        records: list[PageRecord] = []
        previous: Path | None = None
        for scene in scenes:
            record = build(scene, previous)
            records.append(record)
            if chained and record.error is None:
                previous = layout.page(backend.name, strategy, scene.id)
        return records

    # Unchained strategies have no ordering constraint, so fan them out.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda s: build(s, None), scenes))


def run_experiment(
    spec: BenchmarkSpec,
    layout: RunLayout,
    backends: list[str],
    strategies: list[str],
    models: dict[str, str] | None = None,
    limit: int | None = None,
    shared_sheet: Path | None = None,
    workers: int = 1,
    log=print,
    should_stop: Callable[[], bool] | None = None,
) -> list[PageRecord]:
    models = models or {}
    stop = should_stop or (lambda: False)
    scenes = spec.scenes[:limit] if limit else spec.scenes
    total = len(backends) * len(strategies) * len(scenes)
    log(f"{total} images to generate ({len(backends)} backends x "
        f"{len(strategies)} strategies x {len(scenes)} scenes)\n")

    records: list[PageRecord] = []
    for name in backends:
        if stop():
            log("cancelled -- keeping everything generated so far")
            break
        log(f"[{name}]")
        try:
            backend = get_backend(name, models.get(name))
        except BackendError as exc:
            log(f"  skipped: {exc}")
            continue

        try:
            sheet = ensure_sheet(backend, spec, layout, shared_sheet, log=log)
        except Exception as exc:
            log(f"  skipped: could not make a character sheet -- {exc}")
            continue

        for strategy in strategies:
            if stop():
                break
            log(f"  strategy: {strategy}")
            records.extend(
                generate_pages(
                    backend,
                    spec,
                    strategy,
                    sheet,
                    layout,
                    scenes,
                    workers=workers,
                    log=log,
                    should_stop=should_stop,
                )
            )
        log("")

    write_json(layout.pages_json, [r.to_dict() for r in records])
    return records


def load_pages(layout: RunLayout) -> list[PageRecord]:
    import json

    raw = json.loads(layout.pages_json.read_text(encoding="utf-8"))
    return [PageRecord(**item) for item in raw]


def save_pages(layout: RunLayout, records: list[PageRecord]) -> None:
    write_json(layout.pages_json, [r.to_dict() for r in records])


def load_run_spec(layout: RunLayout, fallback: Path) -> BenchmarkSpec:
    """Prefer the spec snapshot inside the run, so scoring matches generation."""
    snapshot = layout.root / "spec.yaml"
    source = snapshot if snapshot.is_file() else fallback
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    return BenchmarkSpec.from_dict(raw)


def eprint(*args) -> None:
    print(*args, file=sys.stderr)
