"""A run has to say what it is: which model drew it, and whether it is real.

The only complete run this project ever produced was stub output, rendered
identically to a paid one. Nothing about the report distinguished them.
"""

import json

from heldenbuch.pipeline import RunLayout
from heldenbuch.report import build_all
from heldenbuch.types import PageRecord


def _record(backend="stub", **kwargs):
    base = dict(scene_id="s1", backend=backend, strategy="sheet_ref",
                image_path=f"{backend}/sheet_ref/page_s1.png", prompt="p")
    base.update(kwargs)
    return PageRecord(**base)


def test_a_stub_run_is_marked_synthetic(tmp_path):
    layout = RunLayout(tmp_path / "run")
    layout.root.mkdir(parents=True)
    build_all(layout, [_record()], "demo", log=lambda *a: None)

    report = layout.report_html.read_text(encoding="utf-8")
    assert "Synthetic run" in report
    assert json.loads((layout.root / "summary.json").read_text())["synthetic"] is True


def test_a_paid_run_carries_provenance_and_no_warning(tmp_path):
    layout = RunLayout(tmp_path / "run")
    layout.root.mkdir(parents=True)
    records = [_record(backend="bfl", model="flux-2-pro",
                       created="2026-08-13T21:00:00", usd=0.04)]
    build_all(layout, records, "demo", log=lambda *a: None)

    report = layout.report_html.read_text(encoding="utf-8")
    assert "Synthetic run" not in report
    assert "flux-2-pro" in report
    assert "2026-08-13T21:00:00" in report

    summary = json.loads((layout.root / "summary.json").read_text())
    assert summary["synthetic"] is False
    assert summary["models"] == ["flux-2-pro"]
    assert summary["usd"] == 0.04


def test_a_mixed_run_still_warns(tmp_path):
    layout = RunLayout(tmp_path / "run")
    layout.root.mkdir(parents=True)
    build_all(layout, [_record(), _record(backend="bfl", model="flux-2-pro")],
              "demo", log=lambda *a: None)

    report = layout.report_html.read_text(encoding="utf-8")
    assert "Some pages here were" in report
