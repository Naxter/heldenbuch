"""One figure out of a reference sheet.

A sheet shows the same character several times on purpose, which is right for
designing an identity and wrong for drawing a page: the model copies what it
sees, and the pages that failed the first finished book failed exactly that
way -- the same character drawn twice on one page.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from heldenbuch.book.solo import figure_count, solo_reference


def _sheet(path, figures, size=(1200, 800), gap=60):
    """Dark figures on off-white paper, evenly spaced, as the real sheets are."""
    array = np.full((size[1], size[0], 3), 250, dtype=np.uint8)
    slot = size[0] // figures
    for i in range(figures):
        x0 = i * slot + gap
        x1 = (i + 1) * slot - gap
        array[120:size[1] - 120, x0:x1] = (40, 90, 160)
    Image.fromarray(array).save(path)
    return path


@pytest.mark.parametrize("figures", [2, 3, 4])
def test_a_multi_figure_sheet_is_cut_down_to_one(tmp_path, figures):
    sheet = _sheet(tmp_path / f"sheet_{figures}.png", figures)
    assert figure_count(sheet) == figures

    solo = solo_reference(sheet)
    assert solo != sheet
    assert figure_count(solo) == 1
    # Roughly one slot wide, not the whole sheet.
    assert Image.open(solo).width < Image.open(sheet).width / (figures - 0.5)


def test_a_single_figure_sheet_is_left_alone(tmp_path):
    """Falling back to the original beats a crop that guessed wrong."""
    sheet = _sheet(tmp_path / "one.png", 1)
    assert solo_reference(sheet) == sheet


def test_a_place_reference_is_never_cropped(tmp_path):
    """A place is one wide establishing view; cropping throws the setting away."""
    path = tmp_path / "place.png"
    rng = np.random.default_rng(3)
    Image.fromarray(rng.integers(0, 255, (800, 1200, 3), dtype=np.uint8)).save(path)
    assert solo_reference(path) == path


def test_the_crop_is_cached_and_refreshed(tmp_path):
    sheet = _sheet(tmp_path / "sheet.png", 4)
    first = solo_reference(sheet)
    assert first.is_file() and first != sheet
    stamp = first.stat().st_mtime_ns

    assert solo_reference(sheet) == first
    assert first.stat().st_mtime_ns == stamp, "should not be redone"

    # Redrawing the sheet invalidates the crop.
    import os
    import time

    time.sleep(0.01)
    _sheet(sheet, 2)
    os.utime(sheet, None)
    assert solo_reference(sheet).stat().st_mtime_ns >= stamp


def test_an_unreadable_sheet_falls_back(tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    assert solo_reference(broken) == broken
    assert solo_reference(tmp_path / "absent.png") == tmp_path / "absent.png"
