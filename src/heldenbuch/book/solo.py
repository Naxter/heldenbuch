"""Cutting one figure out of a reference sheet.

A character sheet deliberately shows the same character several times -- four
views for the hero, two for a cast member -- because that is what pins an
identity down while it is being designed. It is the wrong thing to hand an
illustrator drawing a page, for two reasons.

The first is that the model copies what it sees. Ask for one scene while
showing it a picture containing two Trixis and a page comes back with two
Trixis in it; the pages that failed the first finished book failed exactly this
way. No amount of prose in the prompt outweighs the reference itself.

The second is resolution. Four figures across a 1024 px sheet leaves a face
about seventy pixels wide, and that is the entire identity signal available
when the page is drawn at 2624 px for print.

So the sheet stays as it is for designing a character and for judging one, and
pages are conditioned on a single figure cropped out of it. The crop is derived
once and cached beside the sheet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..imageutil import subject_mask

#: A column counts as part of a figure when this share of its height is not
#: paper. Low enough to catch a thin leg, high enough to ignore stray specks.
_COLUMN_INK = 0.02

#: Runs narrower than this share of the sheet are noise, not a figure.
_MIN_RUN = 0.05

#: Breathing room around the crop, as a share of the figure's larger side.
_PAD = 0.06


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Start/stop pairs for each contiguous True stretch."""
    found: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(flags):
        if value and start is None:
            start = index
        elif not value and start is not None:
            found.append((start, index))
            start = None
    if start is not None:
        found.append((start, len(flags)))
    return found


def figure_count(path: Path) -> int:
    """How many separate figures stand on this sheet."""
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
    except Exception:
        return 0
    mask = subject_mask(rgb)
    height, width = mask.shape
    columns = mask.sum(axis=0) > max(1, int(_COLUMN_INK * height))
    return len([r for r in _runs(columns) if r[1] - r[0] > width * _MIN_RUN])


def solo_reference(sheet: Path) -> Path:
    """One figure from `sheet`, cached beside it. The sheet itself if unsure.

    Falling back to the sheet is deliberate: a reference that is merely
    suboptimal is much better than none, and a crop that guessed wrong would
    be worse than either.
    """
    if not sheet.is_file():
        return sheet
    cached = sheet.with_name(f"{sheet.stem}_solo.png")
    if cached.is_file() and cached.stat().st_mtime >= sheet.stat().st_mtime:
        return cached

    try:
        with Image.open(sheet) as image:
            rgb = np.asarray(image.convert("RGB"))
    except Exception:
        return sheet

    mask = subject_mask(rgb)
    height, width = mask.shape
    columns = mask.sum(axis=0) > max(1, int(_COLUMN_INK * height))
    figures = [r for r in _runs(columns) if r[1] - r[0] > width * _MIN_RUN]
    if len(figures) < 2:
        return sheet  # already a single figure, or nothing legible to cut

    # The first view is the front one on every sheet this app produces, and a
    # front view is the most useful thing to condition a page on.
    x0, x1 = figures[0]
    rows = mask[:, x0:x1].sum(axis=1) > max(1, int(_COLUMN_INK * (x1 - x0)))
    lit = np.where(rows)[0]
    if lit.size == 0:
        return sheet
    y0, y1 = int(lit.min()), int(lit.max()) + 1

    pad = int(_PAD * max(x1 - x0, y1 - y0))
    box = (max(0, x0 - pad), max(0, y0 - pad),
           min(width, x1 + pad), min(height, y1 + pad))
    try:
        with Image.open(sheet) as image:
            image.convert("RGB").crop(box).save(cached)
    except Exception:
        return sheet
    return cached
