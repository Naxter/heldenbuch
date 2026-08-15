"""The export must not scale its memory with the length of the book.

Two things used to make a print export enormous: every finished page was kept
in a list until the PDF was written, and the border check widened a whole
4096 px illustration to float64 -- 400 MB in one array, the single largest
allocation in the program.
"""

import tracemalloc

import numpy as np
import pytest
from PIL import Image

from heldenbuch.book.layout import PRESETS, export_pdf, flat_border
from heldenbuch.book.models import Book, Page


def _book(tmp_path, pages: int) -> tuple[Book, object]:
    art = tmp_path / "art.png"
    Image.new("RGB", (900, 900), "seagreen").save(art)
    book = Book(title={"de": "Test"}, languages=["de"],
                pages=[Page(index=i, text={"de": f"Seite {i}"}, image="art.png")
                       for i in range(1, pages + 1)])
    book.cover = "art.png"
    return book, (lambda rel: tmp_path / rel)


def _peak_mb(book, resolve, tmp_path) -> float:
    tracemalloc.start()
    try:
        export_pdf(book, "de", PRESETS["print_kinderbuch"], resolve,
                   tmp_path / "out.pdf", log=lambda *a: None)
        return tracemalloc.get_traced_memory()[1] / 1024 / 1024
    finally:
        tracemalloc.stop()


def test_peak_memory_does_not_grow_with_the_page_count(tmp_path):
    short_book, resolve = _book(tmp_path, 4)
    long_book, _ = _book(tmp_path, 16)

    short = _peak_mb(short_book, resolve, tmp_path)
    long = _peak_mb(long_book, resolve, tmp_path)

    # One page of this preset is 10 MB, so keeping twelve more would show up
    # as well over 100 MB of growth. Allow one page of slack for the writer.
    assert long < short + 20, f"{short:.0f} MB for 4 pages, {long:.0f} MB for 16"


def test_the_border_check_does_not_widen_the_whole_picture(tmp_path):
    art = tmp_path / "big.png"
    Image.new("RGB", (2000, 2000), "white").save(art)

    tracemalloc.start()
    try:
        flat_border(art)
        peak = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    finally:
        tracemalloc.stop()

    # 2000 x 2000 x 3 is 11 MB as bytes and 92 MB as float64.
    assert peak < 40, f"peak {peak:.0f} MB -- the array is being widened again"


def test_a_real_flat_border_is_still_found(tmp_path):
    """The saving must not cost the measurement it was protecting."""
    array = np.zeros((400, 400, 3), dtype=np.uint8)
    rng = np.random.default_rng(7)
    array[:300] = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    array[300:] = (245, 230, 201)  # the cream band a real page arrived with
    art = tmp_path / "banded.png"
    Image.fromarray(array).save(art)

    left, top, right, bottom = flat_border(art)
    assert bottom == pytest.approx(100, abs=2)
    assert (left, top, right) == (0, 0, 0)
