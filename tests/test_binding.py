"""Binding follows the page count, and the format has to exist at the shop.

A 16-page picture book is stapled, not glued. The cover used to be built with
a glued spine anyway, and spine text switched on at 79 pages -- inside the
range print shops refuse to print a spine at all.
"""

import pytest
from PIL import Image

from heldenbuch.book.handoff import binding_note, trim_supported, trim_warning
from heldenbuch.book.layout import (
    PRESETS,
    binding_for,
    has_spine,
    render_wrap_cover,
    spine_text_allowed,
)
from heldenbuch.book.models import Book, Page


class TestBindingChoice:
    @pytest.mark.parametrize("pages", [12, 16, 20, 24, 31])
    def test_a_picture_book_is_saddle_stitched(self, pages):
        assert binding_for(pages) == "saddle_stitch"
        assert has_spine(pages) is False

    @pytest.mark.parametrize("pages", [32, 64, 200])
    def test_a_thick_book_is_perfect_bound(self, pages):
        assert binding_for(pages) == "perfect_bound"
        assert has_spine(pages) is True

    def test_spine_text_waits_for_a_spine_worth_printing_on(self):
        # 79 pages is where the old 6 mm rule switched text on, and it is
        # inside the band Lulu refuses.
        assert spine_text_allowed(79) is False
        assert spine_text_allowed(129) is False
        assert spine_text_allowed(130) is True


class TestWrapCover:
    @staticmethod
    def _book(tmp_path):
        book = Book(title={"de": "Ein Buch"}, languages=["de"],
                    pages=[Page(index=1, text={"de": "x"})])
        art = tmp_path / "cover.png"
        Image.new("RGB", (600, 600), "steelblue").save(art)
        book.cover = "cover.png"
        return book, (lambda rel: tmp_path / rel)

    def test_a_stapled_book_gets_no_spine_panel(self, tmp_path):
        book, resolve = self._book(tmp_path)
        preset = PRESETS["print_square"]
        _, info = render_wrap_cover(book, "de", preset, resolve, interior_pages=16)

        assert info["binding"] == "saddle_stitch"
        assert info["spine_mm"] == 0
        # Two covers plus bleed, and nothing in between.
        expected = 2 * preset.trim_mm[0] + 2 * preset.bleed_mm
        assert info["size_mm"][0] == pytest.approx(expected, abs=0.2)
        assert "keinen Rücken" in info["note"]

    def test_a_thick_book_keeps_its_spine(self, tmp_path):
        book, resolve = self._book(tmp_path)
        _, info = render_wrap_cover(book, "de", PRESETS["print_square"], resolve,
                                    interior_pages=160)
        assert info["binding"] == "perfect_bound"
        assert info["spine_mm"] > 0
        assert info["note"] is None  # wide enough for text, so nothing to explain


class TestProviderTrim:
    def test_epubli_does_not_sell_the_square_preset(self):
        assert trim_supported(PRESETS["print_square"], "epubli") is False
        warning = trim_warning(PRESETS["print_square"], "epubli")
        assert "epubli" in warning and "215.9" in warning

    def test_the_format_epubli_does_sell_passes(self):
        assert trim_supported(PRESETS["print_kinderbuch"], "epubli") is True
        assert trim_warning(PRESETS["print_kinderbuch"], "epubli") is None

    def test_lulu_sells_the_square_one(self):
        assert trim_supported(PRESETS["print_square"], "lulu") is True


def test_binding_note_explains_why(tmp_path):
    assert "geheftet" in binding_note(16)
    assert "Klebebindung" in binding_note(120)
