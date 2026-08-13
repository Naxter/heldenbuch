"""Tests for the page geometry and typesetting.

Both bugs that reached a finished PDF so far lived in this area: a path
resolved against the wrong root, and a histogram bucket that made near-identical
images look unrelated. These are pure functions over numbers and images, which
makes them the cheapest things in the project to pin down.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from storytime.book.layout import (
    MM_PER_INCH,
    PRESETS,
    PrintPreset,
    effective_dpi,
    find_text_spot,
    fit_text,
    render_story_page,
    wrap,
)


def test_print_square_page_is_trim_plus_bleed_at_300dpi():
    preset = PRESETS["print_square"]
    width, height = preset.page_px()
    # 215.9 mm + 2 x 3.175 mm = 222.25 mm = 8.75 in = 2625 px at 300 dpi
    assert (width, height) == (2625, 2625)
    assert preset.bleed_px == 38  # 3.175 mm at 300 dpi


def test_safety_margin_is_inside_the_trim_edge():
    preset = PRESETS["print_square"]
    # Safety is measured from the trim edge, so the pixel offset from the file
    # edge has to include the bleed as well.
    expected = int(round((12.7 + 3.175) / MM_PER_INCH * 300))
    assert preset.safety_px == expected


def test_home_preset_has_no_bleed():
    preset = PRESETS["home_a4"]
    assert preset.bleed_mm == 0
    assert preset.bleed_px == 0


@pytest.mark.parametrize(
    "pages,expected_mm",
    [
        (24, (24 / 444 + 0.06) * MM_PER_INCH),
        (40, (40 / 444 + 0.06) * MM_PER_INCH),
    ],
)
def test_spine_follows_the_lulu_formula(pages, expected_mm):
    assert PRESETS["print_square"].spine_mm(pages) == pytest.approx(expected_mm)


def test_spine_grows_with_page_count():
    preset = PRESETS["print_square"]
    assert preset.spine_mm(40) > preset.spine_mm(20)


def test_effective_dpi_reports_low_resolution(tmp_path):
    small = tmp_path / "small.png"
    Image.new("RGB", (1024, 1024), "white").save(small)
    # 1024 px across 222.25 mm is about 117 dpi -- far under print quality.
    assert effective_dpi(small, PRESETS["print_square"]) == pytest.approx(117, abs=2)

    big = tmp_path / "big.png"
    Image.new("RGB", (2624, 2624), "white").save(big)
    assert effective_dpi(big, PRESETS["print_square"]) >= 299


def test_wrap_breaks_on_words_not_mid_word():
    from storytime.book.layout import load_font

    font = load_font("georgia", "regular", 40)
    lines = wrap("Mats stapft durch den Garten, plitsch platsch plitsch.", font, 300)
    assert len(lines) > 1
    assert all(" " in line or line.count(" ") == 0 for line in lines)
    # every original word survives, in order
    assert " ".join(lines).split() == \
        "Mats stapft durch den Garten, plitsch platsch plitsch.".split()


def test_fit_text_shrinks_until_it_fits():
    long_text = "Ein sehr langer Satz " * 12
    _, lines, step = fit_text(long_text, "georgia", "regular", (600, 400), max_size=90)
    assert len(lines) * step <= 400


def test_fit_text_never_returns_zero_lines():
    font, lines, step = fit_text("Kurz.", "georgia", "regular", (600, 400), max_size=60)
    assert lines == ["Kurz."]
    assert step > 0


def test_cyrillic_sets_without_falling_back_to_boxes():
    from storytime.book.layout import load_font

    font = load_font("georgia", "regular", 40)
    # A font without Cyrillic coverage reports the same width for every
    # unknown glyph; real coverage gives different widths for different words.
    assert font.getlength("Матс") != font.getlength("Мацуг")


class TestTextPlacement:
    """The text must land where the picture is quiet."""

    @staticmethod
    def _page(busy_at_bottom: bool) -> Image.Image:
        """A 600x600 picture: one half flat, the other half full of detail."""
        array = np.full((600, 600, 3), 235, dtype=np.uint8)
        noise = np.random.default_rng(7).integers(0, 255, (300, 600, 3), dtype=np.uint8)
        if busy_at_bottom:
            array[300:, :, :] = noise
        else:
            array[:300, :, :] = noise
        return Image.fromarray(array)

    def test_text_goes_to_the_bottom_when_the_bottom_is_calm(self):
        spot = find_text_spot(self._page(busy_at_bottom=False), safety=20)
        _, top, _, bottom = spot.box
        assert top > 300, "expected the text below the busy top half"
        assert bottom <= 600

    def test_text_moves_to_the_top_when_the_bottom_is_busy(self):
        spot = find_text_spot(self._page(busy_at_bottom=True), safety=20)
        _, top, _, bottom = spot.box
        assert bottom < 320, "expected the text above the busy bottom half"

    def test_panel_is_dropped_on_a_plain_light_area(self):
        plain = Image.new("RGB", (600, 600), (245, 243, 238))
        spot = find_text_spot(plain, safety=20)
        assert spot.panel is False
        assert spot.ink[0] < 100, "dark ink on a light background"

    def test_light_ink_on_a_plain_dark_area(self):
        dark = Image.new("RGB", (600, 600), (24, 26, 34))
        spot = find_text_spot(dark, safety=20)
        assert spot.panel is False
        assert spot.ink[0] > 200, "light ink on a dark background"

    def test_panel_is_used_when_the_picture_is_busy_everywhere(self):
        noise = np.random.default_rng(3).integers(0, 255, (600, 600, 3), dtype=np.uint8)
        spot = find_text_spot(Image.fromarray(noise), safety=20)
        assert spot.panel is True


class TestRenderStoryPage:
    @staticmethod
    def _art(tmp_path):
        path = tmp_path / "art.png"
        Image.new("RGB", (1024, 1024), (200, 180, 150)).save(path)
        return path

    def test_page_matches_the_preset_size(self, tmp_path):
        preset = PRESETS["print_square"]
        page = render_story_page(self._art(tmp_path), "Hallo Welt.", preset)
        assert page.size == preset.page_px()

    def test_wordless_layout_leaves_the_text_off(self, tmp_path):
        preset = PRESETS["screen"]
        with_text = render_story_page(self._art(tmp_path), "Ein Satz.", preset, layout="full")
        without = render_story_page(self._art(tmp_path), "Ein Satz.", preset, layout="wordless")
        assert with_text.tobytes() != without.tobytes()

    def test_missing_illustration_still_produces_a_page(self, tmp_path):
        preset = PRESETS["screen"]
        page = render_story_page(None, "Nur Text.", preset)
        assert page.size == preset.page_px()

    @pytest.mark.parametrize("layout", ["full", "split", "vignette", "wordless"])
    def test_every_layout_renders(self, tmp_path, layout):
        preset = PRESETS["print_kinderbuch"]
        page = render_story_page(self._art(tmp_path), "Ein kurzer Satz.", preset, layout=layout)
        assert page.size == preset.page_px()

    def test_vignette_keeps_the_text_clear_of_the_picture(self, tmp_path):
        """Long text used to climb over the bottom of the vignette.

        The picture was sized first at a fixed 60% of the page, so any text
        that needed more than the remaining band was drawn on top of it.
        """
        preset = PRESETS["print_square"]
        page_w, page_h = preset.page_px()
        long_text = ("Im weichen Beet entdeckt Mats einen Pfotenabdruck. Daneben liegt "
                     "ein umgeknicktes Salbeiblatt. „Hier lang!“, flüstert er und folgt "
                     "der Spur bis zum Kürbis.")
        page = render_story_page(self._art(tmp_path), long_text, preset, layout="vignette")

        # The picture is opaque colour; paper is not. Find the lowest row that
        # still contains picture, and the highest row that contains ink.
        pixels = np.asarray(page.convert("RGB"))
        art_rows = np.where((np.abs(pixels.astype(int) - np.array([200, 180, 150]))
                             .sum(axis=-1) < 30).any(axis=1))[0]
        dark_rows = np.where((pixels.sum(axis=-1) < 300).any(axis=1))[0]
        assert art_rows.size and dark_rows.size
        assert dark_rows.min() > art_rows.max(), "text must start below the picture"

    def test_split_on_a_square_page_stacks_instead_of_sitting_beside(self, tmp_path):
        """Side by side needs a landscape page; square gets picture over text."""
        square = render_story_page(self._art(tmp_path), "Ein Satz der etwas laenger ist.",
                                   PRESETS["print_square"], layout="split")
        width, height = square.size
        # The bottom strip must be paper, i.e. the art does not reach the floor.
        bottom = np.asarray(square.convert("RGB"))[height - 10:, :, :]
        assert bottom.mean() > 200, "expected a paper band under the picture"

        # On a landscape page the art should instead reach the bottom edge.
        landscape = render_story_page(self._art(tmp_path), "Ein Satz.",
                                      PRESETS["home_a4"], layout="split")
        lw, lh = landscape.size
        left_column = np.asarray(landscape.convert("RGB"))[lh - 10:, : lw // 4, :]
        assert left_column.mean() < 220, "expected picture at the bottom left"


def test_custom_preset_geometry_is_self_consistent():
    preset = PrintPreset(
        key="t", name="t", hint="", trim_mm=(100.0, 200.0),
        bleed_mm=5.0, safety_mm=10.0, dpi=100,
    )
    width, height = preset.page_px()
    assert (width, height) == (round(110 / MM_PER_INCH * 100), round(210 / MM_PER_INCH * 100))
    assert preset.bleed_px == round(5 / MM_PER_INCH * 100)


class TestBodyTypeFloor:
    """Text used to shrink without limit so it always 'fitted'.

    `fit_text`'s default floor is 10 px, which is 2.4 pt at 300 dpi. Only the
    vignette path ever overrode it, so an 8+ page set at 6 pt on the small
    children's format -- inside the product's own word-count spec.
    """

    @staticmethod
    def _art(tmp_path):
        path = tmp_path / "art.png"
        Image.new("RGB", (1024, 1024), (200, 180, 150)).save(path)
        return path

    @staticmethod
    def _art_fraction(page, page_h):
        pixels = np.asarray(page.convert("RGB")).astype(int)
        mask = (np.abs(pixels - np.array([200, 180, 150])).sum(axis=-1) < 30)
        rows = np.where(mask.any(axis=1))[0]
        return 0.0 if not rows.size else (rows.max() - rows.min() + 1) / page_h

    @pytest.mark.parametrize("age,floor_pt", [("2-3", 20), ("4-5", 16), ("6-7", 13), ("8+", 12)])
    @pytest.mark.parametrize("preset_key", ["print_square", "print_kinderbuch"])
    def test_body_type_never_falls_below_the_floor(self, preset_key, age, floor_pt):
        from storytime.book.layout import fit_body, min_body_px

        preset = PRESETS[preset_key]
        page_w, page_h = preset.page_px()
        floor = min_body_px(preset, age)
        assert round(floor / preset.dpi * 72) == floor_pt

        # Far more words than the age band allows, to force the floor.
        text = " ".join(["Silbenwort"] * 400)
        font, _lines, _step, fits = fit_body(
            text, "georgia", (page_w // 2, page_h // 4),
            max_size=int(page_h * 0.045), min_size=floor,
        )
        assert font.size >= floor
        assert fits is False  # and it says so, instead of overflowing silently

    @pytest.mark.parametrize("words", [16, 40, 80, 128, 176, 260, 400])
    def test_a_vignette_never_loses_its_picture(self, tmp_path, words):
        """The regression: a body-type floor with no matching cap on the text
        block let the words grow until the illustration was skipped entirely,
        with no exception and no preflight warning.
        """
        preset = PRESETS["print_square"]
        _page_w, page_h = preset.page_px()
        text = " ".join(["Silbenwort"] * words)
        page = render_story_page(self._art(tmp_path), text, preset,
                                 layout="vignette", age="8+")
        assert self._art_fraction(page, page_h) > 0.30

    def test_a_vignette_too_full_of_words_becomes_a_full_page(self, tmp_path):
        """Rather than shrinking the art away. A page with 300 words is not a
        quiet beat, and the full layout has the whole page for text."""
        preset = PRESETS["print_square"]
        _page_w, page_h = preset.page_px()
        short = render_story_page(self._art(tmp_path), "Drei kurze Worte.",
                                  preset, layout="vignette", age="4-5")
        long = render_story_page(self._art(tmp_path), " ".join(["Silbenwort"] * 300),
                                 preset, layout="vignette", age="4-5")
        assert 0.30 < self._art_fraction(short, page_h) < 0.95   # inset picture
        assert self._art_fraction(long, page_h) > 0.95           # full bleed
