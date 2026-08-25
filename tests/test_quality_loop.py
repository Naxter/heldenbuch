"""The quality loop: variants to choose from, reasons that travel, records
that make drift attributable, and a crop that follows the subject.

Common thread: the person's judgement is the best checker the app has, so
the machinery either asks for it (variants, redraw reasons) or preserves the
evidence it needs (drawn_by, embedding metrics, word budgets).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from heldenbuch.book import illustrate, layout
from heldenbuch.book.library import Library
from heldenbuch.book.models import Book, Hero, Page, Style


@pytest.fixture()
def library(tmp_path) -> Library:
    return Library(tmp_path / "library")


def _png(path: Path, size=(64, 64), colour="white") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)
    return path


def _seeded(library):
    hero = Hero(name="Rusty", sheet="heroes/h/sheet.png")
    style = Style(description="ink")
    book = library.save_book(Book(pages=[
        Page(index=1, illustration="Rusty digs."),
        Page(index=2, illustration="Rusty rests."),
    ], climax=2, cover="pages/cover.png"))
    root = library.book_dir(book.id)
    sheet = _png(library.root / "heroes/h/sheet.png")
    _png(root / "pages/cover.png")
    return hero, style, book, root, sheet


# ------------------------------------------------------------------ variants


def test_variants_are_drawn_and_adopting_one_keeps_the_old_page(library):
    hero, style, book, root, sheet = _seeded(library)
    _png(root / "pages/page_02.png", colour="blue")
    book.pages[1].image = "pages/page_02.png"

    made = illustrate.draw_variants(
        book, hero, style, sheet, pages_dir=root / "pages", index=2,
        backend_name="stub", resolve=lambda rel: root / rel,
        log=lambda *a: None)
    assert len(made) == 3
    assert book.pages[1].variants == made
    assert all((root / rel).is_file() for rel in made)

    illustrate.adopt_variant(book, 2, made[1], root / "pages")
    page = book.pages[1]
    assert page.variants == []
    assert page.image == "pages/page_02.png"
    assert page.check == {}  # a new picture is unreviewed
    assert page.history, "the outgoing picture stays undoable"
    # the losing candidates are gone from disk
    assert not (root / made[0]).is_file()
    assert not (root / made[2]).is_file()


def test_cover_variants_work_the_same_way(library):
    hero, style, book, root, sheet = _seeded(library)
    made = illustrate.draw_variants(
        book, hero, style, sheet, pages_dir=root / "pages", index=0,
        count=2, backend_name="stub", resolve=lambda rel: root / rel,
        log=lambda *a: None)
    assert book.cover_variants == made

    book.cover_check = {"status": "failed"}
    illustrate.adopt_variant(book, 0, made[0], root / "pages")
    assert book.cover == "pages/cover.png"
    assert book.cover_variants == []
    assert book.cover_check == {}
    # The pick dialog promises the previous version survives; the cover used
    # to be the one image that silently broke that promise.
    assert book.cover_history
    assert (root / book.cover_history[-1]).is_file()


def test_a_wrong_variant_name_is_refused(library):
    hero, style, book, root, sheet = _seeded(library)
    with pytest.raises(ValueError):
        illustrate.adopt_variant(book, 2, "pages/nope.png", root / "pages")


def test_a_double_adopt_is_refused_cleanly(library):
    """The second submit used to crash with FileNotFoundError -- after
    copying the already-adopted picture into history as a fake old version."""
    hero, style, book, root, sheet = _seeded(library)
    made = illustrate.draw_variants(
        book, hero, style, sheet, pages_dir=root / "pages", index=2,
        backend_name="stub", resolve=lambda rel: root / rel,
        log=lambda *a: None)
    stale_copy = list(made)
    illustrate.adopt_variant(book, 2, made[0], root / "pages")
    history_before = list(book.pages[1].history)
    book.pages[1].variants = stale_copy  # a second, stale request
    with pytest.raises(ValueError):
        illustrate.adopt_variant(book, 2, stale_copy[0], root / "pages")
    assert book.pages[1].history == history_before


def test_variants_differ_and_the_pick_updates_the_ledger(library):
    """Stub hashed only the prompt, so all candidates were pixel-identical --
    and after a pick, drawn_by still named the replaced picture's service."""
    hero, style, book, root, sheet = _seeded(library)
    book.pages[1].drawn_by = "openai:gpt-image-2"
    made = illustrate.draw_variants(
        book, hero, style, sheet, pages_dir=root / "pages", index=2,
        backend_name="stub", resolve=lambda rel: root / rel,
        log=lambda *a: None)
    digests = {(root / rel).read_bytes() for rel in made}
    assert len(digests) == 3, "three candidates must be three pictures"
    illustrate.adopt_variant(book, 2, made[0], root / "pages")
    assert book.pages[1].drawn_by.startswith("stub:")
    assert book.pages[1].seed is None


def test_discarding_keeps_the_current_picture(library):
    hero, style, book, root, sheet = _seeded(library)
    _png(root / "pages/page_02.png", colour="blue")
    book.pages[1].image = "pages/page_02.png"
    made = illustrate.draw_variants(
        book, hero, style, sheet, pages_dir=root / "pages", index=2,
        backend_name="stub", resolve=lambda rel: root / rel,
        log=lambda *a: None)
    illustrate.discard_variants(book, 2, root / "pages")
    assert book.pages[1].variants == []
    assert book.pages[1].image == "pages/page_02.png"
    assert not any((root / rel).is_file() for rel in made)


# ---------------------------------------------------------- reasons + record


def test_the_redraw_reason_reaches_the_first_attempt(library, monkeypatch):
    hero, style, book, root, sheet = _seeded(library)
    seen: list[dict] = []

    def fake_draw(book_, hero_, style_, page, sheet_, target, **kwargs):
        seen.append({"insist": kwargs.get("insist"),
                     "feedback": list(kwargs.get("feedback") or [])})
        _png(target)

        class R:
            usage = {}

        return R()

    monkeypatch.setattr(illustrate, "draw_page", fake_draw)
    illustrate.illustrate_book(
        book, hero, style, sheet, pages_dir=root / "pages",
        backend_name="stub", check=False, only=[1], redraw=True,
        feedback=["the setting was wrong in the previous version"],
        resolve=lambda rel: root / rel, log=lambda *a: None)
    assert seen[-1]["insist"] is True
    assert "setting was wrong" in seen[-1]["feedback"][0]


def test_every_drawn_page_records_its_service(library):
    hero, style, book, root, sheet = _seeded(library)
    illustrate.illustrate_book(
        book, hero, style, sheet, pages_dir=root / "pages",
        backend_name="stub", check=False,
        resolve=lambda rel: root / rel, log=lambda *a: None)
    assert all(p.drawn_by.startswith("stub:") for p in book.pages)


# ------------------------------------------------------------ words + embed


def test_wordy_pages_flags_only_clear_overruns():
    from heldenbuch.book.preflight import wordy_pages

    fits = "wort " * 40           # at the 4-5 band's budget
    too_much = "wort " * 70       # far over it
    book = Book(age="4-5", pages=[
        Page(index=1, text={"de": fits.strip()}),
        Page(index=2, text={"de": too_much.strip()}),
        Page(index=3, text={"de": ""}, layout="wordless"),
    ])
    assert wordy_pages(book, "de") == [2]


def test_embed_outliers_read_the_stored_metric():
    from heldenbuch.book.preflight import embed_outliers

    def page(i, value):
        return Page(index=i, check={"metrics": {"dino_cosine": value}})

    steady = [page(i, 0.80 + i * 0.001) for i in range(1, 8)]
    book = Book(pages=steady + [page(8, 0.55)])
    assert embed_outliers(book) == [8]
    # without the optional extra no page ever has the metric: silence
    assert embed_outliers(Book(pages=[Page(index=1, check={})])) == []


# --------------------------------------------------------------- crop + font


def test_a_dark_subject_still_steers_the_crop(tmp_path):
    """The saturation mask is blind to near-black subjects; a night scene
    used to fall back to the centred cut this feature exists to replace."""
    source = Image.new("RGB", (400, 400), "white")
    for x in range(20, 120):
        for y in range(150, 250):
            source.putpixel((x, y), (18, 12, 10))
    path = tmp_path / "night.png"
    source.save(path)

    wide = layout.cover_image(path, (200, 100))
    pixels = wide.load()
    darks = sum(1 for x in range(wide.width) for y in range(wide.height)
                if sum(pixels[x, y]) < 90)
    assert darks > 800, "the dark subject must survive the crop"


def test_panel_compositions_keep_the_centred_crop(tmp_path):
    """Title and wrap-cover panels are painted over the top of the crop and
    assume the centred framing -- focus must not pull the head under them."""
    # A textured background, or flat_border trims the scene away first and
    # both crops end up identical for the wrong reason.
    source = Image.new("RGB", (400, 400), "white")
    for x in range(400):
        for y in range(400):
            if (x + y) % 7 == 0:
                source.putpixel((x, y), (235, 228, 210))
    for x in range(150, 250):
        for y in range(20, 120):  # subject high in the frame
            source.putpixel((x, y), (200, 30, 30))
    path = tmp_path / "art.png"
    source.save(path)

    focused = layout.cover_image(path, (400, 200))
    centred = layout.cover_image(path, (400, 200), focus=False)
    assert focused.getpixel((200, 30)) == (200, 30, 30)  # subject kept high
    assert centred.tobytes() != focused.tobytes()


def test_the_crop_follows_the_subject(tmp_path):
    """A square picture cut to a wide box must keep an off-centre subject.
    The centred cut used to walk straight through it."""
    source = Image.new("RGB", (400, 400), "white")
    for x in range(20, 120):
        for y in range(150, 250):
            source.putpixel((x, y), (200, 30, 30))
    path = tmp_path / "art.png"
    source.save(path)

    wide = layout.cover_image(path, (200, 100))
    pixels = wide.load()
    reds = sum(1 for x in range(wide.width) for y in range(wide.height)
               if pixels[x, y][0] > 150 and pixels[x, y][1] < 90)
    assert reds > 800, "the subject must survive the crop"


def test_the_bundled_font_is_found_and_offered():
    assert layout.find_font("Andika-Regular.ttf") is not None
    keys = [f["key"] for f in layout.available_families()]
    assert "andika" in keys


def test_the_preview_locks_type_like_the_export(library, monkeypatch):
    """The proof used to fit type per page while the export locked one size
    for the whole book -- the preview showed a page the PDF then contradicted."""
    book = library.save_book(Book(
        title={"de": "T"}, languages=["de"],
        pages=[Page(index=1, text={"de": "Ein Satz."}, image="pages/page_01.png"),
               Page(index=2, text={"de": "Noch ein Satz."})],
    ))
    root = library.book_dir(book.id)
    _png(root / "pages/page_01.png", size=(256, 256))
    seen: dict = {}
    real = layout.render_story_page

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(layout, "render_story_page", spy)
    layout.render_preview(book, 1, layout.PRESETS["screen"],
                          lambda rel: root / rel, ["de"])
    assert seen.get("body_px"), "the preview must use the book-wide type size"
