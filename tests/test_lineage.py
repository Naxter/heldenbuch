"""Locked references, stale tracking, and the cost ledger.

The through-line: nothing a user did yesterday may silently change or
silently disagree with what they see today. A book keeps its own copies of
its references; every derived artifact remembers the revision it was made
from; every cent leaves a ledger entry.
"""

from __future__ import annotations

import pytest
from PIL import Image

from storytime.book.library import Library
from storytime.book.models import Book, Hero, Page, Style
from storytime.pricing import add as add_spend
from storytime.pricing import summary
from storytime.web.bookapi import BookApi
from storytime.web.bookjobs import BookJobs
from storytime.web.jobs import Job


class _NoJobs:
    def active(self):
        return None

    def pending(self) -> int:
        return 0


@pytest.fixture()
def library(tmp_path) -> Library:
    return Library(tmp_path / "library")


@pytest.fixture()
def api(library) -> BookApi:
    return BookApi(library, _NoJobs())


def _png(path, size=(64, 64), colour="white"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)
    return path


def _hero_style_book(library, styled=True):
    hero = Hero(name="Mats", description="a boy")
    _png(library.hero_dir(hero.id) / "sheet_01.png", colour="red")
    hero.sheet = f"heroes/{hero.id}/sheet_01.png"
    library.save_hero(hero)

    style = Style(name="Aquarell", description="watercolour")
    if styled:
        _png(library.style_dir(style.id) / f"sheet_{hero.id}.png", colour="blue")
        style.sheets[hero.id] = f"styles/{style.id}/sheet_{hero.id}.png"
    library.save_style(style)

    book = Book(hero_id=hero.id, style_id=style.id, title={"de": "T"},
                pages=[Page(index=1, text={"de": "Eins"}, illustration="a meadow")])
    library.lock_references(book, hero, style)
    library.save_book(book)
    return hero, style, book


# ------------------------------------------------------------------ lineage


def test_lock_copies_the_sheets_into_the_book_folder(library):
    hero, style, book = _hero_style_book(library)
    assert book.styled_sheet == "refs/styled_sheet.png"
    assert book.hero_sheet == "refs/hero_sheet.png"
    assert (library.book_dir(book.id) / "refs/styled_sheet.png").is_file()
    assert book.ref_sources["styled"] == style.sheets[hero.id]


def test_a_new_hero_variant_does_not_change_an_existing_book(library):
    hero, style, book = _hero_style_book(library)
    locked = (library.book_dir(book.id) / "refs/styled_sheet.png").read_bytes()

    # The hero picks a new variant; the style is redrawn for it.
    _png(library.hero_dir(hero.id) / "sheet_02.png", colour="green")
    hero.sheet = f"heroes/{hero.id}/sheet_02.png"
    library.save_hero(hero)
    _png(library.style_dir(style.id) / f"sheet_{hero.id}.png", colour="green")

    sheet = BookJobs(library)._locked_sheet(book, hero, style)
    assert sheet.read_bytes() == locked, "the book must keep its frozen reference"


def test_deleting_the_style_leaves_the_book_renderable(library):
    hero, style, book = _hero_style_book(library)
    library.delete_style(style.id)
    sheet = BookJobs(library)._locked_sheet(book, hero, style)
    assert sheet.is_file()
    assert "refs" in str(sheet)


def test_old_books_adopt_their_current_reference_once(library):
    hero, style, book = _hero_style_book(library)
    book.hero_sheet = book.styled_sheet = None  # a book from before the lock
    book.ref_sources = {}
    library.save_book(book)

    BookJobs(library)._locked_sheet(book, hero, style)
    refreshed = library.get_book(book.id)
    assert refreshed.styled_sheet == "refs/styled_sheet.png"


def test_reference_status_notices_a_changed_hero(api, library):
    hero, style, book = _hero_style_book(library)
    _png(library.hero_dir(hero.id) / "sheet_02.png")
    hero.sheet = f"heroes/{hero.id}/sheet_02.png"
    library.save_hero(hero)

    payload = api.book(book.id, {}, None)
    assert payload["references"]["locked"] is True
    assert payload["references"]["hero_changed"] is True

    api.book_update(book.id, {}, {"adopt_references": True})
    payload = api.book(book.id, {}, None)
    assert payload["references"]["hero_changed"] is False


# ------------------------------------------------------------------ staleness


def test_editing_text_makes_only_that_languages_audio_stale(api, library):
    _, _, book = _hero_style_book(library)
    page = book.pages[0]
    page.text = {"de": "Eins", "en": "One"}
    page.audio = {"de": "audio/p1_de.mp3", "en": "audio/p1_en.mp3"}
    library.save_book(book)

    api.book_update(book.id, {}, {"pages": [{"index": 1, "text": {"de": "Neu"}}]})
    refreshed = library.get_book(book.id)
    assert refreshed.pages[0].audio_stale() == ["de"]


def test_editing_the_brief_makes_the_image_stale_and_a_redraw_clears_it(api, library):
    _, _, book = _hero_style_book(library)
    book.pages[0].image = "pages/page_01.png"
    library.save_book(book)

    api.book_update(book.id, {}, {"pages": [{"index": 1, "illustration": "a storm"}]})
    page = library.get_book(book.id).pages[0]
    assert page.image_stale() is True

    page.image_from_rev = page.illustration_rev  # what a redraw records
    assert page.image_stale() is False


def test_saving_the_same_text_bumps_nothing(api, library):
    _, _, book = _hero_style_book(library)
    before = library.get_book(book.id).content_rev
    api.book_update(book.id, {}, {"pages": [{"index": 1, "text": {"de": "Eins"}}]})
    assert library.get_book(book.id).content_rev == before


def test_layout_changes_mark_the_export_stale_but_not_the_image(api, library):
    _, _, book = _hero_style_book(library)
    book.pages[0].image = "pages/page_01.png"
    book.export_rev = book.content_rev
    library.save_book(book)

    api.book_update(book.id, {}, {"pages": [{"index": 1, "layout": "vignette"}]})
    refreshed = library.get_book(book.id)
    assert refreshed.export_stale() is True
    assert refreshed.pages[0].image_stale() is False


def test_old_books_open_without_anything_stale():
    raw = {"id": "book_x", "title": {"de": "Alt"},
           "pages": [{"index": 1, "text": {"de": "x"},
                      "image": "pages/p.png", "audio": {"de": "audio/a.mp3"}}]}
    book = Book.from_dict(raw)
    assert book.pages[0].image_stale() is False
    assert book.pages[0].audio_stale() == []
    assert book.export_stale() is False


# ------------------------------------------------------------------ ledger


def test_every_call_leaves_a_ledger_entry():
    spend: dict = {}
    add_spend(spend, {"model": "gpt-image-2", "backend": "openai",
                      "output_tokens": 100_000, "images": 1}, "pages")
    add_spend(spend, {"model": "gpt-4o-mini-tts", "input_tokens": 100,
                      "output_tokens": 2000}, "narration")

    entries = spend["entries"]
    assert len(entries) == 2
    assert entries[0]["what"] == "pages"
    assert entries[0]["model"] == "gpt-image-2"
    total = summary(spend)
    assert total["usd"] == pytest.approx(round(sum(e["usd"] for e in entries), 2), abs=0.01)
    assert total["entries"] == entries


def test_budget_cap_stops_drawing_before_overspending(library, monkeypatch):
    from storytime.book import illustrate

    hero, style, book = _hero_style_book(library)
    book.pages = [Page(index=i, text={"de": str(i)}, illustration="x")
                  for i in range(1, 6)]
    library.save_book(book)
    sheet = library.resolve(style.sheets[hero.id])

    drawn = []

    class _FakeResult:
        usage = {"model": "gpt-image-2", "usd": 1.0, "images": 1}

    def fake_draw(book_, hero_, style_, page, *args, **kwargs):
        drawn.append(page.index)
        return _FakeResult()

    monkeypatch.setattr(illustrate, "draw_page", fake_draw)
    illustrate.illustrate_book(
        book, hero, style, sheet,
        pages_dir=library.book_dir(book.id) / "pages",
        backend_name="stub", check=False, workers=1, budget_usd=2.5,
        log=lambda *a: None,
    )
    # 1.0 $ per page against a 2.5 $ ceiling: the third page crosses it,
    # everything after is skipped.
    assert len(drawn) == 3
    assert book.spend["usd"] == pytest.approx(3.0)


def test_a_failed_job_keeps_the_spend_already_recorded(library, monkeypatch):
    hero, style, book = _hero_style_book(library)
    jobs = BookJobs(library)

    def explode(book_arg, *args, **kwargs):
        # Spend lands on the book the worker actually loaded, then the run dies.
        from storytime.pricing import add
        add(book_arg.spend, {"model": "gpt-image-2", "usd": 0.5, "images": 1}, "pages")
        raise RuntimeError("provider died mid-run")

    monkeypatch.setattr("storytime.book.illustrate.illustrate_book", explode)
    job = Job(id="1", action="book_illustrate",
              params={"book_id": book.id, "backend": "stub"})
    with pytest.raises(RuntimeError):
        jobs.book_illustrate(job, lambda *a: None)

    refreshed = library.get_book(book.id)
    assert refreshed.spend.get("usd") == pytest.approx(0.5), \
        "the money spent before the crash must survive it"


def test_a_truncated_page_is_redrawn_on_resume(tmp_path, monkeypatch):
    """Resuming skips pages whose file exists. A render killed mid-write left a
    half PNG that was adopted as finished work and could never be redrawn.
    """
    from storytime.book.illustrate import _usable_image

    good = tmp_path / "good.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(good)
    assert _usable_image(good) is True

    truncated = tmp_path / "half.png"
    truncated.write_bytes(good.read_bytes()[: good.stat().st_size // 2])
    assert _usable_image(truncated) is False

    empty = tmp_path / "empty.png"
    empty.touch()
    assert _usable_image(empty) is False

    assert _usable_image(tmp_path / "absent.png") is False


def test_image_writes_are_atomic(tmp_path, monkeypatch):
    """An interrupted write must leave the previous page intact."""
    from storytime.backends.stub import StubBackend
    from storytime.types import GenRequest, OutputSpec

    backend = StubBackend()
    target = tmp_path / "page_01.png"
    backend.generate(GenRequest(prompt="a fox", output=OutputSpec()), target)
    before = target.read_bytes()

    def explode(_src, _dst):
        raise OSError("disk full")

    monkeypatch.setattr("storytime.backends.base.os.replace", explode)
    with pytest.raises(OSError):
        backend.generate(GenRequest(prompt="a different fox", output=OutputSpec()), target)

    assert target.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
