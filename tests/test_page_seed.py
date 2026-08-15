"""A page records the seed it was drawn with -- where that means anything.

Without it a page that came out well cannot be reproduced: the next attempt
at the same prompt is a different picture, and the good one is gone.
"""

from pathlib import Path

from PIL import Image

from heldenbuch.book import illustrate
from heldenbuch.book.library import Library
from heldenbuch.book.models import Book, Hero, Page, Style
from heldenbuch.types import GenResult


def _png(path, size=(32, 32)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return path


def _setup(tmp_path):
    library = Library(tmp_path / "library")
    hero = Hero(name="Simon", description="a boy")
    _png(library.hero_dir(hero.id) / "sheet.png")
    hero.sheet = f"heroes/{hero.id}/sheet.png"
    library.save_hero(hero)
    style = Style(name="S", description="d", sheets={hero.id: hero.sheet})
    book = Book(hero_id=hero.id, style_id=style.id, title={"de": "T"},
                pages=[Page(index=1, text={"de": "eins"}, illustration="a scene")])
    return library, hero, style, book


def _run(monkeypatch, library, hero, style, book, honours_seed: bool):
    """Draw the book with a backend that does or does not take a seed."""
    seen = []

    def fake_draw_page(book_, hero_, style_, page, sheet, target, **kwargs):
        seen.append(kwargs.get("seed"))
        _png(target)
        return GenResult(image_path=target, backend="test", model="test-1",
                         prompt="p", reference_images=[], latency_s=0.0,
                         cost_note="", usage={})

    class FakeBackend:
        default_model = "test-1"
        max_references = 4

    FakeBackend.honours_seed = honours_seed
    monkeypatch.setattr(illustrate, "draw_page", fake_draw_page)
    monkeypatch.setattr(illustrate, "get_backend", lambda *a, **k: FakeBackend())

    illustrate.illustrate_book(
        book, hero, style, library.resolve(hero.sheet),
        pages_dir=library.book_dir(book.id) / "pages",
        check=False, workers=1, log=lambda *a: None,
    )
    return seen


def test_the_seed_is_kept_when_the_service_uses_one(tmp_path, monkeypatch):
    library, hero, style, book = _setup(tmp_path)
    sent = _run(monkeypatch, library, hero, style, book, honours_seed=True)

    assert sent and sent[0] is not None
    assert book.pages[0].seed == sent[0]
    # and it survives a round trip through book.json
    library.save_book(book)
    assert library.get_book(book.id).pages[0].seed == sent[0]


def test_no_seed_is_claimed_when_the_service_ignores_it(tmp_path, monkeypatch):
    library, hero, style, book = _setup(tmp_path)
    _run(monkeypatch, library, hero, style, book, honours_seed=False)

    # A number here would promise a reproducibility the page has not got.
    assert book.pages[0].seed is None


def test_a_redraw_does_not_repeat_the_rejected_picture(tmp_path, monkeypatch):
    library, hero, style, book = _setup(tmp_path)
    first = _run(monkeypatch, library, hero, style, book, honours_seed=True)

    pages_dir = library.book_dir(book.id) / "pages"
    for existing in pages_dir.glob("*.png"):
        Path(existing).unlink()
    second = _run(monkeypatch, library, hero, style, book, honours_seed=True)

    assert first[0] != second[0]
