"""Money spent before a book exists still has to be recorded somewhere.

Character sheets, style previews, the styled sheet and the scout are all paid
for before the first book is written, and are then reused by every book made
from them. None of it was recorded at all: the first euro in the ledger was
the book cover, so a book always looked cheaper than it was.
"""


from PIL import Image

from heldenbuch.book import hero as hero_mod
from heldenbuch.book import look
from heldenbuch.book.library import Library
from heldenbuch.book.models import Book, Hero, Style
from heldenbuch.pricing import add as add_spend
from heldenbuch.types import GenResult

USAGE = {"model": "gpt-image-2", "images": 1, "input_tokens": 100,
         "output_tokens": 1000}


class _Backend:
    """Stands in for a paid image service."""

    max_references = 4
    default_model = "test-1"
    honours_seed = False

    def generate(self, request, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), "white").save(target)
        return GenResult(image_path=target, backend="test", model="test-1",
                         prompt=request.prompt, reference_images=[],
                         latency_s=0.0, cost_note="", usage=USAGE)


def test_character_sheets_are_recorded_on_the_hero(tmp_path, monkeypatch):
    monkeypatch.setattr(hero_mod, "get_backend", lambda *a, **k: _Backend())
    hero = Hero(name="Claudio", description="a boy")

    spend: dict = {}
    hero_mod.generate_variants(
        hero, tmp_path, count=3, log=lambda *a: None,
        spend=lambda usage: add_spend(spend, usage, "hero_sheet"))

    assert spend["images"] == 3
    assert spend["usd"] > 0
    assert spend["by"]["hero_sheet"]["calls"] == 3


def test_style_previews_are_recorded_on_the_style(tmp_path, monkeypatch):
    monkeypatch.setattr(look, "get_backend", lambda *a, **k: _Backend())
    hero = Hero(name="Claudio", description="a boy")
    style = Style(name="S", description="soft")
    sheet = tmp_path / "sheet.png"
    Image.new("RGB", (16, 16), "white").save(sheet)

    look.generate_previews(hero, style, sheet, tmp_path, count=2,
                           spend=lambda usage: add_spend(style.spend, usage, "style_preview"),
                           log=lambda *a: None)

    assert style.spend["images"] == 2
    assert style.spend["by"]["style_preview"]["calls"] == 2


def test_the_library_total_includes_what_no_book_owns(tmp_path):
    library = Library(tmp_path)

    hero = Hero(name="Claudio")
    add_spend(hero.spend, USAGE, "hero_sheet")
    library.save_hero(hero)

    style = Style(name="S")
    add_spend(style.spend, USAGE, "style_preview")
    library.save_style(style)

    book = Book(title={"de": "T"})
    add_spend(book.spend, USAGE, "pages")
    library.save_book(book)

    total = library.totals()
    assert total["images"] == 3          # one each, not just the book's
    assert total["by_area"]["heroes"]["images"] == 1
    assert total["by_area"]["styles"]["images"] == 1
    assert total["by_area"]["books"]["images"] == 1
    # the book's own ledger still reports only its own page
    assert library.get_book(book.id).spend["images"] == 1


def test_a_hero_ledger_survives_a_round_trip(tmp_path):
    library = Library(tmp_path)
    hero = Hero(name="Claudio")
    add_spend(hero.spend, USAGE, "hero_sheet")
    library.save_hero(hero)

    assert library.get_hero(hero.id).spend["by"]["hero_sheet"]["calls"] == 1
