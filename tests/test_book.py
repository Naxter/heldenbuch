"""Tests for the data model, the library, and the story the model sends back.

The library's path handling is what stops a stored relative path from reaching
outside the library, and what made the exported PDFs come out empty when it was
resolved against the wrong root. Both directions are pinned here.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from storytime.book.author import _normalise
from storytime.book.library import Library
from storytime.book.models import Book, CastMember, Hero, Page, Style, slugify
from storytime.pricing import add as add_spend
from storytime.pricing import price, summary
from storytime.types import OutputSpec


# --------------------------------------------------------------------------- model


def test_book_round_trips_through_json(tmp_path):
    book = Book(
        title={"de": "Titel", "ru": "Название"},
        languages=["de", "ru"],
        pages=[Page(index=1, text={"de": "Eins"}, layout="vignette", cast=["Oma"])],
        cast=[CastMember(name="Oma", description="alt", pages=[1])],
    )
    restored = Book.from_dict(json.loads(json.dumps(book.to_dict())))
    assert restored.title == book.title
    assert restored.pages[0].layout == "vignette"
    assert restored.cast[0].name == "Oma"
    assert restored.cast[0].pages == [1]


def test_book_ignores_fields_from_an_older_version():
    """A book written before a field existed -- or after one was removed -- opens."""
    raw = Book(title={"de": "T"}).to_dict()
    raw["some_field_we_dropped"] = 42
    raw["pages"] = [{"index": 1, "text": {"de": "x"}, "obsolete": True}]
    book = Book.from_dict(raw)
    assert book.pages[0].index == 1


def test_cast_for_page_returns_place_always_and_named_characters_only():
    place = CastMember(name="Garten", kind="place", sheet="cast/01.png")
    oma = CastMember(name="Oma", kind="character", sheet="cast/02.png")
    hund = CastMember(name="Hund", kind="character", sheet="cast/03.png")
    book = Book(cast=[place, oma, hund])

    page = Page(index=1, cast=["Oma"])
    names = [m.name for m in book.cast_for(page)]
    assert names == ["Garten", "Oma"]


def test_cast_without_a_sheet_is_not_referenced():
    book = Book(cast=[CastMember(name="Oma", sheet=None)])
    assert book.cast_for(Page(index=1, cast=["Oma"])) == []


def test_display_title_falls_back_across_languages():
    assert Book(title={"de": "", "en": "Only English"},
                languages=["de", "en"]).display_title() == "Only English"


@pytest.mark.parametrize(
    "raw,expected",
    [("Claudio und der Matschstiefel", "mats-und-der-matschstiefel"),
     ("  ", "buch"), ("Ärger!! im Garten", "rger-im-garten")],
)
def test_slugify(raw, expected):
    assert slugify(raw) == expected


# --------------------------------------------------------------------------- library


@pytest.fixture()
def library(tmp_path) -> Library:
    return Library(tmp_path / "library")


def test_library_round_trips_each_kind(library):
    hero = library.save_hero(Hero(name="Claudio", description="d"))
    style = library.save_style(Style(name="Aquarell", description="d"))
    book = library.save_book(Book(hero_id=hero.id, style_id=style.id))

    assert library.get_hero(hero.id).name == "Claudio"
    assert library.get_style(style.id).name == "Aquarell"
    assert library.get_book(book.id).hero_id == hero.id
    assert [h.id for h in library.heroes()] == [hero.id]


def test_relative_and_resolve_are_inverse(library):
    target = library.hero_dir("hero_x") / "sheet_01.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x")
    relative = library.relative(target)
    assert relative == "heroes/hero_x/sheet_01.png"
    assert library.resolve(relative) == target.resolve()


@pytest.mark.parametrize("attempt", ["../secret", "../../.env", "a/../../outside"])
def test_resolve_refuses_to_leave_the_library(library, attempt):
    with pytest.raises(ValueError):
        library.resolve(attempt)


def test_book_paths_are_relative_to_the_book_not_the_library(library):
    """The bug that produced PDFs with no pictures in them.

    Page images are stored relative to the book folder so a book directory stays
    self-contained; resolving them against the library root silently yields a
    path that does not exist, and the exporter treats that as "no picture".
    """
    book = library.save_book(Book())
    pages_dir = library.book_dir(book.id) / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "white").save(pages_dir / "page_01.png")

    stored = "pages/page_01.png"
    assert not (library.root / stored).exists()
    assert (library.book_dir(book.id) / stored).is_file()


def test_delete_removes_the_whole_folder(library):
    book = library.save_book(Book())
    assert library.book_dir(book.id).is_dir()
    library.delete_book(book.id)
    assert not library.book_dir(book.id).exists()


# --------------------------------------------------------------------------- author


def _story(**overrides):
    payload = {
        "title": {"de": "Titel"},
        "dedication": {"de": "Für dich"},
        "cover_illustration": "a garden",
        "cast": [
            {"name": "Oma", "kind": "character", "description": "grey bun, blue apron"},
            {"name": "Garten", "kind": "place", "description": "a walled garden"},
        ],
        "pages": [
            {"index": 1, "text": {"de": "Eins"}, "illustration": "a", "layout": "full",
             "cast": ["Oma"]},
            {"index": 2, "text": {"de": "Zwei"}, "illustration": "b", "layout": "vignette",
             "cast": []},
        ],
    }
    payload.update(overrides)
    return payload


def test_normalise_builds_pages_and_cast():
    story = _normalise(_story(), ["de"], 12)
    assert [p.index for p in story["pages"]] == [1, 2]
    assert {c.name for c in story["cast"]} == {"Oma", "Garten"}
    assert story["pages"][1].layout == "vignette"


def test_normalise_renumbers_pages_the_model_got_wrong():
    payload = _story(pages=[
        {"index": 7, "text": {"de": "A"}, "illustration": "a"},
        {"index": 7, "text": {"de": "B"}, "illustration": "b"},
    ])
    story = _normalise(payload, ["de"], 12)
    assert [p.index for p in story["pages"]] == [1, 2]


def test_normalise_trims_to_the_requested_page_count():
    payload = _story(pages=[
        {"index": i, "text": {"de": str(i)}, "illustration": "x"} for i in range(1, 20)
    ])
    assert len(_normalise(payload, ["de"], 12)["pages"]) == 12


def test_normalise_rejects_an_unknown_layout():
    payload = _story(pages=[{"index": 1, "text": {"de": "A"}, "illustration": "a",
                             "layout": "diagonal-spiral"}])
    assert _normalise(payload, ["de"], 12)["pages"][0].layout == "full"


def test_wordless_pages_carry_no_text():
    payload = _story(pages=[{"index": 1, "text": {"de": "wird verworfen"},
                             "illustration": "a", "layout": "wordless"}])
    assert _normalise(payload, ["de"], 12)["pages"][0].text == {"de": ""}


def test_cast_names_the_model_invented_mid_story_are_dropped():
    payload = _story(pages=[{"index": 1, "text": {"de": "A"}, "illustration": "a",
                             "cast": ["Oma", "Ein Drache Der Nie Vorgestellt Wurde"]}])
    assert _normalise(payload, ["de"], 12)["pages"][0].cast == ["Oma"]


def test_place_is_attached_to_every_page():
    story = _normalise(_story(), ["de"], 12)
    garten = next(c for c in story["cast"] if c.kind == "place")
    assert garten.pages == [1, 2]


def test_normalise_accepts_a_single_language_string():
    payload = _story(pages=[{"index": 1, "text": "Nur ein String", "illustration": "a"}])
    assert _normalise(payload, ["de"], 12)["pages"][0].text == {"de": "Nur ein String"}


def test_normalise_raises_when_there_are_no_pages():
    with pytest.raises(RuntimeError):
        _normalise(_story(pages=[]), ["de"], 12)


# --------------------------------------------------------------------------- sizing


@pytest.mark.parametrize(
    "aspect,long_edge,expected",
    [("1:1", 2624, (2624, 2624)),
     ("4:3", 1024, (1024, 768)),
     ("3:4", 1024, (768, 1024)),
     ("16:9", 1024, (1024, 576))],
)
def test_pixel_size_follows_aspect_ratio(aspect, long_edge, expected):
    spec = OutputSpec(aspect_ratio=aspect, long_edge_px=long_edge)
    assert spec.pixel_size() == expected


def test_pixel_size_snaps_to_a_multiple_of_32():
    width, height = OutputSpec(aspect_ratio="7:3", long_edge_px=1000).pixel_size()
    assert width % 32 == 0 and height % 32 == 0


def test_long_edge_overrides_the_size_label():
    spec = OutputSpec(aspect_ratio="1:1", image_size="1K", long_edge_px=2624)
    assert spec.pixel_size() == (2624, 2624)
    # and without the override the label still decides
    assert OutputSpec(aspect_ratio="1:1", image_size="1K").pixel_size() == (1024, 1024)


def test_openai_size_is_divisible_by_16_and_within_the_ceiling():
    from storytime.backends.openai import MAX_EDGE, _exact_size

    for long_edge in (512, 1024, 2624, 3840, 6000):
        size = _exact_size(OutputSpec(aspect_ratio="4:3", long_edge_px=long_edge))
        width, height = (int(v) for v in size.split("x"))
        assert width % 16 == 0 and height % 16 == 0
        assert max(width, height) <= MAX_EDGE


# --------------------------------------------------------------------------- pricing


def test_price_uses_the_model_rate():
    usage = {"model": "gpt-image-2", "input_tokens": 1_000_000, "output_tokens": 0}
    assert price(usage) == pytest.approx(8.00)


def test_price_prefers_an_exact_amount_when_the_provider_gives_one():
    assert price({"model": "flux-2-pro", "usd": 0.03}) == 0.03


def test_unknown_model_costs_nothing_rather_than_guessing():
    assert price({"model": "some-future-model", "output_tokens": 5000}) == 0.0


def test_spend_accumulates_by_bucket():
    spend: dict = {}
    add_spend(spend, {"model": "gpt-image-2", "output_tokens": 100_000, "images": 1}, "pages")
    add_spend(spend, {"model": "gpt-image-2", "output_tokens": 100_000, "images": 1}, "pages")
    add_spend(spend, {"model": "gpt-image-2", "output_tokens": 100_000, "images": 1}, "cover")

    total = summary(spend)
    assert total["images"] == 3
    assert total["calls"] == 3
    assert total["by"]["pages"]["calls"] == 2
    # 100k output tokens at $30/M is $3 a call, three calls is $9
    assert total["usd"] == pytest.approx(9.0, abs=0.01)
    assert total["by"]["pages"]["usd"] == pytest.approx(6.0, abs=0.01)
    assert total["estimate"] is True
