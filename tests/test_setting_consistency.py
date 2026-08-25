"""The setting machinery: per-page anchors, multi-view places, the chain.

Scene consistency has the same shape as character consistency: whatever has
no reference and no plan is reinvented per page. These tests pin the three
mechanisms that changed that -- the authored setting slot, the multi-place
attachment rule, and the opt-in chained render.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from heldenbuch.book import illustrate
from heldenbuch.book.library import Library
from heldenbuch.book.models import Book, CastMember, Hero, Page, Style


@pytest.fixture()
def library(tmp_path) -> Library:
    return Library(tmp_path / "library")


def _png(path: Path, size=(64, 64), colour="white") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)
    return path


# ------------------------------------------------------------------ the slot


def test_the_setting_rides_in_the_prompt_as_direction():
    hero, style = Hero(name="Rusty"), Style(description="ink")
    page = Page(index=1, illustration="Rusty runs.",
                setting="at the brook, the crooked pine on the left bank")
    prompt = illustrate.page_prompt(Book(), hero, style, page, [])
    assert "Where this stands: at the brook, the crooked pine on the left bank." in prompt
    bare = illustrate.page_prompt(Book(), hero, style, Page(index=1), [])
    assert "Where this stands" not in bare


def test_the_author_reply_keeps_the_setting():
    from heldenbuch.book.author import _normalise

    payload = {"pages": [{"index": 1, "text": {"de": "x"},
                          "illustration": "a brook",
                          "setting": "at the brook, the crooked pine"}]}
    story = _normalise(payload, ["de"], 1)
    assert story["pages"][0].setting == "at the brook, the crooked pine"


def test_the_author_is_told_to_repeat_anchors():
    from heldenbuch.book.author import _story_instructions

    text = _story_instructions(Hero(name="X", description="d"), "4-5", ["de"], 8)
    assert '"setting"' in text
    assert "SAME features in the SAME words" in text


# ------------------------------------------------------------ multiple places


def test_a_single_place_still_rides_on_every_page():
    book = Book(cast=[CastMember(name="Der Garten", kind="place", sheet="c/1.png")])
    page = Page(index=1, illustration="A quiet morning.")
    assert [m.name for m in book.cast_for(page)] == ["Der Garten"]


def test_two_places_follow_the_page_not_each_other():
    """A story that moves between areas must not get both settings in one
    prompt -- two competing places is its own kind of drift."""
    book = Book(cast=[
        CastMember(name="Der Garten", kind="place", sheet="c/1.png"),
        CastMember(name="Der Leuchtturm", kind="place", sheet="c/2.png"),
    ])
    at_sea = Page(index=5, cast=["Der Leuchtturm"],
                  illustration="Waves at the tower.")
    assert [m.name for m in book.cast_for(at_sea)] == ["Der Leuchtturm"]
    # A page that names no place falls back to the first, never to both.
    nowhere = Page(index=2, illustration="A close-up of two paws.")
    assert [m.name for m in book.cast_for(nowhere)] == ["Der Garten"]


def test_the_place_sheet_asks_for_several_views_of_one_place():
    from heldenbuch.book.cast import sheet_prompt

    member = CastMember(name="Der Nebelwald", kind="place", description="mist")
    prompt = sheet_prompt(member, Style(description="ink"))
    assert "SAME single place three times" in prompt
    assert "No text, letters, numbers or labels" in prompt


# ------------------------------------------------------------------ the chain


class _FakeBackend:
    max_references = 3
    honours_seed = False

    def __init__(self, seen: list):
        self._seen = seen

    def generate(self, request, target: Path):
        self._seen.append(request)
        _png(target)

        class R:
            usage = {}

        return R()


def test_the_previous_page_takes_the_last_free_slot(tmp_path, monkeypatch):
    seen: list = []
    monkeypatch.setattr(illustrate, "get_backend",
                        lambda *a, **k: _FakeBackend(seen))
    sheet = _png(tmp_path / "sheet.png")
    prev = _png(tmp_path / "prev.png")
    hero, style = Hero(name="Rusty"), Style(description="ink")
    page = Page(index=2, illustration="Rusty walks on.")

    illustrate.draw_page(Book(), hero, style, page, sheet,
                         tmp_path / "out.png", previous=prev)
    request = seen[-1]
    assert request.reference_images[-1] == prev
    assert f"Image {len(request.reference_images)} is the page that comes directly" \
        in request.prompt


def test_the_cast_outranks_the_previous_page(tmp_path, monkeypatch):
    """With the budget full, the chain image is the one that stays behind:
    a page without its place sheet drifts harder than one without its
    predecessor."""
    seen: list = []
    monkeypatch.setattr(illustrate, "get_backend",
                        lambda *a, **k: _FakeBackend(seen))
    sheet = _png(tmp_path / "sheet.png")
    prev = _png(tmp_path / "prev.png")
    cast = [CastMember(name="Garten", kind="place", sheet="a.png"),
            CastMember(name="Pip", kind="character", sheet="b.png")]
    for name in ("a", "b"):
        _png(tmp_path / f"{name}.png")
    page = Page(index=2, illustration="Pip in the Garten.")

    illustrate.draw_page(Book(cast=cast), Hero(name="R"), Style(description="i"),
                         page, sheet, tmp_path / "out.png",
                         members=cast, previous=prev,
                         resolve=lambda rel: tmp_path / rel)
    request = seen[-1]
    assert prev not in request.reference_images
    assert "the page that comes directly" not in request.prompt


def test_chained_render_passes_the_last_good_page_forward(library, monkeypatch):
    hero = Hero(name="Rusty", sheet="heroes/h/sheet.png")
    style = Style(description="ink")
    book = library.save_book(Book(pages=[
        Page(index=i, illustration=f"scene {i}") for i in (1, 2, 3)
    ]))
    root = library.book_dir(book.id)
    sheet = _png(library.root / "heroes/h/sheet.png")

    order: list[tuple[int, str | None]] = []

    def fake_draw(book_, hero_, style_, page, sheet_, target, **kwargs):
        previous = kwargs.get("previous")
        order.append((page.index, previous.name if previous else None))
        _png(target)

        class R:
            usage = {}

        return R()

    monkeypatch.setattr(illustrate, "draw_page", fake_draw)
    illustrate.illustrate_book(
        book, hero, style, sheet, pages_dir=root / "pages",
        backend_name="stub", check=False, chain=True, workers=4,
        resolve=lambda rel: root / rel, log=lambda *a: None,
    )
    pages = [(i, prev) for i, prev in order if i > 0]
    assert pages == [(1, None), (2, "page_01.png"), (3, "page_02.png")]


# ----------------------------------------------------------------- the judge


def test_the_judge_asks_about_the_setting_only_with_a_place(monkeypatch):
    asked: list[str] = []

    def fake_complete(system, user, **kwargs):
        asked.append(user)
        return {"identity": 5, "style": 5, "extra_or_duplicated_character": False,
                "panelled": False, "setting_consistent": False, "notes": []}

    monkeypatch.setattr(illustrate, "complete_json", fake_complete)
    monkeypatch.setattr(illustrate, "score_page", lambda *a: {})
    monkeypatch.setattr(illustrate, "seam_in_frame", lambda *a, **k: False)

    verdict = illustrate.check_page(Path("p.png"), Path("s.png"), Hero(name="R"),
                                    place_name="Der Nebelwald")
    assert "setting_consistent" in asked[0]
    assert verdict["setting_consistent"] is False
    assert any("does not match the reference for Der Nebelwald" in n
               for n in verdict["notes"])
    # a hint, never a fail, and never counted as a missing fatal fact
    assert verdict["status"] == "passed"

    asked.clear()
    illustrate.check_page(Path("p.png"), Path("s.png"), Hero(name="R"))
    assert "setting_consistent" not in asked[0]


def test_setting_mismatches_land_in_the_preflight_notes(library):
    from heldenbuch.book.layout import PRESETS
    from heldenbuch.book.preflight import validate_export_readiness

    book = library.save_book(Book(title={"de": "T"}, pages=[
        Page(index=1, text={"de": "x"}, image="pages/page_01.png",
             check={"status": "passed", "ok": True, "identity": 5,
                    "setting_consistent": False}),
    ]))
    root = library.book_dir(book.id)
    _png(root / "pages/page_01.png", size=(700, 700))

    report = validate_export_readiness(book, PRESETS["screen"], ["de"],
                                       lambda rel: root / rel)
    assert any(f["code"] == "setting_mismatch" for f in report["notes"])
