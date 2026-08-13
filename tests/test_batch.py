"""The Gemini batch path -- everything that runs without Google.

The transport (upload, submit, poll) needs a live batch job and is exercised
the day one runs. What is pinned here: the request payload shape, the
one-upload-per-reference dedup, and that the orchestration writes images,
meters half price, and keeps redraws undoable.
"""

from __future__ import annotations

import pytest
from PIL import Image

from heldenbuch.backends.gemini import build_inline_requests
from heldenbuch.book import illustrate
from heldenbuch.book.library import Library
from heldenbuch.book.models import Book, Hero, Page, Style
from heldenbuch.types import GenRequest, OutputSpec


@pytest.fixture()
def library(tmp_path) -> Library:
    return Library(tmp_path / "library")


def _png(path, size=(32, 32)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return path


# ------------------------------------------------------------------ payload


def test_inline_requests_carry_prompt_refs_and_image_config(tmp_path):
    sheet = _png(tmp_path / "sheet.png")
    requests = [GenRequest(prompt="draw the fox", reference_images=[sheet],
                           output=OutputSpec(aspect_ratio="1:1", image_size="4K"))]

    calls = []

    def uri_for(path):
        calls.append(path)
        return ("files/abc123", "image/png")

    inline = build_inline_requests(requests, uri_for)
    parts = inline[0]["contents"][0]["parts"]
    assert parts[0] == {"text": "draw the fox"}
    assert parts[1]["file_data"]["file_uri"] == "files/abc123"
    config = inline[0]["config"]
    assert config["response_modalities"] == ["TEXT", "IMAGE"]
    assert config["image_config"] == {"aspect_ratio": "1:1", "image_size": "4K"}


def test_the_shared_sheet_is_resolved_once_per_page_but_cacheable(tmp_path):
    """Sixteen pages share one character sheet; the uri_for cache in
    run_batch uploads it once. Here: the payload asks for it per page, and
    the same path arrives each time, so a dict cache collapses it."""
    sheet = _png(tmp_path / "sheet.png")
    requests = [GenRequest(prompt=f"page {i}", reference_images=[sheet])
                for i in range(3)]

    seen = {}

    def uri_for(path):
        seen[path] = seen.get(path, 0) + 1
        return ("files/x", "image/png")

    build_inline_requests(requests, uri_for)
    assert seen == {sheet: 3}  # same key three times -> one upload in the cache


# ------------------------------------------------------------- orchestration


def _book(library):
    hero = Hero(name="Simon", description="a boy")
    _png(library.hero_dir(hero.id) / "sheet.png")
    hero.sheet = f"heroes/{hero.id}/sheet.png"
    library.save_hero(hero)
    style = Style(name="S", description="d", sheets={hero.id: hero.sheet})
    library.save_style(style)
    book = Book(hero_id=hero.id, style_id=style.id, title={"de": "T"},
                pages=[Page(index=i, text={"de": str(i)}, illustration=f"scene {i}")
                       for i in (1, 2)])
    library.save_book(book)
    return hero, style, book


def test_batch_orchestration_writes_images_at_half_price(library, monkeypatch):
    hero, style, book = _book(library)
    submitted = {}

    def fake_run_batch(model, requests, targets, **kwargs):
        submitted["prompts"] = [r.prompt[:40] for r in requests]
        pixel = Image.new("RGB", (8, 8), "blue")
        import io
        buffer = io.BytesIO(); pixel.save(buffer, "PNG")
        return [{"data": buffer.getvalue(),
                 "usage": {"images": 1, "usd": 0.067, "model": "gemini-3-pro-image",
                           "backend": "gemini"}}
                for _ in requests]

    monkeypatch.setattr("heldenbuch.backends.gemini.run_batch", fake_run_batch)
    illustrate.illustrate_book_batch(
        book, hero, style, library.resolve(hero.sheet),
        pages_dir=library.book_dir(book.id) / "pages",
        check=False, log=lambda *a: None,
    )

    # cover + two pages, each written and metered at the batch rate
    assert len(submitted["prompts"]) == 3
    assert book.cover == "pages/cover.png"
    assert all(p.image for p in book.pages)
    assert (library.book_dir(book.id) / "pages/page_02.png").is_file()
    assert book.spend["usd"] == pytest.approx(3 * 0.067)
    assert book.spend["by"]["pages"]["calls"] == 2
    assert book.spend["by"]["cover"]["calls"] == 1


def test_batch_failures_land_on_the_page_not_in_the_void(library, monkeypatch):
    hero, style, book = _book(library)

    def fake_run_batch(model, requests, targets, **kwargs):
        import io
        pixel = io.BytesIO(); Image.new("RGB", (8, 8)).save(pixel, "PNG")
        results = [{"data": pixel.getvalue(), "usage": {"images": 1, "usd": 0.067}}
                   for _ in requests]
        results[-1] = {"error": "die Antwort enthielt kein Bild"}
        return results

    monkeypatch.setattr("heldenbuch.backends.gemini.run_batch", fake_run_batch)
    illustrate.illustrate_book_batch(
        book, hero, style, library.resolve(hero.sheet),
        pages_dir=library.book_dir(book.id) / "pages",
        check=False, log=lambda *a: None,
    )
    assert book.pages[0].image
    assert book.pages[1].image is None
    assert "kein Bild" in book.pages[1].error
    assert illustrate.flagged_pages(book) == [2]


def test_batch_redraw_keeps_the_old_version_undoable(library, monkeypatch):
    hero, style, book = _book(library)
    pages_dir = library.book_dir(book.id) / "pages"
    _png(pages_dir / "page_01.png")
    _png(pages_dir / "page_02.png")
    _png(pages_dir / "cover.png")
    book.cover = "pages/cover.png"

    def fake_run_batch(model, requests, targets, **kwargs):
        import io
        pixel = io.BytesIO(); Image.new("RGB", (8, 8), "red").save(pixel, "PNG")
        return [{"data": pixel.getvalue(), "usage": {"images": 1, "usd": 0.067}}
                for _ in requests]

    monkeypatch.setattr("heldenbuch.backends.gemini.run_batch", fake_run_batch)
    illustrate.illustrate_book_batch(
        book, hero, style, library.resolve(hero.sheet),
        pages_dir=pages_dir, redraw=True, check=False, log=lambda *a: None,
    )
    assert book.pages[0].history == ["pages/page_01_v1.png"]
    assert (pages_dir / "page_01_v1.png").is_file()


class TestOneBadImageCannotSinkTheBatch:
    """A filtered candidate took down the collection of a finished batch.

    Google had run and billed seventeen print-quality images; one came back
    with `content.parts = None`, iterating it raised TypeError, the old guard
    caught only AttributeError and IndexError, and the whole job failed after
    the money was spent.
    """

    @staticmethod
    def _response(parts=None, content=True, candidates=True, finish=None):
        class Blob:
            def __init__(self, data):
                self.data = data

        class Part:
            def __init__(self, data):
                self.inline_data = Blob(data)

        class Content:
            def __init__(self, parts):
                self.parts = parts

        class Candidate:
            def __init__(self, content, finish):
                self.content = content
                self.finish_reason = finish

        class Response:
            def __init__(self, candidates):
                self.candidates = candidates

        made = None
        if content:
            made = Content([Part(p) for p in parts] if parts is not None else None)
        return Response([Candidate(made, finish)] if candidates else [])

    def test_the_exact_shape_that_crashed(self):
        """content is present, parts is None."""
        from heldenbuch.backends.gemini import _image_from

        data, why = _image_from(self._response(parts=None))
        assert data is None
        assert why, "a reason must be reported, not an exception"

    def test_a_good_response_still_yields_bytes(self):
        from heldenbuch.backends.gemini import _image_from

        data, why = _image_from(self._response(parts=[b"PNGDATA"]))
        assert data == b"PNGDATA"
        assert why == ""

    def test_a_blocked_candidate_says_why(self):
        from heldenbuch.backends.gemini import _image_from

        class Reason:
            name = "IMAGE_SAFETY"

        _data, why = _image_from(self._response(parts=[], finish=Reason()))
        assert "Bildfilter" in why

    @pytest.mark.parametrize("kwargs", [
        {"parts": None},
        {"parts": []},
        {"content": False},
        {"candidates": False},
    ])
    def test_no_shape_raises(self, kwargs):
        from heldenbuch.backends.gemini import _image_from

        data, why = _image_from(self._response(**kwargs))
        assert data is None and isinstance(why, str) and why
