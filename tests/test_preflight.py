"""The export gate and the fail-closed checker.

The product promise is a file a print shop will accept. These tests pin the
two halves of that promise: a broken checker can never produce a "passed"
page, and a broken book can never produce a print PDF.
"""

from __future__ import annotations

import pytest
from PIL import Image

from storytime.book import illustrate
from storytime.book.illustrate import check_page, check_status, flagged_pages, review_split
from storytime.book.layout import PRESETS
from storytime.book.library import Library
from storytime.book.models import Book, Hero, Page
from storytime.book.preflight import validate_export_readiness
from storytime.web.bookjobs import BookJobs
from storytime.web.jobs import Job


@pytest.fixture()
def library(tmp_path) -> Library:
    return Library(tmp_path / "library")


def _png(path, size=(512, 512)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return path


def _book(library, pages=3, check=None, langs=("de",), cover=True, px=512) -> tuple[Book, callable]:
    """A book on disk with `pages` drawn pages, ready for the preflight."""
    book = Book(title={c: "Titel" for c in langs}, languages=list(langs),
                dedication={c: "Für dich" for c in langs})
    book.pages = [
        Page(index=i, text={c: f"Text {i}" for c in langs},
             image=f"pages/page_{i:02d}.png",
             check=dict(check) if check else {})
        for i in range(1, pages + 1)
    ]
    library.save_book(book)
    root = library.book_dir(book.id)
    for i in range(1, pages + 1):
        _png(root / f"pages/page_{i:02d}.png", size=(px, px))
    if cover:
        _png(root / "pages/cover.png", size=(px, px))
        book.cover = "pages/cover.png"
    return book, (lambda rel: root / rel)


PASSED = {"status": "passed", "ok": True, "identity": 5, "style": 5}


# --------------------------------------------------------------------- states


def test_a_healthy_book_passes(library):
    book, resolve = _book(library, check=PASSED)
    report = validate_export_readiness(book, PRESETS["screen"], ["de"], resolve)
    assert report["state"] == "bereit"
    assert report["ok"] is True


def test_a_missing_page_blocks_print(library):
    book, resolve = _book(library, check=PASSED)
    (library.book_dir(book.id) / "pages/page_02.png").unlink()
    report = validate_export_readiness(book, PRESETS["print_square"], ["de"], resolve)
    assert report["state"] == "unvollständig"
    assert report["ok"] is False
    assert report["pages_missing"] == [2]


def test_a_failed_check_blocks_print(library):
    book, resolve = _book(library, check={"status": "failed", "ok": False, "identity": 2}, px=3000)
    report = validate_export_readiness(book, PRESETS["print_square"], ["de"], resolve)
    assert report["ok"] is False
    assert report["pages_failed_check"] == [1, 2, 3]


def test_unknown_checks_block_print_unless_overridden(library):
    book, resolve = _book(library, check={"status": "unknown", "error": "timeout"}, px=3000)
    report = validate_export_readiness(book, PRESETS["print_square"], ["de"], resolve)
    assert report["state"] == "unbekannt"
    assert report["ok"] is False

    forced = validate_export_readiness(book, PRESETS["print_square"], ["de"], resolve,
                                       allow_unknown=True)
    assert forced["ok"] is True
    assert forced["state"] == "warnung"


def test_never_checked_pages_count_as_unknown(library):
    book, resolve = _book(library, check=None, px=3000)  # the stub-book case
    report = validate_export_readiness(book, PRESETS["print_square"], ["de"], resolve)
    assert report["state"] == "unbekannt"
    assert report["pages_unknown_check"] == [1, 2, 3]


def test_low_dpi_blocks_a_print_export(library):
    """It used to be a warning that left the export button enabled.

    Every illustration is scaled up to fill the page, so the PDF reports the
    preset's dpi whatever the artwork actually holds -- a print shop's own
    preflight passes it, and the softness only shows when the book arrives.
    Nothing downstream can catch this, so it has to stop here.
    """
    book, resolve = _book(library, check=PASSED, px=512)  # ~59 dpi at 21.6 cm
    report = validate_export_readiness(book, PRESETS["print_square"], ["de"], resolve)
    assert report["ok"] is False
    assert any("dpi" in e for e in report["errors"])


def test_dpi_is_judged_against_the_selected_format(library):
    """512 px is hopeless on a 21.6 cm page but fine for the screen preset."""
    book, resolve = _book(library, check=PASSED, px=512)
    screen = validate_export_readiness(book, PRESETS["screen"], ["de"], resolve)
    assert not any("dpi" in w for w in screen["warnings"])


def test_missing_language_text_blocks(library):
    book, resolve = _book(library, check=PASSED, langs=("de", "en"))
    book.pages[1].text["en"] = ""
    report = validate_export_readiness(book, PRESETS["print_square"], ["de", "en"], resolve)
    assert report["ok"] is False
    assert any("English" in e for e in report["errors"])


def test_wordless_pages_need_no_text(library):
    book, resolve = _book(library, check=PASSED, px=3000)
    book.pages[0].layout = "wordless"
    book.pages[0].text = {"de": ""}
    report = validate_export_readiness(book, PRESETS["print_square"], ["de"], resolve)
    assert report["ok"] is True


def test_screen_preset_downgrades_blockers_to_warnings(library):
    book, resolve = _book(library, check=PASSED)
    (library.book_dir(book.id) / "pages/page_02.png").unlink()
    report = validate_export_readiness(book, PRESETS["screen"], ["de"], resolve)
    assert report["ok"] is True
    assert report["state"] == "warnung"
    assert report["pages_missing"] == [2]  # still reported, just not blocking


def test_a_corrupt_image_blocks(library):
    book, resolve = _book(library, check=PASSED, px=3000)
    (library.book_dir(book.id) / "pages/page_01.png").write_bytes(b"not a png")
    report = validate_export_readiness(book, PRESETS["print_square"], ["de"], resolve)
    assert report["ok"] is False
    assert any("beschädigt" in e for e in report["errors"])


# ------------------------------------------------------------- export wiring


def test_export_job_refuses_a_blocked_print_book(library):
    book, _ = _book(library, check=PASSED)
    (library.book_dir(book.id) / "pages/page_02.png").unlink()
    job = Job(id="1", action="book_export",
              params={"book_id": book.id, "preset": "print_square", "languages": ["de"]})
    with pytest.raises(ValueError, match="blockiert"):
        BookJobs(library).book_export(job, lambda *a: None)


def test_export_job_builds_a_screen_pdf_despite_warnings(library):
    book, _ = _book(library, check=None)  # unchecked pages: warning on screen
    job = Job(id="1", action="book_export",
              params={"book_id": book.id, "preset": "screen", "languages": ["de"]})
    BookJobs(library).book_export(job, lambda *a: None)
    assert job.result["exports"], "the screen export should have produced a file"


# ------------------------------------------------------------- fail closed


def test_checker_crash_is_unknown_not_a_pass(tmp_path, monkeypatch):
    sheet = _png(tmp_path / "sheet.png")
    page = _png(tmp_path / "page.png")

    def boom(*args, **kwargs):
        raise TimeoutError("provider unreachable")

    monkeypatch.setattr(illustrate, "complete_json", boom)
    verdict = check_page(page, sheet, Hero(name="Claudio"), scene="a meadow")
    assert verdict["status"] == "unknown"
    assert "ok" not in verdict
    assert "provider unreachable" in verdict["error"]


def test_unknown_pages_are_flagged_for_review():
    book = Book(pages=[
        Page(index=1, image="pages/p1.png", check={"status": "passed", "ok": True}),
        Page(index=2, image="pages/p2.png", check={"status": "unknown", "error": "x"}),
        Page(index=3, image="pages/p3.png", check={"status": "failed", "ok": False}),
    ])
    assert flagged_pages(book) == [2, 3]
    split = review_split(book)
    assert split["unknown"] == [2]
    assert split["failed"] == [3]


def test_check_status_maps_books_written_before_the_status_field():
    """A verdict with no scores behind it can only be read from what it stored.

    Where scores *are* present the verdict is worked out from them instead, so
    a rule tightened today reaches a book checked last week -- see
    test_a_stored_verdict_is_re_judged_under_todays_rules.
    """
    assert check_status(Page(check={"ok": True})) == "passed"
    assert check_status(Page(check={"ok": False})) == "failed"
    assert check_status(Page(check={"error": "boom"})) == "unknown"
    assert check_status(Page(check={})) == "unchecked"
    # Scores present, nothing wrong recorded: derived, not guessed.
    assert check_status(Page(check={"identity": 4})) == "passed"
    assert check_status(Page(check={"identity": 2})) == "failed"


def test_a_beanstandetes_titelbild_blocks_the_export(library):
    """The cover is the image on the shelf and the one a buyer sees first.

    It was drawn, paid for, and never checked -- which is how a book called
    "Claudio und Pip" shipped with the dog drawn as a second human boy.
    """
    from storytime.book.illustrate import cover_flagged

    book, resolve = _book(library, check=PASSED)
    book.cover_check = {
        "identity": 5, "style": 5, "scene": 5,
        "extra_or_duplicated_character": True,
        "notes": ["Pip is drawn as a boy, not a dog"],
    }
    assert cover_flagged(book) is True

    # A print preset blocks on it. (A screen file is not a bound book, so
    # there everything is downgraded to a warning by design.)
    report = validate_export_readiness(book, PRESETS["print_square"], ["de"], resolve)
    assert report["ok"] is False
    assert any("Titelbild wurde beanstandet" in e for e in report["errors"])

    screen = validate_export_readiness(book, PRESETS["screen"], ["de"], resolve)
    assert screen["ok"] is True
    assert any("Titelbild wurde beanstandet" in w for w in screen["warnings"])


def test_a_book_with_no_cover_verdict_is_not_nagged(library):
    """Older books carry no evidence either way. Inventing a complaint about
    every one of them just teaches people to ignore the flag."""
    from storytime.book.illustrate import cover_flagged

    book, resolve = _book(library, check=PASSED)
    book.cover_check = {}
    assert cover_flagged(book) is False
    report = validate_export_readiness(book, PRESETS["screen"], ["de"], resolve)
    assert not any("Titelbild wurde beanstandet" in e for e in report["errors"])
