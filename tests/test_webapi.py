"""Tests for the web layer's read API and the offline-capable job workers.

Everything here runs without a key: the workers under test either use the stub
backend or a cached file. The paid paths (real cover redraws, voice samples)
share the same code and are exercised only up to the point where money would
change hands.
"""

from __future__ import annotations

import zipfile

import pytest
from PIL import Image

from heldenbuch.book.library import Library
from heldenbuch.book.models import Book, Hero, Page, Style
from heldenbuch.pricing import image_estimate
from heldenbuch.web.bookapi import BookApi
from heldenbuch.web.bookjobs import BookJobs
from heldenbuch.web.jobs import Job


class _NoJobs:
    """Just enough of a JobManager for the read-only API."""

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


def _job(action: str, **params) -> Job:
    return Job(id="1", action=action, params=params)


# --------------------------------------------------------------------------- estimates


def test_image_estimate_scales_with_pixel_area():
    base = image_estimate("high", 1024)
    print_size = image_estimate("high", 2624)
    assert print_size == pytest.approx(base * (2624 / 1024) ** 2, rel=1e-3)


def test_image_estimate_scales_by_model_rate():
    """The mini model bills output tokens at 8 instead of 30 dollars a million."""
    standard = image_estimate("medium", 1024, "gpt-image-2")
    mini = image_estimate("medium", 1024, "gpt-image-1-mini")
    assert mini == pytest.approx(standard * 8 / 30, rel=1e-2)


def test_status_carries_estimates_for_the_buttons(api):
    status = api.status({}, None)
    estimates = status["estimates"]
    assert estimates["sheet_usd"] > estimates["preview_usd"] > 0
    assert estimates["usd_per_eur"] > 1

    profiles = {p["key"]: p for p in status["render_profiles"]}
    assert profiles["print"]["est_usd"] > profiles["draft"]["est_usd"]


def test_status_offers_models_for_every_backend_kind(api):
    models = api.status({}, None)["image_models"]
    backends = {m["backend"] for m in models}
    assert {"openai", "gemini", "bfl", "comfy"} <= backends
    flat = [m for m in models if m["pricing"] == "flat"]
    scaled = [m for m in models if m["pricing"] == "scaled"]
    # cloud models cost money; the local comfy backend is the one legitimate zero
    assert all(m["usd"] > 0 for m in flat if m["backend"] != "comfy")
    assert all(m["usd"] == 0 for m in flat if m["backend"] == "comfy")
    assert all(m["factor"] > 0 for m in scaled)


# --------------------------------------------------------------------------- scene fidelity


def test_page_prompt_lets_the_scene_override_the_reference():
    """The lost-boot bug: the reference always shows the character complete, so
    the prompt must say the scene wins for what is worn or carried."""
    from heldenbuch.book.illustrate import page_prompt

    hero = Hero(name="Mats", description="a boy")
    style = Style(name="S", description="watercolour")
    book = Book(hero_id=hero.id, style_id=style.id)
    page = Page(index=1, illustration="wearing only one yellow boot")

    prompt = page_prompt(book, hero, style, page, [])
    assert "the scene wins" in prompt
    assert "wearing only one yellow boot" in prompt


def test_author_brief_demands_story_state_in_the_instructions():
    from heldenbuch.book.author import _story_instructions

    text = _story_instructions(Hero(name="Mats", description="d"), "4-5", ["de"], 12)
    assert "boot lost in the mud" in text
    assert "every\nfollowing page" in text or "every following page" in text.replace("\n", " ")


# --------------------------------------------------------------------------- bilingual export


def test_export_pdf_can_set_two_languages_on_one_page(library):
    from heldenbuch.book.layout import PRESETS, export_pdf

    book = library.save_book(Book(
        title={"de": "Titel", "en": "Title"},
        languages=["de", "en"],
        pages=[Page(index=1, text={"de": "Deutscher Satz.", "en": "English sentence."})],
    ))
    root = library.book_dir(book.id)
    target = root / "export" / "bilingual.pdf"

    result = export_pdf(book, "de", PRESETS["screen"], lambda rel: root / rel, target,
                        log=lambda *a: None, secondary=["en"])
    assert target.is_file()
    assert result["language"] == "de+en"
    assert result["language_name"] == "Deutsch + English"


# --------------------------------------------------------------------------- book payload


def test_book_payload_reports_pixel_size_and_history(api, library):
    book = library.save_book(Book(title={"de": "T"}, pages=[
        Page(index=1, text={"de": "x"}, image="pages/page_01.png",
             history=["pages/page_01_v1.png"]),
    ]))
    _png(library.book_dir(book.id) / "pages" / "page_01.png", size=(512, 512))

    payload = api.book(book.id, {}, None)
    page = payload["pages"][0]
    assert page["image_px"] == 512
    assert page["history_urls"] == [f"/library/books/{book.id}/pages/page_01_v1.png"]


def test_book_update_accepts_the_cover_illustration(api, library):
    book = library.save_book(Book(title={"de": "T"}))
    api.book_update(book.id, {}, {"cover_illustration": "  a lighthouse at dusk "})
    assert library.get_book(book.id).cover_illustration == "a lighthouse at dusk"


def test_style_payload_carries_the_reference_url(api, library):
    style = library.save_style(Style(name="S", reference="styles/x/reference.jpg"))
    rows = api.styles({}, None)["styles"]
    assert rows[0]["reference_url"] == "/library/styles/x/reference.jpg"
    assert style.id == rows[0]["id"]


def test_export_files_get_human_labels(api, library):
    """The deliverables panel shows what a file IS, not what it is called."""
    book = library.save_book(Book(title={"de": "Mats im Zoo"}, languages=["de", "en"]))
    folder = library.book_dir(book.id) / "export"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("mats-im-zoo_de-en_print_square.pdf",
                 "mats-im-zoo_umschlag_print_square.pdf",
                 "mats-im-zoo_cover_print_square.jpg",
                 "mats-im-zoo_druckerei_lulu.md"):
        (folder / name).write_bytes(b"x")

    rows = {r["kind"]: r for r in api.book(book.id, {}, None)["exports"]}
    assert rows["interior"]["label"] == "Innenteil Deutsch + English — Druckerei — quadratisch 21,6 cm"
    assert rows["wrap"]["label"].startswith("Umschlag")
    assert rows["cover"]["kind"] == "cover"
    assert rows["handoff"]["label"] == "Anleitung für die Druckerei"


def test_render_preview_typesets_like_the_export(library):
    """The Setzprobe uses the real layout engine at a lighter dpi -- same
    millimetres, fewer pixels, cut guides on print presets."""
    from heldenbuch.book.layout import PRESETS, render_preview

    book = library.save_book(Book(
        title={"de": "Titel"}, languages=["de"],
        pages=[Page(index=1, text={"de": "Ein Satz."}, image="pages/page_01.png")],
    ))
    root = library.book_dir(book.id)
    _png(root / "pages/page_01.png", size=(256, 256))

    def resolve(rel):
        return root / rel

    title = render_preview(book, 0, PRESETS["print_square"], resolve, ["de"])
    page = render_preview(book, 1, PRESETS["print_square"], resolve, ["de"])
    # 110 dpi over 222.25 mm is ~963 px -- screen weight, not print weight
    assert 900 < title.width < 1000
    assert page.size == title.size

    with pytest.raises(ValueError):
        render_preview(book, 99, PRESETS["screen"], resolve, ["de"])


def test_page_preview_endpoint_serves_a_url(api, library):
    book = library.save_book(Book(
        title={"de": "T"}, languages=["de"],
        pages=[Page(index=1, text={"de": "x"})],
    ))
    result = api.book_page_preview(book.id, {"preset": ["screen"], "index": ["1"]}, None)
    assert result["url"].startswith(f"/library/books/{book.id}/preview/")
    assert result["pages"] == 1
    assert (library.book_dir(book.id) / "preview").is_dir()


# --------------------------------------------------------------------------- errors + progress


def test_provider_refusals_get_actionable_advice():
    """The child-safety refusal must arrive as advice, not a JSON blob."""
    from heldenbuch.backends.base import explain_provider_error

    refusal = '{"error": {"code": "moderation_blocked", "message": '
    refusal += '"Your request was rejected by our safety system."}}'
    hint = explain_provider_error(refusal)
    assert hint and "Sicherheitsprüfung" in hint
    assert "noch einmal" in hint  # the honest first tip: retries often pass

    assert "Guthaben" in explain_provider_error("insufficient_quota: ...")
    assert "warten" in explain_provider_error("Rate limit reached for gpt-image-2")
    assert explain_provider_error("something else entirely") is None


def test_domain_errors_carry_no_traceback_into_the_log():
    from heldenbuch.web.jobs import Job, JobManager

    def refuse(job, log):
        raise ValueError("Die Sicherheitsprüfung hat abgelehnt.")

    def crash(job, log):
        raise KeyError("boom")

    manager = JobManager()
    polite = Job(id="1", action="x", params={})
    manager._execute(polite, refuse)
    assert polite.error == "Die Sicherheitsprüfung hat abgelehnt."
    assert not any("Traceback" in line for line in polite.lines)

    rough = Job(id="2", action="x", params={})
    manager._execute(rough, crash)
    assert rough.error.startswith("KeyError")
    assert any("Traceback" in line for line in rough.lines)


def test_job_progress_is_public():
    from heldenbuch.web.jobs import Job

    job = Job(id="1", action="book_illustrate", params={})
    assert job.public()["progress"] is None
    job.progress = (3, 16)
    assert job.public()["progress"] == {"done": 3, "total": 16}


def test_illustrate_reports_progress_per_page(library, monkeypatch):
    from heldenbuch.book import illustrate
    from heldenbuch.book.models import CastMember  # noqa: F401

    hero = Hero(name="M", description="d")
    _png(library.hero_dir(hero.id) / "sheet.png")
    hero.sheet = f"heroes/{hero.id}/sheet.png"
    library.save_hero(hero)
    style = Style(name="S", description="d", sheets={hero.id: hero.sheet})
    library.save_style(style)
    book = Book(hero_id=hero.id, style_id=style.id,
                pages=[Page(index=i, text={"de": "x"}, illustration="a") for i in (1, 2)])
    library.save_book(book)

    class _R:
        usage = {"images": 1, "usd": 0.0}

    monkeypatch.setattr(illustrate, "draw_page", lambda *a, **k: _R())
    seen = []
    illustrate.illustrate_book(
        book, hero, style, library.resolve(hero.sheet),
        pages_dir=library.book_dir(book.id) / "pages",
        backend_name="stub", check=False, workers=1,
        on_progress=lambda d, t: seen.append((d, t)),
        log=lambda *a: None,
    )
    assert seen == [(0, 2), (1, 2), (2, 2)]


# --------------------------------------------------------------------------- backup


def test_backup_zips_the_whole_book_folder(api, library):
    book = library.save_book(Book(title={"de": "Mats im Zoo"}))
    _png(library.book_dir(book.id) / "pages" / "page_01.png")

    info = api.book_backup(book.id, {}, None)
    assert info["url"].startswith("/library/backups/")
    target = library.resolve(info["file"])
    with zipfile.ZipFile(target) as archive:
        names = set(archive.namelist())
    assert "book.json" in names
    assert "pages/page_01.png" in names


# --------------------------------------------------------------------------- workers, offline


def test_voice_preview_serves_the_cached_sample_without_an_api_call(library):
    target = library.root / "voices" / "coral.mp3"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not really audio")

    job = _job("voice_preview", voice="coral")
    BookJobs(library).voice_preview(job, lambda *a: None)
    assert job.result["url"] == "/library/voices/coral.mp3"


def test_voice_preview_rejects_an_unknown_voice(library):
    with pytest.raises(ValueError):
        BookJobs(library).voice_preview(_job("voice_preview", voice="darth"), lambda *a: None)


def test_style_adopt_draws_only_the_missing_sheet(library):
    hero = Hero(name="Mats", description="a boy")
    sheet = _png(library.hero_dir(hero.id) / "sheet_01.png")
    hero.sheet = library.relative(sheet)
    library.save_hero(hero)
    style = library.save_style(Style(name="Aquarell", description="watercolour"))

    job = _job("style_adopt", style_id=style.id, hero_id=hero.id, backend="stub")
    BookJobs(library).style_adopt(job, lambda *a: None)

    refreshed = library.get_style(style.id)
    assert hero.id in refreshed.sheets
    assert library.resolve(refreshed.sheets[hero.id]).is_file()


def test_style_adopt_is_a_no_op_when_the_sheet_exists(library):
    hero = Hero(name="Mats", description="a boy")
    sheet = _png(library.hero_dir(hero.id) / "sheet_01.png")
    hero.sheet = library.relative(sheet)
    library.save_hero(hero)
    style = Style(name="S", description="d", sheets={hero.id: hero.sheet})
    library.save_style(style)

    job = _job("style_adopt", style_id=style.id, hero_id=hero.id, backend="stub")
    BookJobs(library).style_adopt(job, lambda *a: None)
    assert library.get_style(style.id).sheets[hero.id] == hero.sheet  # unchanged


def test_cover_only_redraws_the_cover_and_nothing_else(library):
    hero = Hero(name="Mats", description="a boy")
    sheet = _png(library.hero_dir(hero.id) / "sheet_01.png")
    hero.sheet = library.relative(sheet)
    library.save_hero(hero)
    style = Style(name="S", description="d", sheets={hero.id: hero.sheet})
    library.save_style(style)
    book = library.save_book(Book(
        hero_id=hero.id, style_id=style.id, title={"de": "T"},
        pages=[Page(index=1, text={"de": "x"}, illustration="a meadow")],
    ))

    job = _job("book_illustrate", book_id=book.id, cover_only=True, backend="stub")
    BookJobs(library).book_illustrate(job, lambda *a: None)

    refreshed = library.get_book(book.id)
    assert refreshed.cover == "pages/cover.png"
    assert (library.book_dir(book.id) / "pages" / "cover.png").is_file()
    # the single page was not drawn -- this really was the cover alone
    assert refreshed.pages[0].image is None


# --------------------------------------------------------------------------- one scene per page


def test_single_scene_rewrites_a_diptych_brief_into_one_view():
    """The seam bug: "split" is a page layout, but the author read it as an
    instruction about the artwork and wrote "A split composition: on one
    side... on the other...". The illustrator obeyed literally and three pages
    printed with a join down the middle.
    """
    from heldenbuch.book.models import single_scene

    brief = ("A split composition: on one side, a close low view of the crooked "
             "plank in the bubbling brook; on the other, the nest is visible "
             "beyond the water among ferns. Late morning, curious mood.")
    out = single_scene(brief)

    assert "split composition" not in out.lower()
    assert "on one side" not in out.lower()
    assert "on the other" not in out.lower()
    # Both halves of the original are still described, now in one view.
    assert "crooked plank" in out and "the nest is visible" in out
    assert out.startswith("In the foreground,")
    assert "and beyond it," in out


@pytest.mark.parametrize("brief", [
    "A split composition: one side shows the splash; the other shows Mats on a stump.",
    "Two-panel image: on one side the door, on the other the garden.",
    "A diptych - in one half the cat sleeps, in the other half it wakes.",
])
def test_single_scene_catches_the_other_ways_of_asking_for_two_pictures(brief):
    from heldenbuch.book.models import single_scene

    out = single_scene(brief).lower()
    assert "diptych" not in out and "panel" not in out and "split" not in out
    assert "one side" not in out and "the other" not in out


def test_single_scene_leaves_an_ordinary_brief_alone():
    from heldenbuch.book.models import single_scene

    brief = "Mats kneels by the brook at dusk, holding one boot. Quiet, tired mood."
    assert single_scene(brief) == brief
    assert single_scene("") == ""


def test_page_prompt_forbids_panels_and_duplicate_characters():
    from heldenbuch.book.illustrate import page_prompt

    hero = Hero(name="Mats", description="a boy")
    style = Style(name="S", description="watercolour")
    book = Book(hero_id=hero.id, style_id=style.id)
    page = Page(index=1, layout="split",
                illustration="A split composition: on one side the brook; on the other the nest.")

    prompt = page_prompt(book, hero, style, page, [])
    assert "split composition" not in prompt.lower()
    assert "one single continuous scene" in prompt
    assert "exactly once" in prompt


def test_vignette_prompt_no_longer_asks_for_faded_edges():
    """The art arrived already inset, then the page layout inset it again and
    printed a full-page render at a fraction of the paper."""
    from heldenbuch.book.illustrate import page_prompt

    hero = Hero(name="Mats", description="a boy")
    style = Style(name="S", description="d")
    book = Book(hero_id=hero.id, style_id=style.id)
    page = Page(index=1, layout="vignette", illustration="Mats naps under a hedge")

    prompt = page_prompt(book, hero, style, page, [])
    assert "fading out" not in prompt
    assert "Fill the whole frame" in prompt


# --------------------------------------------------------------------------- the cover knows the cast


def test_cover_prompt_ties_every_cast_name_to_a_reference_image():
    """A cover reading "Simon and Chase" was given only Simon's sheet, so the
    rescue pup had nothing but a name behind it and was drawn as a second boy.
    """
    from heldenbuch.book.illustrate import cover_prompt
    from heldenbuch.book.models import CastMember

    hero = Hero(name="Simon", description="a boy")
    style = Style(name="S", description="d")
    book = Book(hero_id=hero.id, style_id=style.id,
                cover_illustration="Simon and Chase in the valley with Trixi")
    cast = [CastMember(name="Chase", description="a rescue pup", sheet="cast/01.png"),
            CastMember(name="Trixi", description="a triceratops", sheet="cast/02.png")]

    prompt = cover_prompt(book, hero, style, cast)
    assert "Image 2 is the reference for Chase" in prompt
    assert "Image 3 is the reference for Trixi" in prompt
    assert "Simon is image 1" in prompt and "Chase is image 2" in prompt
    assert "a name alone never" in prompt


def test_cover_prompt_without_cast_still_works():
    from heldenbuch.book.illustrate import cover_prompt

    hero = Hero(name="Mats", description="a boy")
    style = Style(name="S", description="d")
    book = Book(hero_id=hero.id, style_id=style.id)

    prompt = cover_prompt(book, hero, style)
    assert "Image 1 is the character reference sheet for Mats" in prompt
    assert "WHO IS WHO" not in prompt  # nothing to disambiguate with one character


def test_attach_drops_the_name_when_its_sheet_is_missing(tmp_path):
    """The prompt numbers references by position, so a member whose sheet is
    not on disk has to vanish from the wording too -- otherwise every name
    after it points at the wrong picture.
    """
    from heldenbuch.book.illustrate import _attach
    from heldenbuch.book.models import CastMember

    sheet = _png(tmp_path / "hero.png")
    real = _png(tmp_path / "trixi.png")
    cast = [CastMember(name="Chase", sheet="missing.png"),
            CastMember(name="Trixi", sheet="trixi.png")]

    paths, named = _attach(sheet, cast, lambda rel: tmp_path / rel, limit=8)
    assert paths == [sheet, real]
    assert [m.name for m in named] == ["Trixi"]


def test_attach_respects_the_backend_reference_limit(tmp_path):
    from heldenbuch.book.illustrate import _attach
    from heldenbuch.book.models import CastMember

    sheet = _png(tmp_path / "hero.png")
    for name in ("a", "b", "c"):
        _png(tmp_path / f"{name}.png")
    cast = [CastMember(name=n.upper(), sheet=f"{n}.png") for n in ("a", "b", "c")]

    paths, named = _attach(sheet, cast, lambda rel: tmp_path / rel, limit=2)
    assert len(paths) == 2
    assert [m.name for m in named] == ["A"]


# --------------------------------------------------------------------------- the quality gate


def test_a_duplicated_character_fails_the_check_whatever_it_scored():
    """The checker wrote "Chase and Trixi are duplicated" into the notes and
    then scored the page a passing 3. It shipped. The note is now read directly.
    """
    from heldenbuch.book.illustrate import duplicate_note

    assert duplicate_note(["Chase and Trixi are duplicated"])
    assert duplicate_note(["an extra uniformed dog appears beside the trunk"])
    assert duplicate_note(["An extra triceratops-like creature is present"])
    assert duplicate_note(["the hero appears twice"])
    assert duplicate_note(["hair is blonde, should be dark brown"]) is None
    assert duplicate_note([]) is None


def test_the_gate_turns_on_facts_not_on_a_blended_score():
    """The scene score mixes a ruined page with a camera quibble.

    Raising its floor to 4 made four framing complaints per book into paid
    redraws while still passing a page whose own notes said two characters
    were duplicated. The faults that matter are asked as yes/no questions and
    fail the page on their own; the score keeps a floor of 3 for "a different
    scene entirely".
    """
    from heldenbuch.book.illustrate import (
        FATAL_FACTS,
        IDENTITY_FLOOR,
        SCENE_FLOOR,
        verdict_from,
    )

    assert (IDENTITY_FLOOR, SCENE_FLOOR) == (4, 3)
    assert set(FATAL_FACTS) == {
        "extra_or_duplicated_character", "panelled", "story_state_ok"}

    good = {"identity": 5, "style": 5, "scene": 5, "notes": []}
    assert verdict_from(good) == "passed"

    # A framing quibble is a note, not a redraw.
    assert verdict_from({**good, "scene": 3,
                         "notes": ["view is wider than the brief asked"]}) == "passed"

    # Each fatal fact fails on its own, at a perfect score.
    assert verdict_from({**good, "extra_or_duplicated_character": True}) == "failed"
    assert verdict_from({**good, "panelled": True}) == "failed"
    assert verdict_from({**good, "story_state_ok": False}) == "failed"
    assert verdict_from({**good, "notes": ["Chase and Trixi are duplicated"]}) == "failed"
    assert verdict_from({**good, "identity": 3}) == "failed"
    assert verdict_from({**good, "scene": 2}) == "failed"


def test_a_stored_verdict_is_re_judged_under_todays_rules():
    """The frozen-boolean bug: every page of the first finished book carried
    ok: true while its notes described duplicated characters, and no later fix
    could reach it because the flag was read instead of the evidence."""
    from heldenbuch.book.illustrate import check_status

    stale = Page(index=1, check={
        "ok": True, "status": "passed", "identity": 5, "style": 5, "scene": 3,
        "notes": ["Chase and Trixi are duplicated: one pair stands on the bridge"],
    })
    assert check_status(stale) == "failed"


def test_a_check_that_errored_is_never_read_as_a_pass():
    from heldenbuch.book.illustrate import check_status

    assert check_status(Page(index=1, check={"error": "timeout"})) == "unknown"
    assert check_status(Page(index=1, check={})) == "unchecked"


@pytest.mark.parametrize("brief", [
    "Mats stands on the other side of the brook, looking back at the nest.",
    "A wide view: the hedge on one side of the path, the gate beyond it.",
])
def test_single_scene_does_not_maul_ordinary_prose(brief):
    """"on the other side OF the brook" already describes one continuous view.
    Rewriting it would produce "and beyond it, of the brook"."""
    from heldenbuch.book.models import single_scene

    assert single_scene(brief) == brief
