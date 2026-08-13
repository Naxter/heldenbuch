"""Drawing the pages, and checking that everyone survived.

Every page points at the *styled character sheet* -- the hero already redrawn
in the chosen look -- plus a sheet for each cast member who appears on that
page and, if the story has one, the place. Identity and style arrive together
as pictures rather than being argued out in words each time.

After each page is drawn it is checked against the hero sheet. The check is the
benchmark machinery from `metrics/`, used here as a quality gate: colour
metrics first because they are free, then a vision model that can actually see
whether the face is the same. A page that drifts is redrawn once, automatically,
with a sharper prompt; only if that also fails is a person bothered.

Pages are drawn several at a time. They used to be chained -- each page shown
the previous one -- but once the styled sheet carries the look, that chain buys
little and costs the whole render being sequential.
"""

from __future__ import annotations

import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..backends import get_backend
from ..llm import complete_json
from ..metrics.cheap import score_page
from ..pricing import add as add_spend
from ..pricing import image_estimate
from ..types import GenRequest, OutputSpec
from .models import Book, CastMember, Hero, Page, Style, single_scene

# Below this the page is redrawn. 4 of 5 means "clearly the same character,
# maybe a small detail off"; 3 means the face has shifted.
IDENTITY_FLOOR = 4

# The scene score is a blunt instrument: it mixes "a stated story change was
# ignored" with "the camera is wider than the brief asked for". Only the first
# ruins a book, so the floor stays at 3 and the faults that actually matter are
# asked about separately, as yes/no questions (see FATAL_FACTS). Raising this
# to 4 instead turned four framing quibbles per book into paid redraws.
SCENE_FLOOR = 3

#: Bumped whenever the pass/fail rule changes, and stored on each verdict, so a
#: book checked under an older rule can be told apart from one checked today.
RULES_REV = 2

#: Yes/no answers from the checker that fail a page on their own, whatever it
#: scored. Each is a defect a reader sees immediately and print cannot undo.
FATAL_FACTS = {
    "extra_or_duplicated_character": True,
    "panelled": True,
    "story_state_ok": False,
}

# A duplicated or invented character is a hard fail whatever the scene scored.
# The vision model reliably *describes* it and then unreliably grades it -- one
# page scored 4 while its own notes said "an extra uniformed dog appears" -- so
# the note is read directly rather than trusting the number.
# The `[\w-]+` runs allow for adjectives the checker slips in: the note that
# shipped a broken page read "an extra *uniformed* dog", and another
# "an extra *triceratops-like* creature".
_DUPLICATE_NOTE = re.compile(
    r"\b(?:duplicat\w*|twice|two\s+copies|second\s+(?:copy|version)|"
    r"(?:extra|additional|another|a\s+second)\s+(?:[\w-]+\s+){0,2}"
    r"(?:character|creature|animal|dog|pup|puppy|person|child|boy|girl|figure|"
    r"dinosaur|triceratops|hero)s?|"
    r"appears?\s+(?:twice|two\s+times)|multiple\s+(?:versions?|copies))\b",
    re.IGNORECASE,
)


def duplicate_note(notes: list[str]) -> str | None:
    """The first note that says a character was drawn twice or invented."""
    for note in notes:
        if _DUPLICATE_NOTE.search(str(note)):
            return str(note)
    return None

# Two render profiles. Draft is for while the story is still moving; print is
# 2624 px, which is exactly 300 dpi across a 21.6 cm page plus its bleed, and
# also the practical ceiling (GPT Image stops at 3840 px on the long edge).
# `image_size` is for backends that take a size label instead of pixels
# (Gemini): print maps to 4K, since its 2K tier falls just short of 300 dpi.
RENDER_PROFILES = {
    "draft": {"long_edge_px": 1024, "quality": "medium", "image_size": "1K",
              "label": "Entwurf — schnell und günstig"},
    "print": {"long_edge_px": 2624, "quality": "high", "image_size": "4K",
              "label": "Druckqualität — 300 dpi, dauert länger"},
}

CHECK_SYSTEM = (
    "You check illustrations for a children's picture book. Image 1 is the "
    "character sheet. Image 2 is a page. You judge two things: whether image 2 "
    "shows the same character as image 1, and whether it shows what the scene "
    "description says. Different pose, angle, background and lighting are "
    "expected and never a fault. But when the scene says something is lost, "
    "missing, taken off or changed, the picture must show that -- a hero who "
    "lost a boot must not be wearing two.\n\n"
    "Two faults matter more than any other, because a printed book cannot "
    "recover from them. Look for both every time, and say so plainly in the "
    "notes if you find one:\n"
    "1. The same character drawn more than once in the picture, or an extra "
    "character, animal or creature that the scene never mentioned.\n"
    "2. The picture divided into panels, halves or side-by-side views, or any "
    "seam or dividing line running across it. A page is one continuous scene.\n"
    "Reply with JSON only."
)


def output_for(profile: str, aspect_ratio: str = "1:1") -> OutputSpec:
    settings = RENDER_PROFILES.get(profile, RENDER_PROFILES["draft"])
    return OutputSpec(
        aspect_ratio=aspect_ratio,
        image_size=settings.get("image_size", "1K"),
        quality=settings["quality"],
        long_edge_px=settings["long_edge_px"],
    )


# --------------------------------------------------------------------------- prompts


# Said on every picture, cover included. A picture book page is one window onto
# one moment; a seam down the middle reads to a small child as a printing fault.
# This also has to survive a SCENE that asks for a diptych in so many words --
# see `single_scene()`, which takes that phrasing back out of the scene text.
SINGLE_SCENE = (
    "Draw one single continuous scene in one frame. Do not divide the picture "
    "into panels, halves, strips or side-by-side views, and do not put a seam, "
    "a border or a dividing line anywhere in it. If the scene mentions two "
    "places, show them in one continuous view -- one in the foreground and the "
    "other further back in the same space, not two pictures next to each "
    "other.\n\n"
    "Draw each character exactly once. The same character must never appear "
    "twice in the picture, and no extra characters, animals or creatures may "
    "be added beyond the ones in the reference images."
)


def _who_is_who(hero: Hero, members: list[CastMember]) -> str:
    """Bind every name the scene can use to the reference image that shows it.

    Without this a name in the SCENE is just a name, and the model is free to
    guess what it belongs to -- which is how a rescue pup called Pip became a
    second boy. It says which image, never what they look like: appearance in
    words competes with the sheet and is what makes a character drift.
    """
    people = [f"{hero.name or 'the hero'} is image 1"]
    people += [f"{m.name} is image {n}" for n, m in enumerate(members, start=2)]
    if len(people) == 1:
        return ""
    return (
        "WHO IS WHO\n"
        f"Every name in the scene is one of the reference images: {', '.join(people)}. "
        "Draw each of them as their image shows them -- a name alone never "
        "means a person. Draw nobody who is not in those images.\n\n"
    )


def _reference_block(hero: Hero, members: list[CastMember]) -> tuple[str, list[str]]:
    """Describe each reference image by number, in the order they are attached."""
    name = hero.name or "the character"
    lines = [
        f"Image 1 is the character reference sheet for {name}. Match that "
        "character exactly: face, hair, build, clothing, colours and "
        "proportions. Do not redesign, restyle, age or simplify them, and do "
        "not add or remove anything they wear or carry -- with one exception: "
        "where the SCENE below says something is lost, missing, taken off, "
        "added or changed, the scene wins. Draw that change and keep "
        "everything else from the reference."
    ]
    for position, member in enumerate(members, start=2):
        if member.kind == "place":
            lines.append(
                f"Image {position} shows {member.name}, the place this story "
                "keeps returning to. Keep its layout, colours and distinctive "
                "features the same, but you may show it from a different angle."
            )
        else:
            lines.append(
                f"Image {position} is the reference for {member.name}. Match "
                "them exactly, the same way as the hero."
            )
    return "\n\n".join(lines), [m.name for m in members]


def page_prompt(book: Book, hero: Hero, style: Style, page: Page,
                members: list[CastMember], insist: bool = False) -> str:
    """The prompt for one page illustration."""
    references, _ = _reference_block(hero, members)

    # The layout decides how the *page* is composed, never how the picture is.
    # All the illustrator is told is where the words will sit on top of it.
    room = (
        "Leave a calm, uncluttered area somewhere in the picture where the "
        "story text can sit -- ideally along the bottom or in one upper corner."
        if page.layout != "wordless"
        else "This page has no text on it, so compose it to fill the whole frame."
    )
    if page.layout == "vignette":
        # No "fading out towards the edges" here: it made the model paint its
        # own pale border, which the page layout then inset a second time and
        # printed as a postage stamp. The insetting is layout.py's job alone.
        room = (
            "This picture will be printed smaller than a full page, so keep it "
            "simple and readable: one clear subject, close in, nothing "
            "important near the edges. Fill the whole frame with the scene."
        )
    elif page.layout == "split":
        room = (
            "This picture will be cropped to a tall portrait shape, so keep "
            "everything important away from the left and right edges. The "
            "picture itself is still one single scene, not two."
        )

    warning = (
        "\n\nThis is a second attempt. The previous try drifted from the "
        "reference or from the scene. Follow image 1 more literally -- same "
        "face shape, same hair, same colours, same clothing -- and follow the "
        "SCENE exactly, including anything it says is missing, lost or "
        "changed. Draw every character once and only once. Nothing invented.\n"
        if insist else ""
    )

    return (
        "A single full-page illustration for a children's picture book.\n\n"
        f"{references}{warning}\n\n"
        f"SCENE\n{single_scene(page.illustration)}\n\n"
        f"{_who_is_who(hero, members)}"
        "Use the reference images only for who and where. Do not copy their "
        "pose, framing or background -- build the scene described above.\n\n"
        f"STYLE\n{style.description}\n\n"
        f"{room}\n\n"
        f"{SINGLE_SCENE}\n\n"
        "Do not put any text, letters, numbers, captions, speech bubbles, "
        "borders, panel frames or watermarks in the image."
    )


def cover_prompt(book: Book, hero: Hero, style: Style,
                 members: list[CastMember] = ()) -> str:
    """The prompt for the cover.

    The cover gets the same reference block as a page. It used to get only the
    hero's sheet, which is how a cover naming "Claudio and Pip" ended up with
    two boys on it: with no picture of Pip attached, a rescue pup reads as a
    second child's name and the model drew one.
    """
    name = hero.name or "the character"
    members = list(members)
    references, _ = _reference_block(hero, members)
    scene = book.cover_illustration or (
        f"{name} in the middle of the story's world, looking out at the reader"
    )
    return (
        "A children's picture book cover illustration.\n\n"
        f"{references}\n\n"
        f"SCENE\n{scene}\n\n"
        f"{_who_is_who(hero, members)}"
        "Compose it as a cover: the characters clearly readable, an inviting "
        "background, and calm empty space in the upper third where the title "
        "will be placed.\n\n"
        f"STYLE\n{style.description}\n\n"
        f"{SINGLE_SCENE}\n\n"
        "Do not put any text, letters, numbers or watermarks in the image."
    )


# --------------------------------------------------------------------------- checking


def check_page(
    page_image: Path,
    sheet: Path,
    hero: Hero,
    provider: str = "openai",
    model: str | None = None,
    scene: str = "",
    spend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Has the hero survived onto this page -- and does the page show the scene?

    The scene check exists because the identity check alone cannot catch a
    *story-state* error: a hero who lost a boot in the mud but is drawn wearing
    both scores a perfect 5 on identity.
    """
    verdict: dict[str, Any] = {"checked_by": provider}

    # Free first: a colour comparison catches gross drift with no API call.
    try:
        verdict["metrics"] = score_page(page_image, sheet)
    except Exception:
        verdict["metrics"] = {}

    # Also free, and measured rather than asked: a page split into two pictures.
    seam = seam_in_frame(page_image)

    scene_block = (
        f"\nThe page was supposed to show this scene:\n{scene}\n\n"
        "scene: 5 = shows exactly what the scene describes; 3 = the framing or "
        "camera angle is not what was asked for, but the right thing is "
        "happening; 1 = a different scene entirely.\n"
        "story_state_ok: false if the scene says something is lost, missing, "
        "taken off or changed and the picture does not show that -- the scene "
        "says one boot is lost, the picture shows both.\n"
        if scene else ""
    )
    # Asked as yes/no rather than folded into a score. The checker reliably
    # *describes* a duplicated character and then unreliably grades it: one
    # page scored 4 while its own notes read "an extra uniformed dog appears".
    facts_block = (
        "extra_or_duplicated_character: true if any character appears more "
        "than once in the picture, or if there is a character, animal or "
        "creature that the scene did not ask for.\n"
        "panelled: true if the picture is divided into panels, halves or "
        "side-by-side views, or has a seam or dividing line across it.\n"
    )
    shape = (
        '{"identity": <1-5>, "style": <1-5>, "scene": <1-5>, '
        '"story_state_ok": <true|false>, "extra_or_duplicated_character": '
        '<true|false>, "panelled": <true|false>, "notes": ["..."]}'
        if scene else
        '{"identity": <1-5>, "style": <1-5>, '
        '"extra_or_duplicated_character": <true|false>, '
        '"panelled": <true|false>, "notes": ["..."]}'
    )

    user = (
        f"The character is {hero.name or 'the child'}.\n\n"
        "Score from 1 to 5:\n"
        "identity: 5 = unmistakably the same character; 4 = same, with one "
        "small detail off; 3 = the face or proportions have shifted; 1 = a "
        "different character or not visible at all.\n"
        "style: 5 = the same illustration technique and palette as image 1.\n"
        f"{scene_block}{facts_block}\n"
        "List every concrete difference you can see as short phrases, for "
        'example "hair is blonde, should be dark brown" or "wears two boots, '
        'the scene says one is lost".\n\n'
        f"Reply as {shape}"
    )

    try:
        payload = complete_json(
            CHECK_SYSTEM, user, images=[sheet, page_image], provider=provider,
            model=model, spend=spend, what="check",
        ) or {}
        verdict["identity"] = max(1, min(5, int(payload.get("identity", 3))))
        verdict["style"] = max(1, min(5, int(payload.get("style", 3))))
        if scene:
            verdict["scene"] = max(1, min(5, int(payload.get("scene", 3))))
        for fact in FATAL_FACTS:
            if fact in payload:
                verdict[fact] = bool(payload[fact])
        notes = payload.get("notes") or []
        verdict["notes"] = [str(n) for n in (notes if isinstance(notes, list) else [notes])][:6]
    except Exception as exc:
        # Fail closed: a checker that timed out or returned junk has verified
        # nothing. "unknown" is not a pass -- the export preflight treats it
        # as unreviewed and the UI says so.
        verdict["error"] = f"{type(exc).__name__}: {exc}"
        verdict["status"] = "unknown"
        verdict["rules_rev"] = RULES_REV
        return verdict

    if seam:
        # The measurement wins over the answer. The checker was asked about
        # panels in as many words and still passed three seamed pages.
        verdict["panelled"] = True
        verdict["notes"] = ["the picture is split by a seam"] + verdict["notes"][:5]
    if duplicate_note(verdict["notes"]):
        verdict["duplicate"] = duplicate_note(verdict["notes"])
    verdict["rules_rev"] = RULES_REV
    verdict["status"] = verdict_from(verdict)
    # Kept for older readers of book.json; `status` is the one to trust.
    verdict["ok"] = verdict["status"] == "passed"
    return verdict


def verdict_from(check: dict[str, Any]) -> str:
    """passed | failed | unknown | unchecked, worked out from the evidence.

    Deriving this rather than storing a boolean is what lets a tightened rule
    reach books that were checked under an older one. Every page of the first
    finished book carried `ok: true` while its own notes read "Pip and Trixi
    are duplicated" -- and because the flag was frozen, no later fix could ever
    change that page's verdict.
    """
    if not check:
        return "unchecked"
    if check.get("error"):
        return "unknown"
    if check.get("identity") is None:
        # No scores to reason from -- an older book, or a check that recorded
        # only its conclusion. Take the stored word rather than inventing one.
        stored = check.get("status")
        if stored in ("passed", "failed", "unknown"):
            return stored
        ok = check.get("ok")
        if ok is True:
            return "passed"
        if ok is False:
            return "failed"
        return "unknown"

    for fact, fatal in FATAL_FACTS.items():
        if fact in check and bool(check[fact]) is fatal:
            return "failed"
    if duplicate_note(check.get("notes") or []):
        return "failed"
    if int(check.get("identity", 0)) < IDENTITY_FLOOR:
        return "failed"
    scene = check.get("scene")
    if scene is not None and int(scene) < SCENE_FLOOR:
        return "failed"
    return "passed"


def check_status(page: Page) -> str:
    """One word for where this page's check stands."""
    return verdict_from(page.check or {})


def _check_rank(check: dict[str, Any]) -> tuple:
    """Sort key for two verdicts on the same page: bigger is better."""
    status = verdict_from(check)
    return (
        {"passed": 3, "unchecked": 2, "unknown": 1, "failed": 0}[status],
        0 if check.get("panelled") else 1,
        0 if (check.get("duplicate")
              or check.get("extra_or_duplicated_character")) else 1,
        0 if check.get("story_state_ok") is False else 1,
        int(check.get("identity") or 0),
        int(check.get("scene") or 0),
        int(check.get("style") or 0),
    )


def _better(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    return _check_rank(candidate) > _check_rank(incumbent)


def seam_in_frame(path: Path, ratio: float = 4.0, floor: float = 20.0) -> bool:
    """Is this two pictures side by side rather than one scene?

    Cheap and deterministic, so it runs on every page for nothing. A diptych
    leaves a hard vertical discontinuity where the two halves meet: the columns
    either side belong to unrelated images, so their difference dwarfs any
    ordinary edge. Comparing that peak against the median column difference
    makes the test independent of how busy the illustration is.

    Measured over the middle half of the frame rather than the exact centre,
    because the split is not always even -- one real diptych joined at 37% of
    the width. On the sixteen pages of the first finished book the three
    diptychs scored 7.9, 8.0 and 14.0 while the widest ordinary edge reached
    2.6, so the threshold sits in a wide gap rather than on a guess.

    The checker was asked about this in words and missed it three times out of
    three, which is why it is also measured here.
    """
    try:
        with Image.open(path) as image:
            grey = np.asarray(image.convert("L"), dtype=np.float64)
    except Exception:
        return False
    if grey.ndim != 2 or grey.shape[1] < 64:
        return False

    # Mean absolute difference between each pair of neighbouring columns.
    steps = np.abs(np.diff(grey, axis=1)).mean(axis=0)
    width = steps.size
    band = steps[int(width * 0.25):int(width * 0.75)]
    if band.size == 0:
        return False
    typical = float(np.median(steps))
    peak = float(band.max())
    return peak >= floor and peak >= ratio * max(typical, 1e-6)


def _usable_image(path: Path) -> bool:
    """Is this a page we can keep, or wreckage from an interrupted render?

    Resuming skips any page whose file exists, so a half-written PNG used to be
    adopted as finished work and could never be redrawn from inside the app --
    the export then blocked on a corrupt image with no way forward.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception:
        return False
    return True


def pick_check_provider(image_backend: str, available: list[str]) -> str:
    """Prefer a checker that did not draw the picture it is grading.

    A model marking its own homework is not a neutral referee. If the only key
    available belongs to the same provider that drew the page, we use it anyway
    and say so, rather than skipping the check.
    """
    others = [p for p in available if p != image_backend]
    return others[0] if others else (available[0] if available else image_backend)


# --------------------------------------------------------------------------- drawing


def draw_page(
    book: Book,
    hero: Hero,
    style: Style,
    page: Page,
    sheet: Path,
    target: Path,
    members: list[CastMember] = (),
    backend_name: str = "openai",
    model: str | None = None,
    output: OutputSpec | None = None,
    insist: bool = False,
    resolve=None,
):
    backend = get_backend(backend_name, model)
    references, named = _attach(sheet, list(members), resolve, backend.max_references)
    return backend.generate(
        GenRequest(
            prompt=page_prompt(book, hero, style, page, named, insist=insist),
            reference_images=references,
            output=output or output_for(book.render_quality),
        ),
        target,
    )


def _attach(
    sheet: Path, members: list[CastMember], resolve, limit: int,
) -> tuple[list[Path], list[CastMember]]:
    """Reference images to send, and the cast they correspond to, in lockstep.

    The prompt numbers the references by position ("Image 3 is Trixi"), so the
    two lists must not drift apart. A member whose sheet is missing from disk,
    or one pushed past the backend's reference limit, has to disappear from the
    prompt as well -- otherwise the numbering slides by one and every name
    after it points at the wrong picture.
    """
    paths = [sheet]
    named: list[CastMember] = []
    for member in members:
        if len(paths) >= limit:
            break
        if not (member.sheet and resolve):
            continue
        try:
            path = resolve(member.sheet)
        except (ValueError, FileNotFoundError):
            continue
        if path.is_file():
            paths.append(path)
            named.append(member)
    return paths, named


def draw_cover(
    book: Book,
    hero: Hero,
    style: Style,
    sheet: Path,
    target: Path,
    backend_name: str = "openai",
    model: str | None = None,
    output: OutputSpec | None = None,
    members: list[CastMember] = (),
    resolve=None,
):
    backend = get_backend(backend_name, model)
    references, named = _attach(sheet, list(members), resolve, backend.max_references)
    return backend.generate(
        GenRequest(
            prompt=cover_prompt(book, hero, style, named),
            reference_images=references,
            output=output or output_for(book.render_quality),
        ),
        target,
    )


def illustrate_book(
    book: Book,
    hero: Hero,
    style: Style,
    sheet: Path,
    pages_dir: Path,
    backend_name: str = "openai",
    model: str | None = None,
    check: bool = True,
    check_provider: str = "openai",
    only: list[int] | None = None,
    redraw: bool = False,
    workers: int = 4,
    auto_retry: bool = True,
    budget_usd: float | None = None,
    resolve=None,
    on_progress=None,
    log=print,
    should_stop=None,
) -> Book:
    """Draw the cover and every page, checking each one.

    `only` limits the work to certain page numbers, `redraw` forces pages that
    already exist to be drawn again. Together they are how a single unhappy
    page gets fixed without paying for the whole book twice. `budget_usd`
    is a hard spending ceiling for this run: once crossed, remaining pages
    are skipped rather than drawn.
    """
    stop = should_stop or (lambda: False)
    pages_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(only) if only else None
    lock = threading.Lock()
    run_spend = {"usd": 0.0, "capped": False}

    def spend(usage: dict, what: str) -> None:
        if not usage:
            return
        with lock:
            before = float(book.spend.get("usd", 0.0))
            add_spend(book.spend, usage, what)
            run_spend["usd"] += float(book.spend.get("usd", 0.0)) - before

    def over_budget() -> bool:
        if budget_usd is None or run_spend["usd"] < budget_usd:
            return False
        with lock:
            if not run_spend["capped"]:
                run_spend["capped"] = True
                log(f"Kostendeckel erreicht (~{run_spend['usd']:.2f} $) — "
                    "weitere Seiten werden übersprungen.")
        return True

    if wanted is None and (redraw or not book.cover):
        if not stop():
            log("Titelbild")
            target = pages_dir / "cover.png"
            try:
                # The cover gets the cast too. Without their sheets it had only
                # a name to go on, and drew the rescue pup as a second boy.
                result = draw_cover(
                    book, hero, style, sheet, target, backend_name, model,
                    members=book.cast, resolve=resolve,
                )
                book.cover = f"pages/{target.name}"
                spend(result.usage, "cover")
            except Exception as exc:
                log(f"  fehlgeschlagen: {exc}")

    todo = [
        page for page in sorted(book.pages, key=lambda p: p.index)
        if (wanted is None or page.index in wanted)
        and (redraw or not _usable_image(pages_dir / f"page_{page.index:02d}.png"))
    ]
    # Anything already on disk and not being redrawn is simply adopted.
    for page in book.pages:
        target = pages_dir / f"page_{page.index:02d}.png"
        if target.is_file() and page not in todo:
            page.image = f"pages/{target.name}"

    if not todo:
        log("Nichts zu zeichnen — alle Seiten sind schon da.")
        return book

    total = len(todo)
    log(f"{total} Bild(er), {min(workers, total)} gleichzeitig")
    done = 0
    if on_progress:
        on_progress(0, total)

    def one(page: Page) -> None:
        nonlocal done
        if stop() or over_budget():
            return
        target = pages_dir / f"page_{page.index:02d}.png"
        members = book.cast_for(page)
        page.error = None

        # Keep the outgoing version so a redraw can be undone.
        if target.is_file():
            keep = pages_dir / f"page_{page.index:02d}_v{len(page.history) + 1}.png"
            shutil.copy2(target, keep)
            page.history.append(f"pages/{keep.name}")

        # The brief the illustrator is actually given, so the checker grades
        # against the same words rather than the raw stored text.
        scene_text = single_scene(page.illustration)
        best: tuple[Path, dict[str, Any]] | None = None

        for attempt in (1, 2):
            try:
                result = draw_page(
                    book, hero, style, page, sheet, target, members=members,
                    backend_name=backend_name, model=model,
                    insist=attempt == 2, resolve=resolve,
                )
                page.image = f"pages/{target.name}"
                page.image_from_rev = page.illustration_rev  # picture matches the brief again
                spend(result.usage, "pages")
            except Exception as exc:
                page.error = f"{type(exc).__name__}: {exc}"
                with lock:
                    log(f"  Seite {page.index}: fehlgeschlagen — {exc}")
                return

            if not check:
                break

            with lock:
                page.check = check_page(target, sheet, hero, provider=check_provider,
                                        scene=scene_text, spend=book.spend)
            page.check["attempts"] = attempt
            if best is None or _better(page.check, best[1]):
                # Keep this attempt's pixels aside before the next one
                # overwrites them, so a worse redraw can be undone.
                spare = pages_dir / f"page_{page.index:02d}_try{attempt}.png"
                shutil.copy2(target, spare)
                best = (spare, dict(page.check))
            if check_status(page) != "failed":
                break
            if attempt == 1 and auto_retry and not stop():
                with lock:
                    log(f"  Seite {page.index}: abgedriftet, zeichne noch einmal nach")
                continue
            break

        # Two paid attempts used to be resolved by keeping whichever came last.
        # Keep whichever came out better instead -- same money, and the second
        # try is not reliably an improvement.
        if best is not None:
            if best[1] is not page.check and _better(best[1], page.check):
                shutil.copy2(best[0], target)
                page.check = best[1]
            for leftover in pages_dir.glob(f"page_{page.index:02d}_try*.png"):
                leftover.unlink(missing_ok=True)

        with lock:
            done += 1
            note = ""
            if check_status(page) == "failed":
                check_data = page.check or {}
                if check_data.get("panelled"):
                    note = " — bitte ansehen (Bild ist geteilt)"
                elif check_data.get("duplicate") or check_data.get(
                        "extra_or_duplicated_character"):
                    note = f" — bitte ansehen ({check_data.get('duplicate', 'Figur doppelt')})"
                elif check_data.get("story_state_ok") is False:
                    note = " — bitte ansehen (Szene stimmt nicht)"
                else:
                    note = f" — bitte ansehen (Ähnlichkeit {check_data.get('identity')}/5)"
            log(f"  [{done}/{total}] Seite {page.index} fertig{note}")
            if on_progress:  # inside the lock: callers may save the book here
                on_progress(done, total)

    if workers > 1 and total > 1:
        with ThreadPoolExecutor(max_workers=min(workers, total)) as pool:
            list(pool.map(one, todo))
    else:
        for page in todo:
            one(page)

    if stop():
        log("abgebrochen — alle fertigen Seiten bleiben erhalten")
    return book


def illustrate_book_batch(
    book: Book,
    hero: Hero,
    style: Style,
    sheet: Path,
    pages_dir: Path,
    model: str | None = None,
    check: bool = True,
    check_provider: str = "openai",
    only: list[int] | None = None,
    redraw: bool = False,
    budget_usd: float | None = None,
    resolve=None,
    on_progress=None,
    log=print,
    should_stop=None,
) -> Book:
    """The whole render as one Gemini batch job -- half price, no hurry.

    Same prompts, same references, same checks as the interactive path; only
    the transport differs: every image goes to Google at once, and the job
    waits for the batch to come back instead of drawing four at a time. No
    automatic retry here -- a second attempt would mean a second batch, so
    flagged pages stay flagged for a person (or a follow-up run) to decide.
    """
    from ..backends import gemini as gemini_mod

    stop = should_stop or (lambda: False)
    pages_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(only) if only else None

    todo: list[tuple[Page | None, Path]] = []
    requests = []
    output = output_for(book.render_quality)

    if wanted is None and (redraw or not book.cover):
        references, named = _attach(sheet, list(book.cast), resolve, 5)
        requests.append(GenRequest(prompt=cover_prompt(book, hero, style, named),
                                   reference_images=references, output=output))
        todo.append((None, pages_dir / "cover.png"))

    for page in sorted(book.pages, key=lambda p: p.index):
        target = pages_dir / f"page_{page.index:02d}.png"
        if wanted is not None and page.index not in wanted:
            continue
        if not redraw and target.is_file():
            page.image = f"pages/{target.name}"
            continue
        references, named = _attach(sheet, book.cast_for(page), resolve, 5)
        requests.append(GenRequest(
            prompt=page_prompt(book, hero, style, page, named),
            reference_images=references, output=output,
        ))
        todo.append((page, target))

    if not requests:
        log("Nichts zu zeichnen — alle Seiten sind schon da.")
        return book

    # The cap has to be applied here, before submitting. A batch is one
    # all-or-nothing request, so there is no mid-run point at which to stop --
    # which is how the cost ceiling ended up doing nothing at all in the one
    # mode chosen specifically to save money.
    if budget_usd is not None:
        each = image_estimate(output.quality, output.long_edge_px,
                              model or gemini_mod.GeminiBackend.default_model) / 2
        affordable = int(budget_usd // each) if each > 0 else len(requests)
        if affordable < len(requests):
            log(f"Kostendeckel {budget_usd:.2f} $ — es werden {affordable} von "
                f"{len(requests)} Bildern beauftragt (~{each:.3f} $ je Bild).")
            requests = requests[:affordable]
            todo = todo[:affordable]
        if not requests:
            log("Der Kostendeckel lässt kein einziges Bild zu.")
            return book

    results = gemini_mod.run_batch(
        model, requests, [target for _, target in todo],
        log=log, should_stop=stop, on_progress=on_progress,
    )

    drawn = 0
    for (page, target), result in zip(todo, results):
        label = "Titelbild" if page is None else f"Seite {page.index}"
        if "error" in result:
            if page is not None:
                page.error = result["error"]
            log(f"  {label}: fehlgeschlagen — {result['error']}")
            continue
        if target.is_file():  # keep the outgoing version, redraws stay undoable
            if page is not None:
                keep = pages_dir / f"page_{page.index:02d}_v{len(page.history) + 1}.png"
                shutil.copy2(target, keep)
                page.history.append(f"pages/{keep.name}")
        target.write_bytes(result["data"])
        add_spend(book.spend, result["usage"], "cover" if page is None else "pages")
        if page is None:
            book.cover = f"pages/{target.name}"
        else:
            page.image = f"pages/{target.name}"
            page.image_from_rev = page.illustration_rev
            page.error = None
        drawn += 1
    log(f"{drawn} von {len(todo)} Bildern angekommen.")

    if check and drawn:
        log("")
        log(f"Jetzt die Prüfung, Seite für Seite — von {check_provider}.")
        for page, target in todo:
            if stop() or page is None or not target.is_file():
                continue
            page.check = check_page(target, sheet, hero, provider=check_provider,
                                    scene=single_scene(page.illustration),
                                    spend=book.spend)
            status = check_status(page)
            word = {"passed": "in Ordnung", "failed": "beanstandet",
                    "unknown": "Prüfung fehlgeschlagen"}.get(status, status)
            log(f"  Seite {page.index}: {word}")

    return book


def flagged_pages(book: Book) -> list[int]:
    """Page numbers a person should look at before printing.

    Failed checks and *unknown* checks both count: a page whose checker
    crashed has been verified by nobody, and treating it as fine is how an
    unreviewed page ends up in a printed book.
    """
    return [
        page.index
        for page in book.pages
        if page.error or (page.image and check_status(page) in ("failed", "unknown"))
    ]


def review_split(book: Book) -> dict[str, list[int]]:
    """The same pages, split by why they need eyes on them."""
    failed, unknown, errors = [], [], []
    for page in book.pages:
        if page.error:
            errors.append(page.index)
        elif page.image:
            status = check_status(page)
            if status == "failed":
                failed.append(page.index)
            elif status == "unknown":
                unknown.append(page.index)
    return {"failed": failed, "unknown": unknown, "errors": errors}
