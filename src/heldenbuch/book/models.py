"""What a book is made of.

Three things are stored separately and on purpose:

  Hero   who the book is about. Made once from photos or a description, then
         reused in every book. This is what keeps him looking like himself.
  Style  how the pictures look. Also made once and reused.
  Book   a story: pages, text in one or more languages, and the illustrations.

Separating them is the whole trick behind "low effort". Getting the hero and
the style right takes a few minutes and costs a little; after that every new
book is just a story idea away.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

# Languages the app can write in. Adding one is a one-line change -- the model
# writes natively, it does not translate.
LANGUAGES = {
    "de": {"name": "Deutsch", "english_name": "German"},
    "en": {"name": "English", "english_name": "English"},
    "ru": {"name": "Русский", "english_name": "Russian"},
    "fr": {"name": "Français", "english_name": "French"},
    "es": {"name": "Español", "english_name": "Spanish"},
    "it": {"name": "Italiano", "english_name": "Italian"},
    "tr": {"name": "Türkçe", "english_name": "Turkish"},
    "pl": {"name": "Polski", "english_name": "Polish"},
    "nl": {"name": "Nederlands", "english_name": "Dutch"},
}

# Age bands drive everything about the writing: how long a sentence may be,
# how many pages, how much can happen on one page.
AGE_BANDS = {
    "2-3": {
        "label": "2–3 Jahre",
        "pages": 12,
        "sentences": "exactly one very short sentence",
        "words_per_page": "5 to 12 words",
        "guidance": (
            "Name everyday objects. Use repetition and a returning refrain. One "
            "single idea per page. No subordinate clauses. No abstract feelings "
            "beyond happy, sad, scared and tired."
        ),
    },
    "4-5": {
        "label": "4–5 Jahre",
        "pages": 16,
        "sentences": "two or three short sentences",
        "words_per_page": "20 to 40 words",
        "guidance": (
            "A small problem appears early and is solved by the end. Concrete "
            "actions the child can picture. Simple feelings named out loud. At "
            "most one subordinate clause per sentence."
        ),
    },
    "6-7": {
        "label": "6–7 Jahre",
        "pages": 20,
        "sentences": "three to five sentences",
        "words_per_page": "45 to 80 words",
        "guidance": (
            "A real plot with a turning point. The hero makes a choice that "
            "costs something. Some humour. Dialogue is welcome. Vocabulary may "
            "stretch slightly beyond everyday words if the context explains it."
        ),
    },
    "8+": {
        "label": "8+ Jahre",
        "pages": 24,
        "sentences": "four to seven sentences",
        "words_per_page": "80 to 140 words",
        "guidance": (
            "A layered plot with a setback before the resolution. Inner life and "
            "motivation matter. Wordplay and irony land at this age. Chapters "
            "may be implied by the page breaks."
        ),
    },
}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def slugify(text: str, fallback: str = "buch") -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", (text or "").lower().strip()).strip("-")
    return cleaned[:60] or fallback


@dataclass
class Hero:
    """The child (or animal, or robot) the books are about."""

    id: str = field(default_factory=lambda: new_id("hero"))
    name: str = ""
    #: written by the LLM from the photos, or typed by hand. Used as the
    #: fallback description and as the yardstick for the consistency check.
    description: str = ""
    #: how the hero came to be: "photo" or "described"
    source: str = "described"
    #: chosen character sheet, relative to the hero folder
    sheet: str | None = None
    #: every sheet generated, so a rejected variant can be picked up later
    variants: list[str] = field(default_factory=list)
    #: uploaded reference photos, relative to the hero folder. Never leave the
    #: machine except for the one call that makes the character sheet.
    photos: list[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Style:
    """How the pictures look. Reused across books."""

    id: str = field(default_factory=lambda: new_id("style"))
    name: str = ""
    #: the text actually appended to every illustration prompt
    description: str = ""
    #: preset key this came from, "custom" (described in words) or "image"
    preset: str = "custom"
    #: an uploaded picture used as the style reference -- a crayon drawing, a
    #: photographed object. Words can only get so close to a look.
    reference: str | None = None
    #: image backend the scout found best for this style, if it has been run
    recommended_backend: str | None = None
    scout: dict[str, Any] = field(default_factory=dict)
    #: preview renders of the hero in this style, relative to the style folder
    previews: list[str] = field(default_factory=list)
    cover_preview: str | None = None
    #: hero_id -> that hero's character sheet redrawn in this style. This is
    #: what the pages actually reference, so identity and look are locked
    #: together in one image instead of fighting each other every page.
    sheets: dict[str, str] = field(default_factory=dict)
    created: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# How a *page* is composed -- where the picture and the words sit on the paper.
# None of these describe the picture itself, and the wording has to keep saying
# so: "split" once read to the author as an instruction about the artwork, and
# it duly wrote "A split composition: on one side..." into the illustration
# brief. The illustrator obeyed, and three pages printed with a seam down the
# middle. See `single_scene()` for the net that catches it if it happens again.
#: The last page's word, per language. Printed on the closing page and read
#: out by the narration, so it lives here rather than inside either one.
CLOSERS = {"de": "Ende", "en": "The End", "ru": "Конец", "fr": "Fin",
           "es": "Fin", "it": "Fine", "tr": "Son", "pl": "Koniec",
           "nl": "Einde"}


def closing_word(language: str) -> str:
    return CLOSERS.get(language, CLOSERS["de"])


LAYOUTS = {
    "full": "picture fills the page, text sits in a quiet corner of it",
    "split": "picture on the upper part of the page, text on plain paper below it",
    "vignette": "picture printed smaller on plain paper, for a quiet beat",
    "wordless": "picture fills the page, no text at all",
}

# Ways an illustration brief can ask for two pictures instead of one. Stripped
# rather than trusted: the model that writes the brief and the model that draws
# from it are different models, and the drawing one takes prose literally.
_DIPTYCH_LEAD = re.compile(
    r"^\s*(?:a|an|the)?\s*"
    r"(?:split|two[\s-]?part|two[\s-]?panel|double|divided|side[\s-]by[\s-]side|diptych)"
    r"[\s-]*(?:composition|image|picture|panel|panels|scene|view|layout|spread|frame)?"
    r"\s*[:,–—-]\s*",
    re.IGNORECASE,
)

# Order matters: the longer phrase has to go first or the shorter one eats it.
#
# Every pattern refuses to fire before "of", because "on the other side OF the
# brook" is ordinary prose describing one continuous view. Rewriting that would
# turn a perfectly good brief into "and beyond it, of the brook".
_DIPTYCH_JOINS = [
    (re.compile(r"\bthe\s+other\s+side\s+shows\s*", re.I), "and beyond it, "),
    (re.compile(r"\bon\s+the\s+other\s+side(?!\s+of)\s*,?\s*", re.I), "and beyond it, "),
    (re.compile(r"\bthe\s+other\s+(?:half\s+)?shows\s*", re.I), "and beyond it, "),
    (re.compile(r"\bin\s+the\s+other\s+half(?!\s+of)\s*,?\s*", re.I), "and beyond it, "),
    (re.compile(r"\bon\s+the\s+other(?!\s+(?:side|half|of))\s*,?\s*", re.I), "and beyond it, "),
    (re.compile(r"\bone\s+side\s+shows\s*", re.I), "in the foreground, "),
    (re.compile(r"\bon\s+one\s+side(?!\s+of)\s*,?\s*", re.I), "in the foreground, "),
    (re.compile(r"\bin\s+one\s+half(?!\s+of)\s*,?\s*", re.I), "in the foreground, "),
]


def single_scene(illustration: str) -> str:
    """Rewrite an illustration brief that asks for two pictures into one scene.

    "A split composition: on one side, the crooked plank; on the other, the
    nest beyond the water" becomes "In the foreground, the crooked plank, and
    beyond it, the nest beyond the water" -- the same two things, in one
    continuous view instead of two panels with a seam between them.

    A brief that never asked for a split comes back untouched, so this is safe
    to run over every page and over books written before it existed.
    """
    text = (illustration or "").strip()
    if not text:
        return text

    cleaned = _DIPTYCH_LEAD.sub("", text, count=1)
    for pattern, replacement in _DIPTYCH_JOINS:
        cleaned = pattern.sub(replacement, cleaned)
    if cleaned == text:
        return text

    # Tidy the joins the substitutions leave behind.
    cleaned = re.sub(r"\s*;\s*and\b", ", and", cleaned)
    cleaned = re.sub(r"\s*,\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned[:1].upper() + cleaned[1:] if cleaned else text


@dataclass
class CastMember:
    """Anyone in the book who is not the hero, and the recurring place.

    Without a reference sheet of their own, a grandmother or a dog is
    reinvented on every page. They get the same treatment as the hero: drawn
    once, then pointed at.
    """

    name: str = ""
    description: str = ""
    kind: str = "character"  # character | place
    sheet: str | None = None
    #: page numbers this one appears on
    pages: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Page:
    """One page of the book: some words, one picture."""

    index: int = 0
    #: language code -> the text on this page
    text: dict[str, str] = field(default_factory=dict)
    #: what the illustration shows. Never mentions the character's appearance --
    #: that comes from the hero sheet, and repeating it in words makes the two
    #: fight each other.
    illustration: str = ""
    #: relative path to the rendered picture
    image: str | None = None
    #: one of LAYOUTS
    layout: str = "full"
    #: names of cast members who appear here, matched against Book.cast
    cast: list[str] = field(default_factory=list)
    #: consistency verdict from the automatic check
    check: dict[str, Any] = field(default_factory=dict)
    #: earlier renders, newest last -- lets a redraw be undone
    history: list[str] = field(default_factory=list)
    #: language code -> narration audio file
    audio: dict[str, str] = field(default_factory=dict)
    #: set when the last render attempt failed
    error: str | None = None

    # -- staleness bookkeeping. Every derived thing on a page (the picture,
    # the narration) remembers which revision of its source it was made from,
    # so an edit can say "the audio still reads the old text" instead of
    # letting old and new quietly disagree. Old books default everything to
    # revision 0, which reads as "current" -- nothing gets marked stale just
    # for having been made before this existed.
    #: language code -> how often that language's text has been edited
    text_rev: dict[str, int] = field(default_factory=dict)
    #: how often the illustration brief has been edited
    illustration_rev: int = 0
    #: the illustration_rev the current picture was drawn from
    image_from_rev: int = 0
    #: language code -> the text_rev that language's narration read
    audio_from_rev: dict[str, int] = field(default_factory=dict)

    def image_stale(self) -> bool:
        return bool(self.image) and self.illustration_rev > self.image_from_rev

    def audio_stale(self) -> list[str]:
        """Languages whose narration reads an older text than the page shows."""
        return [
            code for code in self.audio
            if self.text_rev.get(code, 0) > self.audio_from_rev.get(code, 0)
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Book:
    id: str = field(default_factory=lambda: new_id("book"))
    hero_id: str = ""
    style_id: str = ""
    #: language code -> title
    title: dict[str, str] = field(default_factory=dict)
    #: the one-line idea the story was grown from
    idea: str = ""
    age: str = "4-5"
    languages: list[str] = field(default_factory=lambda: ["de"])
    dedication: dict[str, str] = field(default_factory=dict)
    pages: list[Page] = field(default_factory=list)
    cover: str | None = None
    #: what the cover illustration should show
    cover_illustration: str = ""
    #: "draft" while you are still changing the story, "print" for the final
    #: run. Draft pictures are a quarter of the size and a fraction of the
    #: price; print pictures are 300 dpi on a 21.6 cm page.
    render_quality: str = "draft"
    #: verse instead of prose
    rhyme: bool = False
    #: everyone and everywhere that recurs, each with its own reference sheet
    cast: list[CastMember] = field(default_factory=list)
    #: a real photograph for the last page, with a line under it
    photo_page: dict[str, Any] = field(default_factory=dict)
    #: running tally of what this book has actually cost to make
    spend: dict[str, Any] = field(default_factory=dict)
    #: set when book.json could not be read. The folder is listed anyway so a
    #: damaged book is visible and restorable instead of silently missing.
    broken: bool = False
    #: language code -> back-cover text. Separate from the dedication, which is
    #: written to one child and belongs inside the book, not on the outside.
    blurb: dict[str, str] = field(default_factory=dict)
    #: the consistency verdict for the cover, in the same shape as Page.check.
    #: The cover is the one image everybody sees first and it used to be the
    #: only one nothing checked -- which is how a book called "Claudio und Pip"
    #: shipped with the dog drawn as a second boy.
    cover_check: dict[str, Any] = field(default_factory=dict)
    #: an accepted batch render that has not been collected yet:
    #: {"job": "<provider job name>", "targets": ["pages/page_01.png", ...]}.
    #: Google bills a batch whether or not anyone waits for it, so the handle
    #: outlives the process that submitted it and the next run resumes it.
    pending_batch: dict[str, Any] = field(default_factory=dict)

    # -- locked references. A book keeps its own copies of the character
    # sheet it was drawn from (under refs/ in the book folder), so changing
    # the hero or deleting the style later can never silently change how an
    # existing book renders. `ref_sources` remembers where the copies came
    # from, which is how the UI notices "the hero has moved on since".
    #: book-relative path to this book's copy of the plain hero sheet
    hero_sheet: str | None = None
    #: book-relative path to this book's copy of the styled sheet -- the
    #: image every page actually points at
    styled_sheet: str | None = None
    #: library-relative origins of the copies: {"hero": ..., "styled": ...}
    ref_sources: dict[str, str] = field(default_factory=dict)

    #: bumped on every content edit; exports remember what they were built from
    content_rev: int = 0
    export_rev: int = 0
    #: the voice the existing narration was spoken with
    narration_voice: str | None = None
    #: Narration for the parts of the book that are not numbered pages --
    #: the title, the dedication and the closing word. Printed books have
    #: them, so a reading of the book should have them too.
    #: part -> language -> path, and the text each recording actually says,
    #: so an edited dedication shows as stale the way a page does.
    matter_audio: dict[str, dict[str, str]] = field(default_factory=dict)
    matter_audio_text: dict[str, dict[str, str]] = field(default_factory=dict)

    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Any content edit lands here, so exports can say they are stale."""
        self.content_rev += 1

    def export_stale(self) -> bool:
        return self.content_rev > self.export_rev

    def cast_for(self, page: Page) -> list[CastMember]:
        """The sheets that should be referenced when drawing this page.

        A member is attached when the page lists them *or* when the brief names
        them. Attaching everyone was how a page whose scene mentioned only the
        boy and the dinosaur got the dog's sheet as well, drew him, and was
        then failed by the checker for "a dog that is not mentioned in the
        scene" -- the pipeline inventing a character and marking itself down
        for it.

        A prop is attached only when it is actually named, for the same reason.
        The place stays on every page: it is the setting, not a participant.
        """
        brief = (page.illustration or "").lower()
        drawable = [m for m in self.cast if m.sheet]

        def in_brief(member: CastMember) -> bool:
            return re.search(rf"\b{re.escape(member.name.lower())}\b", brief) is not None

        # The brief is the contract: it is the only thing the illustrator is
        # told to draw. Where it names anyone, that is the guest list. The
        # author's own page.cast is a hint and can disagree with it -- on one
        # page it listed the dog while the brief mentioned only the boy and
        # the dinosaur, so the dog's sheet went along, the dog got drawn, and
        # the checker failed the page for "a dog not mentioned in the scene".
        named = [m for m in drawable if m.kind != "place" and in_brief(m)]
        if not named:
            # A brief that names nobody (or an older book without one) falls
            # back to the list, so a reference is never silently dropped.
            listed = {n.lower() for n in page.cast}
            named = [m for m in drawable
                     if m.kind == "character" and m.name.lower() in listed]

        places = [m for m in drawable if m.kind == "place"]
        return [m for m in drawable if m in named or m in places]

    @property
    def primary_language(self) -> str:
        return self.languages[0] if self.languages else "de"

    def display_title(self) -> str:
        return self.title.get(self.primary_language) or next(
            (t for t in self.title.values() if t), "Ohne Titel"
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["pages"] = [p.to_dict() for p in self.pages]
        data["cast"] = [c.to_dict() for c in self.cast]
        return data

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> Book:
        """Rebuild from stored JSON, ignoring fields written by older versions."""
        known = {f.name for f in fields(Book)}
        page_fields = {f.name for f in fields(Page)}
        cast_fields = {f.name for f in fields(CastMember)}

        payload = {k: v for k, v in raw.items() if k in known}
        payload["pages"] = [
            Page(**{k: v for k, v in p.items() if k in page_fields})
            for p in raw.get("pages", [])
        ]
        payload["cast"] = [
            CastMember(**{k: v for k, v in c.items() if k in cast_fields})
            for c in raw.get("cast", [])
        ]
        return Book(**payload)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    """Write JSON so a reader never sees a half-written file.

    `book.json` is the only index a book has -- page-to-file mapping, story
    text, spend and check results -- and it is rewritten after every drawn
    page. A plain write truncates first, so an interrupted render (or a reader
    polling mid-write) could leave or observe a torn file. Writing beside the
    target and renaming makes the swap atomic on both Windows and POSIX.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
