"""Writing the story.

One call produces the whole book: title, a text for every page in every
language, an illustration instruction per page, the page layout, and the cast —
everyone besides the hero who turns up more than once, plus the place the story
keeps returning to.

Three rules shape the prompt, and all three matter more than they look:

1. **Each language is written, not translated.** Translated children's prose
   reads stiff — rhythm, rhyme and idiom do not survive. The model writes the
   same story natively in each language, so page 7 carries the same beat in
   German and in Russian even when the words differ.

2. **Illustration instructions never describe anyone's appearance.** Everyone
   who recurs gets a reference sheet. Repeating "a boy with brown hair" in the
   prompt makes the text and the reference image compete, and the model splits
   the difference — which is exactly how a character drifts.

3. **The draft is not the book.** `write_story` produces a draft; `revise` reads
   it back the way a parent reads aloud and fixes what does not survive that.
"""

from __future__ import annotations

from typing import Any

from ..llm import complete_json
from .models import (
    AGE_BANDS,
    LANGUAGES,
    LAYOUTS,
    Book,
    CastMember,
    Hero,
    Page,
    single_scene,
)

SYSTEM = """You are an experienced picture book author. You write stories that
are read aloud by a parent to one child, at bedtime or on a sofa.

What good looks like:
- Something happens. A picture book is not a description, it is an event.
- The hero wants something and does something about it. They are not rescued
  by an adult at the end.
- Concrete over abstract. A cold nose, a squeaking door, a lost mitten -- not
  "he felt uncertain".
- Rhythm. Read every line in your head. Short sentences carry small children.
- The last page lands. Warm, quiet, and finished -- not a moral, not a lesson
  spelled out.

What you never do:
- Talk down to the child, or explain the moral of your own story.
- Use frightening peril, injury, death, or anything that would keep a small
  child awake.
- Write the same page twice with different words.
- Mention what anyone looks like. An illustrator already has their picture.

You always reply with valid JSON and nothing else."""

RHYME_GUIDANCE = """
This book is in VERSE. Rules that are not negotiable:
- A steady metre. Read every line out loud and count the beats; a line that
  trips is a broken line.
- True rhymes only. Near-rhymes and eye-rhymes ("Baum"/"Raum" is fine,
  "Baum"/"kaum" is fine, "Hund"/"rund" is fine; "Liebe"/"Triebe" forced into a
  sentence nobody would say is not).
- Never bend word order or invent a word to reach a rhyme. If the rhyme costs
  you natural speech, change the line instead.
- Keep the same rhyme scheme throughout the book.
"""


def _language_block(codes: list[str]) -> str:
    named = [f"{code} ({LANGUAGES.get(code, {}).get('english_name', code)})" for code in codes]
    return ", ".join(named)


def _layout_block() -> str:
    lines = "\n".join(f'  "{key}" — {hint}' for key, hint in LAYOUTS.items())
    return f"""
LAYOUT
The layout says where the picture and the words sit on the printed page. It
says nothing about what the picture shows, and it never changes how you write
that page's "illustration" -- a page laid out as "split" still gets one single
scene, exactly like every other page.
Give each page a layout so the book does not read as identical slides:
{lines}
Use "full" for most pages. Use "split" two or three times. Use "vignette" for a
quiet or small moment. Use "wordless" at most once, at a beat that carries
itself without words -- and then leave the text empty for that page.
The page you name as the "climax" gets a full page. A vignette prints small,
and the moment the whole book has been building to is the one moment that must
not be the smallest picture in it.
"""


def _story_instructions(hero: Hero, age: str, languages: list[str], pages: int,
                        rhyme: bool = False) -> str:
    band = AGE_BANDS.get(age, AGE_BANDS["4-5"])
    return f"""
THE HERO
Name: {hero.name or "the child"}
The illustrator has a character sheet, so do not describe their appearance.
Background the author may use (do not put it in the text):
{hero.description}

THE READER
Age band: {band["label"]}. Per page: {band["sentences"]}, about {band["words_per_page"]}.
{band["guidance"]}
{RHYME_GUIDANCE if rhyme else ""}
LENGTH
Exactly {pages} pages. Give the story a shape across those pages: set-up,
something goes wrong, an attempt that fails or costs something, a turn, and a
quiet ending. Do not stretch a thin idea; make more happen instead.

LANGUAGES
Write the text in every one of these: {_language_block(languages)}.
Write each language natively. Do not translate word for word -- match the
meaning and the rhythm, and let the wording differ where the language wants it
to. Names stay the same in every language.

THE CAST
List everyone besides the hero who appears on more than one page, the one
place the story keeps returning to, and every *object* the story builds,
finds or carries that has to look the same each time it is drawn -- a raft, a
den, a nest, a kite, a lost boot. Give each object kind "prop".

Objects matter as much as characters here and are easy to forget. Each entry
gets its own reference drawing, and anything without one is invented again on
every page it appears in: in one book the whole plot was building a bridge,
the bridge was in eight pictures, and it came out as a fallen mossy log on one
page and a neatly built raft of round logs three pages later.

Describe each the way you would brief an illustrator: concrete, countable
details. Then, for every page, name which of them appear on it.
Keep the cast small -- two or three characters at most, plus the place and at
most two props. A book with eight characters in it is not a picture book for
this age.

ILLUSTRATIONS
For each page write one "illustration" instruction in English describing what
the picture shows: where it happens, what is being done, the time of day and
the mood. Never describe how anybody looks -- not the hero, not the cast.
The one exception is story state: if the story changes what someone wears,
carries or has with them -- a boot lost in the mud, a scarf taken off, a
borrowed umbrella -- say so explicitly in that page's instruction and in every
following page while it lasts ("wearing only one yellow boot, the other foot
in a muddy sock"). The illustrator's reference sheet always shows the
character complete, so an unstated change will silently be drawn back to normal.
Name, in the instruction itself, every cast member and every object that is in
that picture. The instruction is the only thing the illustrator is given, and
only what it names gets a reference image, so a character you list for the
page but leave out of the instruction will be drawn from imagination -- or
left out and invented back in. Equally, do not name anyone who is not in the
picture.
Give each page an "expression": what the hero's face is doing, in two or three
words ("delighted", "close to tears", "narrowing his eyes", "fast asleep").
Left to infer a feeling from the events, an illustrator paints the same
pleasant half-smile on every page, and a story with a real problem in it reads
as sixteen pleasant afternoons. Let it follow the story: the low point should
look like the low point.
Give each page a "direction": which way the picture faces ("moving left to
right", "facing the reader", "looking back over his shoulder"). A picture book
is read left to right, so setting out faces right and coming home faces left;
a page that contradicts the page turn feels wrong even to a child who cannot
say why.
Give each page a "setting": one short line saying where in the story's world
this page stands and what fixed features are in view ("at the brook, the
crooked pine on the left bank"). When several pages play in the same area,
name the SAME features in the SAME words on each of them -- that repetition
is what keeps the place recognisable from page to page. Choose two or three
anchors per area and stick to them; a feature renamed is a feature redrawn
differently.
Never ask for text, letters, numbers or speech bubbles in the picture. Vary the
framing across the book -- close on a face, a wide landscape, a view from
behind, a small hero in a big space -- so the pages do not all look alike.
Every illustration is ONE single continuous scene, seen through one window, at
one moment. Never ask for a split composition, two panels, a diptych, a before
and after, or "on one side... on the other". If a page needs two places, put
them in one view: one in the foreground and the other behind it. A seam down
the middle of a page looks to a small child like the book was printed wrong.
{_layout_block()}""".strip()


def _shape(languages: list[str], count: int) -> str:
    per_language = ", ".join(f'"{c}": "..."' for c in languages)
    return f"""{{
  "title": {{ {per_language} }},
  "dedication": {{ {per_language} }},
  "cover_illustration": "what the cover picture shows",
  "climax": 0,
  "cast": [
    {{ "name": "...", "kind": "character", "description": "concrete details an illustrator can draw" }},
    {{ "name": "...", "kind": "place", "description": "..." }},
    {{ "name": "...", "kind": "prop", "description": "..." }}
  ],
  "pages": [
    {{
      "index": 1,
      "text": {{ {per_language} }},
      "illustration": "...",
      "expression": "what the hero's face is doing, two or three words",
      "direction": "which way the picture faces",
      "setting": "where this page stands, with its fixed anchors named",
      "layout": "full",
      "cast": ["names from the cast list who appear on this page"]
    }}
  ]
}}

The dedication is one short warm line, as a parent would write it.
"climax" is the page number where the story peaks -- the moment everything has
been leading to. Give exactly {count} pages, numbered 1 to {count}."""


def write_story(
    hero: Hero,
    idea: str,
    age: str = "4-5",
    languages: list[str] | None = None,
    pages: int | None = None,
    rhyme: bool = False,
    spend: dict | None = None,
    provider: str = "openai",
    model: str | None = None,
) -> dict[str, Any]:
    """Write a complete book draft."""
    languages = languages or ["de"]
    band = AGE_BANDS.get(age, AGE_BANDS["4-5"])
    count = pages or band["pages"]

    brief = (
        f"Write a picture book from this idea:\n\n{idea.strip()}\n"
        if idea.strip()
        else "Invent the idea yourself. Choose something small and everyday that "
        "matters enormously to a child of this age, and make it happen to this "
        "hero.\n"
    )

    user = (
        f"{brief}\n{_story_instructions(hero, age, languages, count, rhyme)}\n\n"
        f"Reply with exactly this JSON shape:\n\n{_shape(languages, count)}"
    )

    payload = complete_json(SYSTEM, user, provider=provider, model=model,
                            spend=spend, what="story") or {}
    return _normalise(payload, languages, count)


def revise(
    story: dict[str, Any],
    hero: Hero,
    age: str,
    languages: list[str],
    rhyme: bool = False,
    spend: dict | None = None,
    provider: str = "openai",
    model: str | None = None,
    second_reader: bool = False,
) -> dict[str, Any]:
    """Second pass over a draft: rhythm, repetition, and the last line.

    A first draft from any model is serviceable and slightly flat. Reading it
    back with one job -- make this work out loud -- is the cheapest quality
    improvement in the whole pipeline.

    `second_reader` reshapes the pass for a different provider reading the
    text cold: an editor is asked different questions than the author asks
    himself, or the paid second opinion is just the same polish twice.
    """
    band = AGE_BANDS.get(age, AGE_BANDS["4-5"])
    primary = languages[0]
    draft = [{"index": p.index, "text": p.text} for p in story["pages"]]

    editor = (
        "\nYou did not write this draft; you are its first outside reader. "
        "Also check what only fresh eyes catch: whether the story makes "
        "sense read cold, anything a listening child would mishear or "
        "misread, and any page whose meaning leans on something never said."
        if second_reader else ""
    )

    user = f"""Here is a draft picture book for {band["label"]}.{editor}

Title: {story["title"].get(primary, "")}
Pages: {draft}

Read it aloud in your head, then fix it. Specifically:
- Any line that trips when spoken. Rhythm beats cleverness.
- Any two pages that make the same move. Cut the weaker one's content and give
  that page something new to do.
- Any page that only describes and does not advance anything.
- Abstract words a child this age cannot picture. Replace with concrete ones.
- The last page. It must land: warm, quiet, finished. No moral.
- Any place where a language reads like a translation rather than like a book
  originally written in it.
{RHYME_GUIDANCE if rhyme else ""}
Keep the same number of pages, the same page breaks and the same story. You are
polishing, not rewriting. If a page is already good, return it unchanged.
Per page: {band["sentences"]}, about {band["words_per_page"]}.

Reply as {{"title": {{ {", ".join(f'"{c}": "..."' for c in languages)} }},
"pages": [{{"index": 1, "text": {{ {", ".join(f'"{c}": "..."' for c in languages)} }}}}],
"changed": ["short note per page you actually changed"]}}"""

    payload = complete_json(SYSTEM, user, provider=provider, model=model,
                            spend=spend, what="story") or {}

    title = payload.get("title") or {}
    for code in languages:
        if str(title.get(code, "")).strip():
            story["title"][code] = str(title[code]).strip()

    by_index = {int(item.get("index", 0)): item for item in payload.get("pages") or []}
    for page in story["pages"]:
        text = (by_index.get(page.index) or {}).get("text") or {}
        for code in languages:
            new = str(text.get(code, "")).strip()
            if new:
                page.text[code] = new

    notes = payload.get("changed") or []
    story["revision_notes"] = [str(n) for n in notes][:20]
    return story


def _normalise(payload: dict[str, Any], languages: list[str], count: int) -> dict[str, Any]:
    """Make the model's reply safe to store, whatever it actually sent."""
    raw_pages = payload.get("pages") or []
    if not raw_pages:
        raise RuntimeError("the model returned a story with no pages")

    cast: list[CastMember] = []
    for item in payload.get("cast") or []:
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        if not name or not description:
            continue
        raw_kind = str(item.get("kind", "")).lower()
        if raw_kind.startswith("place"):
            kind = "place"
        elif raw_kind.startswith(("prop", "object", "thing", "item")):
            kind = "prop"
        else:
            kind = "character"
        cast.append(CastMember(name=name, description=description, kind=kind))
    # Room for two or three characters, the place, and the props the story
    # builds -- which used to have no reference at all.
    cast = cast[:6]
    known = {member.name.lower() for member in cast}

    pages: list[Page] = []
    for position, item in enumerate(raw_pages[:count], start=1):
        text = item.get("text") or {}
        if isinstance(text, str):  # single-language reply
            text = {languages[0]: text}
        layout = str(item.get("layout", "full")).lower()
        appearing = [
            str(n).strip() for n in (item.get("cast") or [])
            if str(n).strip().lower() in known
        ]
        pages.append(
            Page(
                index=int(item.get("index") or position),
                text={code: str(text.get(code, "")).strip() for code in languages},
                illustration=single_scene(str(item.get("illustration", ""))),
                expression=str(item.get("expression", "")).strip(),
                direction=str(item.get("direction", "")).strip(),
                setting=str(item.get("setting", "")).strip(),
                layout=layout if layout in LAYOUTS else "full",
                cast=appearing,
            )
        )

    # Renumber, in case the model skipped or repeated an index.
    for position, page in enumerate(pages, start=1):
        page.index = position
        if page.layout == "wordless":
            page.text = {code: "" for code in languages}
        # Record where each cast member shows up, for the sheet-drawing step.
        for member in cast:
            if member.kind == "place" or member.name in page.cast:
                member.pages.append(page.index)

    title = payload.get("title") or {}
    if isinstance(title, str):
        title = {languages[0]: title}
    dedication = payload.get("dedication") or {}
    if isinstance(dedication, str):
        dedication = {languages[0]: dedication}

    # The layout is chosen before a single picture exists, so nothing later
    # notices when the story's high point was handed a vignette -- which
    # prints smaller than every page around it. The author knows which page
    # that is; asking it, and then holding it to the answer, is enough.
    climax = _page_index(payload.get("climax"), len(pages))
    if climax and pages[climax - 1].layout == "vignette":
        pages[climax - 1].layout = "full"

    return {
        "title": {code: str(title.get(code, "")).strip() for code in languages},
        "dedication": {code: str(dedication.get(code, "")).strip() for code in languages},
        "cover_illustration": str(payload.get("cover_illustration", "")).strip(),
        "climax": climax,
        "cast": cast,
        "pages": pages,
    }


def _page_index(raw: Any, total: int) -> int:
    """A page number from the model, or 0 when it is not one."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 0
    return value if 1 <= value <= total else 0


def rewrite_page(
    book: Book,
    hero: Hero,
    page: Page,
    note: str = "",
    spend: dict | None = None,
    provider: str = "openai",
    model: str | None = None,
) -> Page:
    """Rewrite one page, keeping it wedged between its neighbours."""
    band = AGE_BANDS.get(book.age, AGE_BANDS["4-5"])
    order = {p.index: p for p in book.pages}
    before = order.get(page.index - 1)
    after = order.get(page.index + 1)
    language = book.primary_language

    context = []
    if before:
        context.append(f"The page before says: {before.text.get(language, '')}")
    if after:
        context.append(f"The page after says: {after.text.get(language, '')}")

    user = f"""This is page {page.index} of a picture book for {band["label"]}.

Current text ({language}): {page.text.get(language, "")}
Current illustration: {page.illustration}
Current layout: {page.layout}

{chr(10).join(context)}

What to change: {note.strip() or "make it better -- sharper, warmer, better rhythm"}

Keep the story working: this page must still lead from the page before into the
page after. Per page: {band["sentences"]}, about {band["words_per_page"]}.
{_story_instructions(hero, book.age, book.languages, len(book.pages), book.rhyme)}

Reply as {{"text": {{ {", ".join(f'"{c}": "..."' for c in book.languages)} }},
"illustration": "...", "layout": "one of {', '.join(LAYOUTS)}"}}"""

    payload = complete_json(SYSTEM, user, provider=provider, model=model,
                            spend=spend, what="story") or {}
    text = payload.get("text") or {}
    if isinstance(text, str):
        text = {language: text}

    page.text = {
        code: str(text.get(code, page.text.get(code, ""))).strip() for code in book.languages
    }
    new_illustration = single_scene(str(payload.get("illustration", "")))
    if new_illustration:
        page.illustration = new_illustration
    new_layout = str(payload.get("layout", "")).lower()
    if new_layout in LAYOUTS:
        page.layout = new_layout
    return page


def add_languages(
    book: Book,
    hero: Hero,
    codes: list[str],
    spend: dict | None = None,
    provider: str = "openai",
    model: str | None = None,
) -> Book:
    """Write an existing book into further languages, reusing the illustrations.

    This is nearly free: the pictures already exist, only the words are new.
    """
    missing = [c for c in codes if c not in book.languages]
    if not missing:
        return book

    band = AGE_BANDS.get(book.age, AGE_BANDS["4-5"])
    source = book.primary_language
    existing = [{"index": p.index, "text": p.text.get(source, "")} for p in book.pages]

    user = f"""Here is a finished picture book in {LANGUAGES.get(source, {}).get("english_name", source)},
for {band["label"]}:

Title: {book.title.get(source, "")}
Dedication: {book.dedication.get(source, "")}
Pages: {existing}

Write this same book in: {_language_block(missing)}.

Write each one natively, as a picture book author of that language would. Keep
the story, the page breaks and the beat of each page identical. Do not
translate literally -- keep the rhythm and the read-aloud quality, and let the
wording change where the language wants it to. Names stay the same.
Per page: {band["sentences"]}, about {band["words_per_page"]}.
{RHYME_GUIDANCE if book.rhyme else ""}
Reply as {{"title": {{...}}, "dedication": {{...}},
"pages": [{{"index": 1, "text": {{ {", ".join(f'"{c}": "..."' for c in missing)} }}}}]}}"""

    payload = complete_json(SYSTEM, user, provider=provider, model=model,
                            spend=spend, what="story") or {}

    title = payload.get("title") or {}
    dedication = payload.get("dedication") or {}
    for code in missing:
        book.title[code] = str(title.get(code, "")).strip()
        book.dedication[code] = str(dedication.get(code, "")).strip()

    by_index = {int(item.get("index", 0)): item for item in payload.get("pages") or []}
    for page in book.pages:
        item = by_index.get(page.index) or {}
        text = item.get("text") or {}
        for code in missing:
            page.text[code] = str(text.get(code, "")).strip()

    book.languages = book.languages + missing
    return book
