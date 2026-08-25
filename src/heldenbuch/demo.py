"""A complete example book, built without any API key.

Every other path through this app calls a paid service: a story needs a text
model, a page needs an image model. That makes the app hard to look at before
you have decided to pay for it, which is the wrong way round.

So this builds one finished book from text that ships with the source and
pictures drawn by the offline `stub` backend. It is not a demonstration of the
image quality -- stub draws labelled placeholders -- it is a demonstration of
the shape of the thing: a hero, a style, a story split into pages with briefs,
a cast with reference sheets, and an export.

It writes to its own library directory and never touches yours.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .book import cast as cast_mod
from .book import hero as hero_mod
from .book import illustrate, look
from .book.library import Library
from .book.models import Book, CastMember, Hero, Page

#: The same fox the benchmark spec uses: invented here, and described in
#: attributes a person can check off a drawing one by one.
HERO = Hero(
    name="Rusty",
    description=(
        "Rusty is a small fox cub, about the size of a house cat, with warm "
        "rust-orange fur and a cream-white chest and muzzle. Three cream-white "
        "rings near the tip of the bushy tail, a moss-green knitted scarf worn "
        "loose with a frayed left end, large amber-yellow eyes, and a small "
        "brass compass on a leather cord around the neck. Rusty walks upright "
        "like a storybook animal, with an alert, curious expression."
    ),
    source="described",
)

CAST = [
    CastMember(
        name="Der Nebelwald",
        kind="place",
        description=(
            "A misty pine forest at dawn: tall dark trunks, low silver fog "
            "between them, moss and fallen needles underfoot."
        ),
    ),
    CastMember(
        name="Die Laterne",
        kind="prop",
        description=(
            "A small brass lantern with four glass panes, one of them cracked, "
            "a stubby candle inside and a wire handle worn smooth."
        ),
    ),
]

#: German and English were both written rather than translated, which is what
#: the app does for a real book too. The last element lists the cast on that
#: page -- the briefs say "the lantern" while the prop is registered as "Die
#: Laterne", which is exactly the wording gap the page list exists to close.
PAGES = [
    ("Rusty wacht auf, als es noch dunkel ist. Der Wald ist ganz still.",
     "Rusty wakes while it is still dark. The forest is completely quiet.",
     "Rusty sits up in a hollow under a pine root at dawn, ears forward, "
     "the misty forest behind.", "just woken, ears up", "facing the reader",
     [], "at the forest edge, the tall split pine on the left"),
    ("Zwischen den Bäumen liegt Nebel. Rusty kann den Weg nicht sehen.",
     "Fog lies between the trees. Rusty cannot see the path.",
     "Rusty stands at the edge of the misty forest, one paw raised, looking "
     "into thick silver fog.", "uncertain", "moving left to right", [],
     "at the forest edge, the tall split pine on the left"),
    ("„Ich habe die Laterne\", sagt Rusty. Sie ist kalt und dunkel.",
     "\"I have the lantern,\" says Rusty. It is cold and dark.",
     "Rusty holds up the small brass lantern in both paws, examining the "
     "cracked pane.", "concentrating", "facing the reader", ["Die Laterne"],
     ""),
    ("Rusty sucht ein Streichholz. In der Tasche ist nur Moos.",
     "Rusty looks for a match. There is only moss in the pocket.",
     "Close on Rusty's paws turning out a small pocket, moss falling out, the "
     "lantern on the ground.", "disappointed", "looking down",
     ["Die Laterne"], ""),
    ("Da leuchtet etwas. Ein Glühwürmchen sitzt auf dem Moos.",
     "Then something glows. A firefly is sitting on the moss.",
     "A single firefly glowing on green moss, Rusty's face lit from below, "
     "eyes wide.", "delighted, wide-eyed", "facing the reader", [], ""),
    ("Rusty öffnet die Laterne. Das Glühwürmchen fliegt hinein.",
     "Rusty opens the lantern. The firefly flies inside.",
     "Rusty holds the lantern open, the firefly drifting in through the "
     "cracked pane, warm light spilling out.", "careful, hopeful",
     "moving left to right", ["Die Laterne"], ""),
    ("Jetzt sieht Rusty den Weg. Er führt zwischen den Bäumen hindurch.",
     "Now Rusty can see the path. It runs between the trees.",
     "Wide view of the misty forest with a narrow path revealed by lantern "
     "light, Rusty small in the middle of it.", "relieved",
     "moving left to right", ["Die Laterne"],
     "at the forest edge, the tall split pine on the left"),
    ("Zu Hause lässt Rusty das Glühwürmchen wieder frei. Gute Nacht.",
     "At home Rusty lets the firefly go again. Good night.",
     "Rusty at the mouth of a burrow at first light, holding the open lantern "
     "up, the firefly flying out.", "content, sleepy", "facing left",
     ["Die Laterne"], ""),
]

TITLE = {"de": "Rusty und das Licht im Nebel", "en": "Rusty and the Light in the Fog"}
DEDICATION = {"de": "Für alle, die im Dunkeln losgehen.",
              "en": "For everyone who sets out in the dark."}
BLURB = {"de": "Ein Fuchs, ein Nebelwald und ein sehr kleines Licht.",
         "en": "A fox, a foggy forest and one very small light."}


def build(root: Path, fresh: bool = False, log=print) -> Book:
    """Create the demo library and return the finished book."""
    if fresh and root.exists():
        shutil.rmtree(root)

    library = Library(root)
    existing = library.books()
    if existing and not fresh:
        log(f"{root} already has {len(existing)} book(s). "
            "Pass --fresh to build it again from scratch.")
        return existing[0]

    log("Drawing the character sheets (offline, no key, no cost) ...")
    hero = Hero(name=HERO.name, description=HERO.description, source=HERO.source)
    folder = library.hero_dir(hero.id)
    folder.mkdir(parents=True, exist_ok=True)
    made = hero_mod.generate_variants(hero, folder, count=2, backend_name="stub",
                                      log=lambda *a: None)
    hero.variants = [library.relative(p) for p in made]
    hero.sheet = hero.variants[0]
    library.save_hero(hero)

    log("Setting up the style ...")
    style = look.preset_style("aquarell")
    style_folder = library.style_dir(style.id)
    style_folder.mkdir(parents=True, exist_ok=True)
    styled = style_folder / f"sheet_{hero.id}.png"
    hero_mod.generate_styled_sheet(hero, style, library.resolve(hero.sheet), styled,
                                   backend_name="stub")
    style.sheets[hero.id] = library.relative(styled)
    style.sheets_from[hero.id] = hero.sheet
    style.previews = [style.sheets[hero.id]]
    library.save_style(style)

    log("Writing the story ...")
    book = Book(
        hero_id=hero.id, style_id=style.id,
        title=dict(TITLE), dedication=dict(DEDICATION), blurb=dict(BLURB),
        languages=["de", "en"], age="4-5", climax=5,
        idea="a fox cub finds a light in the fog",
        cover_illustration="Rusty holding a glowing lantern in a misty forest",
        cast=[CastMember(name=c.name, kind=c.kind, description=c.description)
              for c in CAST],
        pages=[
            Page(index=i, text={"de": de, "en": en}, illustration=brief,
                 expression=face, direction=facing, cast=list(on_page),
                 setting=where, layout="vignette" if i == 4 else "full")
            for i, (de, en, brief, face, facing, on_page, where)
            in enumerate(PAGES, start=1)
        ],
    )
    library.lock_references(book, hero, style)
    library.save_book(book)

    # Sheets before pages, the same order the app's own render job uses: a
    # page drawn while its cast has no sheet is conditioned on nothing, and
    # the forest and the lantern come out different on every page.
    log("Drawing the cast sheets ...")
    cast_mod.generate_all(
        book.cast, style, library.book_dir(book.id) / "cast",
        backend_name="stub", relative_to=library.book_dir(book.id),
        log=lambda *a: None,
    )
    library.save_book(book)

    log("Drawing the pages ...")
    illustrate.illustrate_book(
        book, hero, style, library.resolve(style.sheets[hero.id]),
        pages_dir=library.book_dir(book.id) / "pages",
        backend_name="stub", check=False, workers=1,
        resolve=lambda rel: library.book_dir(book.id) / rel,
        log=lambda *a: None,
    )
    library.save_book(book)
    return book
