"""Reference sheets for everyone who is not the hero, and for the place.

The hero holds together because every page points at one drawing of him. The
grandmother, the dog and the garden get exactly the same treatment, for exactly
the same reason: without a fixed picture to point at, the model reinvents them
every time, and a book where the dog changes breed on page six reads as broken
even when nobody can say why.

Their sheets are simpler than the hero's -- one or two views, drawn straight
into the chosen style, since nobody restyles them later.
"""

from __future__ import annotations

from pathlib import Path

from ..backends import get_backend
from ..types import GenRequest, OutputSpec
from .models import CastMember, Style

CHARACTER_FRAME = (
    "A small character reference sheet for a children's picture book, on a "
    "plain off-white background. Show the SAME character twice: front view and "
    "three-quarter view, full body, standing, evenly lit. No text, letters, "
    "numbers or labels anywhere in the image."
)

PLACE_FRAME = (
    "A location reference for a children's picture book: one wide establishing "
    "view of the place described below, empty of people, evenly lit, drawn so "
    "that its layout and its distinctive features are clear. No text, letters, "
    "numbers or labels anywhere in the image."
)

#: Things the story builds, finds or carries. Without one of these a prop is
#: reinvented on every page it appears on: one book's whole plot is building a
#: bridge, the bridge is in eight pictures, and it came out as a fallen mossy
#: log on one page and a neatly built raft of round logs three pages later.
PROP_FRAME = (
    "An object reference for a children's picture book: the single object "
    "described below on a plain off-white background, seen from one clear "
    "three-quarter angle, whole and evenly lit, with nobody in the picture. "
    "No text, letters, numbers or labels anywhere in the image."
)

_FRAMES = {"place": PLACE_FRAME, "prop": PROP_FRAME}
_LABELS = {"place": "PLACE", "prop": "OBJECT"}


def sheet_prompt(member: CastMember, style: Style) -> str:
    frame = _FRAMES.get(member.kind, CHARACTER_FRAME)
    label = _LABELS.get(member.kind, "CHARACTER")
    return (
        f"{frame}\n\n"
        f"{label}: {member.name}\n{member.description}\n\n"
        f"STYLE\n{style.description}"
    )


def generate_sheet(
    member: CastMember,
    style: Style,
    target: Path,
    backend_name: str = "openai",
    model: str | None = None,
    style_reference: Path | None = None,
    output: OutputSpec | None = None,
):
    """Draw one sheet. Returns the GenResult so callers can meter the cost."""
    backend = get_backend(backend_name, model)
    references = [style_reference] if style_reference and style_reference.is_file() else []
    return backend.generate(
        GenRequest(
            prompt=sheet_prompt(member, style),
            reference_images=references,
            output=output or OutputSpec(aspect_ratio="3:2", image_size="2K", quality="medium"),
            kind="sheet",
        ),
        target,
    )


def generate_all(
    cast: list[CastMember],
    style: Style,
    folder: Path,
    backend_name: str = "openai",
    model: str | None = None,
    style_reference: Path | None = None,
    relative_to: Path | None = None,
    spend=None,
    log=print,
    should_stop=None,
) -> list[CastMember]:
    """Draw a sheet for every cast member that does not have one yet.

    `spend` is called with each call's usage, so cast sheets show up in the
    book's cost ledger like every other image.
    """
    stop = should_stop or (lambda: False)
    folder.mkdir(parents=True, exist_ok=True)
    base = relative_to or folder.parent

    for index, member in enumerate(cast, start=1):
        if stop() or member.sheet:
            continue
        target = folder / f"cast_{index:02d}.png"
        kind = "Ort" if member.kind == "place" else "Figur"
        log(f"  {kind}: {member.name}")
        try:
            result = generate_sheet(
                member, style, target,
                backend_name=backend_name, model=model, style_reference=style_reference,
            )
            member.sheet = str(target.relative_to(base)).replace("\\", "/")
            if spend:
                spend(result.usage)
        except Exception as exc:
            log(f"    fehlgeschlagen: {exc}")
    return cast
