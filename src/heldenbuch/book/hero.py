"""Turning a real child (or an idea) into a picture-book character.

Two paths in, one thing out:

    photos      -> a vision model writes down what an illustrator would need
                   (hair, eyes, skin tone, glasses, build), then an image model
                   draws that as a character, four views, plain background
    description -> the same, skipping the first step

The result is a *character sheet*: one image showing the same character from
four angles. Everything after this points at that image instead of describing
the character in words, because a reference image holds an identity far better
than any amount of text.

On privacy: uploaded photos are stored in the library folder on this machine
and leave it only for this step -- once to the text provider that writes the
description, and once per sheet drawn to the image provider (several variants
means several sends). No page illustration ever sees a photo; they see the
drawn character sheet.
"""

from __future__ import annotations

from pathlib import Path

from ..backends import get_backend
from ..types import GenRequest, OutputSpec
from .models import Hero, Style

# The character sheet itself is drawn in a clean, neutral picture-book style.
# Identity first; the chosen look is applied afterwards in `look.py`.
NEUTRAL_STYLE = (
    "Clean, warm children's picture book illustration. Soft even lighting, "
    "gentle outlines, friendly proportions, flat plain off-white background. "
    "Not photorealistic, not 3D rendered, no photographic texture."
)

SHEET_FRAME = (
    "A character reference sheet for a children's picture book on a plain "
    "off-white background. Show the SAME character four times in a row: front "
    "view, three-quarter view, side profile, and back view. Full body, "
    "standing, relaxed natural pose, evenly lit, no shadows on the background. "
    "Do not put any text, letters, numbers, labels or watermarks in the image."
)

DESCRIBE_SYSTEM = (
    "You help an illustrator turn a photograph of a child into a friendly "
    "picture-book character. You describe only what an illustrator needs in "
    "order to draw a recognisable cartoon: hair, eyes, skin tone, build, and "
    "clothing. You never speculate about identity, ethnicity, health or "
    "family, and you never describe anything that is not visible. Reply with "
    "JSON only."
)


def describe_from_photos(
    photos: list[Path],
    name: str,
    age_hint: str = "",
    provider: str = "openai",
    model: str | None = None,
) -> str:
    """Write the illustrator's brief for a child, from their photos."""
    from ..llm import complete_json

    user = (
        f"These are photos of a child called {name or 'the child'}."
        + (f" Roughly {age_hint} years old." if age_hint else "")
        + "\n\nWrite a short brief an illustrator could draw from. Cover: "
        "approximate age, hair colour and hairstyle, eye colour, skin tone, "
        "build and height for their age, and any consistent visible feature "
        "worth keeping (glasses, freckles, a gap in the teeth). Then invent a "
        "simple everyday outfit that suits them and can be drawn identically "
        "on every page.\n\n"
        'Reply as {"description": "..."} with the brief as one paragraph of '
        "plain, concrete sentences. Write it in English regardless of the "
        "language of this request."
    )
    payload = complete_json(DESCRIBE_SYSTEM, user, images=photos, provider=provider, model=model)
    description = (payload or {}).get("description", "").strip()
    if not description:
        raise RuntimeError("the model returned an empty character description")
    return description


def sheet_prompt(hero: Hero, style_description: str | None = None, from_photos: bool = False) -> str:
    """The prompt that draws a character sheet."""
    style = style_description or NEUTRAL_STYLE
    lines = [SHEET_FRAME, ""]

    if from_photos:
        lines += [
            "The reference images are photographs of a real child. Draw that "
            "child as a warm, friendly illustrated character: keep the hair "
            "colour and hairstyle, eye colour, skin tone and general face "
            "shape recognisable, but make it a drawing, never a photograph and "
            "never photorealistic. Do not copy the background, clothing or "
            "pose from the photographs.",
            "",
        ]

    lines += [
        f"CHARACTER: {hero.name}" if hero.name else "CHARACTER",
        hero.description,
        "",
        "STYLE",
        style,
    ]
    return "\n".join(lines)


def generate_sheet(
    hero: Hero,
    target: Path,
    backend_name: str = "openai",
    model: str | None = None,
    photo_paths: list[Path] | None = None,
    style_description: str | None = None,
    output: OutputSpec | None = None,
) -> Path:
    """Draw one character sheet and write it to `target`."""
    backend = get_backend(backend_name, model)
    photos = photo_paths or []
    request = GenRequest(
        prompt=sheet_prompt(hero, style_description, from_photos=bool(photos)),
        reference_images=list(photos)[: backend.max_references],
        output=output or OutputSpec(aspect_ratio="3:2", image_size="2K", quality="high"),
        kind="sheet",
    )
    backend.generate(request, target)
    return target


def generate_variants(
    hero: Hero,
    folder: Path,
    count: int = 3,
    backend_name: str = "openai",
    model: str | None = None,
    photo_paths: list[Path] | None = None,
    log=print,
) -> list[Path]:
    """Draw several sheets so there is something to choose between.

    Image models vary a lot run to run on a first draw. Showing three and
    letting a person pick is faster and cheaper than trying to prompt-engineer
    the perfect single result.
    """
    made: list[Path] = []
    existing = len(list(folder.glob("sheet_*.png")))
    for index in range(count):
        target = folder / f"sheet_{existing + index + 1:02d}.png"
        log(f"  drawing character sheet {index + 1} of {count}")
        try:
            generate_sheet(
                hero,
                target,
                backend_name=backend_name,
                model=model,
                photo_paths=photo_paths,
            )
            made.append(target)
        except Exception as exc:
            log(f"  variant {index + 1} failed: {exc}")
    if not made:
        raise RuntimeError("no character sheet could be drawn -- see the log above")
    return made


def styled_sheet_prompt(hero: Hero, style: Style) -> str:
    """Redraw an existing character sheet in the chosen look."""
    return (
        f"{SHEET_FRAME}\n\n"
        f"Image 1 is the existing character sheet for {hero.name or 'this character'}. "
        "Keep the character exactly as they are -- same face, same hair, same "
        "build, same clothing, same colours. Change only the drawing style, and "
        "keep the same four views in the same order.\n\n"
        f"STYLE\n{style.description}"
    )


def generate_styled_sheet(
    hero: Hero,
    style: Style,
    base_sheet: Path,
    target: Path,
    backend_name: str = "openai",
    model: str | None = None,
    output: OutputSpec | None = None,
) -> Path:
    """Lock identity and look into a single reference the pages can point at."""
    backend = get_backend(backend_name, model)
    request = GenRequest(
        prompt=styled_sheet_prompt(hero, style),
        reference_images=[base_sheet],
        output=output or OutputSpec(aspect_ratio="3:2", image_size="2K", quality="high"),
        kind="sheet",
    )
    backend.generate(request, target)
    return target
