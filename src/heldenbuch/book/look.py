"""Choosing how the book looks.

A style is just a paragraph appended to every illustration prompt -- but which
paragraph makes an enormous difference, and reading one tells you almost
nothing about how it will actually look with *your* character. So the app
renders the chosen hero in a candidate style and shows you that.

The presets below are written the way image models respond best: naming the
physical medium and its artefacts (paper grain, brush edges, pencil tooth)
rather than adjectives like "beautiful" or "high quality", and saying plainly
what to avoid.
"""

from __future__ import annotations

from pathlib import Path

from ..backends import get_backend
from ..types import GenRequest, OutputSpec
from .models import Hero, Style

PRESETS: dict[str, dict[str, str]] = {
    "aquarell": {
        "name": "Aquarell",
        "hint": "weich, warm, klassisch",
        "description": (
            "Soft watercolour and coloured-pencil children's picture book "
            "illustration. Visible paper grain, gentle washes with slightly "
            "uneven edges, warm muted palette, soft natural light, light pencil "
            "outlines. No harsh black outlines, no digital gloss, no 3D render."
        ),
    },
    "buntstift": {
        "name": "Buntstift",
        "hint": "wie selbst gemalt",
        "description": (
            "Coloured pencil illustration on textured paper. Visible pencil "
            "strokes and cross-hatching, slightly loose lines, colours built up "
            "in layers so the paper shows through. Warm and handmade, a little "
            "imperfect. No airbrush smoothness, no digital gradients."
        ),
    },
    "papercut": {
        "name": "Papercollage",
        "hint": "grafisch, kräftig",
        "description": (
            "Cut-paper collage illustration. Shapes look like torn and cut "
            "coloured paper layered on top of each other, with visible paper "
            "edges and soft drop shadows between layers. Bold flat colours, "
            "simple geometric shapes, minimal detail. No gradients, no outlines, "
            "no photographic texture."
        ),
    },
    "gouache": {
        "name": "Gouache",
        "hint": "satt, mid-century",
        "description": (
            "Mid-century gouache picture book illustration. Opaque matte paint "
            "with visible brush marks, limited palette of five or six colours, "
            "flat stylised shapes, slightly off-register printing feel. Retro "
            "1960s children's book. No fine detail, no photorealism, no glow."
        ),
    },
    "knuffig3d": {
        "name": "Knuffig 3D",
        "hint": "modern, wie ein Animationsfilm",
        "description": (
            "Soft stylised 3D render in the manner of a modern animated feature. "
            "Rounded friendly shapes, subsurface-scattering skin, soft global "
            "illumination, shallow depth of field, gentle pastel palette. Clean "
            "and cosy. No photorealism, no hard shadows, no uncanny detail."
        ),
    },
    "tusche": {
        "name": "Tusche & Aquarell",
        "hint": "klassisch europäisch",
        "description": (
            "Loose ink line with watercolour wash, in the European picture book "
            "tradition. Lively slightly scratchy pen outlines that do not "
            "perfectly follow the colour, transparent washes that run past the "
            "line, generous white paper. No digital smoothness, no heavy "
            "uniform outlines."
        ),
    },
    "filz": {
        "name": "Filz & Wolle",
        "hint": "warm, taktil, ungewöhnlich",
        "description": (
            "Needle-felted wool diorama photographed under soft light. "
            "Everything looks made of fuzzy felted wool and fabric, with visible "
            "fibres and hand-stitched details, sitting in a small handmade set. "
            "Warm, tactile and slightly imperfect. No flat 2D drawing, no "
            "digital art look."
        ),
    },
    "nachtblau": {
        "name": "Nachtblau",
        "hint": "ruhig, für Gute-Nacht-Geschichten",
        # Palette and technique only. The first version also asked for "stars
        # and warm windows" and "dark calm space" -- scenery, not style -- and
        # every page inherited the same night sky whatever its brief said.
        "description": (
            "Quiet bedtime picture book illustration in a deep blue and warm "
            "amber palette. Soft painterly shapes, muted edges, generous calm "
            "negative space. Gentle and hushed. No bright saturated primaries, "
            "no busy detail, no harsh contrast."
        ),
    },
}

# What the preview shows. Neutral enough to judge the style, specific enough to
# reveal how it handles a face, a full figure and a background.
PREVIEW_SCENES = [
    "standing in a sunlit meadow with tall grass and a few butterflies, looking up",
    "sitting on the floor of a cosy room at night, reading a book by lamplight",
    "walking along a path at the edge of a wood on a windy autumn day",
]


def preset_style(key: str) -> Style:
    preset = PRESETS.get(key)
    if not preset:
        raise ValueError(f"unknown style preset {key!r}; valid: {', '.join(PRESETS)}")
    return Style(name=preset["name"], description=preset["description"], preset=key)


def preview_prompt(hero: Hero, style: Style, scene: str, with_reference: bool = False) -> str:
    """One illustration, used to judge the style with the real character in it."""
    style_note = (
        "\n\nImage 2 is a style reference. Copy its medium, palette, line "
        "quality and level of detail exactly. Ignore what image 2 depicts -- "
        "take only how it is made."
        if with_reference else ""
    )
    return (
        "A single full-page illustration for a children's picture book.\n\n"
        f"Image 1 is the character reference sheet for "
        f"{hero.name or 'the character'}. Match that character exactly: face, "
        "hair, build, clothing and colours. Do not redesign them, and do not "
        "copy the pose, framing or background of the reference."
        f"{style_note}\n\n"
        f"SCENE\n{hero.name or 'The character'} is {scene}.\n\n"
        f"STYLE\n{style.description}\n\n"
        "Do not put any text, letters, numbers, captions, borders or "
        "watermarks in the image."
    )


def generate_previews(
    hero: Hero,
    style: Style,
    base_sheet: Path,
    folder: Path,
    count: int = 2,
    backend_name: str = "openai",
    model: str | None = None,
    output: OutputSpec | None = None,
    style_reference: Path | None = None,
    spend=None,
    log=print,
) -> list[Path]:
    """Render the hero in this style so the combination can be judged for real.

    `spend` is called with each preview's usage. Previews are drawn before any
    book exists, so without it they were paid for and recorded nowhere.
    """
    backend = get_backend(backend_name, model)
    spec = output or OutputSpec(aspect_ratio="4:3", image_size="1K", quality="medium")
    made: list[Path] = []

    references = [base_sheet]
    if style_reference and style_reference.is_file():
        references.append(style_reference)

    for index, scene in enumerate(PREVIEW_SCENES[:count]):
        target = folder / f"preview_{hero.id}_{index + 1}.png"
        log(f"  Vorschau {index + 1} von {count}")
        try:
            result = backend.generate(
                GenRequest(
                    prompt=preview_prompt(hero, style, scene,
                                          with_reference=len(references) > 1),
                    reference_images=references[: backend.max_references],
                    output=spec,
                ),
                target,
            )
            if spend is not None:
                spend(result.usage)
            made.append(target)
        except Exception as exc:
            log(f"  Vorschau {index + 1} fehlgeschlagen: {exc}")

    if not made:
        raise RuntimeError("keine Vorschau möglich — siehe Protokoll oben")
    return made


DESCRIBE_IMAGE_SYSTEM = (
    "You look at a picture and write down its visual technique so that an "
    "image generation model can reproduce the same look on completely "
    "different subjects. You describe the medium and its artefacts -- paper, "
    "brush, pencil, print, render, fabric -- along with line quality and level "
    "of detail. You never describe what the picture depicts. You never name an "
    "artist or a copyrighted character.\n\n"
    "A style is a medium, not a set. This description is appended to every "
    "page of a book, so anything about *where* the reference picture happens "
    "or *when* overrides what each page asked for. Never mention a location, "
    "a habitat, a time of day, a weather condition or a specific lighting "
    "set-up: no jungle, no forest, no sunset, no dappled sunlight through "
    "leaves, no golden hour. Describe how light behaves in the medium -- soft "
    "or hard edges, flat or volumetric shading, how highlights fall -- not "
    "what is lighting the scene. Palette means the character of the colour "
    "(muted, chalky, high-contrast, warm-biased), not a list of the scenery's "
    "colours.\n\n"
    "Reply with JSON only."
)


def describe_style_from_image(
    path: Path, provider: str = "openai", model: str | None = None,
    spend: dict | None = None,
) -> dict[str, str]:
    """Turn an uploaded picture into a style description.

    Words can only get so close to a look. A photograph of a felt toy, or a
    crayon drawing your child made, carries information no adjective does.
    """
    from ..llm import complete_json

    user = (
        "Describe the visual style of this picture so it can be applied to "
        "other scenes.\n\n"
        'Reply as {"name": "...", "description": "..."} where name is two or '
        "three words in German for a menu entry, and description is three to "
        "five English sentences naming the medium, the character of the "
        "palette, line quality, how the medium renders light, and level of "
        "detail, ending with what to avoid. Say nothing about the subject of "
        "the picture, and nothing about where or when it happens."
    )
    payload = complete_json(DESCRIBE_IMAGE_SYSTEM, user, images=[path],
                            provider=provider, model=model,
                            spend=spend, what="style") or {}
    description = (payload.get("description") or "").strip()
    if not description:
        raise RuntimeError("aus diesem Bild ließ sich kein Stil ableiten")
    return {"name": (payload.get("name") or "Nach Vorlage").strip(), "description": description}


def describe_custom_style(
    wish: str, provider: str = "openai", model: str | None = None,
    spend: dict | None = None,
) -> dict[str, str]:
    """Turn a rough wish ("wie Janosch", "neon space comic") into a usable style.

    People describe styles by reference and by feeling. Image models want the
    medium and its artefacts. This bridges the two.
    """
    from ..llm import complete_json

    system = (
        "You write style descriptions for an image generation model that will "
        "illustrate a children's picture book. You describe the physical medium "
        "and its visible artefacts -- paper, brush, pencil, print, render -- "
        "never vague praise. You always state what to avoid. You never name a "
        "living artist or a copyrighted character; translate any such reference "
        "into the concrete visual technique behind it. Reply with JSON only."
    )
    user = (
        f"The wish is: {wish!r}\n\n"
        'Reply as {"name": "...", "description": "..."} where name is two or '
        "three words in German for a menu entry, and description is three to "
        "five English sentences naming the medium, palette, line quality, "
        "lighting and level of detail, ending with what to avoid."
    )
    payload = complete_json(system, user, provider=provider, model=model,
                            spend=spend, what="style") or {}
    name = (payload.get("name") or wish[:30]).strip()
    description = (payload.get("description") or "").strip()
    if not description:
        raise RuntimeError("could not turn that wish into a style description")
    return {"name": name, "description": description}
