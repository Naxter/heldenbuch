"""Prompt construction -- the actual independent variable of the experiment.

Three strategies, kept deliberately clean so the comparison means something:

  text_only        the character is described in words, no reference image
  sheet_ref        the character sheet is the only definition of the character
  sheet_plus_prev  the sheet, plus the previously generated page

`sheet_ref` and `sheet_plus_prev` intentionally leave the character description
*out* of the prompt. Both Black Forest Labs and Google advise keeping identity
descriptors out once you pass a reference, because the text and the image
otherwise compete and the model splits the difference. Including it in one arm
and not the other would also confound the comparison.
"""

from __future__ import annotations

from .types import BenchmarkSpec, Scene

NO_TEXT = (
    "Do not put any text, letters, numbers, captions, page borders, panel "
    "frames or watermarks in the image."
)

DEFAULT_SHEET_PROMPT = (
    "A character reference sheet for a children's picture book, on a plain "
    "off-white background. Show the same character four times in a row: front "
    "view, three-quarter view, side profile, and back view. Full body, "
    "standing, evenly lit."
)


def sheet_prompt(spec: BenchmarkSpec) -> str:
    """The one image everything else is measured against."""
    frame = spec.character.sheet_prompt or DEFAULT_SHEET_PROMPT
    return (
        f"{frame}\n\n"
        f"CHARACTER: {spec.character.name}\n{spec.character.description}\n\n"
        f"STYLE\n{spec.style.description}\n\n"
        f"{NO_TEXT}"
    )


def scene_prompt(spec: BenchmarkSpec, scene: Scene, strategy: str, has_previous: bool) -> str:
    name = spec.character.name
    header = "A single full-page illustration for a children's picture book."

    if strategy == "text_only":
        return (
            f"{header}\n\n"
            f"SCENE\n{scene.action}\n\n"
            f"CHARACTER: {name} -- every detail below must be exactly right\n"
            f"{spec.character.description}\n\n"
            f"STYLE\n{spec.style.description}\n\n"
            f"{NO_TEXT}"
        )

    if strategy == "sheet_ref" or (strategy == "sheet_plus_prev" and not has_previous):
        return (
            f"{header}\n\n"
            f"Image 1 is the character reference sheet for {name}. Treat image 1 "
            f"as the exact and complete definition of this character: fur, "
            f"markings, colours, clothing, props, proportions and face must match "
            f"image 1 precisely. Do not redesign, restyle, age or simplify the "
            f"character, and do not add or remove anything they wear or carry.\n\n"
            f"SCENE\n{scene.action}\n\n"
            f"Use image 1 only for the character. Do not copy its pose, framing, "
            f"lighting or background -- build the new scene described above.\n\n"
            f"STYLE\n{spec.style.description}\n\n"
            f"{NO_TEXT}"
        )

    # sheet_plus_prev, with a previous page available
    return (
        f"{header}\n\n"
        f"Image 1 is the character reference sheet for {name}. Image 2 is the "
        f"previous page of the same book.\n\n"
        f"Take the character's identity from image 1 -- fur, markings, colours, "
        f"clothing, props, proportions and face must match it precisely. Take "
        f"the painting style, palette, brush quality and level of detail from "
        f"image 2, so the two pages sit next to each other in one book.\n\n"
        f"SCENE\n{scene.action}\n\n"
        f"Do not copy the pose, framing or background of either image -- build "
        f"the new scene described above.\n\n"
        f"{NO_TEXT}"
    )


def judge_prompt(spec: BenchmarkSpec) -> str:
    """Rubric for the model that scores a page against the sheet."""
    return f"""You are grading illustrations for a children's picture book.

Image 1 is the CHARACTER REFERENCE SHEET. Image 2 is a PAGE from the book.
Both are meant to show the same character, {spec.character.name}.

The character is defined as:
{spec.character.description}

Judge only whether image 2 shows the SAME character as image 1, and whether it
belongs in the same book. A different pose, camera angle, background or time of
day is fine and expected -- do not penalise those.

Score each dimension from 1 to 5:

identity   5 = unmistakably the same character; 3 = same species and rough
           idea but the face or proportions have shifted; 1 = a different
           character.
attributes 5 = every signature attribute is present, correct and appears
           exactly once; 3 = one is wrong, miscoloured, missing or duplicated;
           1 = several wrong. Count anything countable, and check explicitly
           for props that appear twice or extra items that were never in the
           character's description.
style      5 = the same illustration technique and palette as image 1;
           3 = related but noticeably different; 1 = a different medium.

Then list every concrete discrepancy you can see, as short phrases naming the
attribute and what went wrong, for example "scarf is blue, should be
moss-green" or "four tail rings instead of three". If the character is not
visible in image 2 at all, set identity to 1 and say so.

Reply with JSON only, no prose and no code fences:
{{"identity": <1-5>, "attributes": <1-5>, "style": <1-5>, "discrepancies": ["..."]}}"""
