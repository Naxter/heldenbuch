"""Which image model should draw this book?

The benchmark half of this project exists to answer exactly one question --
which backend keeps a character most consistent -- and there is no reason the
app should not simply ask it.

The scout draws the same two test scenes with every backend that has a key,
checks each against the hero's sheet, and records a winner on the style. It is
a small, cheap version of the full benchmark: same idea, same checker, four
images instead of ninety.

Worth being honest about what this measures: two scenes is a small sample and
image models vary run to run. Treat the answer as a nudge, not a verdict, which
is why the app suggests the winner rather than silently switching to it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..backends import REQUIRED_KEY, get_backend
from ..config import load_dotenv
from ..pricing import add as add_spend
from ..types import GenRequest, OutputSpec
from .illustrate import check_page
from .models import Hero, Style

# Deliberately awkward: a back view hides the face, and a close-up exposes every
# small prop. Backends that survive both survive a book.
TEST_SCENES = [
    "seen from behind, walking away along a narrow path between tall trees",
    "a close-up of their face and shoulders, looking down at something small "
    "held in both hands",
]


def available_backends() -> list[str]:
    from ..backends import comfy_available

    load_dotenv()
    ready = [
        name for name, key in REQUIRED_KEY.items()
        if name != "stub" and key and os.environ.get(key, "").strip()
    ]
    # The local backend has no key; it counts when its server is running.
    if comfy_available():
        ready.append("comfy")
    return ready


def _prompt(hero: Hero, style: Style, scene: str) -> str:
    return (
        "A single full-page illustration for a children's picture book.\n\n"
        f"Image 1 is the character reference sheet for {hero.name or 'the character'}. "
        "Match that character exactly: face, hair, build, clothing and colours.\n\n"
        f"SCENE\n{hero.name or 'The character'} is {scene}.\n\n"
        f"STYLE\n{style.description}\n\n"
        "Do not put any text, letters, numbers or watermarks in the image."
    )


def run(
    hero: Hero,
    style: Style,
    sheet: Path,
    folder: Path,
    backends: list[str] | None = None,
    check_provider: str = "openai",
    spend: dict | None = None,
    log=print,
    should_stop=None,
) -> dict[str, Any]:
    """Draw the test scenes with each backend and score them."""
    stop = should_stop or (lambda: False)
    backends = backends or available_backends()
    if len(backends) < 2:
        raise RuntimeError(
            "Für einen Vergleich brauche ich mindestens zwei Bilddienste mit "
            "Schlüssel. Vorhanden: " + (", ".join(backends) or "keiner")
        )

    folder.mkdir(parents=True, exist_ok=True)
    spec = OutputSpec(aspect_ratio="1:1", image_size="1K", quality="medium",
                      long_edge_px=1024)
    results: dict[str, Any] = {}

    for name in backends:
        if stop():
            break
        log(f"{name}")
        scores, images = [], []
        try:
            backend = get_backend(name)
        except Exception as exc:
            log(f"  übersprungen: {exc}")
            continue

        for index, scene in enumerate(TEST_SCENES, start=1):
            if stop():
                break
            target = folder / f"scout_{name}_{index}.png"
            try:
                result = backend.generate(
                    GenRequest(prompt=_prompt(hero, style, scene),
                               reference_images=[sheet], output=spec),
                    target,
                )
                if spend is not None:
                    add_spend(spend, result.usage, "scout")
                verdict = check_page(target, sheet, hero, provider=check_provider)
                scores.append(verdict.get("identity", 0))
                images.append(str(target.name))
                log(f"  Szene {index}: Ähnlichkeit {verdict.get('identity')}/5")
            except Exception as exc:
                log(f"  Szene {index} fehlgeschlagen: {exc}")

        if scores:
            results[name] = {
                "identity": round(sum(scores) / len(scores), 2),
                "scenes": len(scores),
                "images": images,
            }

    if not results:
        raise RuntimeError("kein Vergleich möglich — siehe Protokoll oben")

    winner = max(results, key=lambda k: results[k]["identity"])
    log("")
    for name, row in sorted(results.items(), key=lambda kv: -kv[1]["identity"]):
        mark = "  ←" if name == winner else ""
        log(f"{name:<8} Ähnlichkeit {row['identity']}/5{mark}")
    log("")
    log(f"Empfehlung für „{style.name}“: {winner}")
    log("Zwei Szenen sind eine kleine Stichprobe — nimm es als Hinweis, nicht als Urteil.")

    return {"winner": winner, "results": results, "scenes": len(TEST_SCENES),
            "checked_by": check_provider}
