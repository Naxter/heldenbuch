"""Offline stub backend -- draws placeholder images, calls no API, costs nothing.

Its job is to exercise the whole pipeline (generation -> metrics -> judge ->
report) without a key, so the harness itself can be tested and debugged.

It is not a toy: it deliberately *simulates drift*. Character attributes (fur
colour, scarf colour, number of tail rings, ear notch) are derived from a hash
of the prompt, so they wander from page to page. When reference images are
supplied it reads the character's colours back off the reference and pulls the
drift in. So a correct harness should score `sheet_ref` above `text_only` even
in stub mode -- which is exactly the check that proves the scoring works.
"""

from __future__ import annotations

import colorsys
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..types import GenRequest
from .base import Backend

# How far attributes are allowed to wander, by number of reference images.
DRIFT_BY_REFERENCES = {0: 0.55, 1: 0.18, 2: 0.09}

# The character as originally designed. Pages without a reference drift from it.
CANON_FUR = (206, 106, 48)
CANON_SCARF = (94, 128, 84)
CANON_EYES = (240, 186, 60)

PAPER = (247, 243, 234)


def _hash_floats(text: str, count: int) -> list[float]:
    """Deterministic pseudo-random floats in [0, 1) derived from a string."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    while len(digest) < count * 4:
        digest += hashlib.sha256(digest).digest()
    return [int.from_bytes(digest[i * 4 : i * 4 + 4], "big") / 2**32 for i in range(count)]


def _shift(colour: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    """Rotate hue and nudge saturation by `amount` (0 = unchanged)."""
    r, g, b = (c / 255 for c in colour)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + amount * 0.5) % 1.0
    s = min(1.0, max(0.15, s + amount * 0.3))
    v = min(1.0, max(0.2, v - amount * 0.15))
    return tuple(int(round(c * 255)) for c in colorsys.hsv_to_rgb(h, s, v))  # type: ignore[return-value]


class _Geometry:
    """Where every part of the figure sits, as a function of canvas size.

    Drawing and reference-sampling both go through this, so the stub reads the
    colours back from exactly the pixels it painted -- which is what makes
    `sheet_ref` behave like real reference conditioning instead of averaging
    the background in.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.body = int(min(width, height) * 0.20)
        self.cx = width // 2
        self.cy = int(height * 0.58)
        self.neck_y = self.cy - int(self.body * 0.75)
        self.head_r = int(self.body * 0.78)
        self.head_y = self.neck_y - int(self.head_r * 0.75)
        self.eye_r = max(3, self.head_r // 7)

    @property
    def fur_point(self) -> tuple[int, int]:
        return self.cx, self.cy + int(self.body * 0.55)

    @property
    def scarf_point(self) -> tuple[int, int]:
        return self.cx - int(self.body * 0.75), self.neck_y + int(self.body * 0.17)

    @property
    def eye_point(self) -> tuple[int, int]:
        return (
            self.cx + self.head_r // 2 + int(self.eye_r * 0.6),
            self.head_y - self.head_r // 8,
        )


def _sample(image: Image.Image, point: tuple[int, int], radius: int = 3) -> tuple[int, int, int]:
    """Median colour of a small patch, so one stray pixel cannot swing it."""
    x, y = point
    x = max(radius, min(image.width - radius - 1, x))
    y = max(radius, min(image.height - radius - 1, y))
    patch = np.asarray(
        image.convert("RGB").crop((x - radius, y - radius, x + radius + 1, y + radius + 1)),
        dtype=np.uint8,
    ).reshape(-1, 3)
    return tuple(int(v) for v in np.median(patch, axis=0))  # type: ignore[return-value]


def _read_reference(path: Path) -> tuple[tuple[int, int, int], ...]:
    """Recover (fur, scarf, eyes) from a previously generated stub image."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        geometry = _Geometry(image.width, image.height)
        return (
            _sample(image, geometry.fur_point),
            _sample(image, geometry.scarf_point),
            _sample(image, geometry.eye_point, radius=1),
        )


class StubBackend(Backend):
    name = "stub"
    max_references = 8
    honours_seed = True  # deterministic by construction

    @property
    def default_model(self) -> str:
        return "stub-v1"

    def _generate(self, req: GenRequest) -> tuple[bytes, str]:
        width, height = req.output.pixel_size()
        width, height = min(width, 768), min(height, 768)  # keep stub runs fast

        is_sheet = req.kind == "sheet"
        drift = 0.0 if is_sheet else DRIFT_BY_REFERENCES.get(len(req.reference_images), 0.05)
        # The seed joins the hash: real services vary per seed, and the
        # variants feature draws the same prompt several times expecting
        # different candidates -- hashing the prompt alone made all of them
        # pixel-identical in the no-key harness.
        rolls = _hash_floats(f"{req.prompt}|{req.output.seed or ''}", 8)

        fur, scarf, eyes = CANON_FUR, CANON_SCARF, CANON_EYES
        if req.reference_images:
            fur, scarf, eyes = _read_reference(Path(req.reference_images[0]))

        fur = _shift(fur, (rolls[0] - 0.5) * drift)
        scarf = _shift(scarf, (rolls[1] - 0.5) * drift * 2)
        eyes = _shift(eyes, (rolls[2] - 0.5) * drift)

        # Two attributes a judge can count: tail rings and the ear notch.
        rings = 3 if rolls[3] > drift else (2 if rolls[4] > 0.5 else 4)
        notch = rolls[5] > drift * 0.8

        image = self._draw(width, height, fur, scarf, eyes, rings, notch, rolls, is_sheet)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        kind = "sheet" if is_sheet else "page"
        # Free, but still counted, so the spend meter can be exercised without
        # spending anything.
        self.last_usage = {"images": 1}
        return buffer.getvalue(), f"stub {kind} drift={drift:.2f} rings={rings} notch={notch}"

    @staticmethod
    def _draw(
        width: int,
        height: int,
        fur: tuple[int, int, int],
        scarf: tuple[int, int, int],
        eyes: tuple[int, int, int],
        rings: int,
        notch: bool,
        rolls: list[float],
        is_sheet: bool,
    ) -> Image.Image:
        image = Image.new("RGB", (width, height), PAPER)
        draw = ImageDraw.Draw(image)
        g = _Geometry(width, height)

        # Pages get a scene wash so they differ from each other the way real
        # ones would. The reference sheet stays on plain paper, as its prompt asks.
        if not is_sheet:
            wash = _shift((196, 214, 196), (rolls[6] - 0.5) * 0.8)
            draw.rectangle([0, int(height * 0.66), width, height], fill=wash)

        cx, cy, body = g.cx, g.cy, g.body

        # Tail with its countable rings, tucked in from the frame edge.
        tail_x = cx - int(body * 1.7)
        for index in range(6):
            t = index / 5
            radius = int(body * (0.40 - 0.05 * t))
            x = tail_x - int(body * 0.42 * t)
            y = cy + int(body * 0.30) - int(body * 0.75 * t)
            colour = fur if index < 6 - rings else (246, 241, 229)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=colour)

        draw.ellipse([cx - body, cy - body, cx + body, cy + int(body * 1.5)], fill=fur)

        # Scarf across the neck.
        draw.rectangle(
            [cx - int(body * 1.05), g.neck_y, cx + int(body * 1.05), g.neck_y + int(body * 0.34)],
            fill=scarf,
        )

        head_r, head_y = g.head_r, g.head_y
        draw.ellipse([cx - head_r, head_y - head_r, cx + head_r, head_y + head_r], fill=fur)

        # Ears. The left one carries the notch.
        ear = int(head_r * 0.62)
        left = [
            (cx - head_r + ear // 3, head_y - head_r + ear // 4),
            (cx - head_r // 3, head_y - head_r - ear),
            (cx - head_r // 8, head_y - head_r + ear // 3),
        ]
        right = [(2 * cx - x, y) for x, y in left]
        draw.polygon(left, fill=fur)
        draw.polygon(right, fill=fur)
        if notch:
            nx, ny = left[1]
            draw.polygon(
                [(nx - ear // 4, ny + ear // 3), (nx, ny + ear // 10), (nx + ear // 4, ny + ear // 3)],
                fill=PAPER,
            )

        # Muzzle and eyes.
        draw.ellipse(
            [cx - head_r // 2, head_y + head_r // 6, cx + head_r // 2, head_y + head_r],
            fill=(248, 244, 236),
        )
        eye_r = g.eye_r
        for sign in (-1, 1):
            ex = cx + sign * head_r // 2
            ey = head_y - head_r // 8
            draw.ellipse([ex - eye_r, ey - eye_r, ex + eye_r, ey + eye_r], fill=eyes)
            draw.ellipse(
                [ex - eye_r // 3, ey - eye_r // 3, ex + eye_r // 3, ey + eye_r // 3],
                fill=(40, 32, 28),
            )

        # Brass compass on its cord.
        draw.ellipse(
            [cx - body // 5, g.neck_y + int(body * 0.42), cx + body // 5, g.neck_y + int(body * 0.82)],
            fill=(198, 160, 78),
            outline=(120, 92, 40),
        )

        return image.filter(ImageFilter.SMOOTH_MORE)
