"""Small image helpers shared by the backends, the metrics and the report."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

# Pixels this bright are treated as paper/background, not part of the character.
WHITE_CUTOFF = 232
# Pixels this dark carry almost no hue information.
BLACK_CUTOFF = 24


def load_rgb(path: Path | str, max_edge: int | None = 512) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max_edge and max(image.size) > max_edge:
            scale = max_edge / max(image.size)
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        else:
            image = image.copy()
    return image


def subject_mask(rgb: np.ndarray, min_saturation: int = 0) -> np.ndarray:
    """True where a pixel probably belongs to the subject rather than the paper.

    `min_saturation` (0-255) additionally drops washed-out pixels. The character
    sheet sits on plain paper while pages have full backgrounds, so comparing
    the two only works if the muted background is filtered out first -- in a
    picture book the character is almost always the most saturated thing on the
    page.
    """
    mask = (rgb.min(axis=-1) < WHITE_CUTOFF) & (rgb.max(axis=-1) > BLACK_CUTOFF)
    if min_saturation > 0:
        # HSV saturation for uint8 RGB, without a colour-space conversion.
        # int32 is not optional: (255 - 0) * 255 overflows int16 and comes back
        # negative, which silently excludes every strongly coloured pixel --
        # precisely the ones this mask exists to keep.
        high = rgb.max(axis=-1).astype(np.int32)
        low = rgb.min(axis=-1).astype(np.int32)
        saturation = np.where(high > 0, (high - low) * 255 // np.maximum(high, 1), 0)
        mask &= saturation >= min_saturation
    return mask


def dominant_colors(
    path: Path | str, count: int = 5, min_saturation: int = 0
) -> list[tuple[int, int, int]]:
    """The most common non-background colours, most frequent first."""
    image = load_rgb(path, max_edge=96)
    quantised = image.quantize(colors=16, method=Image.Quantize.MEDIANCUT).convert("RGB")
    pixels = np.asarray(quantised, dtype=np.uint8).reshape(-1, 3)
    keep = subject_mask(pixels, min_saturation=min_saturation)

    counts: dict[tuple[int, int, int], int] = {}
    for pixel in map(tuple, pixels[keep]):
        counts[pixel] = counts.get(pixel, 0) + 1  # type: ignore[index]
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [tuple(int(c) for c in colour) for colour, _ in ranked[:count]] or [(128, 128, 128)]  # type: ignore[misc]


def to_data_uri(path: Path | str, max_edge: int = 768) -> tuple[str, str]:
    """Return (base64_png, mime) for sending an image to a judge model.

    Downscaled first -- judges do not need 2K pixels to see that a scarf is the
    wrong colour, and image tokens are the bulk of the cost.
    """
    image = load_rgb(path, max_edge=max_edge)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii"), "image/png"
