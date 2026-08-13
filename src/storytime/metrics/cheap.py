"""Cheap, offline metrics. No API calls, no torch -- numpy and Pillow only.

Be honest about what these measure: they compare *colour and texture* between a
page and the character sheet. That catches the most common and most visible
kind of drift (the scarf turns blue, the fur goes pink, the style shifts from
watercolour to 3D render) and it costs nothing to compute.

They do not measure identity. A page with the right palette and a completely
wrong face scores well here. For identity you need the VLM judge in
`judge.py`, or the embedding metric in `embed.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..imageutil import dominant_colors, load_rgb, subject_mask

HUE_BINS = 24
SAT_BINS = 6
# Below this saturation a pixel is background wash or paper, not the character.
MIN_SATURATION = 55


def _smooth(histogram: np.ndarray) -> np.ndarray:
    """Blur the histogram so a one-bin colour shift does not read as a total mismatch.

    Without this the metric is worthless: nudge a hue by 15 degrees and every
    pixel jumps to the neighbouring bin, so two nearly identical images score
    near zero. Hue wraps around, so it is smoothed circularly.
    """
    kernel = np.array([0.25, 0.5, 0.25])
    for _ in range(2):
        histogram = sum(
            weight * np.roll(histogram, offset, axis=0)
            for offset, weight in zip((-1, 0, 1), kernel)
        )
    padded = np.pad(histogram, ((0, 0), (1, 1)), mode="edge")
    histogram = (
        kernel[0] * padded[:, :-2] + kernel[1] * padded[:, 1:-1] + kernel[2] * padded[:, 2:]
    )
    return histogram


def _hue_sat_histogram(image: Image.Image) -> np.ndarray:
    """Normalised, smoothed hue x saturation histogram over saturated subject pixels."""
    rgb = np.asarray(image, dtype=np.uint8)
    mask = subject_mask(rgb, min_saturation=MIN_SATURATION)
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)

    hue = hsv[..., 0][mask].astype(np.int32) * HUE_BINS // 256
    sat = hsv[..., 1][mask].astype(np.int32) * SAT_BINS // 256
    if hue.size == 0:
        return np.zeros(HUE_BINS * SAT_BINS, dtype=np.float64)

    flat = np.bincount(hue * SAT_BINS + sat, minlength=HUE_BINS * SAT_BINS)
    grid = flat.astype(np.float64).reshape(HUE_BINS, SAT_BINS)
    return _smooth(grid / grid.sum()).ravel()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def _edge_density(image: Image.Image) -> float:
    """Fraction of pixels sitting on a strong intensity gradient.

    A rough stand-in for how detailed and how sharply drawn an image is, which
    is what separates a soft watercolour from a crisp vector illustration.
    """
    grey = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    dy, dx = np.gradient(grey)
    magnitude = np.hypot(dx, dy)
    return float((magnitude > 0.08).mean())


def _signature_coverage(image: Image.Image, palette: list[tuple[int, int, int]]) -> float:
    """Share of subject pixels that are close to one of the sheet's key colours.

    If the fox's rust fur and moss scarf survived onto the page, this is high.
    If the page recoloured everything, it collapses.
    """
    rgb = np.asarray(image, dtype=np.uint8)
    mask = subject_mask(rgb, min_saturation=MIN_SATURATION)
    subject = rgb.astype(np.int16)[mask]
    if subject.size == 0 or not palette:
        return 0.0

    reference = np.asarray(palette, dtype=np.int16)  # (k, 3)
    # Euclidean distance in RGB is crude but adequate at this tolerance.
    distances = np.linalg.norm(subject[:, None, :] - reference[None, :, :], axis=-1)
    return float((distances.min(axis=1) < 60).mean())


def score_page(page_path: Path | str, sheet_path: Path | str) -> dict[str, float]:
    """Compare one page against the character sheet."""
    page = load_rgb(page_path, max_edge=256)
    sheet = load_rgb(sheet_path, max_edge=256)

    palette_cosine = _cosine(_hue_sat_histogram(page), _hue_sat_histogram(sheet))

    page_edges = _edge_density(page)
    sheet_edges = _edge_density(sheet)
    # 1.0 when the two match, falling off symmetrically in either direction.
    edge_match = (
        float(min(page_edges, sheet_edges) / max(page_edges, sheet_edges))
        if max(page_edges, sheet_edges) > 0
        else 0.0
    )

    coverage = _signature_coverage(
        page, dominant_colors(sheet_path, count=4, min_saturation=MIN_SATURATION)
    )

    return {
        "palette_cosine": round(palette_cosine, 4),
        "signature_coverage": round(coverage, 4),
        "edge_match": round(edge_match, 4),
        "edge_density": round(page_edges, 4),
    }


def score_run(records, run_root: Path) -> None:
    """Attach cheap metrics to every record in place."""
    for record in records:
        if record.error:
            continue
        sheet = run_root / record.backend / "sheet.png"
        page = run_root / record.image_path
        if not (sheet.is_file() and page.is_file()):
            continue
        record.metrics.update(score_page(page, sheet))
