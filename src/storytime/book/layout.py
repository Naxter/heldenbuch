"""Putting words on the pictures and making a file a printer will accept.

The text is typeset here, with real fonts, and never asked of the image model
-- image models produce convincing-looking gibberish instead of letters, in
every language.

Two things make the pages stop looking machine-made:

  * The text goes where the *picture* is quiet, measured rather than assumed.
    Asking the image model to keep the lower third calm helps; it does not
    always listen, and when it does not, a white box lands on someone's face.
  * Where that quiet area is also plain and light, the panel is dropped and the
    words sit straight on the illustration. That single difference is most of
    what separates a book from a slide.

Print presets carry the numbers a print shop needs:

  bleed      the picture must run 3.175 mm past the edge on all sides, because
             the trimming blade wanders. Without it you get white slivers.
  safety     nothing important within 12.7 mm of the trim edge, for the same
             reason. All text lives inside this.
  dpi        300 for print. The effective resolution is reported so you know
             when a draft image is being stretched too far.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

MM_PER_INCH = 25.4

# Font families available on a normal Windows install, all with Cyrillic
# coverage so Russian sets correctly.
FONT_FAMILIES: dict[str, dict[str, str]] = {
    "georgia": {"name": "Georgia", "regular": "georgia.ttf", "bold": "georgiab.ttf",
                "italic": "georgiai.ttf"},
    "candara": {"name": "Candara", "regular": "Candara.ttf", "bold": "Candarab.ttf",
                "italic": "Candarai.ttf"},
    "segoe": {"name": "Segoe UI", "regular": "segoeui.ttf", "bold": "segoeuib.ttf",
              "italic": "segoeuii.ttf"},
    "verdana": {"name": "Verdana", "regular": "verdana.ttf", "bold": "verdanab.ttf",
                "italic": "verdanai.ttf"},
    "comic": {"name": "Comic Sans", "regular": "comic.ttf", "bold": "comicbd.ttf",
              "italic": "comici.ttf"},
}

FONT_DIRS = [
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts"),
    Path("/Library/Fonts"),
    Path.home() / "AppData/Local/Microsoft/Windows/Fonts",
]


@dataclass(frozen=True)
class PrintPreset:
    key: str
    name: str
    hint: str
    trim_mm: tuple[float, float]
    bleed_mm: float
    safety_mm: float
    dpi: int
    layout: str = "band"  # "band" = text on the picture, "split" = beside it
    pad_to_multiple: int = 4
    #: thickness of one sheet, for working out the spine. Lulu's perfect-bound
    #: rule is pages/444 inch plus a flat 0.06 inch cover allowance.
    pages_per_inch: float = 444.0
    spine_allowance_in: float = 0.06

    def page_px(self) -> tuple[int, int]:
        width = self.trim_mm[0] + 2 * self.bleed_mm
        height = self.trim_mm[1] + 2 * self.bleed_mm
        return (self._px(width), self._px(height))

    def _px(self, mm: float) -> int:
        return int(round(mm / MM_PER_INCH * self.dpi))

    @property
    def bleed_px(self) -> int:
        return self._px(self.bleed_mm)

    @property
    def safety_px(self) -> int:
        return self._px(self.safety_mm + self.bleed_mm)

    def spine_mm(self, pages: int) -> float:
        return (pages / self.pages_per_inch + self.spine_allowance_in) * MM_PER_INCH


PRESETS: dict[str, PrintPreset] = {
    "print_square": PrintPreset(
        key="print_square",
        name="Druckerei — quadratisch 21,6 cm",
        hint="Standardformat für Kinderbücher (Lulu, Gelato). Mit Beschnitt, 300 dpi.",
        trim_mm=(215.9, 215.9), bleed_mm=3.175, safety_mm=12.7, dpi=300,
    ),
    "print_kinderbuch": PrintPreset(
        key="print_kinderbuch",
        name="Druckerei — Kinderbuch 15,6 × 14,8 cm",
        hint="epubli-Kinderbuchformat, kleiner und günstiger. Mit Beschnitt, 300 dpi.",
        trim_mm=(156.0, 148.0), bleed_mm=3.0, safety_mm=10.0, dpi=300,
    ),
    "home_a4": PrintPreset(
        key="home_a4",
        name="Zuhause drucken — A4 quer",
        hint="Auf jedem normalen Drucker. Bild links, Text rechts, kein Beschnitt.",
        trim_mm=(297.0, 210.0), bleed_mm=0.0, safety_mm=12.0, dpi=200,
        layout="split", pad_to_multiple=1,
    ),
    "screen": PrintPreset(
        key="screen",
        name="Zum Vorlesen am Bildschirm",
        hint="Querformat für Tablet oder Laptop, kleine Datei.",
        trim_mm=(254.0, 190.5), bleed_mm=0.0, safety_mm=8.0, dpi=110,
        pad_to_multiple=1,
    ),
}

PAPER = (252, 250, 245)
INK = (38, 34, 30)
LIGHT_INK = (252, 250, 245)


# --------------------------------------------------------------------------- fonts


def find_font(filename: str) -> Path | None:
    for folder in FONT_DIRS:
        candidate = folder / filename
        if candidate.is_file():
            return candidate
    return None


def available_families() -> list[dict[str, str]]:
    """Only offer families actually installed on this machine."""
    return [
        {"key": key, "name": family["name"]}
        for key, family in FONT_FAMILIES.items()
        if find_font(family["regular"])
    ]


def load_font(family: str, weight: str, size: int) -> ImageFont.FreeTypeFont:
    spec = FONT_FAMILIES.get(family) or FONT_FAMILIES["georgia"]
    path = find_font(spec.get(weight, spec["regular"])) or find_font(spec["regular"])
    if path is None:  # last resort: whatever Pillow ships
        return ImageFont.load_default(size)
    return ImageFont.truetype(str(path), size)


def wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Greedy word wrap. Long words overflow rather than being broken, which
    reads better in a children's book."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words, current = paragraph.split(), ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if font.getlength(candidate) <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return [line for line in lines if line != "" or len(lines) == 1]


#: Smallest body type we are willing to print, in points, by age band. A book
#: is held by an adult reading aloud and followed by a child looking on, so
#: there is a floor below which the page stops working for either. `fit_text`
#: on its own will shrink as far as it is told to, and its old default of 10 px
#: is 2.4 pt at 300 dpi -- text that technically fits and cannot be read.
MIN_BODY_PT = {"2-3": 20.0, "4-5": 16.0, "6-7": 13.0, "8+": 12.0}
DEFAULT_MIN_BODY_PT = 14.0

#: A vignette must keep at least this much of the page height for the picture.
#: Past that it is no longer a small picture on paper, it is a text page with a
#: stamp on it -- and beyond that again the art was dropped entirely.
MIN_VIGNETTE_ART = 0.40


def min_body_px(preset: PrintPreset, age: str = "") -> int:
    """The body-type floor for this page size and reader, in pixels."""
    points = MIN_BODY_PT.get(age, DEFAULT_MIN_BODY_PT)
    return max(8, int(round(points / 72.0 * preset.dpi)))


def fit_body(
    text: str,
    family: str,
    box: tuple[int, int],
    max_size: int,
    min_size: int,
    line_spacing: float = 1.35,
) -> tuple[ImageFont.FreeTypeFont, list[str], int, bool]:
    """Fit body text, and say whether it actually fitted at the floor.

    `fit_text` always returns something; when the text is too long it returns
    it at `min_size` and lets it overflow. Callers that have somewhere else to
    put the words need to know that happened, so this reports it.
    """
    font, lines, step = fit_text(text, family, "regular", box,
                                 max_size=max_size, min_size=min_size,
                                 line_spacing=line_spacing)
    return font, lines, step, len(lines) * step <= box[1]


def fit_text(
    text: str,
    family: str,
    weight: str,
    box: tuple[int, int],
    max_size: int,
    min_size: int = 10,
    line_spacing: float = 1.35,
) -> tuple[ImageFont.FreeTypeFont, list[str], int]:
    """Largest font size at which the text still fits the box."""
    width, height = box
    size = max_size
    while size > min_size:
        font = load_font(family, weight, size)
        lines = wrap(text, font, width)
        step = int(size * line_spacing)
        if len(lines) * step <= height:
            return font, lines, step
        size -= max(1, size // 20)
    font = load_font(family, weight, min_size)
    return font, wrap(text, font, width), int(min_size * line_spacing)


def draw_lines(
    canvas: Image.Image,
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    step: int,
    origin: tuple[int, int],
    width: int,
    colour: tuple[int, int, int] = INK,
    align: str = "left",
    shadow: tuple[int, int, int] | None = None,
) -> int:
    draw = ImageDraw.Draw(canvas)
    x, y = origin
    offset_px = max(2, step // 22)
    for line in lines:
        dx = int((width - font.getlength(line)) / 2) if align == "center" else 0
        if shadow:
            draw.text((x + dx + offset_px, y + offset_px), line, font=font, fill=shadow)
        draw.text((x + dx, y), line, font=font, fill=colour)
        y += step
    return y


# --------------------------------------------------------------------------- placement


@dataclass
class TextSpot:
    """Where the words go on a picture, and how they should be drawn."""

    box: tuple[int, int, int, int]  # left, top, right, bottom
    panel: bool
    ink: tuple[int, int, int]
    shadow: tuple[int, int, int] | None
    align: str = "left"


def _region_stats(grey: np.ndarray, box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Mean detail and mean brightness of a fractional region, both 0..1."""
    height, width = grey.shape
    left = max(0, int(box[0] * width))
    top = max(0, int(box[1] * height))
    right = min(width, int(box[2] * width))
    bottom = min(height, int(box[3] * height))
    if right - left < 2 or bottom - top < 2:
        return 1.0, 0.5

    patch = grey[top:bottom, left:right]
    dy, dx = np.gradient(patch)
    detail = float(np.hypot(dx, dy).mean())
    return detail, float(patch.mean())


def find_text_spot(
    canvas: Image.Image,
    safety: int,
    want_fraction: float = 0.30,
) -> TextSpot:
    """Find the calmest place on the picture for the text.

    Six candidates -- bottom, top, and the left and right halves of each -- are
    scored on how much detail they contain. The quietest wins. If it is also
    plain and bright (or plain and dark) the panel is dropped and the words go
    straight onto the illustration.
    """
    width, height = canvas.size
    small = np.asarray(
        ImageOps.grayscale(canvas.resize((96, 96), Image.Resampling.BILINEAR)),
        dtype=np.float32,
    ) / 255.0

    top_band = (0.0, 1.0 - want_fraction, 1.0, 1.0)
    high_band = (0.0, 0.0, 1.0, want_fraction)
    candidates = {
        "bottom": top_band,
        "top": high_band,
        "bottom-left": (0.0, 1.0 - want_fraction, 0.58, 1.0),
        "bottom-right": (0.42, 1.0 - want_fraction, 1.0, 1.0),
        "top-left": (0.0, 0.0, 0.58, want_fraction),
        "top-right": (0.42, 0.0, 1.0, want_fraction),
    }

    scored = []
    for name, box in candidates.items():
        detail, brightness = _region_stats(small, box)
        # Prefer the bottom slightly: it is where a reader expects the words.
        bias = 0.0 if name.startswith("bottom") else 0.004
        scored.append((detail + bias, detail, brightness, name, box))
    scored.sort()

    _, detail, brightness, name, box = scored[0]

    left = max(safety, int(box[0] * width) + (safety if box[0] > 0 else 0))
    right = min(width - safety, int(box[2] * width) - (safety if box[2] < 1 else 0))
    top = max(safety, int(box[1] * height) + (safety if box[1] > 0 else 0))
    bottom = min(height - safety, int(box[3] * height) - (safety if box[3] < 1 else 0))

    # Quiet enough to skip the box? Thresholds picked so a flat sky or a plain
    # wall qualifies but grass or foliage does not.
    plain = detail < 0.020
    if plain and brightness > 0.70:
        return TextSpot((left, top, right, bottom), False, INK, (255, 255, 255))
    if plain and brightness < 0.32:
        return TextSpot((left, top, right, bottom), False, LIGHT_INK, (0, 0, 0))
    return TextSpot((left, top, right, bottom), True, INK, None)


# --------------------------------------------------------------------------- page art


def cover_image(source: Path, size: tuple[int, int]) -> Image.Image:
    """Scale an illustration to fill a box, cropping the overflow centrally."""
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image.convert("RGB"))
        scale = max(size[0] / image.width, size[1] / image.height)
        resized = image.resize(
            (max(1, math.ceil(image.width * scale)), max(1, math.ceil(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    left = (resized.width - size[0]) // 2
    top = (resized.height - size[1]) // 2
    return resized.crop((left, top, left + size[0], top + size[1]))


def fit_inside(source: Path, size: tuple[int, int]) -> Image.Image:
    """Scale to fit entirely inside a box, keeping the whole picture."""
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image.convert("RGB"))
        scale = min(size[0] / image.width, size[1] / image.height)
        return image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )


def effective_dpi(source: Path, preset: PrintPreset) -> int:
    """What resolution the picture really has once it fills the page."""
    with Image.open(source) as image:
        longest = max(image.size)
    page_mm = max(preset.trim_mm[0] + 2 * preset.bleed_mm,
                  preset.trim_mm[1] + 2 * preset.bleed_mm)
    return int(longest / (page_mm / MM_PER_INCH))


def _panel(canvas: Image.Image, box: tuple[int, int, int, int], radius: int,
           fill=(255, 253, 248, 222)) -> Image.Image:
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rounded_rectangle(box, radius=radius, fill=fill)
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


def render_story_page(
    illustration: Path | None,
    text: str,
    preset: PrintPreset,
    family: str = "georgia",
    layout: str = "full",
    age: str = "",
) -> Image.Image:
    """One page of the book, composed according to its layout."""
    page_w, page_h = preset.page_px()
    safety = preset.safety_px
    has_art = bool(illustration and illustration.is_file())
    floor = min_body_px(preset, age)

    # The A4 home preset always sets picture beside text; it is a different
    # shape of page and the band layout wastes it.
    if preset.layout == "split" or layout == "split":
        canvas = Image.new("RGB", (page_w, page_h), PAPER)

        # Side by side only works on a landscape page. On a square or portrait
        # one it crops the art to a sliver and leaves a text column so narrow
        # that three words fill a line, so stack instead.
        if page_w / page_h > 1.15:
            art_w = int(page_w * 0.56)
            if has_art:
                canvas.paste(cover_image(illustration, (art_w - preset.bleed_px, page_h)), (0, 0))
            text_x = art_w + safety
            box = (page_w - text_x - safety, page_h - 2 * safety)
            if text.strip():
                font, lines, step = fit_text(text, family, "regular", box,
                                             max_size=int(page_h * 0.055))
                start_y = max(safety, (page_h - len(lines) * step) // 2)
                draw_lines(canvas, lines, font, step, (text_x, start_y), box[0])
            return canvas

        art_h = int(page_h * 0.62)
        if has_art:
            canvas.paste(cover_image(illustration, (page_w, art_h)), (0, 0))
        if text.strip():
            box = (page_w - 3 * safety, page_h - art_h - 2 * safety)
            font, lines, step = fit_text(text, family, "regular", box,
                                         max_size=int(page_h * 0.05))
            start_y = art_h + max(safety, (page_h - art_h - len(lines) * step) // 2)
            draw_lines(canvas, lines, font, step, (int(safety * 1.5), start_y), box[0])
        return canvas

    if layout == "vignette":
        # The words are measured first and the picture gets what is left, so
        # the two cannot collide. But the picture also has a floor: the text
        # block may never grow past what MIN_VIGNETTE_ART leaves it. If the
        # words genuinely do not fit in that band at readable size then this is
        # not a quiet beat, and the page is composed as a full one instead --
        # which has the whole page for text rather than a quarter of it.
        band_h = int(page_h * (1.0 - MIN_VIGNETTE_ART)) - 3 * safety
        lines: list[str] = []
        step = 0
        fits = True
        if text.strip():
            font, lines, step, fits = fit_body(
                text, family, (page_w - 3 * safety, band_h),
                max_size=int(page_h * 0.045), min_size=floor,
            )

        if fits:
            canvas = Image.new("RGB", (page_w, page_h), PAPER)
            block_h = len(lines) * step
            # Top margin, one gap under the picture, bottom margin.
            art_h = page_h - block_h - 3 * safety

            if has_art:
                art = fit_inside(illustration, (page_w - 2 * safety, art_h))
                # Centred in the space above the words rather than pinned to
                # the top, so a wide picture does not leave a hole under it.
                top = safety + (art_h - art.height) // 2
                canvas.paste(art, ((page_w - art.width) // 2, top))
            if lines:
                draw_lines(canvas, lines, font, step,
                           (int(safety * 1.5), page_h - safety - block_h),
                           page_w - 3 * safety, align="center")
            return canvas
        layout = "full"  # too many words for a vignette; fall through

    canvas = (
        cover_image(illustration, (page_w, page_h))
        if has_art
        else Image.new("RGB", (page_w, page_h), PAPER)
    )
    if layout == "wordless" or not text.strip():
        return canvas

    spot = (
        find_text_spot(canvas, safety)
        if has_art
        else TextSpot((safety, int(page_h * 0.66), page_w - safety, page_h - safety),
                      False, INK, None)
    )
    left, top, right, bottom = spot.box
    inner_w = right - left
    pad = int(safety * 0.5)

    font, lines, step, fits = fit_body(
        text, family, (inner_w - 2 * pad, (bottom - top) - pad),
        max_size=int(page_h * 0.045), min_size=floor,
    )
    if not fits:
        # The quiet area found in the picture is too small for this much text
        # at a readable size. Take a taller band across the page instead and
        # set it on a panel -- growing the box is always better than shrinking
        # the type, which is what produced 6 pt pages for the oldest band.
        left, right = safety, page_w - safety
        top, bottom = int(page_h * 0.42), page_h - safety
        inner_w = right - left
        spot = TextSpot((left, top, right, bottom), True, INK, None)
        font, lines, step, _ = fit_body(
            text, family, (inner_w - 2 * pad, (bottom - top) - pad),
            max_size=int(page_h * 0.045), min_size=floor,
        )
    block_h = len(lines) * step

    if spot.panel:
        panel_top = bottom - block_h - int(pad * 1.6)
        canvas = _panel(canvas, (left, panel_top, right, bottom), radius=int(safety * 0.6))
        text_top = panel_top + int(pad * 0.8)
    else:
        text_top = bottom - block_h

    draw_lines(canvas, lines, font, step, (left + pad, text_top), inner_w - 2 * pad,
               colour=spot.ink, shadow=spot.shadow)
    return canvas


def render_title_page(
    cover: Path | None,
    title: str,
    subtitle: str,
    preset: PrintPreset,
    family: str = "georgia",
) -> Image.Image:
    page_w, page_h = preset.page_px()
    safety = preset.safety_px
    canvas = (
        cover_image(cover, (page_w, page_h))
        if cover and cover.is_file()
        else Image.new("RGB", (page_w, page_h), PAPER)
    )

    inner_w = page_w - 2 * safety - int(safety)
    # Keep the title band inside the top quarter. The cover art is generated
    # with calm space up there; a taller panel starts covering the character.
    font, lines, step = fit_text(
        title, family, "bold", (inner_w, int(page_h * 0.17)),
        max_size=int(page_h * 0.075), line_spacing=1.18,
    )
    block_h = len(lines) * step + (int(page_h * 0.035) if subtitle else 0)
    panel_top = safety
    panel_bottom = panel_top + block_h + int(safety * 1.1)

    canvas = _panel(canvas, (safety, panel_top, page_w - safety, panel_bottom),
                    radius=int(safety * 0.6), fill=(255, 253, 248, 228))

    y = draw_lines(canvas, lines, font, step,
                   (safety + int(safety * 0.5), panel_top + int(safety * 0.55)),
                   inner_w, align="center")
    if subtitle:
        small = load_font(family, "italic", int(page_h * 0.030))
        draw = ImageDraw.Draw(canvas)
        offset = int((inner_w - small.getlength(subtitle)) / 2)
        draw.text((safety + int(safety * 0.5) + offset, y + int(page_h * 0.008)),
                  subtitle, font=small, fill=(92, 84, 74))
    return canvas


def render_plain_page(
    text: str, preset: PrintPreset, family: str = "georgia", italic: bool = True
) -> Image.Image:
    page_w, page_h = preset.page_px()
    safety = preset.safety_px
    canvas = Image.new("RGB", (page_w, page_h), PAPER)
    if not text.strip():
        return canvas
    box = (page_w - 2 * safety - int(safety), int(page_h * 0.4))
    font, lines, step = fit_text(
        text, family, "italic" if italic else "regular", box, max_size=int(page_h * 0.042)
    )
    start_y = (page_h - len(lines) * step) // 2
    draw_lines(canvas, lines, font, step, (safety + int(safety * 0.5), start_y), box[0],
               colour=(92, 84, 74), align="center")
    return canvas


def render_photo_page(
    photo: Path, caption: str, preset: PrintPreset, family: str = "georgia"
) -> Image.Image:
    """The last page: a real photograph, framed on paper, with a line under it."""
    page_w, page_h = preset.page_px()
    safety = preset.safety_px
    canvas = Image.new("RGB", (page_w, page_h), PAPER)

    frame = int(safety * 0.35)
    art = fit_inside(photo, (page_w - 4 * safety, int(page_h * 0.62)))
    x = (page_w - art.width) // 2
    y = int(page_h * 0.10)
    ImageDraw.Draw(canvas).rectangle(
        [x - frame, y - frame, x + art.width + frame, y + art.height + frame],
        fill=(255, 255, 255),
    )
    canvas.paste(art, (x, y))

    if caption.strip():
        box = (page_w - 4 * safety, int(page_h * 0.18))
        font, lines, step = fit_text(caption, family, "italic", box,
                                     max_size=int(page_h * 0.038))
        draw_lines(canvas, lines, font, step,
                   (2 * safety, y + art.height + frame + safety), box[0],
                   colour=(92, 84, 74), align="center")
    return canvas


# --------------------------------------------------------------------------- cover


def render_wrap_cover(
    book,
    language: str,
    preset: PrintPreset,
    resolve,
    interior_pages: int,
    family: str = "georgia",
) -> tuple[Image.Image, dict[str, Any]]:
    """Front, spine and back as one sheet, which is what a print shop wants.

    Spine width comes from the page count: `pages / 444 inch + 0.06 inch` is
    Lulu's perfect-bound rule. Get it wrong and the front artwork wraps around
    onto the spine.
    """
    trim_w, trim_h = preset.trim_mm
    spine_mm = preset.spine_mm(interior_pages)
    bleed = preset.bleed_mm

    total_w_mm = 2 * trim_w + spine_mm + 2 * bleed
    total_h_mm = trim_h + 2 * bleed
    to_px = lambda mm: int(round(mm / MM_PER_INCH * preset.dpi))  # noqa: E731
    page_w, page_h = to_px(total_w_mm), to_px(total_h_mm)
    spine_px = to_px(spine_mm)
    panel_w = to_px(trim_w + bleed)
    safety = preset.safety_px

    canvas = Image.new("RGB", (page_w, page_h), PAPER)

    cover_art = None
    if book.cover:
        try:
            candidate = resolve(book.cover)
            cover_art = candidate if candidate.is_file() else None
        except (ValueError, FileNotFoundError):
            cover_art = None

    # Front is on the right; the back is on the left, as the sheet wraps.
    front_x = panel_w + spine_px
    if cover_art:
        canvas.paste(cover_image(cover_art, (page_w - front_x, page_h)), (front_x, 0))
        # A soft, heavily blurred wash of the same art behind the back cover
        # keeps the two sides related without competing with the blurb.
        from PIL import ImageFilter

        back = cover_image(cover_art, (panel_w, page_h)).filter(ImageFilter.GaussianBlur(
            radius=max(8, page_h // 90)))
        canvas.paste(Image.blend(back, Image.new("RGB", back.size, PAPER), 0.62), (0, 0))

    # Title on the front.
    title = book.title.get(language) or book.display_title()
    inner_w = (page_w - front_x) - 2 * safety
    font, lines, step = fit_text(title, family, "bold", (inner_w, int(page_h * 0.17)),
                                 max_size=int(page_h * 0.075), line_spacing=1.18)
    panel_bottom = safety + len(lines) * step + int(safety * 1.1)
    canvas = _panel(canvas, (front_x + safety, safety, page_w - safety, panel_bottom),
                    radius=int(safety * 0.6), fill=(255, 253, 248, 228))
    draw_lines(canvas, lines, font, step,
               (front_x + safety + int(safety * 0.5), safety + int(safety * 0.55)),
               inner_w - int(safety), align="center")

    # Blurb on the back.
    blurb = book.dedication.get(language, "") or title
    box = (panel_w - 3 * safety, int(page_h * 0.30))
    font, lines, step = fit_text(blurb, family, "italic", box, max_size=int(page_h * 0.034))
    draw_lines(canvas, lines, font, step,
               (int(safety * 1.5), int(page_h * 0.34)), box[0],
               colour=(70, 62, 54), align="center")

    # Spine, but only when it is wide enough to hold readable type.
    spine_note = None
    if spine_mm >= 6.0:
        spine = Image.new("RGB", (page_h - 2 * safety, spine_px), PAPER)
        sfont, slines, sstep = fit_text(title, family, "bold",
                                        (spine.width, spine_px - int(spine_px * 0.3)),
                                        max_size=max(12, int(spine_px * 0.45)))
        draw_lines(spine, slines[:1], sfont, sstep,
                   (0, (spine_px - sstep) // 2), spine.width, align="center")
        canvas.paste(spine.rotate(90, expand=True), (panel_w, safety))
    else:
        spine_note = (
            f"Der Rücken ist nur {spine_mm:.1f} mm breit — dafür ist kein Text "
            "aufgedruckt. Unter etwa 6 mm drucken die meisten Druckereien nichts."
        )

    info = {
        "spine_mm": round(spine_mm, 2),
        "interior_pages": interior_pages,
        "size_mm": (round(total_w_mm, 1), round(total_h_mm, 1)),
        "size_px": (page_w, page_h),
        "note": spine_note,
    }
    return canvas, info


# --------------------------------------------------------------------------- export


def join_languages(mapping: dict, langs: list[str], separator: str = "\n") -> str:
    """One string carrying every requested language, in order.

    Duplicates are dropped ("Fin" is the closing word in Spanish *and*
    French), so a bilingual page never says the same thing twice.
    """
    parts = [str((mapping or {}).get(code, "")).strip() for code in langs]
    seen: list[str] = []
    for part in parts:
        if part and part not in seen:
            seen.append(part)
    return separator.join(seen)


def render_preview(
    book,
    index: int,
    preset: PrintPreset,
    resolve,
    languages: list[str],
    family: str = "georgia",
    dpi: int = 110,
    guides: bool = True,
) -> Image.Image:
    """One page, typeset by the same code the export runs -- at screen size.

    Until now the exported PDF was the first time anyone saw a *set* page;
    this is the "so wird die Seite gesetzt" view. Everything in the layout is
    proportional to the preset's physical size, so rendering the same
    millimetres at a lower dpi is faithful, just lighter. `index` 0 is the
    title page; page numbers are the story pages.

    For print presets, `guides` draws where the blade will cut (red) and the
    safety boundary text must stay inside (blue) -- the two lines a print
    shop assumes you know about.
    """
    from dataclasses import replace

    small = replace(preset, dpi=dpi) if preset.dpi > dpi else preset
    langs = [c for c in languages if c in book.languages] or [book.primary_language]

    def art(relative):
        if not relative:
            return None
        try:
            path = resolve(relative)
        except (ValueError, FileNotFoundError):
            return None
        return path if path.is_file() else None

    if index == 0:
        image = render_title_page(
            art(book.cover),
            join_languages(book.title, langs, " · ") or book.display_title(),
            "", small, family)
    else:
        page = next((p for p in book.pages if p.index == index), None)
        if page is None:
            raise ValueError(f"Seite {index} gibt es nicht.")
        image = render_story_page(
            art(page.image), join_languages(page.text, langs), small,
            family, layout=page.layout, age=book.age)

    if guides and small.bleed_px > 0:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        trim = small.bleed_px
        safety = small.safety_px
        draw.rectangle([trim, trim, image.width - trim - 1, image.height - trim - 1],
                       outline=(200, 60, 40, 200), width=2)
        draw.rectangle([safety, safety, image.width - safety - 1, image.height - safety - 1],
                       outline=(60, 120, 200, 140), width=1)
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    return image


def export_pdf(
    book,
    language: str,
    preset: PrintPreset,
    resolve,
    target: Path,
    family: str = "georgia",
    include_cover: bool = True,
    log=print,
    secondary: list[str] | None = None,
) -> dict:
    """Build the whole book as one PDF. `resolve` turns a stored relative path
    into an absolute one.

    `secondary` lists further languages to set on the *same* page, under the
    primary text -- a bilingual book for a child growing up with two languages,
    rather than one file per language.
    """
    from .models import LANGUAGES

    langs = [language] + [code for code in (secondary or [])
                          if code != language and code in book.languages]

    def joined(mapping: dict, separator: str = "\n") -> str:
        return join_languages(mapping, langs, separator)

    pages: list[Image.Image] = []
    warnings: list[str] = []

    def art(relative: str | None) -> Path | None:
        if not relative:
            return None
        try:
            path = resolve(relative)
        except (ValueError, FileNotFoundError):
            return None
        return path if path.is_file() else None

    cover_art = art(book.cover)
    sample = cover_art or next((art(p.image) for p in book.pages if art(p.image)), None)
    if sample and preset.dpi >= 300:
        dpi = effective_dpi(sample, preset)
        if dpi < 200:
            warnings.append(
                f"Die Bilder haben in dieser Größe nur etwa {dpi} dpi, eine Druckerei "
                "möchte 300. Vor dem Bestellen das Buch noch einmal in Druckqualität "
                "zeichnen lassen."
            )

    if include_cover:
        log("  Titelseite")
        pages.append(
            render_title_page(cover_art, joined(book.title, " · ") or book.display_title(),
                              "", preset, family)
        )

    dedication = joined(book.dedication)
    if dedication:
        pages.append(render_plain_page(dedication, preset, family))

    for page in sorted(book.pages, key=lambda p: p.index):
        log(f"  Seite {page.index}")
        pages.append(
            render_story_page(art(page.image), joined(page.text), preset,
                              family, layout=page.layout, age=book.age)
        )

    photo = (book.photo_page or {}).get("image")
    if photo and art(photo):
        log("  Fotoseite")
        caption = (book.photo_page or {}).get("caption", {})
        caption = joined(caption) if isinstance(caption, dict) else str(caption)
        pages.append(render_photo_page(art(photo), caption, preset, family))

    closers = {"de": "Ende", "en": "The End", "ru": "Конец", "fr": "Fin", "es": "Fin",
               "it": "Fine", "tr": "Son", "pl": "Koniec", "nl": "Einde"}
    closing = joined({code: closers.get(code, "Ende") for code in langs}, " · ")
    pages.append(render_plain_page(closing, preset, family))

    interior = len(pages)
    if preset.pad_to_multiple > 1:
        blank = Image.new("RGB", preset.page_px(), PAPER)
        while len(pages) % preset.pad_to_multiple:
            pages.append(blank.copy())

    target.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(target, "PDF", save_all=True, append_images=pages[1:],
                  resolution=float(preset.dpi),
                  title=book.title.get(language) or book.display_title())

    return {
        "path": target,
        "pages": len(pages),
        "content_pages": interior,
        "preset": preset.key,
        "language": "+".join(langs),
        "language_name": " + ".join(
            LANGUAGES.get(code, {}).get("name", code) for code in langs),
        "warnings": warnings,
    }


def export_cover_image(book, language: str, preset: PrintPreset, resolve, target: Path,
                       family: str = "georgia") -> Path:
    """Front cover on its own, for shops that want it separately."""
    cover_art = None
    if book.cover:
        try:
            candidate = resolve(book.cover)
            cover_art = candidate if candidate.is_file() else None
        except (ValueError, FileNotFoundError):
            cover_art = None
    image = render_title_page(cover_art, book.title.get(language) or book.display_title(),
                              "", preset, family)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, "JPEG", quality=94, dpi=(preset.dpi, preset.dpi))
    return target
