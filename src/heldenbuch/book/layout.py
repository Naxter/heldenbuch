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

from .models import closing_word, single_scene

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


# ----------------------------------------------------------------- binding

#: Saddle stitch -- folded down the middle and stapled -- is what a 12 to 24
#: page picture book actually gets. It has no spine at all: the fold is the
#: spine. Perfect binding (a glued flat spine) needs enough paper to glue,
#: and no print-on-demand shop offers it below this.
PERFECT_BOUND_MIN_PAGES = 32

#: Above this a saddle-stitched book will not close flat.
SADDLE_STITCH_MAX_PAGES = 48

#: Lulu prints spine text only from here up. Below it the spine is too narrow
#: for the fold to land reliably, and the text creeps onto the front cover.
#: The old rule started printing at a 6 mm spine, which is 79 pages -- one
#: page inside the range the shop refuses.
SPINE_TEXT_MIN_PAGES = 130


def binding_for(pages: int) -> str:
    """Which binding a book of this many pages is actually made with."""
    if pages < PERFECT_BOUND_MIN_PAGES:
        return "saddle_stitch"
    return "perfect_bound"


def has_spine(pages: int) -> bool:
    """A saddle-stitched book is folded, not glued: there is no flat spine to
    print on, and a cover built with one is wider than the press expects."""
    return binding_for(pages) == "perfect_bound"


def spine_text_allowed(pages: int) -> bool:
    return has_spine(pages) and pages >= SPINE_TEXT_MIN_PAGES


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
    if len(lines) == 1:
        return lines
    # Blank lines between paragraphs are kept -- on a multilingual page they
    # are the only thing separating one language from the next -- but leading
    # and trailing ones are just stray whitespace.
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return lines or [""]


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
    #: which of the six candidate zones this is, so a whole book can be held
    #: to one of them instead of moving the words on every turn
    zone: str = "bottom"
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


def readable_on(canvas: Image.Image, box: tuple[int, int, int, int],
                ink: tuple[int, int, int]) -> bool:
    """Would text in `ink` actually be readable over this patch of picture?

    Measured at full resolution and at the fifth percentile, not on a 96 px
    thumbnail and not on the mean. A thumbnail averages away exactly the
    texture that breaks legibility -- dappled grass reads as a calm pale patch
    at 96 px -- and a mean hides the darkest pixels a letter has to sit on.
    """
    left, top, right, bottom = box
    if right - left < 4 or bottom - top < 4:
        return False
    patch = np.asarray(
        ImageOps.grayscale(canvas.crop((left, top, right, bottom))), dtype=np.float32
    ) / 255.0
    # WCAG relative luminance, close enough on greyscale for a go/no-go.
    ink_lum = (0.2126 * ink[0] + 0.7152 * ink[1] + 0.0722 * ink[2]) / 255.0
    worst = float(np.percentile(patch, 5 if ink_lum < 0.5 else 95))
    lighter, darker = max(worst, ink_lum), min(worst, ink_lum)
    return (lighter + 0.05) / (darker + 0.05) >= 4.5


def find_text_spot(
    canvas: Image.Image,
    safety: int,
    want_fraction: float = 0.30,
    prefer: str | None = None,
) -> TextSpot:
    """Find the calmest place on the picture for the text.

    Six candidates -- bottom, top, and the left and right halves of each -- are
    scored on how much detail they contain. The quietest wins. If it is also
    plain and bright (or plain and dark) the panel is dropped and the words go
    straight onto the illustration.

    `prefer` pins the choice to one zone so the words do not move to a
    different corner on every turn. A reader's eye should land on the text
    without hunting for it, and the search used to be run per page.
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

    if prefer in candidates:
        name, box = prefer, candidates[prefer]
        detail, brightness = _region_stats(small, box)
    else:
        scored = []
        for candidate, region in candidates.items():
            detail, brightness = _region_stats(small, region)
            # Prefer the bottom slightly: it is where a reader expects words.
            bias = 0.0 if candidate.startswith("bottom") else 0.004
            scored.append((detail + bias, detail, brightness, candidate, region))
        scored.sort()
        _, detail, brightness, name, box = scored[0]

    left = max(safety, int(box[0] * width) + (safety if box[0] > 0 else 0))
    right = min(width - safety, int(box[2] * width) - (safety if box[2] < 1 else 0))
    top = max(safety, int(box[1] * height) + (safety if box[1] > 0 else 0))
    bottom = min(height - safety, int(box[3] * height) - (safety if box[3] < 1 else 0))
    where = (left, top, right, bottom)

    # Quiet enough to skip the panel? The detail figure decides whether it is
    # worth asking, and then the contrast is measured for real: a light patch
    # with fine dark texture -- grass, pebbles, dappled light, which is most of
    # what these illustrations contain -- averages bright and reads terribly.
    plain = detail < 0.020
    if plain and brightness > 0.70 and readable_on(canvas, where, INK):
        return TextSpot(where, False, INK, (255, 255, 255), name)
    if plain and brightness < 0.32 and readable_on(canvas, where, LIGHT_INK):
        return TextSpot(where, False, LIGHT_INK, (0, 0, 0), name)
    return TextSpot(where, True, INK, None, name)


# --------------------------------------------------------------------------- page art


def flat_border(source: Path, tol: float = 2.0, min_share: float = 0.02
                ) -> tuple[int, int, int, int]:
    """How much dead flat colour sits around the edges of this picture.

    Returns (left, top, right, bottom) in pixels. Image models sometimes paint
    their own matte -- one real page arrived with 149 rows of flat cream across
    the bottom sixth, which then printed as a white sliver at the trim edge,
    the exact failure bleed exists to prevent. Trimming it costs nothing and
    the picture is scaled to fill the page afterwards anyway.
    """
    try:
        with Image.open(source) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float64)
    except Exception:
        return (0, 0, 0, 0)
    height, width = array.shape[:2]

    def run(lines) -> int:
        count = 0
        for line in lines:
            if float(line.std(axis=0).mean()) > tol:
                break
            count += 1
        return count

    top = run(array[i] for i in range(height))
    bottom = run(array[height - 1 - i] for i in range(height))
    left = run(array[:, i] for i in range(width))
    right = run(array[:, width - 1 - i] for i in range(width))

    # A picture that is flat all the way through is not a border, and a couple
    # of stray rows are not worth cropping.
    if top + bottom >= height or left + right >= width:
        return (0, 0, 0, 0)
    return (
        left if left > width * min_share else 0,
        top if top > height * min_share else 0,
        right if right > width * min_share else 0,
        bottom if bottom > height * min_share else 0,
    )


def cover_image(source: Path, size: tuple[int, int]) -> Image.Image:
    """Scale an illustration to fill a box, cropping the overflow centrally."""
    inset = flat_border(source)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image.convert("RGB"))
        if any(inset):
            # Drop a matte the image model painted for itself before scaling,
            # or it survives to the page and prints at the trim edge.
            image = image.crop((inset[0], inset[1],
                                image.width - inset[2], image.height - inset[3]))
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


def book_body_size(texts: list[str], preset: PrintPreset, family: str,
                   age: str = "") -> int:
    """One body-type size for the whole book, set by its wordiest page.

    Fitting each page on its own made the size swing two and a half times
    across sixteen pages -- 27 pt on a short page, 11 pt on the goodnight one
    -- so the reader's eye has to re-accommodate on every turn. Books do not
    do that: one size, chosen so the longest page still fits.
    """
    page_w, page_h = preset.page_px()
    safety = preset.safety_px
    box = (int((page_w - 2 * safety) * 0.58) - safety, int(page_h * 0.30) - safety)
    ceiling = int(page_h * 0.045)
    smallest = ceiling
    for text in texts:
        if not (text or "").strip():
            continue
        font, _lines, _step = fit_text(text, family, "regular", box,
                                       max_size=ceiling,
                                       min_size=min_body_px(preset, age))
        smallest = min(smallest, font.size)
    return max(min_body_px(preset, age), smallest)


def text_fits(text: str, preset: PrintPreset, age: str = "",
              family: str = "georgia") -> bool:
    """Can this page's words be set at a readable size at all?

    Measured against the widest band the renderer will fall back to, so a
    False here means no layout can hold them -- not that the first choice was
    a poor one.
    """
    if not (text or "").strip():
        return True
    page_w, page_h = preset.page_px()
    safety = preset.safety_px
    box = (page_w - 3 * safety, int(page_h * 0.58) - safety)
    _font, _lines, _step, fits = fit_body(
        text, family, box, max_size=int(page_h * 0.045),
        min_size=min_body_px(preset, age),
    )
    return fits


def book_text_zone(images: list[Path | None], preset: PrintPreset) -> str:
    """The one zone the words sit in, across every page of this book.

    Each page votes with the quietest zone in its own illustration and the
    majority wins, so the text lands in the same place on every turn while
    still following what the pictures actually look like. Ties go to the
    bottom, where a reader expects the words.
    """
    votes: dict[str, int] = {}
    page_size = preset.page_px()
    for image in images:
        if image is None or not image.is_file():
            continue
        try:
            canvas = cover_image(image, page_size)
        except Exception:
            continue
        spot = find_text_spot(canvas, preset.safety_px)
        votes[spot.zone] = votes.get(spot.zone, 0) + 1
    if not votes:
        return "bottom"
    best = max(votes.values())
    winners = [zone for zone, count in votes.items() if count == best]
    return "bottom" if "bottom" in winners else sorted(winners)[0]


def render_story_page(
    illustration: Path | None,
    text: str,
    preset: PrintPreset,
    family: str = "georgia",
    layout: str = "full",
    age: str = "",
    body_px: int | None = None,
    zone: str | None = None,
) -> Image.Image:
    """One page of the book, composed according to its layout.

    `body_px` and `zone` come from the book rather than the page, so the type
    size and the position of the words stay put from one spread to the next.
    """
    page_w, page_h = preset.page_px()
    safety = preset.safety_px
    has_art = bool(illustration and illustration.is_file())
    floor = min_body_px(preset, age)
    ceiling = body_px or int(page_h * 0.045)

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
                                         max_size=ceiling, min_size=floor)
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
                max_size=ceiling, min_size=floor,
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
        find_text_spot(canvas, safety, prefer=zone)
        if has_art
        else TextSpot((safety, int(page_h * 0.66), page_w - safety, page_h - safety),
                      False, INK, None)
    )
    left, top, right, bottom = spot.box
    inner_w = right - left
    pad = int(safety * 0.5)

    font, lines, step, fits = fit_body(
        text, family, (inner_w - 2 * pad, (bottom - top) - pad),
        max_size=ceiling, min_size=floor,
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
            max_size=ceiling, min_size=floor,
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
    # A saddle-stitched book is folded, not glued: there is no spine panel at
    # all, and building the sheet with one made it wider than the trim the
    # shop cuts to. Every book this app makes at 12-24 pages is in that band.
    spined = has_spine(interior_pages)
    spine_mm = preset.spine_mm(interior_pages) if spined else 0.0
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

    # Blurb on the back. Never the dedication: that is written to one child and
    # printed on the outside of a book that gets handed around.
    blurb = (book.blurb or {}).get(language, "") or title
    box = (panel_w - 3 * safety, int(page_h * 0.30))
    font, lines, step = fit_text(blurb, family, "italic", box, max_size=int(page_h * 0.034))
    draw_lines(canvas, lines, font, step,
               (int(safety * 1.5), int(page_h * 0.34)), box[0],
               colour=(70, 62, 54), align="center")

    # Spine text, only where a shop will actually print it. Printing it on a
    # narrow spine pasted a cream strip over the artwork and put the title
    # somewhere the fold does not reliably land.
    spine_note = None
    if not spined:
        spine_note = (
            f"{interior_pages} Seiten werden geheftet, nicht geklebt — dieses "
            "Buch hat keinen Rücken. Der Umschlag ist deshalb ohne Rückenbreite "
            "angelegt: Rückseite, Falz, Vorderseite."
        )
    elif spine_text_allowed(interior_pages):
        spine = Image.new("RGB", (page_h - 2 * safety, spine_px), PAPER)
        sfont, slines, sstep = fit_text(title, family, "bold",
                                        (spine.width, spine_px - int(spine_px * 0.3)),
                                        max_size=max(12, int(spine_px * 0.45)))
        draw_lines(spine, slines[:1], sfont, sstep,
                   (0, (spine_px - sstep) // 2), spine.width, align="center")
        canvas.paste(spine.rotate(90, expand=True), (panel_w, safety))
    else:
        spine_note = (
            f"Der Rücken ist {spine_mm:.1f} mm breit ({interior_pages} Seiten) — "
            f"darauf ist kein Text gedruckt. Druckereien setzen Rückentext erst "
            f"ab etwa {SPINE_TEXT_MIN_PAGES} Seiten, darunter wandert er beim "
            "Falzen auf die Vorderseite."
        )

    info = {
        "spine_mm": round(spine_mm, 2),
        "binding": binding_for(interior_pages),
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

    When there is more than one language the blocks are separated by a blank
    line. They used to run together as consecutive lines in identical type,
    with nothing to show where German ended and Russian began -- readable only
    by recognising the script, which fails entirely between German and English.
    """
    parts = [str((mapping or {}).get(code, "")).strip() for code in langs]
    seen: list[str] = []
    for part in parts:
        if part and part not in seen:
            seen.append(part)
    if len(seen) > 1 and separator == "\n":
        return "\n\n".join(seen)
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

    # What each page says, and what its picture shows, collected as the pages
    # are composed. The words are typeset into the artwork, so without this
    # the finished PDF is a stack of photographs with no text in it at all --
    # nothing to select, search, or read out.
    texts: list[str] = []
    alts: list[str] = []

    if include_cover:
        log("  Titelseite")
        pages.append(
            render_title_page(cover_art, joined(book.title, " · ") or book.display_title(),
                              "", preset, family)
        )
        texts.append(joined(book.title, " · ") or book.display_title())
        alts.append(book.cover_illustration or "")

    dedication = joined(book.dedication)
    if dedication:
        pages.append(render_plain_page(dedication, preset, family))
        texts.append(dedication)
        alts.append("")

    ordered = sorted(book.pages, key=lambda p: p.index)
    # One type size and one text position for the whole book, decided before
    # the first page is composed. Fitted per page, both used to change on
    # every turn.
    body_px = book_body_size([joined(p.text) for p in ordered], preset, family, book.age)
    zone = book_text_zone(
        [art(p.image) for p in ordered if p.layout not in ("wordless", "vignette")],
        preset,
    )
    for page in ordered:
        log(f"  Seite {page.index}")
        pages.append(
            render_story_page(art(page.image), joined(page.text), preset,
                              family, layout=page.layout, age=book.age,
                              body_px=body_px, zone=zone)
        )
        texts.append(joined(page.text))
        # The illustration brief describes the picture in a sentence. It has
        # been sitting in book.json all along as the alt text nobody used.
        alts.append(single_scene(page.illustration or ""))

    photo = (book.photo_page or {}).get("image")
    if photo and art(photo):
        log("  Fotoseite")
        caption = (book.photo_page or {}).get("caption", {})
        caption = joined(caption) if isinstance(caption, dict) else str(caption)
        pages.append(render_photo_page(art(photo), caption, preset, family))
        texts.append(caption)
        alts.append("")

    closing = joined({code: closing_word(code) for code in langs}, " · ")
    pages.append(render_plain_page(closing, preset, family))
    texts.append(closing)
    alts.append("")

    interior = len(pages)
    if preset.pad_to_multiple > 1:
        # Bare paper, not the cream the designed pages sit on. PAPER converts
        # to a four-colour tint at about 9% coverage, so a page the reader
        # thinks is empty would print as a screened wash -- billed as colour,
        # and prone to mottling, which is exactly what light tints do on press.
        blank = Image.new("RGB", preset.page_px(), (255, 255, 255))
        while len(pages) % preset.pad_to_multiple:
            pages.append(blank.copy())

    target.parent.mkdir(parents=True, exist_ok=True)
    # Pillow's PDF writer JPEGs every page at its default quality of 75 with
    # chroma subsampling, which puts ringing around typeset letterforms at
    # 300 dpi. Neither is visible on screen and both are visible on paper.
    pages[0].save(target, "PDF", save_all=True, append_images=pages[1:],
                  resolution=float(preset.dpi), quality=95, subsampling=0,
                  title=book.title.get(language) or book.display_title())

    if embed_srgb(target, language=langs[0] if langs else ""):
        log("  sRGB-Profil eingebettet")
        tagged = tag_pdf(target, texts, alts)
        if tagged:
            log(f"  Textebene gesetzt ({tagged} von {len(texts)} Seiten)")
        if tagged < len([t for t in texts if t.strip()]):
            warnings.append(
                "Auf einigen Seiten steht Text, den die eingebaute PDF-Schrift "
                "nicht abbilden kann (etwa Kyrillisch) — diese Seiten haben "
                "keine auswählbare Textebene. Das Bild ist davon unberührt."
            )
    elif preset.bleed_mm > 0:
        # Only print presets care; a screen PDF without a profile is fine.
        warnings.append('Kein Farbprofil im PDF — für farbverbindlichen Druck '
                        '`pip install "heldenbuch[print]"` und neu exportieren.')

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


def embed_srgb(pdf_path: Path, language: str = "") -> bool:
    """Stamp an sRGB OutputIntent and the document language onto a PDF.

    Pillow writes untagged DeviceRGB, which a press is free to interpret --
    the one colour-management gap in the export. Rewriting the PDF needs the
    optional `pikepdf` dependency (`pip install "heldenbuch[print]"`); without
    it the export stays exactly as it was and the caller may warn.
    """
    try:
        import pikepdf
    except ImportError:
        return False
    from PIL import ImageCms

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        profile = pdf.make_stream(icc)
        profile.N = 3
        intent = pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name.OutputIntent,
            S=pikepdf.Name.GTS_PDFA1,
            OutputConditionIdentifier=pikepdf.String("sRGB IEC61966-2.1"),
            Info=pikepdf.String("sRGB IEC61966-2.1"),
            DestOutputProfile=profile,
        ))
        pdf.Root.OutputIntents = pikepdf.Array([intent])
        if language:
            pdf.Root.Lang = pikepdf.String(language)
        pdf.save(pdf_path)
    return True


def _pdf_string(text: str) -> bytes | None:
    """Text as a PDF literal string, or None if the built-in font cannot say it.

    The text layer uses Helvetica, which every reader already has, so nothing
    has to be embedded. Its encoding covers Latin script; Cyrillic and Greek
    would need a real embedded font, and writing them as mangled Latin-1 would
    put nonsense where a screen reader looks. Better to leave those pages
    without a text layer and say so.
    """
    try:
        raw = text.encode("cp1252")
    except UnicodeEncodeError:
        return None
    out = bytearray(b"(")
    for byte in raw:
        if byte in b"()\\":
            out += b"\\" + bytes([byte])
        elif byte < 32:
            out += b" "
        else:
            out.append(byte)
    out += b")"
    return bytes(out)


def tag_pdf(pdf_path: Path, texts: list[str], alts: list[str] | None = None) -> int:
    """Give a picture-book PDF a real text layer and a structure tree.

    The words are typeset into the artwork, so the exported file was a stack
    of pictures: nothing to select, nothing to search, and nothing for a
    screen reader to say. This lays each page's own words over it as
    invisible text, marks the artwork as a figure with the illustration brief
    as its alternate description, and links both into a structure tree so the
    reading order is stated rather than guessed.

    Returns how many pages got a text layer. Needs the optional `print`
    extra; without it the PDF is left exactly as it was.
    """
    try:
        import pikepdf
    except ImportError:
        return 0

    alts = alts or []
    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        font = pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1,
            BaseFont=pikepdf.Name.Helvetica, Encoding=pikepdf.Name.WinAnsiEncoding,
        ))
        struct_root = pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name.StructTreeRoot))
        document = pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name.StructElem, S=pikepdf.Name.Document, P=struct_root))

        kids, parent_pairs, written = [], [], 0
        for index, page in enumerate(pdf.pages):
            text = (texts[index] if index < len(texts) else "").strip()
            alt = (alts[index] if index < len(alts) else "").strip()
            encoded = _pdf_string(text) if text else None

            elements = []
            # The whole existing content is the artwork, so it is wrapped as
            # one figure rather than picked apart.
            figure_mcid = 0
            page.contents_add(pikepdf.Stream(
                pdf, b"/Figure <</MCID 0>> BDC\n"), prepend=True)
            page.contents_add(pikepdf.Stream(pdf, b"\nEMC\n"))
            figure = pdf.make_indirect(pikepdf.Dictionary(
                Type=pikepdf.Name.StructElem, S=pikepdf.Name.Figure,
                P=document, Pg=page.obj, K=figure_mcid,
                Alt=pikepdf.String(alt or text or " "),
            ))
            elements.append(figure)

            if encoded:
                height = float(page.mediabox[3]) - float(page.mediabox[1])
                width = float(page.mediabox[2]) - float(page.mediabox[0])
                size = max(9.0, min(16.0, height / 45))
                lines = _wrap_pdf_text(text, width, size)
                body = bytearray(b"/P <</MCID 1>> BDC\nBT\n/HbF1 %f Tf\n3 Tr\n"
                                 % size)
                top = height - size * 3
                for offset, line in enumerate(lines):
                    literal = _pdf_string(line)
                    if literal is None:
                        continue
                    body += b"1 0 0 1 %f %f Tm\n%s Tj\n" % (
                        size, top - offset * size * 1.25, literal)
                body += b"ET\nEMC\n"
                page.contents_add(pikepdf.Stream(pdf, bytes(body)))

                page.Resources = page.get("/Resources", pikepdf.Dictionary())
                fonts = page.Resources.get("/Font", pikepdf.Dictionary())
                fonts["/HbF1"] = font
                page.Resources["/Font"] = fonts

                paragraph = pdf.make_indirect(pikepdf.Dictionary(
                    Type=pikepdf.Name.StructElem, S=pikepdf.Name.P,
                    P=document, Pg=page.obj, K=1,
                    ActualText=pikepdf.String(text),
                ))
                elements.append(paragraph)
                written += 1

            page.StructParents = index
            parent_pairs += [index, pdf.make_indirect(pikepdf.Array(elements))]
            kids.extend(elements)

        document.K = pikepdf.Array(kids)
        struct_root.K = pikepdf.Array([document])
        struct_root.ParentTree = pdf.make_indirect(pikepdf.Dictionary(
            Nums=pikepdf.Array(parent_pairs)))
        struct_root.ParentTreeNextKey = len(pdf.pages)
        pdf.Root.StructTreeRoot = struct_root
        pdf.Root.MarkInfo = pikepdf.Dictionary(Marked=True)
        pdf.save(pdf_path)
    return written


def _wrap_pdf_text(text: str, page_width: float, size: float) -> list[str]:
    """Break the text into lines that fit the page, for the invisible layer.

    Helvetica averages about half its point size per character; the layer is
    invisible, so this only has to be close enough that selecting a page
    yields its words in reading order.
    """
    per_line = max(20, int(page_width / (size * 0.5)))
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words, current = paragraph.split(), ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) <= per_line or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


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
