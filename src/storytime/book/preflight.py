"""Is this book actually ready to become a file a print shop will accept?

The export panel used to appear as soon as one page existed, and the exporter
would happily build a PDF with blank pages, failed checks, or a checker that
had crashed and verified nothing. This module is the gate: one server-side
function that looks at everything, called by the export job *before* any PDF
is written, and by the UI to say what still stands in the way.

Severity is honest about what each problem means on paper:

  errors     break the printed book -- a missing picture, a page whose render
             failed, missing text in a selected language. For a print preset
             these block the export; for a screen preset they merely warn,
             because a preview on a laptop is not a bound book.
  unknown    the consistency checker crashed or never ran, so nobody -- human
             or model -- has verified those pages. Blocks print by default;
             `allow_unknown=True` is the explicit override.
  warnings   worth knowing, not worth blocking: low resolution (soft print,
             your call), an empty dedication, a blank spine.
"""

from __future__ import annotations

from typing import Any

from .illustrate import check_status
from .layout import FONT_FAMILIES, PrintPreset, effective_dpi, find_font
from .models import LANGUAGES, Book

#: below this a 300-dpi print preset reads as visibly soft on paper
DPI_WARN = 280


def validate_export_readiness(
    book: Book,
    preset: PrintPreset,
    languages: list[str],
    resolve,
    allow_unknown: bool = False,
) -> dict[str, Any]:
    """One structured verdict for the whole export.

    `resolve` maps a stored book-relative path to an absolute one, exactly as
    the exporter itself will. Returns errors / warnings / unknowns plus the
    page lists the UI links to, and `state`: bereit | warnung | unbekannt |
    unvollständig.
    """
    is_print = preset.bleed_mm > 0
    errors: list[str] = []
    warnings: list[str] = []
    unknowns: list[str] = []
    pages_missing: list[int] = []
    pages_failed: list[int] = []
    pages_unknown: list[int] = []

    def art(relative):
        if not relative:
            return None
        try:
            path = resolve(relative)
        except (ValueError, FileNotFoundError):
            return None
        return path if path.is_file() else None

    def openable(path) -> bool:
        try:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
            return True
        except Exception:
            return False

    # ---- cover -------------------------------------------------------------
    cover = art(book.cover)
    if cover is None:
        (errors if is_print else warnings).append(
            "Es gibt kein Titelbild — die Titelseite und der Umschlag blieben leer."
        )
    elif not openable(cover):
        errors.append("Die Titelbild-Datei ist beschädigt und lässt sich nicht öffnen.")
    elif is_print and effective_dpi(cover, preset) < DPI_WARN:
        errors.append(
            f"Das Titelbild hat nur etwa {effective_dpi(cover, preset)} dpi — "
            "im Druck wird es weich."
        )

    # ---- pages -------------------------------------------------------------
    low_dpi: list[int] = []
    for page in sorted(book.pages, key=lambda p: p.index):
        image = art(page.image)
        if image is None:
            pages_missing.append(page.index)
            continue
        if not openable(image):
            pages_missing.append(page.index)
            errors.append(f"Seite {page.index}: die Bilddatei ist beschädigt.")
            continue
        if page.error:
            pages_failed.append(page.index)
            errors.append(f"Seite {page.index}: der letzte Zeichenversuch schlug fehl.")
            continue
        status = check_status(page)
        if status == "failed":
            pages_failed.append(page.index)
        elif status in ("unknown", "unchecked"):
            pages_unknown.append(page.index)
        if is_print and effective_dpi(image, preset) < DPI_WARN:
            low_dpi.append(page.index)

    if pages_missing:
        errors.append(
            f"{len(pages_missing)} Seite(n) ohne Bild: "
            + ", ".join(map(str, pages_missing)) + "."
        )
    if pages_failed:
        failed_only = [i for i in pages_failed
                       if not any(f"Seite {i}:" in e for e in errors)]
        if failed_only:
            errors.append(
                "Die Ähnlichkeitsprüfung hat Seite(n) "
                + ", ".join(map(str, failed_only))
                + " beanstandet — erst ansehen, dann exportieren."
            )
    if pages_unknown:
        unknowns.append(
            f"{len(pages_unknown)} Seite(n) sind ungeprüft — die Prüfung lief "
            "nicht oder brach ab: " + ", ".join(map(str, pages_unknown)) + "."
        )
    if low_dpi:
        # An error, not a warning. The PDF reports the resolution of the
        # *upscaled* bitmap, so a 1024 px illustration stretched onto a 300 dpi
        # page passes a print shop's preflight and only reveals itself as a
        # soft, blurry book when the box arrives. Nothing downstream catches
        # this, so it has to stop the export here.
        errors.append(
            f"{len(low_dpi)} Seite(n) liegen unter {DPI_WARN} dpi für dieses "
            "Format: " + ", ".join(map(str, low_dpi))
            + ". Die Druckerei merkt das nicht — im Buch sieht man es. "
            "Vor dem Bestellen in Druckqualität neu zeichnen."
        )

    stale = [p.index for p in book.pages if p.image_stale()]
    if stale:
        warnings.append(
            f"Seite(n) {', '.join(map(str, stale))}: die Bildanweisung wurde "
            "seit dem Zeichnen geändert — das Bild zeigt noch die alte Fassung."
        )

    # ---- text --------------------------------------------------------------
    for code in languages:
        name = LANGUAGES.get(code, {}).get("name", code)
        if not (book.title.get(code) or "").strip():
            errors.append(f"Es fehlt der Titel auf {name}.")
        if not (book.dedication.get(code) or "").strip():
            warnings.append(f"Keine Widmung auf {name} — die Seite entfällt.")
        empty = [p.index for p in book.pages
                 if p.layout != "wordless" and not (p.text.get(code) or "").strip()]
        if empty:
            errors.append(
                f"Seite(n) {', '.join(map(str, empty))} haben keinen Text auf {name}."
            )

    # ---- selections --------------------------------------------------------
    if not languages:
        errors.append("Keine Sprache ausgewählt.")

    return finish(errors, warnings, unknowns, pages_missing, pages_failed,
                  pages_unknown, is_print, allow_unknown)


def check_font(family: str) -> str | None:
    """A warning line when the chosen font is not on this machine."""
    spec = FONT_FAMILIES.get(family)
    if spec is None:
        return f"Unbekannte Schrift „{family}“ — Georgia wird verwendet."
    if find_font(spec["regular"]) is None:
        return f"{spec['name']} ist hier nicht installiert — Georgia wird verwendet."
    return None


def finish(errors, warnings, unknowns, pages_missing, pages_failed,
           pages_unknown, is_print: bool, allow_unknown: bool) -> dict[str, Any]:
    """Fold the findings into one state the UI and the export job both obey."""
    if not is_print:
        # A screen or home-printer file is not a bound book: everything that
        # would block print is still worth saying, but nothing blocks.
        warnings = errors + unknowns + warnings
        errors, unknowns = [], []
    if allow_unknown and unknowns:
        warnings = warnings + unknowns
        unknowns = []

    if errors:
        state = "unvollständig"
    elif unknowns:
        state = "unbekannt"
    elif warnings:
        state = "warnung"
    else:
        state = "bereit"

    return {
        "ok": state in ("bereit", "warnung"),
        "state": state,
        "errors": errors,
        "warnings": warnings,
        "unknowns": unknowns,
        "pages_missing": pages_missing,
        "pages_failed_check": pages_failed,
        "pages_unknown_check": pages_unknown,
    }
