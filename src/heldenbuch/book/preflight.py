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

from .illustrate import check_status, cover_flagged
from .layout import (
    FONT_FAMILIES,
    PrintPreset,
    effective_dpi,
    find_font,
    join_languages,
    text_fits,
)
from .models import LANGUAGES, Book

#: below this a 300-dpi print preset reads as visibly soft on paper
DPI_WARN = 280


def finding(code: str, text: str, **params: Any) -> dict[str, Any]:
    """One preflight result, in a form both readers can use.

    `code` and `params` are what the browser needs: it looks the code up in
    its own locale table and fills the slots, so the export panel speaks the
    language the person chose. `text` is the same sentence in German, which
    the job log prints and the browser falls back to for any code it does not
    know yet. One German source, one place to translate.
    """
    return {"code": code, "params": params, "text": text}


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
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    unknowns: list[dict[str, Any]] = []
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
        (errors if is_print else warnings).append(finding(
            "cover_missing",
            "Es gibt kein Titelbild — die Titelseite und der Umschlag blieben leer."))
    elif not openable(cover):
        errors.append(finding(
            "cover_broken",
            "Die Titelbild-Datei ist beschädigt und lässt sich nicht öffnen."))
    elif is_print and effective_dpi(cover, preset) < DPI_WARN:
        dpi = effective_dpi(cover, preset)
        errors.append(finding(
            "cover_low_dpi",
            f"Das Titelbild hat nur etwa {dpi} dpi — im Druck wird es weich.",
            dpi=dpi))
    if cover is not None and cover_flagged(book):
        note = (book.cover_check.get("notes") or ["es weicht von der Vorlage ab"])[0]
        errors.append(finding(
            "cover_flagged",
            f"Das Titelbild wurde beanstandet: {note}. Es ist das Bild, das man "
            "zuerst sieht — erst ansehen, dann exportieren.",
            note=note))

    # ---- pages -------------------------------------------------------------
    low_dpi: list[int] = []
    #: pages that already have a complaint of their own, so the summary lines
    #: below do not say the same thing twice
    named: set[int] = set()
    for page in sorted(book.pages, key=lambda p: p.index):
        image = art(page.image)
        if image is None:
            pages_missing.append(page.index)
            continue
        if not openable(image):
            pages_missing.append(page.index)
            named.add(page.index)
            errors.append(finding(
                "page_image_broken",
                f"Seite {page.index}: die Bilddatei ist beschädigt.", n=page.index))
            continue
        if page.error:
            pages_failed.append(page.index)
            named.add(page.index)
            errors.append(finding(
                "page_draw_failed",
                f"Seite {page.index}: der letzte Zeichenversuch schlug fehl.",
                n=page.index))
            continue
        status = check_status(page)
        if status == "failed":
            pages_failed.append(page.index)
        elif status in ("unknown", "unchecked"):
            pages_unknown.append(page.index)
        if is_print and effective_dpi(image, preset) < DPI_WARN:
            low_dpi.append(page.index)

    if pages_missing:
        listed = ", ".join(map(str, pages_missing))
        errors.append(finding(
            "pages_missing",
            f"{len(pages_missing)} Seite(n) ohne Bild: {listed}.",
            count=len(pages_missing), pages=listed))
    if pages_failed:
        failed_only = [i for i in pages_failed if i not in named]
        if failed_only:
            listed = ", ".join(map(str, failed_only))
            errors.append(finding(
                "pages_flagged",
                f"Die Ähnlichkeitsprüfung hat Seite(n) {listed} beanstandet — "
                "erst ansehen, dann exportieren.",
                count=len(failed_only), pages=listed))
    if pages_unknown:
        listed = ", ".join(map(str, pages_unknown))
        unknowns.append(finding(
            "pages_unchecked",
            f"{len(pages_unknown)} Seite(n) sind ungeprüft — die Prüfung lief "
            f"nicht oder brach ab: {listed}.",
            count=len(pages_unknown), pages=listed))
    # Text that cannot fit at a readable size. The renderer widens the box and
    # falls back from vignette to a full page before giving up, so reaching
    # here means the page is genuinely overfull -- and silently shrinking it
    # instead is what printed 6 pt type for the oldest age band.
    overfull = [
        page.index for page in sorted(book.pages, key=lambda p: p.index)
        if page.layout != "wordless"
        and not text_fits(join_languages(page.text, languages), preset, book.age)
    ]
    if overfull:
        listed = ", ".join(map(str, overfull))
        (errors if is_print else warnings).append(finding(
            "text_overfull",
            f"Zu viel Text für {len(overfull)} Seite(n) bei lesbarer "
            f"Schriftgröße: {listed}. Kürzen oder auf zwei Seiten teilen.",
            count=len(overfull), pages=listed))

    if low_dpi:
        # An error, not a warning. The PDF reports the resolution of the
        # *upscaled* bitmap, so a 1024 px illustration stretched onto a 300 dpi
        # page passes a print shop's preflight and only reveals itself as a
        # soft, blurry book when the box arrives. Nothing downstream catches
        # this, so it has to stop the export here.
        listed = ", ".join(map(str, low_dpi))
        errors.append(finding(
            "pages_low_dpi",
            f"{len(low_dpi)} Seite(n) liegen unter {DPI_WARN} dpi für dieses "
            f"Format: {listed}. Die Druckerei merkt das nicht — im Buch sieht "
            "man es. Vor dem Bestellen in Druckqualität neu zeichnen.",
            count=len(low_dpi), pages=listed, dpi=DPI_WARN))

    stale = [p.index for p in book.pages if p.image_stale()]
    if stale:
        listed = ", ".join(map(str, stale))
        warnings.append(finding(
            "pages_image_stale",
            f"Seite(n) {listed}: die Bildanweisung wurde seit dem Zeichnen "
            "geändert — das Bild zeigt noch die alte Fassung.",
            pages=listed))

    # ---- text --------------------------------------------------------------
    for code in languages:
        name = LANGUAGES.get(code, {}).get("name", code)
        if not (book.title.get(code) or "").strip():
            errors.append(finding("title_missing",
                                  f"Es fehlt der Titel auf {name}.", language=name))
        if not (book.dedication.get(code) or "").strip():
            warnings.append(finding(
                "dedication_missing",
                f"Keine Widmung auf {name} — die Seite entfällt.", language=name))
        empty = [p.index for p in book.pages
                 if p.layout != "wordless" and not (p.text.get(code) or "").strip()]
        if empty:
            listed = ", ".join(map(str, empty))
            errors.append(finding(
                "pages_text_missing",
                f"Seite(n) {listed} haben keinen Text auf {name}.",
                pages=listed, language=name))

    # ---- selections --------------------------------------------------------
    if not languages:
        errors.append(finding("no_language", "Keine Sprache ausgewählt."))

    return finish(errors, warnings, unknowns, pages_missing, pages_failed,
                  pages_unknown, is_print, allow_unknown)


def check_font(family: str) -> dict[str, Any] | None:
    """A warning when the chosen font is not on this machine."""
    spec = FONT_FAMILIES.get(family)
    if spec is None:
        return finding("font_unknown",
                       f"Unbekannte Schrift „{family}“ — Georgia wird verwendet.",
                       family=family)
    if find_font(spec["regular"]) is None:
        return finding("font_missing",
                       f"{spec['name']} ist hier nicht installiert — Georgia "
                       "wird verwendet.", name=spec["name"])
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
