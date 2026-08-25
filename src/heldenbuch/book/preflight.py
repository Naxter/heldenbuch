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

import statistics
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
from .models import AGE_BANDS, LANGUAGES, Book

#: below this a 300-dpi print preset reads as visibly soft on paper
DPI_WARN = 280

#: how far a page's palette may sit from the book's own median before it is
#: worth a human look. Symmetric on purpose: the per-page check compares each
#: page to the *sheet* and tolerates small deltas, so a page can pass alone
#: and still sit visibly apart from its neighbours -- on the first finished
#: book this rule flags exactly one page, and it is one the owner had
#: complained about before the rule existed.
PALETTE_DRIFT = 0.15

#: Fewer scored pages than this and the median says nothing worth acting on.
MIN_SCORED_PAGES = 6


#: how far a page's embedding similarity may sit from the book's median.
#: Tighter than the palette rule because DINO cosines cluster high; like it,
#: unvalidated until the benchmark runs, which is why both feed only notes.
#: Known confound, stated in embed.py's own docstring: the similarity is
#: whole-image and measured against the sheet, so a busy scene scores lower
#: than a plain one regardless of identity. The median-of-the-book framing
#: softens that (every page shares the handicap) but does not remove it --
#: page-to-neighbour distance would, and needs the benchmark to calibrate.
EMBED_DRIFT = 0.12


def _metric_outliers(book: Book, key: str, drift: float) -> list[int]:
    """Pages whose stored per-page metric stands apart from the book's median.

    Derived from stored evidence, so it costs nothing and reaches books
    already on disk. Needs enough scored pages for a median to mean anything.
    """
    scored = [(p.index, (p.check or {}).get("metrics", {}).get(key))
              for p in book.pages]
    values = sorted(v for _, v in scored if isinstance(v, (int, float)))
    if len(values) < MIN_SCORED_PAGES:
        return []
    median = statistics.median(values)
    return sorted(index for index, v in scored
                  if isinstance(v, (int, float)) and abs(v - median) > drift)


def palette_outliers(book: Book) -> list[int]:
    return _metric_outliers(book, "palette_cosine", PALETTE_DRIFT)


def wordy_pages(book: Book, language: str) -> list[int]:
    """Pages clearly over the age band's word budget, in the given language.

    The 1.4 headroom keeps this quiet for ordinary variance; only pages a
    listener would actually feel as long are named.
    """
    limit = AGE_BANDS.get(book.age, {}).get("words_max")
    if not limit:
        return []
    return sorted(
        page.index for page in book.pages
        if page.layout != "wordless"
        and len((page.text or {}).get(language, "").split()) > limit * 1.4)


def embed_outliers(book: Book) -> list[int]:
    """The sharper cross-page look, where the embedding extra was installed.

    `dino_cosine` measures whole-image likeness to the reference sheet; a
    page whose likeness sits far from the book's own median looks different
    from its siblings in a way the palette histogram cannot see -- style
    drift, a recoloured character, a wrong setting.
    """
    return _metric_outliers(book, "dino_cosine", EMBED_DRIFT)


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
    family: str = "georgia",
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
    #: facts about the chosen format worth knowing, never gating
    notes: list[dict[str, Any]] = []
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
    # The pictures are drawn square; a page far from square fills itself by
    # cropping them. Said here rather than discovered in the printed book,
    # where a cut-off composition reads as "the model ignored my prompt".
    # A note, not a warning: it is a property of the format, true on every
    # export, and a state that cries "warnung" forever teaches people to
    # stop reading it.
    page_w, page_h = preset.page_px()
    crop_share = 1.0 - min(page_w, page_h) / max(page_w, page_h)
    if crop_share > 0.10:
        percent = round(crop_share * 100)
        notes.append(finding(
            "aspect_crop",
            f"Die Bilder sind quadratisch, dieses Format ist es nicht — pro "
            f"Bild werden etwa {percent} % mittig weggeschnitten. Die "
            "Druckbogen-Vorschau zeigt den Ausschnitt.",
            percent=percent))

    def pages_note(code: str, template: str, indices: list[int]) -> None:
        if indices:
            listed = ", ".join(map(str, indices))
            notes.append(finding(code, template.format(pages=listed),
                                 pages=listed, count=len(indices)))

    # Pages whose check said the location plainly is not the place reference.
    # A hint like the palette rule: the judge's word alone must not gate an
    # export, but a person should hear it before ordering the book.
    pages_note("setting_mismatch",
               "Seite(n) {pages}: Der Schauplatz passt laut Prüfung nicht "
               "zur Orts-Referenz — beim Durchblättern prüfen.",
               sorted(page.index for page in book.pages
                      if (page.check or {}).get("setting_consistent") is False))

    # The cross-page look: every check compares a page to the reference sheet
    # alone, so drift between pages -- the thing a reader flipping the book
    # actually sees -- shows up nowhere else. The embedding rule sees more
    # (style, setting, a recoloured figure) where the extra is installed; the
    # palette rule is the floor that always works.
    drifted = palette_outliers(book)
    pages_note("palette_outlier",
               "Seite(n) {pages} weichen farblich deutlich vom Rest des "
               "Buches ab — beim Durchblättern prüfen.", drifted)
    pages_note("embed_outlier",
               "Seite(n) {pages} sehen der Vorlage deutlich unähnlicher als "
               "der Rest des Buches — beim Durchblättern prüfen.",
               [i for i in embed_outliers(book) if i not in drifted])

    # The words carry half the book, in every chosen language: translations
    # rarely match lengths, and checking only the first shipped an
    # over-budget second language unflagged into a bilingual book.
    chosen = languages or [book.primary_language]
    pages_note("wordy_pages",
               "Seite(n) {pages} liegen deutlich über dem Wortbudget der "
               "Altersstufe — beim Vorlesen prüfen, ob sie zu lang geraten.",
               sorted({i for code in chosen for i in wordy_pages(book, code)}))

    # Measured in the face the export will actually set, not the default:
    # Georgia's metrics said "fits" for pages that Comic Sans then overflowed.
    overfull = [
        page.index for page in sorted(book.pages, key=lambda p: p.index)
        if page.layout != "wordless"
        and not text_fits(join_languages(page.text, languages), preset,
                          book.age, family=family)
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
                  pages_unknown, is_print, allow_unknown, notes=notes)


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
           pages_unknown, is_print: bool, allow_unknown: bool,
           notes=()) -> dict[str, Any]:
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
        "notes": list(notes),
        "pages_missing": pages_missing,
        "pages_failed_check": pages_failed,
        "pages_unknown_check": pages_unknown,
    }
