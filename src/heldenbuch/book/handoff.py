"""What to actually do with the files, at a real print shop.

Handing someone a PDF and wishing them luck is where most home-made books
stop. This writes the missing page: which product to choose, which options,
which file goes where, and what to check on the proof.

The numbers come from the preset the book was exported with, so they are the
book's real numbers rather than a generic guide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .layout import MM_PER_INCH, PrintPreset

PROVIDERS: dict[str, dict[str, Any]] = {
    "lulu": {
        "name": "Lulu",
        "url": "https://www.lulu.com/create/print-books",
        "where": "weltweit, druckt in Europa",
        "steps": [
            'Buchtyp "Photo Book" oder "Print Book" wählen, Bindung Hardcover '
            "oder Softcover.",
            "Format: das quadratische 8,5 × 8,5 Zoll entspricht 21,6 × 21,6 cm.",
            'Papier: das schwerste angebotene ("premium" / 80#), sonst scheint '
            "das Bild der Rückseite durch.",
            "Innenteil als eine PDF hochladen — die Datei mit dem Sprachkürzel "
            "im Namen.",
            "Umschlag getrennt hochladen: die Wrap-Datei enthält Rücken und "
            "Rückseite und ist schon in der richtigen Gesamtbreite angelegt.",
        ],
        "warnings": [
            "Lulu akzeptiert keine doppelseitigen Bilder über den Bund hinweg. "
            "Jede Seite ist einzeln — so ist das Buch auch angelegt.",
            "Keine Schnittmarken einschalten. Die Datei hat schon Beschnitt, "
            "Marken würden mitgedruckt.",
        ],
    },
    "epubli": {
        "name": "epubli",
        "url": "https://www.epubli.com/buch/drucken",
        "where": "Berlin, druckt in Deutschland, ab 1 Exemplar",
        "steps": [
            'Produkt "Fotobuch" oder "Buch drucken" wählen.',
            "Format: Kinderbuch 15,6 × 14,8 cm oder Quadrat, je nachdem womit "
            "exportiert wurde.",
            "Papier: 170 g oder 250 g weiß matt.",
            "Innenteil als PDF, Cover als JPG mit 300 dpi hochladen.",
        ],
        "warnings": [
            "epubli will das Cover als Bilddatei, nicht als PDF — dafür ist die "
            "JPG-Datei da.",
        ],
    },
    "gelato": {
        "name": "Gelato",
        "url": "https://www.gelato.com/products/childrens-books",
        "where": "druckt lokal in 32 Ländern",
        "steps": [
            "Children's Book wählen, Hardcover oder Softcover.",
            "Format an die exportierte Größe anpassen.",
            "Innenteil und Umschlag getrennt hochladen.",
        ],
        "warnings": [],
    },
}


def sheet(book, preset: PrintPreset, exports: list[dict], cover_info: dict | None = None,
          provider: str = "lulu", measured_dpi: int | None = None) -> str:
    """A markdown page telling you exactly what to do next.

    `measured_dpi` is the real resolution of the weakest page. The table used
    to print the preset's nominal 300 dpi as though it were a fact about the
    file, directly above the instruction to upload it -- which is how a 117 dpi
    book gets sent to a print shop with a sheet certifying it as print-ready.
    """
    spec = PROVIDERS.get(provider, PROVIDERS["lulu"])
    trim_w, trim_h = preset.trim_mm
    bleed = preset.bleed_mm
    pages = max((e.get("pages", 0) for e in exports), default=0)
    if measured_dpi is None:
        resolution = f"{preset.dpi} dpi (Sollwert — nicht nachgemessen)"
    elif measured_dpi >= preset.dpi:
        resolution = f"{measured_dpi} dpi"
    else:
        resolution = (f"**{measured_dpi} dpi** — gefordert sind {preset.dpi}. "
                      "Die Vorprüfung der Druckerei merkt das nicht.")

    lines = [
        f"# {book.display_title()} — ab in den Druck",
        "",
        f"Empfohlen: **{spec['name']}** ({spec['where']}) — {spec['url']}",
        "",
        "## Die Zahlen für das Bestellformular",
        "",
        "| | |",
        "|---|---|",
        f"| Endformat (beschnitten) | {trim_w:.1f} × {trim_h:.1f} mm |",
        f"| Datei inkl. Beschnitt | {trim_w + 2 * bleed:.1f} × {trim_h + 2 * bleed:.1f} mm |",
        f"| Beschnittzugabe | {bleed:.3f} mm auf jeder Seite |",
        f"| Sicherheitsabstand | {preset.safety_mm:.1f} mm — nichts Wichtiges dichter an den Rand |",
        f"| Auflösung | {resolution} |",
        f"| Seitenzahl Innenteil | {pages} |",
        "| Farbraum | RGB, ohne eingebettetes Profil |",
    ]

    if cover_info:
        spine = cover_info.get("spine_mm", 0)
        width, height = cover_info.get("size_mm", (0, 0))
        lines += [
            f"| Rückenbreite | {spine:.1f} mm (bei {cover_info.get('interior_pages', pages)} Seiten) |",
            f"| Umschlag gesamt | {width:.1f} × {height:.1f} mm |",
        ]

    lines += ["", "## Dateien", ""]
    for item in exports:
        lines.append(
            f"- `{Path(item['file']).name}` — Innenteil, {item.get('language_name', item['language'])}"
            f", {item.get('pages', '?')} Seiten"
        )
    if cover_info and cover_info.get("file"):
        lines.append(f"- `{Path(cover_info['file']).name}` — Umschlag, komplett mit Rücken und Rückseite")
    if cover_info and cover_info.get("front_file"):
        lines.append(f"- `{Path(cover_info['front_file']).name}` — nur die Vorderseite, als JPG")

    lines += ["", "## Schritt für Schritt", ""]
    lines += [f"{i}. {step}" for i, step in enumerate(spec["steps"], start=1)]

    warnings = list(spec["warnings"])
    if cover_info and cover_info.get("note"):
        warnings.append(cover_info["note"])
    for item in exports:
        warnings.extend(item.get("warnings", []))
    if warnings:
        lines += ["", "## Aufpassen", ""]
        lines += [f"- {w}" for w in dict.fromkeys(warnings)]

    checks = [
        "- Ist an allen vier Rändern Bild bis zur Kante, ohne weiße Streifen?",
        "- Steht der Text weit genug vom Rand und vom Bund weg?",
    ]
    # Only ask about the spine when there is actually something printed on it.
    if cover_info and not cover_info.get("note"):
        checks.append("- Sitzt der Titel auf dem Rücken mittig?")
    checks.append(
        "- Sind die Bilder scharf? Wenn sie weich wirken, war das Buch im "
        "Entwurfsmodus gerendert — noch einmal in Druckqualität zeichnen lassen."
    )

    lines += ["", "## Am Probeexemplar prüfen", ""] + checks
    lines += ["", f"_Erzeugt von Heldenbuch. Seitenformat {preset.name}._"]
    return "\n".join(lines)


def spine_note(preset: PrintPreset, pages: int) -> str:
    inches = pages / preset.pages_per_inch + preset.spine_allowance_in
    return (
        f"{pages} Seiten ÷ {preset.pages_per_inch:.0f} + "
        f"{preset.spine_allowance_in} Zoll = {inches:.3f} Zoll "
        f"= {inches * MM_PER_INCH:.1f} mm Rücken"
    )
