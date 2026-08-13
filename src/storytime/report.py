"""Turning a scored run into something you can actually look at.

Two outputs:

  contact_<backend>_<strategy>.png   the character sheet followed by every
                                     page, so drift is visible at a glance
  report.html                        the numbers, next to the pictures
"""

from __future__ import annotations

import html
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from .pipeline import RunLayout
from .types import PageRecord, write_json

JUDGE_DIMS = ("identity", "attributes", "style")
CHEAP_KEYS = ("palette_cosine", "signature_coverage", "edge_match")

THUMB = 320
LABEL_H = 34


def _font(size: int = 15) -> ImageFont.ImageFont:
    for candidate in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _thumb(path: Path, width: int = THUMB) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        height = max(1, round(image.height * width / image.width))
        return image.resize((width, height), Image.Resampling.LANCZOS)


# --------------------------------------------------------------------------- aggregation


def _mean(values: Iterable[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 3) if values else None


def aggregate(records: list[PageRecord]) -> list[dict[str, Any]]:
    """Mean scores per (backend, strategy), best first."""
    groups: dict[tuple[str, str], list[PageRecord]] = defaultdict(list)
    for record in records:
        groups[(record.backend, record.strategy)].append(record)

    rows: list[dict[str, Any]] = []
    for (backend, strategy), group in groups.items():
        judged = [r for r in group if r.judge and "error" not in r.judge]
        row: dict[str, Any] = {
            "backend": backend,
            "strategy": strategy,
            "pages": len(group),
            "failed": sum(1 for r in group if r.error),
            "judged": len(judged),
        }
        for dim in JUDGE_DIMS:
            row[dim] = _mean(r.judge.get(dim) for r in judged)
        row["judge_mean"] = _mean(r.judge.get("mean") for r in judged)
        for key in CHEAP_KEYS:
            row[key] = _mean(r.metrics.get(key) for r in group if not r.error)
        if any("dino_cosine" in r.metrics for r in group):
            row["dino_cosine"] = _mean(r.metrics.get("dino_cosine") for r in group)
        rows.append(row)

    # Rank by the judge when it ran, otherwise by the colour metric.
    rows.sort(
        key=lambda r: (
            r["judge_mean"] if r["judge_mean"] is not None else -1,
            r["palette_cosine"] if r["palette_cosine"] is not None else -1,
        ),
        reverse=True,
    )
    return rows


def hardest_scenes(records: list[PageRecord], limit: int = 5) -> list[dict[str, Any]]:
    """Which scenes broke consistency most often, across all backends."""
    by_scene: dict[str, list[PageRecord]] = defaultdict(list)
    for record in records:
        by_scene[record.scene_id].append(record)

    rows = []
    for scene_id, group in by_scene.items():
        judged = [r for r in group if r.judge and "error" not in r.judge]
        if not judged:
            continue
        rows.append(
            {
                "scene_id": scene_id,
                "stress": next((r.stress for r in group if r.stress), ""),
                "judge_mean": _mean(r.judge.get("mean") for r in judged),
                "identity": _mean(r.judge.get("identity") for r in judged),
            }
        )
    rows.sort(key=lambda r: r["judge_mean"] if r["judge_mean"] is not None else 99)
    return rows[:limit]


# --------------------------------------------------------------------------- contact sheets


def contact_sheet(layout: RunLayout, records: list[PageRecord], backend: str, strategy: str) -> Path | None:
    """Character sheet plus every page in one image, labelled with its score."""
    selected = sorted(
        (r for r in records if r.backend == backend and r.strategy == strategy),
        key=lambda r: r.scene_id,
    )
    tiles: list[tuple[Image.Image, str]] = []

    sheet = layout.sheet(backend)
    if sheet.is_file():
        tiles.append((_thumb(sheet), "REFERENCE SHEET"))

    for record in selected:
        page = layout.root / record.image_path
        if not page.is_file():
            continue
        label = f"page {record.scene_id}"
        if record.judge and "error" not in record.judge:
            label += (f"  id {record.judge['identity']}"
                      f" attr {record.judge['attributes']}"
                      f" sty {record.judge['style']}")
        elif "palette_cosine" in record.metrics:
            label += f"  palette {record.metrics['palette_cosine']:.2f}"
        tiles.append((_thumb(page), label))

    if not tiles:
        return None

    columns = min(4, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    cell_h = max(t.height for t, _ in tiles) + LABEL_H
    canvas = Image.new("RGB", (columns * THUMB, rows * cell_h), (250, 249, 246))
    draw = ImageDraw.Draw(canvas)
    font = _font()

    for index, (image, label) in enumerate(tiles):
        x = (index % columns) * THUMB
        y = (index // columns) * cell_h
        canvas.paste(image, (x, y))
        draw.rectangle([x, y + image.height, x + THUMB, y + cell_h], fill=(238, 235, 228))
        draw.text((x + 8, y + image.height + 8), label, fill=(40, 38, 34), font=font)

    out = layout.root / f"contact_{backend}_{strategy}.png"
    canvas.save(out)
    return out


# --------------------------------------------------------------------------- html


CSS = """
:root { color-scheme: light dark; --bg:#faf9f6; --fg:#1d1b18; --muted:#6b675f;
        --line:#dedad0; --card:#ffffff; --good:#1f7a45; --bad:#a8322a; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16150f; --fg:#ece8dd; --muted:#9d968a; --line:#332f26;
          --card:#1e1c15; --good:#5fd394; --bad:#f08b80; }
}
* { box-sizing: border-box; }
body { margin:0; padding:2rem 1.5rem 5rem; background:var(--bg); color:var(--fg);
       font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif; }
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size:1.7rem; margin:0 0 .3rem; }
h2 { font-size:1.15rem; margin:2.6rem 0 .8rem; }
h3 { font-size:.95rem; margin:1.6rem 0 .5rem; font-weight:600; }
p.sub { color:var(--muted); margin:0 0 1.6rem; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:14px; min-width:720px; }
th,td { text-align:left; padding:.5rem .7rem; border-bottom:1px solid var(--line); }
th { font-weight:600; color:var(--muted); font-size:12px; text-transform:uppercase;
     letter-spacing:.04em; }
tbody tr:first-child td { font-weight:600; }
td.num { font-variant-numeric: tabular-nums; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); gap:1rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
        overflow:hidden; }
.card img { width:100%; display:block; background:#fff; }
.meta { padding:.55rem .7rem; font-size:12.5px; }
.meta .row { display:flex; justify-content:space-between; gap:.5rem; color:var(--muted); }
.meta b { color:var(--fg); }
.notes { margin:.4rem 0 0; padding-left:1rem; color:var(--bad); font-size:12px; }
.err { color:var(--bad); padding:.7rem; font-size:12.5px; }
.sheet { max-width:420px; border:1px solid var(--line); border-radius:10px; }
footer { margin-top:3rem; color:var(--muted); font-size:12.5px; }
"""


def _cell(value: Any) -> str:
    if value is None:
        return "<td class='num'>&mdash;</td>"
    if isinstance(value, float):
        return f"<td class='num'>{value:.2f}</td>"
    return f"<td class='num'>{html.escape(str(value))}</td>"


def _summary_table(rows: list[dict[str, Any]]) -> str:
    head = (
        "<tr><th>backend</th><th>strategy</th><th>identity</th><th>attributes</th>"
        "<th>style</th><th>judge mean</th><th>palette</th><th>signature</th>"
        "<th>edge match</th><th>failed</th></tr>"
    )
    body = "".join(
        "<tr><td>" + html.escape(r["backend"]) + "</td><td>" + html.escape(r["strategy"]) + "</td>"
        + _cell(r["identity"]) + _cell(r["attributes"]) + _cell(r["style"])
        + _cell(r["judge_mean"]) + _cell(r["palette_cosine"]) + _cell(r["signature_coverage"])
        + _cell(r["edge_match"]) + _cell(r["failed"]) + "</tr>"
        for r in rows
    )
    return f"<div class='scroll'><table><thead>{head}</thead><tbody>{body}</tbody></table></div>"


def _page_cards(layout: RunLayout, records: list[PageRecord], backend: str, strategy: str) -> str:
    selected = sorted(
        (r for r in records if r.backend == backend and r.strategy == strategy),
        key=lambda r: r.scene_id,
    )
    cards = []
    for record in selected:
        if record.error:
            cards.append(
                f"<div class='card'><div class='err'>page {html.escape(record.scene_id)}<br>"
                f"{html.escape(record.error)}</div></div>"
            )
            continue

        judge = record.judge or {}
        lines = [f"<div class='row'><span>page {html.escape(record.scene_id)}</span>"
                 f"<b>{judge.get('mean', '')}</b></div>"]
        if "error" in judge:
            lines.append(f"<div class='row'><span>judge failed</span></div>")
        elif judge:
            lines.append(
                f"<div class='row'><span>identity / attributes / style</span>"
                f"<b>{judge.get('identity')} &middot; {judge.get('attributes')}"
                f" &middot; {judge.get('style')}</b></div>"
            )
        if "palette_cosine" in record.metrics:
            lines.append(
                f"<div class='row'><span>palette</span>"
                f"<b>{record.metrics['palette_cosine']:.2f}</b></div>"
            )
        if record.stress:
            lines.append(f"<div class='row'><span>{html.escape(record.stress)}</span></div>")
        notes = judge.get("discrepancies") or []
        if notes:
            items = "".join(f"<li>{html.escape(str(n))}</li>" for n in notes[:4])
            lines.append(f"<ul class='notes'>{items}</ul>")

        cards.append(
            f"<div class='card'><img loading='lazy' src='{html.escape(record.image_path)}' "
            f"alt='page {html.escape(record.scene_id)}'>"
            f"<div class='meta'>{''.join(lines)}</div></div>"
        )
    return f"<div class='grid'>{''.join(cards)}</div>"


def build_html(layout: RunLayout, records: list[PageRecord], spec_name: str) -> Path:
    rows = aggregate(records)
    hardest = hardest_scenes(records)

    parts = [
        f"<div class='wrap'><h1>Character consistency &mdash; {html.escape(spec_name)}</h1>",
        f"<p class='sub'>{len(records)} pages &middot; "
        f"{len({r.backend for r in records})} backends &middot; "
        f"{len({r.strategy for r in records})} strategies. "
        f"Judge scores are 1&ndash;5, higher is better.</p>",
        "<h2>Summary</h2>",
        _summary_table(rows),
    ]

    if hardest:
        items = "".join(
            f"<tr><td>{html.escape(h['scene_id'])}</td>"
            f"<td>{html.escape(h['stress'])}</td>"
            + _cell(h["identity"]) + _cell(h["judge_mean"]) + "</tr>"
            for h in hardest
        )
        parts.append(
            "<h2>Hardest scenes</h2><div class='scroll'><table><thead><tr><th>scene</th>"
            "<th>what it stresses</th><th>identity</th><th>judge mean</th></tr></thead>"
            f"<tbody>{items}</tbody></table></div>"
        )

    for backend in sorted({r.backend for r in records}):
        parts.append(f"<h2>{html.escape(backend)}</h2>")
        sheet = layout.sheet(backend)
        if sheet.is_file():
            rel = str(sheet.relative_to(layout.root)).replace("\\", "/")
            parts.append(f"<img class='sheet' src='{html.escape(rel)}' alt='character sheet'>")
        for strategy in sorted({r.strategy for r in records if r.backend == backend}):
            parts.append(f"<h3>{html.escape(strategy)}</h3>")
            parts.append(_page_cards(layout, records, backend, strategy))

    parts.append(
        "<footer>Generated by StoryTime. Judge scores come from a vision model and "
        "carry its biases &mdash; spot-check the pictures before trusting the table.</footer></div>"
    )

    document = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>StoryTime &mdash; {html.escape(spec_name)}</title>"
        f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>"
    )
    layout.report_html.write_text(document, encoding="utf-8")
    return layout.report_html


def build_all(layout: RunLayout, records: list[PageRecord], spec_name: str, log=print) -> Path:
    for backend in sorted({r.backend for r in records}):
        for strategy in sorted({r.strategy for r in records if r.backend == backend}):
            path = contact_sheet(layout, records, backend, strategy)
            if path:
                log(f"  wrote {path.name}")

    write_json(
        layout.root / "summary.json",
        {"summary": aggregate(records), "hardest_scenes": hardest_scenes(records)},
    )
    path = build_html(layout, records, spec_name)
    log(f"  wrote {path.name}")
    return path


def print_summary(records: list[PageRecord], log=print) -> None:
    rows = aggregate(records)
    if not rows:
        log("no results to summarise")
        return
    header = f"{'backend':<10} {'strategy':<17} {'ident':>6} {'attr':>6} {'style':>6} {'palette':>8} {'fail':>5}"
    log(header)
    log("-" * len(header))
    for row in rows:
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else ("-" if v is None else str(v))  # noqa: E731
        log(
            f"{row['backend']:<10} {row['strategy']:<17} {fmt(row['identity']):>6} "
            f"{fmt(row['attributes']):>6} {fmt(row['style']):>6} "
            f"{fmt(row['palette_cosine']):>8} {row['failed']:>5}"
        )
