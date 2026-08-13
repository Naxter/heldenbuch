"""HTTP surface for the book app.

Read operations answer immediately; anything that costs money or takes time is
pushed through the shared job manager instead.
"""

from __future__ import annotations

from typing import Any

from ..backends import REQUIRED_KEY
from ..book import preflight
from ..book.handoff import PROVIDERS as PRINT_SHOPS
from ..book.illustrate import RENDER_PROFILES, check_status, flagged_pages, review_split
from ..book.layout import PRESETS as PRINT_PRESETS
from ..book.layout import available_families
from ..book.library import Library
from ..book.look import PRESETS as STYLE_PRESETS
from ..book.models import AGE_BANDS, LANGUAGES, LAYOUTS, slugify
from ..book.narrate import VOICES
from ..book.scout import available_backends
from ..llm import available_providers
from ..pricing import FLAT_IMAGE_USD, RATES, USD_PER_EUR, image_estimate
from ..pricing import summary as spend_summary


class BookApi:
    def __init__(self, library: Library, jobs) -> None:
        self.library = library
        self.jobs = jobs

    # -------------------------------------------------------------- reference

    def status(self, _query, _body) -> dict[str, Any]:
        import os

        from ..config import load_dotenv

        from ..backends import comfy_available

        load_dotenv()
        image_backends = [
            {"name": name,
             # comfy has no key; ready means a local ComfyUI server answers
             "ready": (comfy_available() if name == "comfy"
                       else key is None or bool(os.environ.get(key, "").strip()))}
            for name, key in REQUIRED_KEY.items()
            if name != "stub"
        ]
        text = available_providers()
        active = self.jobs.active()

        can_draw = any(b["ready"] for b in image_backends)
        return {
            "ready": can_draw and bool(text),
            #: what actually works right now, so the UI can disable exactly
            #: the affected actions and name exactly the missing piece
            "capabilities": {
                "writing": bool(text),
                "drawing": can_draw,
                "style_analysis": bool(text),  # vision rides the text providers
                "narration": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
            },
            "image_backends": image_backends,
            "text_providers": text,
            "languages": [
                {"code": code, "name": info["name"]} for code, info in LANGUAGES.items()
            ],
            "ages": [
                {"key": key, "label": band["label"], "pages": band["pages"]}
                for key, band in AGE_BANDS.items()
            ],
            "style_presets": [
                {"key": key, "name": preset["name"], "hint": preset["hint"]}
                for key, preset in STYLE_PRESETS.items()
            ],
            "print_presets": [
                {
                    "key": preset.key,
                    "name": preset.name,
                    "hint": preset.hint,
                    "bleed": preset.bleed_mm > 0,
                }
                for preset in PRINT_PRESETS.values()
            ],
            "render_profiles": [
                {"key": key, "label": profile["label"], "px": profile["long_edge_px"],
                 "est_usd": image_estimate(profile["quality"], profile["long_edge_px"])}
                for key, profile in RENDER_PROFILES.items()
            ],
            #: rough per-image prices so buttons can say what they will spend
            "estimates": {
                "sheet_usd": image_estimate("high"),      # hero + styled sheets
                "preview_usd": image_estimate("medium"),  # previews, cast, scout
                "narration_page_usd": 0.003,              # measured: 12 pages ≈ $0.03
                "usd_per_eur": USD_PER_EUR,
            },
            #: models the picker offers, with what one image roughly costs.
            #: "scaled": gpt-image pricing, factor × the measured profile price.
            #: "flat": per-image list price as (draft, print).
            "image_models": [
                {"backend": "openai", "key": "gpt-image-2",
                 "label": "gpt-image-2 — Standard", "pricing": "scaled", "factor": 1.0},
                {"backend": "openai", "key": "gpt-image-1-mini",
                 "label": "gpt-image-1-mini — deutlich günstiger, einfachere Bilder",
                 "pricing": "scaled",
                 "factor": round(RATES["gpt-image-1-mini"]["output"]
                                 / RATES["gpt-image-2"]["output"], 3)},
            ] + [
                {"backend": ("gemini" if key.startswith("gemini") else "bfl"),
                 "key": key,
                 "label": {"gemini-3-pro-image": "Nano Banana Pro — referenztreu, bis 4K",
                           "gemini-3.1-flash-image": "Nano Banana Flash — schnell, günstig",
                           "flux-2-pro": "FLUX.2 pro — Flatpreis je Bild",
                           }.get(key, key),
                 "pricing": "flat", "usd": draft, "usd_print": print_}
                for key, (draft, print_) in FLAT_IMAGE_USD.items()
            ] + [
                {"backend": "comfy", "key": "flux-2-klein",
                 "label": "FLUX.2 klein — lokal auf deiner GPU, kostenlos",
                 "pricing": "flat", "usd": 0.0, "usd_print": 0.0},
            ],
            "fonts": available_families(),
            "voices": [{"key": key, "hint": hint} for key, hint in VOICES.items()],
            "layouts": [{"key": key, "hint": hint} for key, hint in LAYOUTS.items()],
            "print_shops": [
                {"key": key, "name": shop["name"], "where": shop["where"]}
                for key, shop in PRINT_SHOPS.items()
            ],
            #: a scout only makes sense with two or more image backends
            "can_scout": len(available_backends()) >= 2,
            "counts": {
                "heroes": len(self.library.heroes()),
                "styles": len(self.library.styles()),
                "books": len(self.library.books()),
            },
            "active_job": active.id if active else None,
            "queued": self.jobs.pending(),
            #: for the job center: everything recent, queued and finished alike
            "jobs": self.jobs.recent(10) if hasattr(self.jobs, "recent") else [],
        }

    # ------------------------------------------------------------------ heroes

    def heroes(self, _query, _body) -> dict[str, Any]:
        return {"heroes": [self._hero_public(h) for h in self.library.heroes()]}

    def hero(self, hero_id: str, _query, _body) -> dict[str, Any]:
        return self._hero_public(self.library.get_hero(hero_id))

    def hero_update(self, hero_id: str, _query, body) -> dict[str, Any]:
        hero = self.library.get_hero(hero_id)
        body = body or {}
        if body.get("sheet") in hero.variants:
            hero.sheet = body["sheet"]
        if "name" in body:
            hero.name = str(body["name"]).strip()
        if "description" in body:
            hero.description = str(body["description"]).strip()
        self.library.save_hero(hero)
        return self._hero_public(hero)

    def hero_delete(self, hero_id: str, _query, _body) -> dict[str, Any]:
        self.library.delete_hero(hero_id)
        return {"deleted": hero_id}

    def _hero_public(self, hero) -> dict[str, Any]:
        data = hero.to_dict()
        data["sheet_url"] = f"/library/{hero.sheet}" if hero.sheet else None
        data["variant_urls"] = [
            {"path": path, "url": f"/library/{path}"} for path in hero.variants
        ]
        data.pop("photos", None)  # the browser never needs to see them again
        data["photo_count"] = len(hero.photos)
        return data

    # ------------------------------------------------------------------ styles

    def styles(self, query, _body) -> dict[str, Any]:
        hero_id = (query.get("hero") or [None])[0]
        styles = self.library.styles()
        if hero_id:
            styles = [s for s in styles if hero_id in s.sheets]
        return {"styles": [self._style_public(s) for s in styles]}

    def style_delete(self, style_id: str, _query, _body) -> dict[str, Any]:
        self.library.delete_style(style_id)
        return {"deleted": style_id}

    def _style_public(self, style) -> dict[str, Any]:
        data = style.to_dict()
        data["preview_urls"] = [f"/library/{p}" for p in style.previews]
        data["sheet_urls"] = {
            hero_id: f"/library/{path}" for hero_id, path in style.sheets.items()
        }
        data["reference_url"] = f"/library/{style.reference}" if style.reference else None
        return data

    # ------------------------------------------------------------------- books

    def books(self, _query, _body) -> dict[str, Any]:
        rows = []
        for book in self.library.books():
            cover = f"/library/books/{book.id}/{book.cover}" if book.cover else None
            rows.append(
                {
                    "id": book.id,
                    "title": book.display_title(),
                    "languages": book.languages,
                    "age": book.age,
                    "pages": len(book.pages),
                    "drawn": sum(1 for p in book.pages if p.image),
                    "cover_url": cover,
                    "hero_id": book.hero_id,
                    "style_id": book.style_id,
                    "updated": book.updated,
                }
            )
        return {"books": rows}

    def book(self, book_id: str, _query, _body) -> dict[str, Any]:
        book = self.library.get_book(book_id)
        prefix = f"/library/books/{book.id}"

        hero = style = None
        hero_obj = style_obj = None
        try:
            hero_obj = self.library.get_hero(book.hero_id)
            hero = self._hero_public(hero_obj)
        except FileNotFoundError:
            pass
        try:
            style_obj = self.library.get_style(book.style_id)
            style = self._style_public(style_obj)
        except FileNotFoundError:
            pass

        data = book.to_dict()
        data["display_title"] = book.display_title()
        data["primary_language"] = book.primary_language
        data["cover_url"] = f"{prefix}/{book.cover}" if book.cover else None
        book_root = self.library.book_dir(book.id)
        for page, raw in zip(sorted(book.pages, key=lambda p: p.index), data["pages"]):
            raw["image_url"] = f"{prefix}/{page.image}" if page.image else None
            raw["can_undo"] = bool(page.history)
            raw["history_urls"] = [f"{prefix}/{path}" for path in page.history]
            raw["check_status"] = check_status(page)
            raw["image_stale"] = page.image_stale()
            raw["audio_stale"] = page.audio_stale()
            raw["audio_urls"] = {
                code: f"{prefix}/{path}" for code, path in page.audio.items()
            }
            # Long edge in pixels, so the UI can say whether this survives print.
            if page.image:
                try:
                    from PIL import Image

                    with Image.open(book_root / page.image) as im:
                        raw["image_px"] = max(im.size)
                except Exception:
                    pass
        for member, raw in zip(book.cast, data["cast"]):
            raw["sheet_url"] = f"{prefix}/{member.sheet}" if member.sheet else None
        if (book.photo_page or {}).get("image"):
            data["photo_page"]["image_url"] = f"{prefix}/{book.photo_page['image']}"
        data["spend"] = spend_summary(book.spend)
        data["flagged"] = flagged_pages(book)
        data["review"] = review_split(book)
        data["export_stale"] = book.export_stale()
        # Has the hero or style moved on since this book froze its references?
        references: dict[str, Any] = {"locked": bool(book.styled_sheet or book.hero_sheet)}
        if hero_obj and book.ref_sources.get("hero"):
            references["hero_changed"] = hero_obj.sheet != book.ref_sources["hero"]
        if style_obj is None:
            references["style_missing"] = True
        elif book.ref_sources.get("styled"):
            references["style_changed"] = (
                style_obj.sheets.get(book.hero_id) != book.ref_sources["styled"])
        data["references"] = references
        data["hero"] = hero
        data["style"] = style
        data["exports"] = [
            self._export_public(p)
            for p in sorted((self.library.book_dir(book.id) / "export").glob("*"))
            if p.is_file()
        ]
        return data

    def _export_public(self, path) -> dict[str, Any]:
        """An export file with a label a person understands, not a filename.

        The filenames encode language and preset (`stem_de-en_print_square.pdf`);
        this decodes them back into "Innenteil Deutsch + English — Druckerei …".
        """
        relative = self.library.relative(path)
        row: dict[str, Any] = {
            "file": relative, "url": f"/library/{relative}", "name": path.name,
            "size_kb": path.stat().st_size // 1024,
        }
        stem = path.stem
        if path.suffix == ".md":
            row["kind"], row["label"] = "handoff", "Anleitung für die Druckerei"
            return row
        if "_umschlag_" in stem:
            row["kind"], row["label"] = "wrap", "Umschlag — Rücken und Rückseite"
            return row
        if "_cover_" in stem:
            row["kind"], row["label"] = "cover", "Titelbild einzeln (JPG)"
            return row

        preset = next((p for key, p in PRINT_PRESETS.items() if stem.endswith(key)), None)
        if preset is not None:
            remainder = stem[: -len(preset.key)].rstrip("_")
            codes = remainder.rsplit("_", 1)[-1].split("-")
            names = [LANGUAGES.get(c, {}).get("name", c) for c in codes]
            row["kind"] = "interior"
            row["label"] = f"Innenteil {' + '.join(names)} — {preset.name}"
        else:
            row["kind"], row["label"] = "file", path.name
        return row

    def book_open_folder(self, book_id: str, _query, _body) -> dict[str, Any]:
        """Open the export folder in the system file manager.

        A localhost tool may do this: the button, the machine and the folder
        all belong to the same person. This is the single feature that stops
        "where did my PDF go" from needing a file-path conversation.
        """
        import os
        import subprocess
        import sys

        folder = self.library.book_dir(book_id) / "export"
        folder.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(folder)  # noqa: S606 -- deliberate, local desktop app
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
        return {"opened": self.library.relative(folder)}

    def book_update(self, book_id: str, _query, body) -> dict[str, Any]:
        """Edit page text or the title by hand -- the last word stays with a person.

        Every *actual* change bumps a revision, which is how old narration,
        old images and old exports get their "veraltet" label. Saving the
        same value twice bumps nothing.
        """
        book = self.library.get_book(book_id)
        body = body or {}
        changed = False

        if isinstance(body.get("title"), dict):
            fresh = {k: str(v) for k, v in body["title"].items()}
            changed |= any(book.title.get(k) != v for k, v in fresh.items())
            book.title.update(fresh)
        if isinstance(body.get("dedication"), dict):
            fresh = {k: str(v) for k, v in body["dedication"].items()}
            changed |= any(book.dedication.get(k) != v for k, v in fresh.items())
            book.dedication.update(fresh)
        if isinstance(body.get("cover_illustration"), str):
            fresh_cover = body["cover_illustration"].strip()
            changed |= fresh_cover != book.cover_illustration
            book.cover_illustration = fresh_cover
        if body.get("render_quality") in RENDER_PROFILES:
            book.render_quality = body["render_quality"]

        if body.get("adopt_references"):
            # The explicit "yes, move this book to the hero/style as they are
            # now" -- the opposite of what happens by default.
            hero = self.library.get_hero(book.hero_id)
            style = self.library.get_style(book.style_id)
            self.library.lock_references(book, hero, style)

        for edit in body.get("pages") or []:
            page = next((p for p in book.pages if p.index == int(edit.get("index", -1))), None)
            if page is None:
                continue
            if isinstance(edit.get("text"), dict):
                for code, value in edit["text"].items():
                    if str(value) != page.text.get(code):
                        page.text_rev[code] = page.text_rev.get(code, 0) + 1
                        changed = True
                page.text.update({k: str(v) for k, v in edit["text"].items()})
            if edit.get("illustration"):
                fresh_brief = str(edit["illustration"])
                if fresh_brief != page.illustration:
                    page.illustration_rev += 1
                    changed = True
                page.illustration = fresh_brief
            if edit.get("layout") in LAYOUTS and edit["layout"] != page.layout:
                page.layout = edit["layout"]
                changed = True

        # Cast corrections: rename, describe, fix page membership, remove.
        # Renames follow through to every page that names the member --
        # otherwise the pages keep pointing at a name that no longer exists.
        for edit in body.get("cast") or []:
            position = int(edit.get("index", -1))
            if not (0 <= position < len(book.cast)):
                continue
            member = book.cast[position]
            if edit.get("remove"):
                for page in book.pages:
                    page.cast = [n for n in page.cast
                                 if n.lower() != member.name.lower()]
                book.cast.remove(member)
                changed = True
                continue
            fresh_name = str(edit.get("name") or "").strip()
            if fresh_name and fresh_name != member.name:
                old = member.name.lower()
                member.name = fresh_name
                for page in book.pages:
                    page.cast = [fresh_name if n.lower() == old else n
                                 for n in page.cast]
                changed = True
            fresh_desc = str(edit.get("description") or "").strip()
            if fresh_desc and fresh_desc != member.description:
                member.description = fresh_desc
                changed = True
            if isinstance(edit.get("pages"), list):
                wanted = {int(i) for i in edit["pages"]}
                member.pages = sorted(wanted)
                for page in book.pages:
                    named = any(n.lower() == member.name.lower() for n in page.cast)
                    if page.index in wanted and not named:
                        page.cast.append(member.name)
                    elif page.index not in wanted and named:
                        page.cast = [n for n in page.cast
                                     if n.lower() != member.name.lower()]
                changed = True

        if changed:
            book.touch()
        self.library.save_book(book)
        return {"ok": True, "book_id": book.id}

    def book_delete(self, book_id: str, _query, _body) -> dict[str, Any]:
        self.library.delete_book(book_id)
        return {"deleted": book_id}

    def book_preflight(self, book_id: str, query, _body) -> dict[str, Any]:
        """The same readiness check the export job enforces, for display.

        The client calls this to render the bereit/warnung/unbekannt/
        unvollständig state; the export job re-runs it server-side, so a stale
        tab can never sneak a broken book past it.
        """
        book = self.library.get_book(book_id)
        preset = PRINT_PRESETS.get((query.get("preset") or ["print_square"])[0])
        if preset is None:
            preset = PRINT_PRESETS["print_square"]
        languages = [c for c in (query.get("languages") or []) if c in book.languages]
        if not languages:
            languages = book.languages

        root = self.library.book_dir(book.id)

        def resolve(relative: str):
            target = (root / relative).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError(relative)
            return target

        report = preflight.validate_export_readiness(book, preset, languages, resolve)
        font_note = preflight.check_font((query.get("font") or ["georgia"])[0])
        if font_note:
            report["warnings"] = report["warnings"] + [font_note]
        return report

    def book_page_preview(self, book_id: str, query, _body) -> dict[str, Any]:
        """Typeset one page exactly as the export would, and hand back a URL.

        The image lands under preview/ in the book folder (one file per
        page-language-preset combination, overwritten on re-render), so the
        regular file server delivers it and nothing needs streaming.
        """
        from ..book.layout import render_preview

        book = self.library.get_book(book_id)
        preset = PRINT_PRESETS.get((query.get("preset") or ["screen"])[0],
                                   PRINT_PRESETS["screen"])
        languages = [c for c in (query.get("languages") or []) if c in book.languages]
        family = (query.get("font") or ["georgia"])[0]
        index = int((query.get("index") or ["0"])[0])

        root = self.library.book_dir(book.id)

        def resolve(relative: str):
            target = (root / relative).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError(relative)
            return target

        image = render_preview(book, index, preset, resolve, languages, family)
        codes = "-".join(languages) or book.primary_language
        target = root / "preview" / f"set_{index:02d}_{codes}_{preset.key}_{family}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target)

        relative = self.library.relative(target)
        return {"url": f"/library/{relative}?v={int(target.stat().st_mtime)}",
                "index": index, "pages": len(book.pages),
                "print": preset.bleed_mm > 0}

    def book_backup(self, book_id: str, _query, _body) -> dict[str, Any]:
        """Zip the whole book folder -- genuinely self-contained.

        Since the reference lock, a book folder carries its own copies of the
        character sheets under refs/, so this archive really is everything:
        pages, refs, cast sheets, audio, exports. A manifest with hashes makes
        the restore verifiable, and carries the hero/style descriptions so a
        restore on a fresh library still knows what it was drawn from.
        """
        import hashlib
        import json as json_mod
        import zipfile

        from .. import __version__

        book = self.library.get_book(book_id)
        root = self.library.book_dir(book.id)
        folder = self.library.root / "backups"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{slugify(book.display_title())}_{book.id}.zip"

        manifest: dict[str, Any] = {
            "kind": "storytime-book-backup",
            "version": __version__,
            "book_id": book.id,
            "title": book.display_title(),
            "files": {},
        }
        for source in ("hero", "style"):
            try:
                obj = (self.library.get_hero(book.hero_id) if source == "hero"
                       else self.library.get_style(book.style_id))
                manifest[source] = {"id": obj.id, "name": obj.name,
                                    "description": obj.description}
            except FileNotFoundError:
                pass

        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                relative = str(path.relative_to(root)).replace("\\", "/")
                if relative.startswith("preview/"):
                    continue  # regenerable typeset previews, not book substance
                manifest["files"][relative] = hashlib.sha256(
                    path.read_bytes()).hexdigest()
                archive.write(path, relative)
            archive.writestr("manifest.json",
                             json_mod.dumps(manifest, indent=2, ensure_ascii=False))

        relative = self.library.relative(target)
        return {"file": relative, "url": f"/library/{relative}",
                "size_kb": target.stat().st_size // 1024,
                "files": len(manifest["files"])}

    def backups(self, _query, _body) -> dict[str, Any]:
        """The ZIPs sitting in library/backups, restorable with one click."""
        folder = self.library.root / "backups"
        rows = []
        for path in sorted(folder.glob("*.zip")) if folder.is_dir() else []:
            rows.append({"file": f"backups/{path.name}", "name": path.name,
                         "size_kb": path.stat().st_size // 1024})
        return {"backups": rows}

    def book_restore(self, _query, body) -> dict[str, Any]:
        """Bring a backup ZIP back as a book -- validated, never blind.

        The manifest is the contract: every file must be listed and its hash
        must match, and no path may escape the target folder. A backup whose
        book id already exists restores as a copy, so a restore can never
        silently overwrite a live book.
        """
        import hashlib
        import json as json_mod
        import zipfile

        from ..book.models import new_id

        relative = str((body or {}).get("file") or "")
        if not relative.startswith("backups/") or not relative.endswith(".zip"):
            raise ValueError("Bitte eine ZIP aus library/backups auswählen.")
        source = self.library.resolve(relative)
        if not source.is_file():
            raise FileNotFoundError(relative)

        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            if "manifest.json" not in names:
                raise ValueError(
                    "Diese ZIP hat kein Manifest — sie stammt nicht aus der "
                    "Buch-Sicherung (oder ist von vor deren Einführung).")
            manifest = json_mod.loads(archive.read("manifest.json"))
            if manifest.get("kind") != "storytime-book-backup":
                raise ValueError("Das Manifest gehört nicht zu einer Buch-Sicherung.")
            if "book.json" not in manifest.get("files", {}):
                raise ValueError("Der Sicherung fehlt die book.json.")

            book_id = str(manifest.get("book_id") or "")
            if not book_id or (self.library.book_dir(book_id) / "book.json").is_file():
                book_id = new_id("book")  # restore as a copy, never overwrite
            target_root = self.library.book_dir(book_id)

            restored = 0
            for name, expected in manifest["files"].items():
                target = (target_root / name).resolve()
                if not target.is_relative_to(target_root.resolve()):
                    raise ValueError(f"Pfad verlässt den Buchordner: {name}")
                data = archive.read(name)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(
                        f"Die Datei {name} ist beschädigt — Prüfsumme stimmt "
                        "nicht. Die Sicherung wird nicht eingespielt.")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                restored += 1

        # The book.json inside the archive may carry the old id; the folder
        # decides, so a restored copy is consistent with where it lives.
        book_json = target_root / "book.json"
        raw = json_mod.loads(book_json.read_text(encoding="utf-8"))
        raw["id"] = book_id
        book_json.write_text(json_mod.dumps(raw, indent=2, ensure_ascii=False),
                             encoding="utf-8")

        return {"book_id": book_id, "files": restored,
                "title": manifest.get("title", "")}

    # -------------------------------------------------------------------- file

    def file(self, relative: str):
        import mimetypes
        from urllib.parse import unquote

        try:
            target = self.library.resolve(unquote(relative))
        except ValueError as exc:  # path tried to escape the library
            raise FileNotFoundError(relative) from exc
        if not target.is_file():
            raise FileNotFoundError(relative)
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return target.read_bytes(), mime
