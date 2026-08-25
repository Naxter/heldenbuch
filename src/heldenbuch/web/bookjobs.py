"""Job workers for the book app: hero, look, story, pages, sound, export."""

from __future__ import annotations

import base64
import binascii
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ..book import author, handoff, illustrate, look, narrate, preflight, scout
from ..book import cast as cast_mod
from ..book import hero as hero_mod
from ..book import layout as layout_mod
from ..book.library import Library
from ..book.models import AGE_BANDS, Book, Hero, Style, slugify
from ..llm import available_providers
from ..pricing import add as add_spend
from ..pricing import summary as spend_summary
from .jobs import Job

MAX_UPLOAD_BYTES = 16 * 1024 * 1024
#: Ceiling on decoded pixels, well under Pillow's own bomb threshold. A phone
#: photo is around 12 megapixels, so this leaves plenty of headroom.
MAX_UPLOAD_PIXELS = 50_000_000
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def save_upload(item: dict, folder: Path, stem: str) -> Path | None:
    """Write one browser upload to disk, upright.

    Phone photos carry their rotation in an EXIF tag rather than in the pixels.
    Without `exif_transpose` a portrait photo arrives on its side, and you get
    a sideways character with no obvious cause.
    """
    raw = (item.get("data") or "").split(",")[-1]
    try:
        blob = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not blob or len(blob) > MAX_UPLOAD_BYTES:
        return None

    suffix = Path(item.get("name") or "").suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".jpg"

    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{stem}{suffix}"
    target.write_bytes(blob)

    try:
        with Image.open(target) as image:
            # A small file can decode to an enormous bitmap. Pillow raises
            # above roughly 178 megapixels, but the band below that still
            # materialises hundreds of megabytes at four bytes a pixel, so
            # check the declared size before any decoding happens.
            if image.width * image.height > MAX_UPLOAD_PIXELS:
                raise ValueError("image too large")
            image.verify()
        # verify() consumes the file object, so reopen to do the real work.
        with Image.open(target) as image:
            upright = ImageOps.exif_transpose(image)
            if upright is not image or image.getexif():
                upright.convert("RGB").save(target)
    except Exception:
        # Not a usable image. Delete it rather than keeping bytes that will
        # fail later, further from the upload that caused them.
        target.unlink(missing_ok=True)
        return None
    return target


class BookJobs:
    def __init__(self, library: Library, image_backend: str = "openai",
                 text_provider: str = "openai") -> None:
        self.library = library
        self.image_backend = image_backend
        self.text_provider = text_provider

    def workers(self) -> dict[str, Any]:
        return {
            "hero_create": self.hero_create,
            "hero_more": self.hero_more,
            "style_try": self.style_try,
            "style_adopt": self.style_adopt,
            "style_scout": self.style_scout,
            "voice_preview": self.voice_preview,
            "story_write": self.story_write,
            "story_rewrite": self.story_rewrite,
            "story_languages": self.story_languages,
            "book_illustrate": self.book_illustrate,
            "book_check": self.book_check,
            "cast_redraw": self.cast_redraw,
            "book_undo": self.book_undo,
            "book_photo": self.book_photo,
            "book_narrate": self.book_narrate,
            "book_export": self.book_export,
        }

    # ------------------------------------------------------------------ helpers

    def _backend(self, job: Job, style: Style | None = None) -> str:
        if job.params.get("backend"):
            return job.params["backend"]
        if style and style.recommended_backend:
            return style.recommended_backend
        return self.image_backend

    def _provider(self, job: Job) -> str:
        return job.params.get("provider") or self.text_provider

    def _checker(self, job: Job, image_backend: str) -> str:
        """Prefer a checker that did not draw the picture it is grading."""
        if job.params.get("check_provider"):
            return job.params["check_provider"]
        return illustrate.pick_check_provider(image_backend, available_providers())

    def _book_resolver(self, book: Book):
        root = self.library.book_dir(book.id)

        def resolve(relative: str) -> Path:
            target = (root / relative).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError(f"path escapes the book folder: {relative}")
            return target

        return resolve

    def _locked_sheet(self, book: Book, hero: Hero, style: Style) -> Path:
        """The styled sheet this book draws from -- its own frozen copy.

        Books from before the lock existed adopt whatever they are using
        today, once, and are frozen from then on. Only if the frozen copy is
        gone from disk does the current library state step in.
        """
        if not book.styled_sheet and not book.hero_sheet:
            self.library.lock_references(book, hero, style)
            self.library.save_book(book)

        resolve = self._book_resolver(book)
        for relative in (book.styled_sheet, book.hero_sheet):
            if relative:
                candidate = resolve(relative)
                if candidate.is_file():
                    return candidate

        fallback = style.sheets.get(hero.id) or hero.sheet
        if not fallback:
            raise ValueError("Es fehlt das Charakterblatt für diesen Helden und Stil.")
        return self.library.resolve(fallback)

    @staticmethod
    def _report_shared_spend(spend: dict, log) -> None:
        """What a hero or a style cost. Said once, where it is spent.

        This money buys something every book made from it reuses, so it is
        never charged to a single book -- but it used to be reported nowhere
        at all, which made a first book look cheaper than it was.
        """
        total = spend_summary(spend)
        if total["calls"]:
            log(f"Dafür bezahlt: ~{total['usd']:.2f} $ / ~{total['eur']:.2f} € "
                f"({total['images']} Bilder, geschätzt). Das gilt für jedes "
                "Buch, das damit gemacht wird.")
            log("")

    @staticmethod
    def _report_spend(book: Book, log) -> None:
        total = spend_summary(book.spend)
        if total["calls"]:
            log("")
            log(f"Bisher für dieses Buch: ~{total['usd']:.2f} $ / ~{total['eur']:.2f} € "
                f"({total['images']} Bilder, {total['calls']} Aufrufe, geschätzt)")

    # --------------------------------------------------------------------- hero

    def hero_create(self, job: Job, log) -> None:
        params = job.params
        name = (params.get("name") or "").strip()
        hero = Hero(name=name, description=(params.get("description") or "").strip())

        folder = self.library.hero_dir(hero.id)
        folder.mkdir(parents=True, exist_ok=True)

        photos: list[Path] = []
        offered = (params.get("photos") or [])[:4]
        for index, item in enumerate(offered, start=1):
            saved = save_upload(item, folder / "photos", f"photo_{index}")
            if saved:
                photos.append(saved)
        hero.photos = [self.library.relative(p) for p in photos]
        if len(photos) < len(offered):
            # A rejected upload used to vanish without a word.
            log(f"{len(offered) - len(photos)} Datei(en) waren keine lesbaren "
                "Bilder und wurden verworfen.")

        if photos:
            hero.source = "photo"
            log(f"{len(photos)} Foto(s) gespeichert und aufrecht gedreht — "
                "sie gehen nur für diesen Schritt raus.")
            log("Ich schaue mir an, was ein Illustrator wissen muss …")
            hero.description = hero_mod.describe_from_photos(
                photos, name, age_hint=params.get("age_hint", ""),
                provider=self._provider(job), spend=hero.spend,
            )
            log("")
            log(hero.description)
            log("")
        elif not hero.description:
            raise ValueError("Ohne Foto brauche ich wenigstens eine kurze Beschreibung.")

        self.library.save_hero(hero)

        made = hero_mod.generate_variants(
            hero, folder, count=max(1, min(4, int(params.get("variants") or 3))),
            backend_name=self._backend(job), model=job.params.get("model"),
            photo_paths=photos,
            spend=lambda usage: add_spend(hero.spend, usage, "hero_sheet"),
            log=log,
        )
        hero.variants = [self.library.relative(p) for p in made]
        hero.sheet = hero.variants[0]
        self.library.save_hero(hero)

        job.result["hero_id"] = hero.id
        log("")
        self._report_shared_spend(hero.spend, log)
        log("Fertig. Such dir die Version aus, die am besten passt.")

    def hero_more(self, job: Job, log) -> None:
        hero = self.library.get_hero(job.params["hero_id"])
        folder = self.library.hero_dir(hero.id)
        photos = [p for p in (self.library.resolve(x) for x in hero.photos) if p.is_file()]

        made = hero_mod.generate_variants(
            hero, folder, count=max(1, min(4, int(job.params.get("variants") or 2))),
            backend_name=self._backend(job), model=job.params.get("model"),
            photo_paths=photos,
            spend=lambda usage: add_spend(hero.spend, usage, "hero_sheet"),
            log=log,
        )
        hero.variants += [self.library.relative(p) for p in made]
        self.library.save_hero(hero)
        job.result["hero_id"] = hero.id

    # -------------------------------------------------------------------- style

    def style_try(self, job: Job, log) -> None:
        """Build a style and immediately show the hero rendered in it."""
        params = job.params
        hero = self.library.get_hero(params["hero_id"])
        if not hero.sheet:
            raise ValueError("Für diesen Helden gibt es noch kein Charakterblatt.")

        preset_key = params.get("preset")
        wish = (params.get("wish") or "").strip()
        upload = (params.get("reference") or [None])
        upload = upload[0] if isinstance(upload, list) else params.get("reference")

        style: Style
        reference_path: Path | None = None

        if preset_key:
            style = look.preset_style(preset_key)
        elif upload:
            style = Style(preset="image")
            folder = self.library.style_dir(style.id)
            reference_path = save_upload(upload, folder, "reference")
            if reference_path is None:
                raise ValueError("Diese Bilddatei konnte ich nicht lesen.")
            log("Ich schaue mir an, wie dieses Bild gemacht ist …")
            described = look.describe_style_from_image(
                reference_path, provider=self._provider(job), spend=style.spend
            )
            style.name = described["name"]
            style.description = described["description"]
            style.reference = self.library.relative(reference_path)
            log("")
            log(style.description)
            log("")
        elif wish:
            log(f"Ich übersetze „{wish}“ in eine Stilbeschreibung …")
            # The style does not exist yet, so its ledger starts here and is
            # carried over once it does.
            wish_spend: dict = {}
            described = look.describe_custom_style(wish, provider=self._provider(job),
                                                   spend=wish_spend)
            style = Style(name=described["name"], description=described["description"],
                          preset="custom", spend=wish_spend)
            log("")
            log(style.description)
            log("")
        else:
            raise ValueError("Bitte einen Stil auswählen, beschreiben oder ein Bild hochladen.")

        folder = self.library.style_dir(style.id)
        folder.mkdir(parents=True, exist_ok=True)
        self.library.save_style(style)

        base = self.library.resolve(hero.sheet)
        previews = look.generate_previews(
            hero, style, base, folder,
            count=max(1, min(3, int(params.get("previews") or 2))),
            backend_name=self._backend(job), model=params.get("model"),
            style_reference=reference_path,
            spend=lambda usage: add_spend(style.spend, usage, "style_preview"),
            log=log,
        )
        style.previews = [self.library.relative(p) for p in previews]

        log("Ich zeichne das Charakterblatt in diesem Stil …")
        styled = folder / f"sheet_{hero.id}.png"
        hero_mod.generate_styled_sheet(
            hero, style, base, styled,
            backend_name=self._backend(job), model=params.get("model"),
            spend=lambda usage: add_spend(style.spend, usage, "styled_sheet"))
        style.sheets[hero.id] = self.library.relative(styled)
        self.library.save_style(style)

        job.result["style_id"] = style.id
        log("")
        self._report_shared_spend(style.spend, log)
        log("Fertig. Wenn dir der Look gefällt, geht es zur Geschichte.")

    def style_adopt(self, job: Job, log) -> None:
        """Set up an existing style for another hero.

        Styles are locked to a hero through the styled character sheet, so a
        sibling could not reuse a style without rebuilding it from scratch.
        This draws only the missing sheet -- one image instead of four.
        """
        style = self.library.get_style(job.params["style_id"])
        hero = self.library.get_hero(job.params["hero_id"])
        if not hero.sheet:
            raise ValueError("Für diesen Helden gibt es noch kein Charakterblatt.")
        if style.sheets.get(hero.id):
            job.result["style_id"] = style.id
            log("Dieser Stil ist für diesen Helden schon eingerichtet.")
            return

        log(f"Ich zeichne das Charakterblatt von {hero.name or 'dem Helden'} "
            f"im Stil „{style.name}“ …")
        styled = self.library.style_dir(style.id) / f"sheet_{hero.id}.png"
        hero_mod.generate_styled_sheet(
            hero, style, self.library.resolve(hero.sheet), styled,
            backend_name=self._backend(job, style), model=job.params.get("model"),
            spend=lambda usage: add_spend(style.spend, usage, "styled_sheet"),
        )
        style.sheets[hero.id] = self.library.relative(styled)
        self.library.save_style(style)
        job.result["style_id"] = style.id
        log("Fertig — der Stil steht jetzt auch für diesen Helden bereit.")

    def style_scout(self, job: Job, log) -> None:
        """Ask the benchmark which image model suits this style best."""
        style = self.library.get_style(job.params["style_id"])
        hero = self.library.get_hero(job.params["hero_id"])
        sheet_rel = style.sheets.get(hero.id) or hero.sheet
        if not sheet_rel:
            raise ValueError("Es fehlt das Charakterblatt für diese Kombination.")

        log("Ich zeichne dieselben zwei Testszenen mit jedem verfügbaren Bilddienst.")
        log("")
        outcome = scout.run(
            hero, style, self.library.resolve(sheet_rel),
            self.library.style_dir(style.id) / "scout",
            check_provider=self._checker(job, self.image_backend),
            spend=style.spend,
            log=log, should_stop=lambda: job.cancelled,
        )
        # A run that found no clear winner must not leave an old recommendation
        # standing, and must not invent a new one.
        style.recommended_backend = outcome["winner"]
        style.scout = outcome
        self.library.save_style(style)
        job.result["style_id"] = style.id
        job.result["winner"] = outcome["winner"]

    # -------------------------------------------------------------------- story

    def story_write(self, job: Job, log) -> None:
        params = job.params
        hero = self.library.get_hero(params["hero_id"])
        style = self.library.get_style(params["style_id"])
        languages = [c for c in (params.get("languages") or ["de"]) if c]
        age = params.get("age") or "4-5"
        rhyme = bool(params.get("rhyme"))

        log(f"Ich schreibe eine Geschichte für {AGE_BANDS.get(age, {}).get('label', age)}"
            f"{' in Reimen' if rhyme else ''} …")
        if len(languages) > 1:
            log(f"in {len(languages)} Sprachen, jede eigenständig geschrieben — nicht übersetzt.")

        story = author.write_story(
            hero, idea=params.get("idea", ""), age=age, languages=languages,
            pages=int(params["pages"]) if params.get("pages") else None,
            rhyme=rhyme, provider=self._provider(job),
        )

        if params.get("revise", True):
            log("Ich lese den Entwurf noch einmal laut und feile daran …")
            story = author.revise(story, hero, age, languages, rhyme=rhyme,
                                  provider=self._provider(job))
            for note in story.get("revision_notes", [])[:6]:
                log(f"  · {note}")

        book = Book(
            hero_id=hero.id, style_id=style.id,
            title=story["title"], dedication=story["dedication"],
            cover_illustration=story["cover_illustration"],
            idea=params.get("idea", ""), age=age, languages=languages,
            pages=story["pages"], cast=story["cast"], rhyme=rhyme,
            render_quality=params.get("render_quality") or "draft",
        )
        # Freeze the exact references this book will be drawn from. Changing
        # the hero or the style tomorrow must not change this book.
        self.library.lock_references(book, hero, style)
        self.library.save_book(book)
        job.result["book_id"] = book.id

        primary = book.primary_language
        log("")
        log(f"„{book.title.get(primary, '')}“ — {len(book.pages)} Seiten")
        if book.cast:
            people = ", ".join(f"{c.name} ({c.kind})" for c in book.cast)
            log(f"Weitere Figuren und Orte: {people}")
        log("")
        for page in book.pages:
            marker = {"wordless": " [ohne Text]", "vignette": " [klein]",
                      "split": " [Bild/Text]"}.get(page.layout, "")
            log(f"  {page.index:>2}. {page.text.get(primary, '')[:100]}{marker}")

    def story_rewrite(self, job: Job, log) -> None:
        book = self.library.get_book(job.params["book_id"])
        hero = self.library.get_hero(book.hero_id)
        index = int(job.params["index"])
        page = next((p for p in book.pages if p.index == index), None)
        if page is None:
            raise ValueError(f"Seite {index} gibt es nicht.")

        log(f"Ich schreibe Seite {index} neu …")
        before_text = dict(page.text)
        before_brief = page.illustration
        author.rewrite_page(book, hero, page, note=job.params.get("note", ""),
                            provider=self._provider(job))
        # Whatever actually changed makes its derived output stale.
        for code in book.languages:
            if page.text.get(code) != before_text.get(code):
                page.text_rev[code] = page.text_rev.get(code, 0) + 1
        if page.illustration != before_brief:
            page.illustration_rev += 1
        book.touch()
        self.library.save_book(book)
        job.result["book_id"] = book.id
        log("")
        log(page.text.get(book.primary_language, ""))

    def story_languages(self, job: Job, log) -> None:
        book = self.library.get_book(job.params["book_id"])
        hero = self.library.get_hero(book.hero_id)
        codes = [c for c in (job.params.get("languages") or []) if c]

        log("Ich schreibe das Buch in weiteren Sprachen. Die Bilder bleiben, "
            "nur die Worte sind neu.")
        author.add_languages(book, hero, codes, provider=self._provider(job))
        book.touch()  # existing exports are missing the new language now
        self.library.save_book(book)
        job.result["book_id"] = book.id
        log(f"Jetzt vorhanden: {', '.join(book.languages)}")

    # -------------------------------------------------------------------- pages

    def book_illustrate(self, job: Job, log) -> None:
        params = job.params
        book = self.library.get_book(params["book_id"])
        hero = self.library.get_hero(book.hero_id)
        style = self.library.get_style(book.style_id)

        if params.get("render_quality"):
            book.render_quality = params["render_quality"]

        sheet = self._locked_sheet(book, hero, style)

        backend = self._backend(job, style)

        # Redraw only the cover -- one image, not the whole book. The full
        # renderer only touches the cover as part of "draw everything".
        if params.get("cover_only"):
            log("Nur das Titelbild wird neu gezeichnet …")
            target = self.library.book_dir(book.id) / "pages" / "cover.png"
            result = illustrate.draw_cover(
                book, hero, style, sheet, target, backend_name=backend,
                model=params.get("model"),
                output=illustrate.output_for(book.render_quality),
                # The cast belongs on the cover too, and especially here: this
                # is the button someone presses *because* the cover is wrong.
                members=book.cast, resolve=self._book_resolver(book),
            )
            book.cover = f"pages/{target.name}"
            add_spend(book.spend, result.usage, "cover")
            self.library.save_book(book)
            job.result["book_id"] = book.id
            log("Fertig.")
            self._report_spend(book, log)
            return

        checker = self._checker(job, backend)
        profile = illustrate.RENDER_PROFILES.get(book.render_quality, {})
        log(profile.get("label", book.render_quality))
        checking = bool(params.get("check", True))
        if not checking:
            log(f"Gezeichnet von {backend}. Ähnlichkeitsprüfung ist aus.")
        elif checker == backend:
            log(f"Gezeichnet und geprüft von {checker} — derselbe Dienst. Mit einem "
                "zweiten Schlüssel wäre das Urteil unabhängiger.")
        else:
            log(f"Gezeichnet von {backend}, geprüft von {checker}.")
        log("")

        style_reference = None
        if style.reference:
            try:
                candidate = self.library.resolve(style.reference)
                style_reference = candidate if candidate.is_file() else None
            except (ValueError, FileNotFoundError):
                style_reference = None

        pending = [c for c in book.cast if not c.sheet]
        if pending:
            log(f"Zuerst {len(pending)} Referenzblatt/-blätter für Nebenfiguren und Orte:")
            cast_mod.generate_all(
                book.cast, style, self.library.book_dir(book.id) / "cast",
                backend_name=backend, style_reference=style_reference,
                relative_to=self.library.book_dir(book.id),
                spend=lambda usage: add_spend(book.spend, usage, "cast"),
                log=log, should_stop=lambda: job.cancelled,
            )
            self.library.save_book(book)
            log("")

        def report(done: int, total: int) -> None:
            job.progress = (done, total)
            # Saving per page lets finished pictures appear in the browser
            # while the rest are still being drawn. The render's copy is
            # minutes old by now: adopt any text edited in the meantime.
            self.library.save_book(book, adopt="editorial")

        only = [int(i) for i in params.get("only") or []] or None

        # Batch: everything to Google at once, half price, no hurry. Only
        # gemini offers this; the interactive path below stays the default.
        if params.get("batch") and backend == "gemini":
            try:
                illustrate.illustrate_book_batch(
                    book, hero, style, sheet,
                    pages_dir=self.library.book_dir(book.id) / "pages",
                    model=params.get("model"),
                    check=bool(params.get("check", True)),
                    check_provider=checker,
                    only=only,
                    redraw=bool(params.get("redraw")),
                    # The UI offers the cap and the batch checkbox side by
                    # side and sends both; only the interactive path was
                    # reading it, so the cap did nothing in batch mode.
                    budget_usd=(float(params["budget_usd"])
                                if params.get("budget_usd") else None),
                    resolve=self._book_resolver(book),
                    on_progress=report,
                    # So the batch handle reaches disk before the wait starts.
                    save=lambda b: self.library.save_book(b, adopt="editorial"),
                    log=log, should_stop=lambda: job.cancelled,
                )
            finally:
                self.library.save_book(book, adopt="editorial")
            job.result["book_id"] = book.id
            flagged_now = illustrate.flagged_pages(book)
            log("")
            if flagged_now:
                log("Diese Seiten solltest du dir ansehen: "
                    + ", ".join(map(str, flagged_now)))
            else:
                log("Alle Seiten sehen stimmig aus.")
            self._report_spend(book, log)
            return

        try:
            illustrate.illustrate_book(
                book, hero, style, sheet,
                pages_dir=self.library.book_dir(book.id) / "pages",
                backend_name=backend,
                model=params.get("model"),
                check=bool(params.get("check", True)),
                check_provider=checker,
                only=only,
                redraw=bool(params.get("redraw")),
                workers=max(1, min(6, int(params.get("workers") or 4))),
                auto_retry=bool(params.get("auto_retry", True)),
                budget_usd=(float(params["budget_usd"]) if params.get("budget_usd") else None),
                resolve=self._book_resolver(book),
                on_progress=report,
                log=log, should_stop=lambda: job.cancelled,
            )
        finally:
            # A crash halfway through must not lose the pages -- and the
            # money -- that were already spent.
            self.library.save_book(book, adopt="editorial")
        job.result["book_id"] = book.id

        flagged = illustrate.flagged_pages(book)
        log("")
        if flagged:
            log(f"Diese Seiten solltest du dir ansehen: {', '.join(map(str, flagged))}")
        else:
            log("Alle Seiten sehen stimmig aus.")
        self._report_spend(book, log)

    def cast_redraw(self, job: Job, log) -> None:
        """Redraw one cast member's reference sheet -- after the description
        was corrected, or because the first draw missed."""
        book = self.library.get_book(job.params["book_id"])
        position = int(job.params["cast_index"])
        if not (0 <= position < len(book.cast)):
            raise ValueError("Diese Figur gibt es nicht mehr.")
        member = book.cast[position]
        style = self.library.get_style(book.style_id)

        style_reference = None
        if style.reference:
            try:
                candidate = self.library.resolve(style.reference)
                style_reference = candidate if candidate.is_file() else None
            except (ValueError, FileNotFoundError):
                style_reference = None

        target = self.library.book_dir(book.id) / "cast" / f"cast_{position + 1:02d}.png"
        kind = "Ort" if member.kind == "place" else "Figur"
        log(f"{kind}: {member.name} wird neu gezeichnet …")
        result = cast_mod.generate_sheet(
            member, style, target,
            backend_name=self._backend(job, style), model=job.params.get("model"),
            style_reference=style_reference,
        )
        # The draw took long enough for someone to edit text meanwhile:
        # apply the result to a fresh copy instead of saving the stale one.
        book = self.library.get_book(book.id)
        if position < len(book.cast):
            book.cast[position].sheet = f"cast/{target.name}"
        add_spend(book.spend, result.usage, "cast")
        self.library.save_book(book)
        job.result["book_id"] = book.id
        log("Fertig — Seiten, auf denen "
            f"{member.name} vorkommt, zeigen die Änderung erst nach dem Neuzeichnen.")

    def book_check(self, job: Job, log) -> None:
        """Re-run the consistency check without redrawing anything.

        This is the recovery path for "unknown" pages: when the checker was
        down mid-render, the pictures are fine but unverified. Checking again
        costs a vision call per page, not an image.
        """
        book = self.library.get_book(job.params["book_id"])
        hero = self.library.get_hero(book.hero_id)
        style = self.library.get_style(book.style_id)
        sheet = self._locked_sheet(book, hero, style)
        checker = self._checker(job, self._backend(job, style))

        wanted = {int(i) for i in job.params.get("only") or []} or None
        resolve = self._book_resolver(book)
        pages = [p for p in sorted(book.pages, key=lambda p: p.index)
                 if p.image and (wanted is None or p.index in wanted)]
        if not pages:
            raise ValueError("Keine gezeichnete Seite zum Prüfen.")

        log(f"{len(pages)} Seite(n) werden geprüft — von {checker}.")
        for page in pages:
            if job.cancelled:
                log("abgebrochen")
                break
            target = resolve(page.image)
            if not target.is_file():
                continue
            page.check = illustrate.check_page(
                target, sheet, hero, provider=checker, scene=page.illustration)
            status = illustrate.check_status(page)
            word = {"passed": "in Ordnung", "failed": "beanstandet",
                    "unknown": "Prüfung fehlgeschlagen"}.get(status, status)
            log(f"  Seite {page.index}: {word}")

        self.library.save_book(book)
        job.result["book_id"] = book.id
        flagged = illustrate.flagged_pages(book)
        log("")
        log("Diese Seiten solltest du dir ansehen: " + ", ".join(map(str, flagged))
            if flagged else "Alle geprüften Seiten sehen stimmig aus.")

    def book_undo(self, job: Job, log) -> None:
        """Put back the previous version of one page."""
        book = self.library.get_book(job.params["book_id"])
        index = int(job.params["index"])
        page = next((p for p in book.pages if p.index == index), None)
        if page is None or not page.history:
            raise ValueError(f"Für Seite {index} gibt es keine frühere Fassung.")

        resolve = self._book_resolver(book)
        previous = resolve(page.history.pop())
        current = resolve(page.image) if page.image else None
        if current:
            shutil.copy2(previous, current)
        previous.unlink(missing_ok=True)
        page.check = {}
        self.library.save_book(book)
        job.result["book_id"] = book.id
        log(f"Seite {index}: vorherige Fassung wiederhergestellt.")

    def book_photo(self, job: Job, log) -> None:
        """Attach a real photograph and a line for the last page."""
        book = self.library.get_book(job.params["book_id"])
        upload = job.params.get("photo")
        caption = job.params.get("caption") or {}

        if upload:
            saved = save_upload(upload, self.library.book_dir(book.id) / "pages", "photo_page")
            if saved is None:
                raise ValueError("Diese Bilddatei konnte ich nicht lesen.")
            book.photo_page["image"] = f"pages/{saved.name}"
            log("Foto gespeichert und aufrecht gedreht.")
        if isinstance(caption, dict) and caption:
            book.photo_page.setdefault("caption", {})
            book.photo_page["caption"].update({k: str(v) for k, v in caption.items()})
        if job.params.get("remove"):
            book.photo_page = {}
            log("Fotoseite entfernt.")

        self.library.save_book(book)
        job.result["book_id"] = book.id

    # -------------------------------------------------------------------- sound

    def book_narrate(self, job: Job, log) -> None:
        book = self.library.get_book(job.params["book_id"])
        try:
            narrate.narrate_book(
                book,
                audio_dir=self.library.book_dir(book.id) / "audio",
                languages=job.params.get("languages"),
                voice=job.params.get("voice") or "coral",
                speed=float(job.params.get("speed") or 0.95),
                redo=bool(job.params.get("redo")),
                log=log, should_stop=lambda: job.cancelled,
            )
        finally:
            # Keep what was already narrated and paid; adopt any text edited
            # while the narration ran.
            self.library.save_book(book, adopt="editorial")
        job.result["book_id"] = book.id
        self._report_spend(book, log)

    def voice_preview(self, job: Job, log) -> None:
        """One short sample per voice, cached forever, so the voice menu means
        something before money is spent on a whole book."""
        voice = job.params.get("voice") or "coral"
        if voice not in narrate.VOICES:
            raise ValueError(f"Unbekannte Stimme: {voice}")
        target = self.library.root / "voices" / f"{voice}.mp3"
        if not target.is_file():
            log(f"Stimmprobe für „{voice}“ wird erzeugt …")
            narrate.speak(
                "Hallo! So klinge ich, wenn ich abends eine Geschichte vorlese.",
                target, voice=voice, language="de",
            )
        job.result["url"] = f"/library/{self.library.relative(target)}"
        log("Fertig.")

    # ------------------------------------------------------------------- export

    def book_export(self, job: Job, log) -> None:
        params = job.params
        book = self.library.get_book(params["book_id"])
        preset = layout_mod.PRESETS.get(params.get("preset") or "print_square")
        if preset is None:
            raise ValueError(f"Unbekanntes Format: {params.get('preset')}")

        languages = [c for c in (params.get("languages") or book.languages)
                     if c in book.languages]
        if not languages:
            raise ValueError("Keine gültige Sprache ausgewählt.")
        family = params.get("font") or "georgia"
        book_root = self.library.book_dir(book.id)
        folder = book_root / "export"
        stem = slugify(book.display_title())
        resolve = self._book_resolver(book)

        # The gate: nothing is written until the book survives the preflight.
        # The client runs the same check for display, but the job is the
        # authority -- a stale browser tab must not produce a broken print file.
        report = preflight.validate_export_readiness(
            book, preset, languages, resolve,
            allow_unknown=bool(params.get("allow_unknown")),
        )
        for mark, group in (("✗", "errors"), ("?", "unknowns"), ("⚠", "warnings")):
            for item in report[group]:
                log(f"{mark} {item['text']}")
        font_note = preflight.check_font(family)
        if font_note:
            log(f"⚠ {font_note['text']}")
        if not report["ok"]:
            raise ValueError(
                "Der Export ist blockiert — das Buch ist noch nicht druckreif. "
                "Die Punkte oben zeigen, was fehlt."
            )
        if report["warnings"]:
            log("")

        # One PDF per language -- or, for a bilingual child, every chosen
        # language together on each page in a single file.
        combine = bool(params.get("combine")) and len(languages) > 1
        wanted = ([(languages[0], languages[1:])] if combine
                  else [(code, None) for code in languages])

        results = []
        for language, secondary in wanted:
            label = "+".join([language] + (secondary or []))
            log(f"{preset.name} — {label}")
            target = folder / f"{stem}_{label.replace('+', '-')}_{preset.key}.pdf"
            result = layout_mod.export_pdf(book, language, preset, resolve, target,
                                           family=family, log=log, secondary=secondary)
            for warning in result["warnings"]:
                log(f"  ⚠ {warning}")
            results.append({
                "language": result["language"],
                "language_name": result["language_name"],
                "pages": result["pages"],
                "file": self.library.relative(target),
                "size_kb": target.stat().st_size // 1024,
                "warnings": result["warnings"],
            })

        cover_info: dict[str, Any] | None = None
        if preset.bleed_mm > 0:
            interior = results[0]["pages"]
            log("")
            log("Umschlag mit Rücken und Rückseite …")
            log("  " + handoff.spine_note(preset, interior))
            wrap, info = layout_mod.render_wrap_cover(
                book, languages[0], preset, resolve, interior, family=family
            )
            wrap_path = folder / f"{stem}_umschlag_{preset.key}.pdf"
            wrap.save(wrap_path, "PDF", resolution=float(preset.dpi))
            info["file"] = self.library.relative(wrap_path)
            if info.get("note"):
                log(f"  ⚠ {info['note']}")

            front = folder / f"{stem}_cover_{preset.key}.jpg"
            layout_mod.export_cover_image(book, languages[0], preset, resolve, front,
                                          family=family)
            info["front_file"] = self.library.relative(front)
            cover_info = info
            job.result["cover"] = info["file"]

        provider = params.get("provider_hint") or "lulu"
        # Only a file going to a print shop needs shop instructions. The
        # screen and home-printer presets have no bleed, no spine and no
        # provider, so the sheet was pure noise beside them.
        note_path = None
        if preset.bleed_mm > 0:
            # The weakest page decides what the sheet may claim. Every
            # illustration is scaled up to fill the page, so the PDF always
            # *reports* the preset's dpi regardless of what the artwork holds.
            measured = [layout_mod.effective_dpi(p, preset)
                        for p in (resolve(page.image) for page in book.pages if page.image)
                        if p.is_file()]
            note = handoff.sheet(book, preset, results, cover_info, provider=provider,
                                 measured_dpi=min(measured) if measured else None)
            note_path = folder / f"{stem}_druckerei_{provider}.md"
            note_path.write_text(note, encoding="utf-8")
            log("")
            log(f"Anleitung für {handoff.PROVIDERS[provider]['name']} geschrieben.")

        # The files on disk now match this content revision -- until the next
        # edit, which makes them visibly stale again.
        book.export_rev = book.content_rev
        self.library.save_book(book)

        job.result["book_id"] = book.id
        job.result["exports"] = results
        if note_path is not None:
            job.result["handoff"] = self.library.relative(note_path)
        log("")
        log("Fertig.")
        self._report_spend(book, log)
