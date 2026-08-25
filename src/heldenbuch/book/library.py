"""Where heroes, styles and books live on disk.

    library/
      heroes/<hero_id>/hero.json, photos/, sheet_*.png
      styles/<style_id>/style.json, preview_*.png
      books/<book_id>/book.json, pages/page_01.png, export/*.pdf

Plain folders and JSON, nothing to migrate and nothing to install. Everything
the app shows can be read with a file browser, and a book can be backed up by
copying one directory.
"""

from __future__ import annotations

import shutil
import threading
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

from .models import Book, Hero, Style, load_json, save_json

T = TypeVar("T")


def _build(cls: type[T], raw: dict[str, Any]) -> T:
    """Construct a dataclass from stored JSON, ignoring keys it no longer has.

    Books written by an earlier version must keep opening after the model
    grows a field or loses one.
    """
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in raw.items() if k in known})  # type: ignore[return-value]


class Library:
    def __init__(self, root: Path) -> None:
        self.root = root
        for name in ("heroes", "styles", "books"):
            (root / name).mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def book_lock(self, book_id: str) -> threading.Lock:
        """One lock per book, shared by every thread in this process."""
        with self._locks_guard:
            return self._locks.setdefault(str(book_id), threading.Lock())

    # ---------------------------------------------------------------- paths

    #: folder name -> what one of them is called, for error messages
    _KINDS = {"heroes": "hero", "styles": "style", "books": "book"}

    def _member_dir(self, kind: str, ident: str) -> Path:
        """The folder for one hero, style or book -- and nothing else.

        The id arrives from a URL and ends up in `shutil.rmtree`, so it has to
        be a single path segment. `..`, a backslash, an absolute path or a
        Windows drive-relative path like `C:foo` all resolve somewhere other
        than directly inside `library/<kind>/`, and every one of them is
        refused here rather than at each call site.
        """
        ident = str(ident)
        # A backslash separates paths on Windows and is an ordinary filename
        # character on Linux. Refuse it on both, explicitly: the resolve()
        # check below only catches it where it separates, and a library must
        # stay safe to copy between the two systems.
        if not ident or "/" in ident or "\\" in ident or ident in (".", ".."):
            raise ValueError(f"invalid {self._KINDS.get(kind, kind)} id: {ident!r}")
        parent = (self.root / kind).resolve()
        target = (parent / ident).resolve()
        if target.parent != parent or target == parent:
            raise ValueError(f"invalid {self._KINDS.get(kind, kind)} id: {ident!r}")
        return target

    def hero_dir(self, hero_id: str) -> Path:
        return self._member_dir("heroes", hero_id)

    def style_dir(self, style_id: str) -> Path:
        return self._member_dir("styles", style_id)

    def book_dir(self, book_id: str) -> Path:
        return self._member_dir("books", book_id)

    def relative(self, path: Path) -> str:
        """A path the web layer can serve, using forward slashes on every OS."""
        return str(path.resolve().relative_to(self.root.resolve())).replace("\\", "/")

    def resolve(self, relative: str) -> Path:
        """Turn a stored relative path back into an absolute one, safely."""
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root.resolve()):
            raise ValueError(f"path escapes the library: {relative}")
        return target

    # ---------------------------------------------------------------- heroes

    def save_hero(self, hero: Hero) -> Hero:
        save_json(self.hero_dir(hero.id) / "hero.json", hero.to_dict())
        return hero

    def get_hero(self, hero_id: str) -> Hero:
        # An id that is not a single path segment names nothing, which to a
        # reader is the same as a hero that is not there. Only the writing and
        # deleting paths need the louder ValueError.
        try:
            path = self.hero_dir(hero_id) / "hero.json"
        except ValueError as exc:
            raise FileNotFoundError(str(exc)) from exc
        if not path.is_file():
            raise FileNotFoundError(f"no hero {hero_id}")
        return _build(Hero, load_json(path))

    def heroes(self) -> list[Hero]:
        found = []
        for path in sorted((self.root / "heroes").glob("*/hero.json")):
            try:
                found.append(_build(Hero, load_json(path)))
            except (TypeError, ValueError, OSError):
                # OSError: the folder can vanish between glob and read when a
                # delete races a listing; one stale entry must not 404 the
                # whole shelf.
                continue
        return sorted(found, key=lambda h: h.created, reverse=True)

    def delete_hero(self, hero_id: str) -> None:
        shutil.rmtree(self.hero_dir(hero_id), ignore_errors=True)

    # ---------------------------------------------------------------- styles

    def save_style(self, style: Style) -> Style:
        save_json(self.style_dir(style.id) / "style.json", style.to_dict())
        return style

    def get_style(self, style_id: str) -> Style:
        try:
            path = self.style_dir(style_id) / "style.json"
        except ValueError as exc:
            raise FileNotFoundError(str(exc)) from exc
        if not path.is_file():
            raise FileNotFoundError(f"no style {style_id}")
        return _build(Style, load_json(path))

    def styles(self) -> list[Style]:
        found = []
        for path in sorted((self.root / "styles").glob("*/style.json")):
            try:
                found.append(_build(Style, load_json(path)))
            except (TypeError, ValueError, OSError):
                # Same race as heroes(): a concurrent delete is one missing
                # tile, not a failed listing.
                continue
        return sorted(found, key=lambda s: s.created, reverse=True)

    def delete_style(self, style_id: str) -> None:
        shutil.rmtree(self.style_dir(style_id), ignore_errors=True)

    # ----------------------------------------------------------------- books

    def lock_references(self, book: Book, hero: Hero, style: Style) -> Book:
        """Copy the exact reference sheets into the book's own folder.

        A book depends on one specific styled character sheet. If it merely
        *pointed* at the hero/style folders, picking a new hero variant or
        deleting a style would silently change or break every book drawn from
        the old one. A copy under refs/ freezes what this book looks like;
        `ref_sources` remembers the origin so the UI can offer an update when
        the hero moves on.
        """
        refs = self.book_dir(book.id) / "refs"
        styled_rel = style.sheets.get(hero.id) if style else None
        for kind, source_rel in (("hero", hero.sheet if hero else None),
                                 ("styled", styled_rel)):
            if not source_rel:
                continue
            try:
                source = self.resolve(source_rel)
            except ValueError:
                continue
            if not source.is_file():
                continue
            refs.mkdir(parents=True, exist_ok=True)
            target = refs / f"{kind}_sheet.png"
            shutil.copy2(source, target)
            relative = f"refs/{target.name}"
            if kind == "hero":
                book.hero_sheet = relative
            else:
                book.styled_sheet = relative
            book.ref_sources[kind] = source_rel
        return book

    #: Fields the person editing writes. A background job that saves a copy it
    #: has held for minutes adopts these from disk first, so a text corrected
    #: mid-render is not reverted by the render's next save.
    # `cast` is deliberately absent: a cast member is written from both sides
    # like a page is, so it is merged field by field below. Adopting the whole
    # list threw away a sheet the render had just paid for whenever an editor
    # save landed between two of the job's saves.
    _EDITORIAL_BOOK = ("title", "idea", "age", "languages", "dedication",
                       "rhyme", "render_quality", "narration_voice",
                       "photo_page", "blurb", "content_rev")
    # `expression`, `direction` and `setting` are brief fields like
    # `illustration`: the person writes them, so a job holding a minutes-old
    # copy must adopt them or its next save silently reverts the edit.
    _EDITORIAL_PAGE = ("text", "text_rev", "illustration", "illustration_rev",
                       "layout", "cast", "expression", "direction", "setting")
    #: Fields the background jobs write. The editor adopts these from disk so
    #: saving a text edit cannot throw away a page drawn since the screen was
    #: opened.
    _RENDERED_BOOK = ("cover", "cover_check", "spend", "pending_batch")
    # `seed` belongs to the render like `image` does: without it here, an
    # editor save could clobber the number a good page was drawn with.
    _RENDERED_PAGE = ("image", "image_from_rev", "check", "history", "error",
                      "audio", "audio_from_rev", "seed")
    #: The person names and describes a cast member; the render draws their
    #: sheet. Matched by position, the handle the rest of the app already uses
    #: for a cast member (the API's `index`, the sheet's own file name).
    _EDITORIAL_CAST = ("name", "description", "kind", "pages")
    _RENDERED_CAST = ("sheet", "sheet_error")

    def _merge_cast(self, book: Book, disk: Book, adopt: str) -> None:
        """Bring the other writer's cast fields over, without losing ours."""
        fields = (self._EDITORIAL_CAST if adopt == "editorial"
                  else self._RENDERED_CAST)
        if len(book.cast) == len(disk.cast):
            for member, other in zip(book.cast, disk.cast):
                for name in fields:
                    setattr(member, name, getattr(other, name))
            return
        if adopt != "editorial":
            # Membership is the editor's own doing, so there is nothing to
            # reconcile: take the drawn sheets for whoever still matches.
            drawn = {m.name.lower(): m for m in disk.cast}
            for member in book.cast:
                other = drawn.get(member.name.lower())
                if other is not None:
                    for name in fields:
                        setattr(member, name, getattr(other, name))
            return
        # Someone was added or removed while this job ran. That list is the
        # one that counts; keep the sheets this run drew for whoever is in it.
        ours = {m.name.lower(): m for m in book.cast}
        for other in disk.cast:
            mine = ours.get(other.name.lower())
            if mine is not None:
                for name in self._RENDERED_CAST:
                    setattr(other, name, getattr(mine, name))
        book.cast = list(disk.cast)

    def save_book(self, book: Book, adopt: str | None = None) -> Book:
        """Write the book, without clobbering the other writer.

        Two parties save whole books: the editor (seconds-old copies) and
        background jobs (minutes-old copies). `adopt` names what this caller
        does NOT own -- "editorial" for jobs, "rendered" for the editor. When
        the disk copy is newer than what this copy was loaded from, the other
        party's fields are adopted from disk before writing, so neither side's
        work is lost. Saves without `adopt` keep the old clobbering semantics
        and should hold only briefly-loaded books.
        """
        import time

        with self.book_lock(book.id):
            path = self.book_dir(book.id) / "book.json"
            loaded = getattr(book, "_loaded_updated", None)
            if adopt and loaded is not None and path.is_file():
                try:
                    disk = Book.from_dict(load_json(path))
                except (TypeError, ValueError, OSError):
                    disk = None
                if disk is not None and disk.updated > loaded:
                    book_fields, page_fields = (
                        (self._EDITORIAL_BOOK, self._EDITORIAL_PAGE)
                        if adopt == "editorial"
                        else (self._RENDERED_BOOK, self._RENDERED_PAGE))
                    for name in book_fields:
                        setattr(book, name, getattr(disk, name))
                    self._merge_cast(book, disk, adopt)
                    by_index = {p.index: p for p in disk.pages}
                    for page in book.pages:
                        other = by_index.get(page.index)
                        if other is None:
                            continue
                        for name in page_fields:
                            setattr(page, name, getattr(other, name))
            # Strictly greater than the copy this was loaded from, even when
            # two saves land inside one clock tick -- the staleness test
            # above is `>`, and an equal timestamp would hide an intervening
            # save. 0.1 ms is far above float granularity at epoch scale and
            # far below any real editing timescale.
            book.updated = max(time.time(), (loaded or 0.0) + 1e-4)
            save_json(path, book.to_dict())
            book._loaded_updated = book.updated
        return book

    def get_book(self, book_id: str) -> Book:
        try:
            path = self.book_dir(book_id) / "book.json"
        except ValueError as exc:
            raise FileNotFoundError(str(exc)) from exc
        if not path.is_file():
            raise FileNotFoundError(f"no book {book_id}")
        book = Book.from_dict(load_json(path))
        # The id steers every write path for this book, so take it from the
        # folder we actually loaded from rather than from the file's contents.
        book.id = path.parent.name
        # Remembered so save_book can tell whether someone saved in between.
        book._loaded_updated = book.updated
        return book

    def books(self) -> list[Book]:
        """Every book on the shelf, including any that will not load.

        A book whose JSON is unreadable used to be skipped in silence, so a
        half-written file made the book disappear from the shelf with no error
        anywhere. It comes back as a placeholder instead: the id is the folder
        name, and `broken` tells the UI to offer a restore rather than a story.
        """
        found = []
        for path in sorted((self.root / "books").glob("*/book.json")):
            try:
                book = Book.from_dict(load_json(path))
                book.id = path.parent.name
            except (TypeError, ValueError, OSError):
                book = Book(id=path.parent.name, broken=True,
                            updated=path.stat().st_mtime if path.exists() else 0.0)
            found.append(book)
        return sorted(found, key=lambda b: b.updated, reverse=True)

    def delete_book(self, book_id: str) -> None:
        shutil.rmtree(self.book_dir(book_id), ignore_errors=True)

    # ---------------------------------------------------------------- money

    def totals(self) -> dict[str, Any]:
        """Everything this library has cost, by where the money went.

        A book's own ledger is only part of the bill: the character sheets and
        the style previews are drawn before any book exists and are paid for
        once, then reused. Summing all three is the only figure that matches
        what the providers actually charged.
        """
        from ..pricing import summary

        parts = {
            "heroes": [h.spend for h in self.heroes()],
            "styles": [s.spend for s in self.styles()],
            "books": [b.spend for b in self.books()],
        }
        combined: dict[str, Any] = {}
        by_area: dict[str, Any] = {}
        for area, ledgers in parts.items():
            area_total = {"usd": 0.0, "calls": 0, "images": 0}
            for ledger in ledgers:
                for key in area_total:
                    area_total[key] += (ledger or {}).get(key, 0) or 0
                for key in ("usd", "calls", "images"):
                    combined[key] = combined.get(key, 0) + ((ledger or {}).get(key, 0) or 0)
            area_total["usd"] = round(area_total["usd"], 6)
            by_area[area] = summary(area_total)
        combined["usd"] = round(combined.get("usd", 0.0), 6)

        total = summary(combined)
        total["by_area"] = by_area
        return total
