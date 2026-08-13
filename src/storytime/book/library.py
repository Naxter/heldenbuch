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
        parent = (self.root / kind).resolve()
        target = (parent / str(ident)).resolve()
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
            except (TypeError, ValueError):
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
            except (TypeError, ValueError):
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

    def save_book(self, book: Book) -> Book:
        import time

        book.updated = time.time()
        save_json(self.book_dir(book.id) / "book.json", book.to_dict())
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
