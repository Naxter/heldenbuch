"""Backup with manifest, validated restore, and cast editing.

A backup that cannot be restored is a file, not a backup. The restore path
refuses anything it cannot verify, and never overwrites a living book.
"""

from __future__ import annotations

import json
import zipfile

import pytest
from PIL import Image

from heldenbuch.book.library import Library
from heldenbuch.book.models import Book, CastMember, Page
from heldenbuch.web.bookapi import BookApi


class _NoJobs:
    def active(self):
        return None

    def pending(self) -> int:
        return 0


@pytest.fixture()
def library(tmp_path) -> Library:
    return Library(tmp_path / "library")


@pytest.fixture()
def api(library) -> BookApi:
    return BookApi(library, _NoJobs())


def _book_with_page(library) -> Book:
    book = Book(title={"de": "Sicherungstest"},
                pages=[Page(index=1, text={"de": "Eins"}, image="pages/page_01.png")])
    library.save_book(book)
    target = library.book_dir(book.id) / "pages/page_01.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), "white").save(target)
    return book


# ------------------------------------------------------------------- backup


def test_backup_carries_a_manifest_with_hashes(api, library):
    book = _book_with_page(library)
    info = api.book_backup(book.id, {}, None)

    with zipfile.ZipFile(library.resolve(info["file"])) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["kind"] == "heldenbuch-book-backup"
    assert manifest["book_id"] == book.id
    assert "book.json" in manifest["files"]
    assert "pages/page_01.png" in manifest["files"]
    assert all(len(digest) == 64 for digest in manifest["files"].values())


def test_restore_round_trips_as_a_copy_when_the_book_still_exists(api, library):
    book = _book_with_page(library)
    info = api.book_backup(book.id, {}, None)

    result = api.book_restore({}, {"file": info["file"]})
    assert result["book_id"] != book.id, "a living book must never be overwritten"
    restored = library.get_book(result["book_id"])
    assert restored.title == {"de": "Sicherungstest"}
    assert (library.book_dir(result["book_id"]) / "pages/page_01.png").is_file()


def test_restore_recreates_a_deleted_book_under_its_own_id(api, library):
    book = _book_with_page(library)
    info = api.book_backup(book.id, {}, None)
    library.delete_book(book.id)

    result = api.book_restore({}, {"file": info["file"]})
    assert result["book_id"] == book.id


def test_restore_refuses_a_tampered_archive(api, library):
    book = _book_with_page(library)
    info = api.book_backup(book.id, {}, None)
    target = library.resolve(info["file"])

    # Rewrite the archive with one file changed but the manifest untouched.
    with zipfile.ZipFile(target) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    entries["pages/page_01.png"] = b"tampered bytes"
    with zipfile.ZipFile(target, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)

    with pytest.raises(ValueError, match="Prüfsumme"):
        api.book_restore({}, {"file": info["file"]})


def test_restore_refuses_a_zip_without_a_manifest(api, library):
    folder = library.root / "backups"
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(folder / "plain.zip", "w") as archive:
        archive.writestr("book.json", "{}")
    with pytest.raises(ValueError, match="Manifest"):
        api.book_restore({}, {"file": "backups/plain.zip"})


def test_restore_refuses_paths_outside_the_backups_folder(api):
    with pytest.raises(ValueError):
        api.book_restore({}, {"file": "../.env"})


# ------------------------------------------------------------------- cast


def _cast_book(library) -> Book:
    book = Book(
        title={"de": "T"},
        cast=[CastMember(name="Oma", description="grey bun", pages=[1, 2])],
        pages=[Page(index=1, text={"de": "a"}, cast=["Oma"]),
               Page(index=2, text={"de": "b"}, cast=["Oma"]),
               Page(index=3, text={"de": "c"})],
    )
    library.save_book(book)
    return book


def test_renaming_a_cast_member_follows_through_to_the_pages(api, library):
    book = _cast_book(library)
    api.book_update(book.id, {}, {"cast": [{"index": 0, "name": "Omi"}]})
    refreshed = library.get_book(book.id)
    assert refreshed.cast[0].name == "Omi"
    assert refreshed.pages[0].cast == ["Omi"]
    assert refreshed.pages[1].cast == ["Omi"]


def test_page_membership_edits_update_both_sides(api, library):
    book = _cast_book(library)
    api.book_update(book.id, {}, {"cast": [{"index": 0, "pages": [2, 3]}]})
    refreshed = library.get_book(book.id)
    assert refreshed.cast[0].pages == [2, 3]
    assert refreshed.pages[0].cast == []
    assert "Oma" in refreshed.pages[2].cast


def test_removing_a_cast_member_clears_the_page_lists(api, library):
    book = _cast_book(library)
    api.book_update(book.id, {}, {"cast": [{"index": 0, "remove": True}]})
    refreshed = library.get_book(book.id)
    assert refreshed.cast == []
    assert refreshed.pages[0].cast == []


def test_a_backup_written_under_the_old_name_still_restores(library):
    """The app was renamed after the first backups were written, and for some
    books a ZIP in library/backups is the only copy that exists."""
    import json
    import zipfile

    from heldenbuch.web.bookapi import BACKUP_KINDS

    assert "storytime-book-backup" in BACKUP_KINDS

    api = BookApi(library, _NoJobs())
    book = library.save_book(Book(title={"de": "Alt"}, pages=[]))
    api.book_backup(book.id, {}, None)
    archive = next((library.root / "backups").glob("*.zip"))

    # Rewrite the manifest with the pre-rename marker, as an old ZIP has.
    with zipfile.ZipFile(archive) as source:
        items = {n: source.read(n) for n in source.namelist()}
    manifest = json.loads(items["manifest.json"])
    manifest["kind"] = "storytime-book-backup"
    items["manifest.json"] = json.dumps(manifest).encode("utf-8")
    legacy = library.root / "backups" / "legacy.zip"
    with zipfile.ZipFile(legacy, "w") as out:
        for name, blob in items.items():
            out.writestr(name, blob)

    library.delete_book(book.id)
    result = api.book_restore({}, {"file": "backups/legacy.zip"})
    assert result.get("book_id")
