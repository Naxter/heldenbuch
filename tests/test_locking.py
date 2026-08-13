"""The two-writers problem: the editor and a render job both save whole books.

Each holds its own copy -- the render's for minutes. Without the ownership
merge in `Library.save_book`, whichever saved last silently reverted the
other's work.
"""

from heldenbuch.book.library import Library
from heldenbuch.book.models import Book, Page


def _seed(library: Library) -> str:
    book = Book(id="book_test",
                pages=[Page(index=1, text={"de": "alt"}, illustration="ein Haus")])
    library.save_book(book)
    return book.id


def test_render_save_adopts_text_edited_meanwhile(tmp_path):
    library = Library(tmp_path)
    book_id = _seed(library)

    render_copy = library.get_book(book_id)  # the job starts and holds this
    editor_copy = library.get_book(book_id)  # the user opens the editor

    editor_copy.pages[0].text["de"] = "neu"
    library.save_book(editor_copy, adopt="rendered")

    render_copy.pages[0].image = "pages/page_01.png"
    library.save_book(render_copy, adopt="editorial")

    final = library.get_book(book_id)
    assert final.pages[0].text["de"] == "neu"
    assert final.pages[0].image == "pages/page_01.png"


def test_editor_save_adopts_pages_drawn_meanwhile(tmp_path):
    library = Library(tmp_path)
    book_id = _seed(library)

    editor_copy = library.get_book(book_id)
    render_copy = library.get_book(book_id)

    render_copy.pages[0].image = "pages/page_01.png"
    render_copy.cover = "pages/cover.png"
    library.save_book(render_copy, adopt="editorial")

    editor_copy.pages[0].text["de"] = "neu"
    library.save_book(editor_copy, adopt="rendered")

    final = library.get_book(book_id)
    assert final.pages[0].image == "pages/page_01.png"
    assert final.cover == "pages/cover.png"
    assert final.pages[0].text["de"] == "neu"


def test_plain_save_keeps_old_semantics(tmp_path):
    library = Library(tmp_path)
    book_id = _seed(library)

    copy_a = library.get_book(book_id)
    copy_b = library.get_book(book_id)
    copy_a.idea = "a"
    library.save_book(copy_a)
    copy_b.idea = "b"
    library.save_book(copy_b)

    assert library.get_book(book_id).idea == "b"
