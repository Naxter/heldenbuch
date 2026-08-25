"""The two-writers problem: the editor and a render job both save whole books.

Each holds its own copy -- the render's for minutes. Without the ownership
merge in `Library.save_book`, whichever saved last silently reverted the
other's work.
"""

from heldenbuch.book.library import Library
from heldenbuch.book.models import Book, CastMember, Page


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


def test_face_and_seed_survive_the_other_writers_save(tmp_path):
    """expression/direction belong to the editor, seed to the render; each
    used to be missing from its owner's field list and was silently reverted
    by the other writer's save."""
    library = Library(tmp_path)
    book_id = _seed(library)

    render_copy = library.get_book(book_id)
    editor_copy = library.get_book(book_id)

    editor_copy.pages[0].expression = "close to tears"
    editor_copy.pages[0].direction = "facing left"
    library.save_book(editor_copy, adopt="rendered")

    render_copy.pages[0].image = "pages/page_01.png"
    render_copy.pages[0].seed = 12345
    library.save_book(render_copy, adopt="editorial")

    final = library.get_book(book_id)
    assert final.pages[0].expression == "close to tears"
    assert final.pages[0].direction == "facing left"
    assert final.pages[0].seed == 12345

    editor_again = library.get_book(book_id)
    editor_again.pages[0].expression = "smiling"
    library.save_book(editor_again, adopt="rendered")
    after = library.get_book(book_id)
    assert after.pages[0].expression == "smiling"  # the later edit lands
    assert after.pages[0].seed == 12345            # without losing the render's


def _seed_cast(library: Library) -> str:
    book = Book(id="book_cast",
                cast=[CastMember(name="Pip", kind="character"),
                      CastMember(name="Der Garten", kind="place")],
                pages=[Page(index=1, text={"de": "alt"}, illustration="Pip")])
    library.save_book(book)
    return book.id


def test_a_drawn_cast_sheet_survives_an_edit_landing_mid_render(tmp_path):
    """A cast member is written from both sides: the person names them, the
    render draws their sheet. Adopting the whole list threw away a sheet that
    had just been paid for whenever an edit landed between two job saves."""
    library = Library(tmp_path)
    book_id = _seed_cast(library)

    render_copy = library.get_book(book_id)   # the job starts and holds this
    editor_copy = library.get_book(book_id)   # a tab is already open

    # The job draws the sheets and saves them.
    render_copy.cast[0].sheet = "cast/cast_01.png"
    render_copy.cast[1].sheet = "cast/cast_02.png"
    library.save_book(render_copy)

    # The editor, holding a copy from before that, renames someone.
    editor_copy.cast[0].name = "Max"
    library.save_book(editor_copy, adopt="rendered")

    # The job's next progress save must not put the sheet-less cast back.
    render_copy.pages[0].image = "pages/page_01.png"
    library.save_book(render_copy, adopt="editorial")

    final = library.get_book(book_id)
    assert [c.name for c in final.cast] == ["Max", "Der Garten"]
    assert [c.sheet for c in final.cast] == ["cast/cast_01.png", "cast/cast_02.png"]


def test_a_member_added_mid_render_is_not_dropped(tmp_path):
    """Membership belongs to the person. A prop added while the pages were
    drawing must still be there afterwards -- with the sheets kept."""
    library = Library(tmp_path)
    book_id = _seed_cast(library)

    render_copy = library.get_book(book_id)
    editor_copy = library.get_book(book_id)

    render_copy.cast[0].sheet = "cast/cast_01.png"
    library.save_book(render_copy)

    editor_copy.cast.append(CastMember(name="Die Laterne", kind="prop"))
    library.save_book(editor_copy, adopt="rendered")

    library.save_book(render_copy, adopt="editorial")

    final = library.get_book(book_id)
    assert [c.name for c in final.cast] == ["Pip", "Der Garten", "Die Laterne"]
    assert final.cast[0].sheet == "cast/cast_01.png"


def test_an_editor_save_adopts_sheets_drawn_meanwhile(tmp_path):
    """The mirror case: the editor must not clobber a sheet either."""
    library = Library(tmp_path)
    book_id = _seed_cast(library)

    editor_copy = library.get_book(book_id)
    render_copy = library.get_book(book_id)

    render_copy.cast[1].sheet = "cast/cast_02.png"
    library.save_book(render_copy, adopt="editorial")

    editor_copy.cast[1].description = "ein verwunschener Garten"
    library.save_book(editor_copy, adopt="rendered")

    final = library.get_book(book_id)
    assert final.cast[1].sheet == "cast/cast_02.png"
    assert final.cast[1].description == "ein verwunschener Garten"
