"""A page can say what the hero's face is doing and which way it faces.

The brief describes what happens. Left to infer a feeling from events, the
illustrator paints the same pleasant half-smile on every page, and a story
with a real problem in it reads as sixteen pleasant afternoons. Facing matters
too: a picture book is read left to right, so a page that contradicts the page
turn feels wrong even to a child who cannot say why.
"""

from heldenbuch.book.illustrate import page_prompt
from heldenbuch.book.models import Book, Hero, Page, Style


def _prompt(**page_kwargs) -> str:
    page = Page(index=1, illustration="the boy crosses the plank", **page_kwargs)
    return page_prompt(Book(pages=[page]), Hero(name="Claudio"),
                       Style(description="soft watercolour"), page, [])


def test_the_illustrator_is_told_the_expression():
    out = _prompt(expression="close to tears")
    assert "The hero's face: close to tears." in out


def test_the_illustrator_is_told_the_direction():
    out = _prompt(direction="moving left to right")
    assert "Direction: moving left to right." in out


def test_a_page_without_them_says_nothing_extra():
    """Books written before these existed must not grow empty instructions."""
    out = _prompt()
    assert "The hero's face:" not in out
    assert "Direction:" not in out


def test_they_sit_outside_the_scene_text():
    """So they read as direction, not as something to draw a label about."""
    out = _prompt(expression="delighted")
    scene = out.split("SCENE\n", 1)[1]
    assert scene.startswith("the boy crosses the plank\n")
    assert "The hero's face: delighted." in scene.split("\n\n", 1)[0]


def test_blank_values_are_ignored():
    assert "Direction:" not in _prompt(direction="   ")


def test_the_author_is_asked_for_both():
    from heldenbuch.book.author import _shape

    shape = _shape(["de"], 12)
    assert '"expression"' in shape
    assert '"direction"' in shape


def test_they_survive_a_round_trip(tmp_path):
    from heldenbuch.book.library import Library

    library = Library(tmp_path)
    book = Book(title={"de": "T"},
                pages=[Page(index=1, expression="asleep", direction="facing the reader")])
    library.save_book(book)

    page = library.get_book(book.id).pages[0]
    assert page.expression == "asleep"
    assert page.direction == "facing the reader"
