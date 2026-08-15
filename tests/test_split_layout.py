"""The `split` layout must never reach the illustrator as "draw two pictures".

It is a page layout -- picture above, words below -- but the word invited both
the author and the image model to compose a diptych, and three of three split
pages came back with a seam down the middle. Two guards stop that now, and
this pins both: the brief is rewritten before it is drawn, and the prompt says
in words that the picture is one scene.
"""

import pytest

from heldenbuch.book.illustrate import page_prompt
from heldenbuch.book.models import Book, Hero, Page, Style, single_scene


def _prompt(layout: str, illustration: str = "the boy crosses the plank") -> str:
    book = Book(title={"de": "T"}, languages=["de"],
                pages=[Page(index=1, text={"de": "x"}, illustration=illustration,
                            layout=layout)])
    hero = Hero(name="Claudio", description="a boy")
    style = Style(name="S", description="soft watercolour")
    return page_prompt(book, hero, style, book.pages[0], [])


def test_a_split_page_is_told_it_is_still_one_picture():
    prompt = _prompt("split")
    assert "one single scene, not two" in prompt


@pytest.mark.parametrize("layout", ["full", "split", "vignette", "wordless"])
def test_no_layout_ever_asks_for_panels(layout):
    prompt = _prompt(layout).lower()
    for forbidden in ("two panels", "diptych", "side by side", "split composition"):
        assert forbidden not in prompt, f"{layout} prompt suggests {forbidden}"


def test_a_diptych_brief_is_collapsed_before_it_is_drawn():
    """Even if the author writes one anyway."""
    brief = ("A split composition: on one side the crooked plank, on the "
             "other the nest beyond the water.")
    prompt = _prompt("split", brief)

    assert "split composition" not in prompt.lower()
    assert single_scene(brief) in prompt
