"""The biggest moment must not get the smallest picture.

Layout is chosen while the story is written, before a single picture exists,
so nothing downstream notices when the page everything builds towards was
handed a vignette -- which prints smaller than every page around it.
"""

from heldenbuch.book.author import _normalise


def _payload(pages, climax=None):
    body = {
        "title": {"de": "T"},
        "dedication": {"de": "D"},
        "cover_illustration": "a boy",
        "cast": [],
        "pages": [
            {"index": i + 1, "text": {"de": f"Seite {i + 1}"},
             "illustration": "a scene", "layout": layout}
            for i, layout in enumerate(pages)
        ],
    }
    if climax is not None:
        body["climax"] = climax
    return body


def test_a_vignette_on_the_peak_is_promoted():
    story = _normalise(_payload(["full", "vignette", "full"], climax=2), ["de"], 3)
    assert story["pages"][1].layout == "full"
    assert story["climax"] == 2


def test_other_vignettes_are_left_alone():
    """They are the point of having the layout at all."""
    story = _normalise(_payload(["vignette", "full", "vignette"], climax=2), ["de"], 3)
    assert [p.layout for p in story["pages"]] == ["vignette", "full", "vignette"]


def test_a_wordless_peak_is_respected():
    """A silent high point is a deliberate choice, and still a full page."""
    story = _normalise(_payload(["full", "wordless", "full"], climax=2), ["de"], 3)
    assert story["pages"][1].layout == "wordless"


def test_a_missing_climax_changes_nothing():
    story = _normalise(_payload(["full", "vignette", "full"]), ["de"], 3)
    assert story["pages"][1].layout == "vignette"
    assert story["climax"] == 0


def test_a_nonsense_climax_is_ignored():
    for bad in ("later", None, 0, 99, -1):
        story = _normalise(_payload(["full", "vignette"], climax=bad), ["de"], 2)
        assert story["climax"] == 0
        assert story["pages"][1].layout == "vignette"


def test_the_author_is_asked_where_the_story_peaks():
    from heldenbuch.book.author import _shape

    assert '"climax"' in _shape(["de"], 12)
