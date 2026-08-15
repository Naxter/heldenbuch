"""The exported PDF has to contain its own words.

The text is typeset into the artwork, so without a text layer the file is a
stack of photographs: nothing to select, nothing to search, and nothing for a
screen reader to say.
"""

import pytest
from PIL import Image

from heldenbuch.book.layout import tag_pdf


def _pdf(tmp_path, pages=2):
    path = tmp_path / "book.pdf"
    sheets = [Image.new("RGB", (300, 300), "white") for _ in range(pages)]
    sheets[0].save(path, "PDF", save_all=True, append_images=sheets[1:])
    return path


def test_without_pikepdf_the_pdf_is_left_alone(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_pikepdf(name, *args, **kwargs):
        if name == "pikepdf":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pikepdf)
    path = _pdf(tmp_path)
    before = path.read_bytes()
    assert tag_pdf(path, ["Es regnet."]) == 0
    assert path.read_bytes() == before


def test_the_words_end_up_in_the_pdf(tmp_path):
    pikepdf = pytest.importorskip("pikepdf")

    path = _pdf(tmp_path, pages=2)
    written = tag_pdf(path, ["Es regnet auf den Matschstiefel.", "Ende"],
                      ["Ein Junge im Regen", ""])
    assert written == 2

    with pikepdf.open(path) as pdf:
        first = b"".join(bytes(s.read_bytes()) for s in
                         (pdf.pages[0].Contents if isinstance(pdf.pages[0].Contents, pikepdf.Array)
                          else [pdf.pages[0].Contents]))
        # the page's own words are in its content stream ...
        assert b"Matschstiefel" in first
        # ... written in invisible render mode, over the artwork
        assert b"3 Tr" in first
        assert b"/Figure" in first and b"BDC" in first

        # the document says it is tagged, and the structure tree exists
        assert bool(pdf.Root.MarkInfo.Marked) is True
        root = pdf.Root.StructTreeRoot
        document = root.K[0]
        kinds = [str(el.S) for el in document.K]
        assert "/Figure" in kinds and "/P" in kinds
        # the illustration brief is the figure's alternate description
        figure = next(el for el in document.K if str(el.S) == "/Figure")
        assert "Ein Junge im Regen" in str(figure.Alt)


def test_a_page_the_builtin_font_cannot_write_is_skipped(tmp_path):
    pytest.importorskip("pikepdf")

    path = _pdf(tmp_path, pages=2)
    # Cyrillic needs an embedded font; mangled Latin-1 would be worse than
    # nothing, so that page gets no text layer and the count says so.
    written = tag_pdf(path, ["Es regnet.", "Идёт дождь."])
    assert written == 1


def test_pages_beyond_the_supplied_text_are_still_valid(tmp_path):
    pikepdf = pytest.importorskip("pikepdf")

    path = _pdf(tmp_path, pages=3)  # padding pages have no text
    assert tag_pdf(path, ["Nur die erste Seite"]) == 1
    with pikepdf.open(path) as pdf:
        assert len(pdf.pages) == 3
        assert int(pdf.pages[2].StructParents) == 2
