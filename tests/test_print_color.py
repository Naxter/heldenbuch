"""The sRGB OutputIntent stamped onto exported PDFs (optional pikepdf path)."""

import pytest
from PIL import Image

from heldenbuch.book.layout import embed_srgb


def test_embed_srgb_returns_false_without_pikepdf(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_pikepdf(name, *args, **kwargs):
        if name == "pikepdf":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pikepdf)
    pdf = tmp_path / "book.pdf"
    Image.new("RGB", (40, 40), "white").save(pdf, "PDF")
    before = pdf.read_bytes()
    assert embed_srgb(pdf) is False
    assert pdf.read_bytes() == before  # untouched without the dependency


def test_embed_srgb_stamps_output_intent_and_language(tmp_path):
    pikepdf = pytest.importorskip("pikepdf")

    pdf = tmp_path / "book.pdf"
    Image.new("RGB", (40, 40), "white").save(pdf, "PDF")
    assert embed_srgb(pdf, language="de") is True

    with pikepdf.open(pdf) as result:
        intents = result.Root.OutputIntents
        assert len(intents) == 1
        assert str(intents[0].OutputConditionIdentifier) == "sRGB IEC61966-2.1"
        assert intents[0].DestOutputProfile.N == 3
        assert str(result.Root.Lang) == "de"
