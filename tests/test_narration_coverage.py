"""Narration covers the whole book, not only the numbered pages.

A printed book opens with a title page and ends with a closing word, and this
one can carry a dedication. Reading only the pages dropped a listener into the
middle of the story and abandoned them before the end.
"""

from heldenbuch.book import narrate
from heldenbuch.book.models import Book, Page


def _book():
    return Book(
        title={"de": "Mats und der Matschstiefel"},
        dedication={"de": "Für Mats, Weihnachten 2026"},
        languages=["de"],
        pages=[Page(index=1, text={"de": "Es regnet."}),
               Page(index=2, text={"de": "Der Stiefel ist weg."})],
    )


def _record(monkeypatch, book, audio_dir, **kwargs):
    """Run the narration with the provider replaced by a recorder."""
    spoken = []

    def fake_speak(text, target, voice="coral", speed=0.95, language="de"):
        spoken.append((target.name, text))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"ID3fake")
        return {"model": "test", "input_tokens": 1, "output_tokens": 1, "audio_files": 1}

    monkeypatch.setattr(narrate, "speak", fake_speak)
    narrate.narrate_book(book, audio_dir, log=lambda *a: None, **kwargs)
    return spoken


def test_title_dedication_pages_and_closing_are_all_read(tmp_path, monkeypatch):
    book = _book()
    spoken = _record(monkeypatch, book, tmp_path / "audio")

    said = [text for _, text in spoken]
    assert "Mats und der Matschstiefel" in said
    assert "Für Mats, Weihnachten 2026" in said
    assert "Es regnet." in said
    assert "Ende" in said

    # and in the order the book is read
    assert said.index("Mats und der Matschstiefel") < said.index("Es regnet.")
    assert said.index("Ende") == len(said) - 1

    assert book.matter_audio["title"]["de"] == "audio/title_de.mp3"
    assert book.matter_audio["dedication"]["de"] == "audio/dedication_de.mp3"
    assert book.matter_audio["closing"]["de"] == "audio/closing_de.mp3"


def test_a_book_without_a_dedication_skips_it(tmp_path, monkeypatch):
    book = _book()
    book.dedication = {}
    spoken = _record(monkeypatch, book, tmp_path / "audio")

    assert not any(name.startswith("dedication") for name, _ in spoken)
    assert "dedication" not in book.matter_audio
    assert book.matter_audio["closing"]["de"] == "audio/closing_de.mp3"


def test_a_second_run_reuses_what_still_matches(tmp_path, monkeypatch):
    book = _book()
    audio = tmp_path / "audio"
    _record(monkeypatch, book, audio)
    again = _record(monkeypatch, book, audio)
    assert again == []  # nothing changed, so nothing is paid for twice


def test_an_edited_dedication_is_recorded_again(tmp_path, monkeypatch):
    book = _book()
    audio = tmp_path / "audio"
    _record(monkeypatch, book, audio)

    book.dedication["de"] = "Für Mats, zum sechsten Geburtstag"
    spoken = _record(monkeypatch, book, audio)

    assert [text for _, text in spoken] == ["Für Mats, zum sechsten Geburtstag"]


def test_the_closing_word_follows_the_language(tmp_path, monkeypatch):
    book = _book()
    book.languages = ["de", "en"]
    book.title["en"] = "Mats and the muddy boot"
    book.pages[0].text["en"] = "It is raining."
    book.pages[1].text["en"] = "The boot is gone."
    spoken = _record(monkeypatch, book, tmp_path / "audio")

    said = dict((name, text) for name, text in spoken)
    assert said["closing_de.mp3"] == "Ende"
    assert said["closing_en.mp3"] == "The End"
