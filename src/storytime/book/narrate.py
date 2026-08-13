"""Reading the book out loud.

One audio file per page per language, so the book can be listened to as well as
read. This is the payoff for having written each language natively rather than
translating: the German and the Russian both scan when spoken, because they
were written to be spoken.

The voice gets an instruction, not just text. Told to read slowly and warmly to
a small child, the same model produces something quite different from its
default news-reader delivery.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

from ..config import require_key
from ..pricing import add as add_spend
from .models import LANGUAGES

API_URL = "https://api.openai.com/v1/audio/speech"
MODEL = "gpt-4o-mini-tts"

# A few voices that suit reading to a child. The names are OpenAI's.
VOICES = {
    "coral": "warm, freundlich",
    "sage": "ruhig, gelassen",
    "nova": "hell, lebendig",
    "alloy": "neutral",
    "fable": "erzählerisch",
    "onyx": "tief, ruhig",
}

INSTRUCTIONS = (
    "Read this aloud to a small child at bedtime. Slow, warm and unhurried. "
    "Pause at full stops and let them land. Never rush, never perform, never "
    "use a cartoon voice. Sound like a parent who has read this book before "
    "and likes it."
)


class NarrationError(RuntimeError):
    pass


def speak(text: str, target: Path, voice: str = "coral", speed: float = 0.95,
          language: str = "de") -> dict:
    """Render one piece of text to an mp3. Returns usage for the spend meter."""
    if not text.strip():
        raise NarrationError("nothing to read")

    key = require_key("OPENAI_API_KEY", "narration")
    language_name = LANGUAGES.get(language, {}).get("english_name", language)
    body = json.dumps({
        "model": MODEL,
        "voice": voice if voice in VOICES else "coral",
        "input": text,
        "instructions": f"{INSTRUCTIONS} The text is in {language_name}.",
        "speed": max(0.5, min(1.2, speed)),
        "response_format": "mp3",
    }).encode("utf-8")

    request = urllib.request.Request(
        API_URL, data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            audio = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise NarrationError(f"Sprachausgabe abgelehnt ({exc.code}): {detail}") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(audio)

    # Billed on text tokens in and audio tokens out; roughly four characters
    # per token is the usual rule of thumb, and audio runs far longer.
    text_tokens = max(1, len(text) // 4)
    return {
        "model": MODEL,
        "input_tokens": text_tokens,
        "output_tokens": text_tokens * 20,
        "audio_files": 1,
    }


def narrate_book(
    book,
    audio_dir: Path,
    languages: list[str] | None = None,
    voice: str = "coral",
    speed: float = 0.95,
    redo: bool = False,
    log=print,
    should_stop=None,
) -> None:
    """Narrate the title and every page, in each requested language."""
    stop = should_stop or (lambda: False)
    languages = [c for c in (languages or book.languages) if c in book.languages]
    audio_dir.mkdir(parents=True, exist_ok=True)

    for language in languages:
        if stop():
            break
        name = LANGUAGES.get(language, {}).get("name", language)
        log(f"{name} — Stimme {voice}")

        for page in sorted(book.pages, key=lambda p: p.index):
            if stop():
                log("abgebrochen — die fertigen Dateien bleiben")
                return
            text = page.text.get(language, "").strip()
            if not text:
                continue
            target = audio_dir / f"page_{page.index:02d}_{language}.mp3"
            # Reuse an existing take only while it still reads the current
            # text; a stale one falls through and is recorded again.
            fresh = page.text_rev.get(language, 0) <= page.audio_from_rev.get(language, 0)
            if target.is_file() and not redo and fresh:
                page.audio[language] = f"audio/{target.name}"
                continue
            try:
                usage = speak(text, target, voice=voice, speed=speed, language=language)
                page.audio[language] = f"audio/{target.name}"
                # The narration now reads exactly this revision of the text;
                # the next text edit makes it visibly stale.
                page.audio_from_rev[language] = page.text_rev.get(language, 0)
                add_spend(book.spend, usage, "narration")
                log(f"  Seite {page.index}")
            except Exception as exc:
                log(f"  Seite {page.index}: fehlgeschlagen — {exc}")

    book.narration_voice = voice
