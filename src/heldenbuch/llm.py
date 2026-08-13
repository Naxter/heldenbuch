"""Text LLM client -- writes the stories, describes the heroes, translates.

Separate from `metrics/judge.py` on purpose: that one grades pictures, this one
produces prose and structured JSON. Both talk to the same providers, both use
only the standard library.

Every call goes through `complete_json`, because everything the app asks for
has a shape: a story is a list of pages, a hero description is a set of fields.
Free prose comes back inside a JSON field, never as the whole reply.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import require_key
from .imageutil import to_data_uri

DEFAULT_MODELS = {
    "openai": "gpt-5.6-terra",
    "anthropic": "claude-opus-5",
    "gemini": "gemini-3-pro",
}

PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class LLMError(RuntimeError):
    pass


def _post(url: str, body: dict, headers: dict, timeout: int = 300) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:600]
        raise LLMError(f"{url.split('/')[2]} returned {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"could not reach {url.split('/')[2]}: {exc.reason}") from exc


def extract_json(text: str) -> Any:
    """Parse JSON out of a reply that may be wrapped in prose or code fences."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"[\{\[].*[\}\]]", cleaned, flags=re.DOTALL)
    if not match:
        raise LLMError(f"the model did not return JSON: {(text or '')[:300]}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMError(f"the model returned broken JSON: {exc}") from exc


# --------------------------------------------------------------------------- providers


#: Field names the three providers use for the same two numbers.
_TOKEN_FIELDS = (
    ("input_tokens", "prompt_tokens", "prompt_token_count", "input_token_count"),
    ("output_tokens", "completion_tokens", "candidates_token_count", "output_token_count"),
)


def _usage(model: str, payload: Any) -> dict[str, Any]:
    """What one text call cost, in the shape `pricing.price` expects.

    Every provider reports its token counts and this was thrown away, so the
    story, the revision pass, each translation and -- most of all -- the
    per-page vision check all cost real money and recorded none of it. On a
    finished book that is roughly a tenth of the true total, missing from the
    one number the app shows.
    """
    raw = payload.get("usage") if isinstance(payload, dict) else payload
    if raw is None:
        return {}
    get = raw.get if isinstance(raw, dict) else lambda k, d=0: getattr(raw, k, d)
    counts = {}
    for canonical, *aliases in _TOKEN_FIELDS:
        for name in (canonical, *aliases):
            value = get(name, None)
            if value:
                counts[canonical] = int(value)
                break
    return {"model": model, **counts} if counts else {}


def _openai(system: str, user: str, images: list[Path], model: str) -> str:
    key = require_key("OPENAI_API_KEY", "openai")
    content: list[dict[str, Any]] = [{"type": "text", "text": user}]
    for path in images:
        data, _ = to_data_uri(path, max_edge=768)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})

    payload = _post(
        "https://api.openai.com/v1/chat/completions",
        {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        },
        {"Authorization": f"Bearer {key}"},
    )
    try:
        return payload["choices"][0]["message"]["content"], _usage(model, payload)
    except (KeyError, IndexError) as exc:
        raise LLMError(f"unexpected OpenAI reply: {json.dumps(payload)[:300]}") from exc


def _anthropic(system: str, user: str, images: list[Path], model: str) -> str:
    key = require_key("ANTHROPIC_API_KEY", "anthropic")
    content: list[dict[str, Any]] = [{"type": "text", "text": user}]
    for path in images:
        data, _ = to_data_uri(path, max_edge=768)
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": data},
            }
        )

    payload = _post(
        "https://api.anthropic.com/v1/messages",
        {
            "model": model,
            "max_tokens": 8192,
            "system": system,
            "messages": [{"role": "user", "content": content}],
        },
        {"x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    try:
        text = "".join(block.get("text", "") for block in payload["content"])
        return text, _usage(model, payload)
    except (KeyError, TypeError) as exc:
        raise LLMError(f"unexpected Anthropic reply: {json.dumps(payload)[:300]}") from exc


def _gemini(system: str, user: str, images: list[Path], model: str) -> str:
    try:
        from google import genai
    except ImportError as exc:
        raise LLMError("the gemini provider needs: pip install 'google-genai>=2.3.0'") from exc

    client = genai.Client(api_key=require_key("GEMINI_API_KEY", "gemini"))
    payload: list[dict[str, Any]] = [{"type": "text", "text": f"{system}\n\n{user}"}]
    for path in images:
        data, mime = to_data_uri(path, max_edge=768)
        payload.append({"type": "image", "data": data, "mime_type": mime})
    interaction = client.interactions.create(model=model, input=payload)
    meta = getattr(interaction, "usage", None) or getattr(
        interaction, "usage_metadata", None)
    return interaction.output_text or "", _usage(model, meta)


PROVIDERS = {"openai": _openai, "anthropic": _anthropic, "gemini": _gemini}


# --------------------------------------------------------------------------- entry point


def complete_json(
    system: str,
    user: str,
    images: list[Path] | None = None,
    provider: str = "openai",
    model: str | None = None,
    spend: dict[str, Any] | None = None,
    what: str = "text",
) -> Any:
    """Ask for structured output and return it parsed.

    Pass `spend` to record what the call cost into a running tally; without it
    the cost is simply not tracked, which is what used to happen everywhere.
    """
    if provider not in PROVIDERS:
        raise LLMError(f"unknown provider {provider!r}; valid: {', '.join(PROVIDERS)}")
    model = model or DEFAULT_MODELS[provider]
    reply, usage = PROVIDERS[provider](system, user, images or [], model)
    if spend is not None and usage:
        from .pricing import add as add_spend

        add_spend(spend, usage, what)
    return extract_json(reply)


def available_providers() -> list[str]:
    """Which text providers have a key set, so the UI can offer only those."""
    import os

    from .config import load_dotenv

    load_dotenv()
    ready = [p for p, key in PROVIDER_KEYS.items() if os.environ.get(key, "").strip()]
    # Gemini also needs its SDK installed.
    if "gemini" in ready:
        try:
            __import__("google.genai")
        except ImportError:
            ready.remove("gemini")
    return ready
