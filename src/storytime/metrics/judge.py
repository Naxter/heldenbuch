"""VLM judge -- a vision model scores each page against the character sheet.

This is the metric that actually answers the question. The colour metrics can
tell you the palette drifted; only something that can *look* at the picture can
tell you the fox has the wrong number of tail rings.

Three providers are supported. Prefer one that is not also under test: a model
grading its own output is not a neutral referee, and the CLI warns when that
happens.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..config import require_key
from ..imageutil import to_data_uri
from ..prompts import judge_prompt
from ..types import BenchmarkSpec

DEFAULT_MODELS = {
    "openai": "gpt-5.6-terra",
    "anthropic": "claude-opus-5",
    "gemini": "gemini-3-pro",
}

DIMENSIONS = ("identity", "attributes", "style")


class JudgeError(RuntimeError):
    pass


def _post(url: str, body: dict, headers: dict, timeout: int = 180) -> dict:
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
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise JudgeError(f"judge request failed ({exc.code}): {detail}") from exc


def _parse(text: str) -> dict[str, Any]:
    """Pull the JSON verdict out of a reply that may be wrapped in prose."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise JudgeError(f"judge did not return JSON: {text[:200]}")
    parsed = json.loads(match.group(0))

    verdict: dict[str, Any] = {}
    for key in DIMENSIONS:
        try:
            verdict[key] = max(1, min(5, int(round(float(parsed[key])))))
        except (KeyError, TypeError, ValueError) as exc:
            raise JudgeError(f"judge verdict is missing a valid '{key}': {parsed}") from exc

    raw_notes = parsed.get("discrepancies") or []
    if isinstance(raw_notes, str):
        raw_notes = [raw_notes]
    verdict["discrepancies"] = [str(n) for n in raw_notes][:12]
    verdict["mean"] = round(sum(verdict[k] for k in DIMENSIONS) / len(DIMENSIONS), 3)
    return verdict


# --------------------------------------------------------------------------- providers


def _judge_openai(prompt: str, sheet: Path, page: Path, model: str) -> dict[str, Any]:
    key = require_key("OPENAI_API_KEY", "openai judge")
    sheet_b64, _ = to_data_uri(sheet)
    page_b64, _ = to_data_uri(page)
    payload = _post(
        "https://api.openai.com/v1/chat/completions",
        {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{sheet_b64}"}},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{page_b64}"}},
                    ],
                }
            ],
        },
        {"Authorization": f"Bearer {key}"},
    )
    try:
        return _parse(payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError) as exc:
        raise JudgeError(f"unexpected OpenAI response: {json.dumps(payload)[:300]}") from exc


def _judge_anthropic(prompt: str, sheet: Path, page: Path, model: str) -> dict[str, Any]:
    key = require_key("ANTHROPIC_API_KEY", "anthropic judge")
    sheet_b64, _ = to_data_uri(sheet)
    page_b64, _ = to_data_uri(page)
    image_block = lambda data: {  # noqa: E731
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }
    payload = _post(
        "https://api.anthropic.com/v1/messages",
        {
            "model": model,
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        image_block(sheet_b64),
                        image_block(page_b64),
                    ],
                }
            ],
        },
        {"x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    try:
        text = "".join(b.get("text", "") for b in payload["content"])
    except (KeyError, TypeError) as exc:
        raise JudgeError(f"unexpected Anthropic response: {json.dumps(payload)[:300]}") from exc
    return _parse(text)


def _judge_gemini(prompt: str, sheet: Path, page: Path, model: str) -> dict[str, Any]:
    try:
        from google import genai
    except ImportError as exc:
        raise JudgeError(
            "the gemini judge needs the google-genai SDK: pip install 'google-genai>=2.3.0'"
        ) from exc

    client = genai.Client(api_key=require_key("GEMINI_API_KEY", "gemini judge"))
    sheet_b64, mime = to_data_uri(sheet)
    page_b64, _ = to_data_uri(page)
    interaction = client.interactions.create(
        model=model,
        input=[
            {"type": "text", "text": prompt},
            {"type": "image", "data": sheet_b64, "mime_type": mime},
            {"type": "image", "data": page_b64, "mime_type": mime},
        ],
    )
    return _parse(interaction.output_text or "")


PROVIDERS = {
    "openai": _judge_openai,
    "anthropic": _judge_anthropic,
    "gemini": _judge_gemini,
}


# --------------------------------------------------------------------------- entry point


def score_run(
    records,
    run_root: Path,
    spec: BenchmarkSpec,
    provider: str,
    model: str | None = None,
    log=print,
    should_stop=None,
) -> None:
    """Attach a judge verdict to every record in place."""
    if provider == "none":
        log("  judge: disabled")
        return
    if provider not in PROVIDERS:
        raise JudgeError(
            f"unknown judge provider {provider!r}; valid: {', '.join(PROVIDERS)}, none"
        )

    model = model or DEFAULT_MODELS[provider]
    prompt = judge_prompt(spec)
    log(f"  judge: {provider}/{model}")

    stop = should_stop or (lambda: False)
    for record in records:
        if stop():
            log("    cancelled -- keeping the verdicts already collected")
            return
        # Keep good verdicts, but retry ones that previously errored.
        if record.error or (record.judge and "error" not in record.judge):
            continue
        sheet = run_root / record.backend / "sheet.png"
        page = run_root / record.image_path
        if not (sheet.is_file() and page.is_file()):
            continue
        try:
            verdict = PROVIDERS[provider](prompt, sheet, page, model)
            verdict["judge"] = f"{provider}/{model}"
            record.judge = verdict
            log(f"    {record.backend}/{record.strategy}/page {record.scene_id}: "
                f"identity={verdict['identity']} attributes={verdict['attributes']} "
                f"style={verdict['style']}")
        except Exception as exc:
            record.judge = {"error": f"{type(exc).__name__}: {exc}"}
            log(f"    {record.backend}/{record.strategy}/page {record.scene_id}: "
                f"judge failed -- {exc}")
