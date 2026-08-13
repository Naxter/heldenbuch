"""OpenAI image backend (GPT Image).

Two endpoints are used depending on the request:

  * no reference images -> POST /v1/images/generations
  * one or more references -> POST /v1/images/edits (multipart, `image[]`)

`gpt-image-2` handles every input image at high fidelity automatically, so
there is no `input_fidelity` knob to set. Only the *first* reference gets the
extra texture-preservation pass, which is why the pipeline always puts the
character sheet first.

Uses only the standard library -- the multipart body is built by hand below.
"""

from __future__ import annotations

import json
import mimetypes
import os
import urllib.error
import urllib.request
import uuid
from base64 import b64decode
from pathlib import Path

from ..config import require_key
from ..types import GenRequest
from .base import Backend, BackendError, explain_provider_error

API_BASE = "https://api.openai.com/v1"

MODELS = {
    "2": "gpt-image-2",
    "1.5": "gpt-image-1.5",
    "1": "gpt-image-1",
    "mini": "gpt-image-1-mini",
}

# gpt-image-2 takes an arbitrary size as long as both edges are divisible by 16
# and the longest is at most 3840. That matters: a 21.6 cm page at 300 dpi
# needs 2624 px, which the old three fixed sizes could not reach.
SIZE_STEP = 16
MAX_EDGE = 3840
MIN_EDGE = 256

# The older models only accept these three.
LEGACY_SIZES = {"1024x1024": 1.0, "1536x1024": 1.5, "1024x1536": 1 / 1.5}
LEGACY_MODELS = {"gpt-image-1", "gpt-image-1-mini"}


def _snap_legacy(aspect_ratio: str) -> str:
    try:
        a, b = (float(x) for x in aspect_ratio.split(":"))
        wanted = a / b
    except (ValueError, ZeroDivisionError):
        wanted = 1.0
    return min(LEGACY_SIZES, key=lambda s: abs(LEGACY_SIZES[s] - wanted))


def _exact_size(output) -> str:
    """Honour the requested pixels, rounded to what the API will accept."""
    width, height = output.pixel_size()
    longest = max(width, height)
    if longest > MAX_EDGE:  # scale the whole thing down, keep the ratio
        scale = MAX_EDGE / longest
        width, height = int(width * scale), int(height * scale)

    def snap(value: int) -> int:
        return max(MIN_EDGE, min(MAX_EDGE, int(round(value / SIZE_STEP)) * SIZE_STEP))

    return f"{snap(width)}x{snap(height)}"


def _encode_multipart(fields: dict[str, str], files: list[tuple[str, Path]]) -> tuple[bytes, str]:
    """Build a multipart/form-data body. Returns (body, content_type)."""
    boundary = f"----storytime{uuid.uuid4().hex}"
    parts: list[bytes] = []

    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )

    for name, path in files:
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        header = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
            f'filename="{path.name}"\r\nContent-Type: {mime}\r\n\r\n'
        ).encode()
        parts.append(header + path.read_bytes() + b"\r\n")

    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _request(url: str, body: bytes, content_type: str, key: str, timeout: int = 300) -> dict:
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": content_type, "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        if exc.code == 429 or exc.code >= 500:
            raise RuntimeError(f"OpenAI transient error ({exc.code}): {detail}") from exc
        # Surface the human-readable message, not the JSON envelope -- and
        # lead with advice when we recognise the failure.
        try:
            message = json.loads(detail).get("error", {}).get("message") or detail
        except (json.JSONDecodeError, AttributeError):
            message = detail
        hint = explain_provider_error(detail)
        raise BackendError(
            (f"{hint}\n\n" if hint else "") + f"OpenAI ({exc.code}): {message[:300]}"
        ) from exc


class OpenAIBackend(Backend):
    name = "openai"
    max_references = 8

    @property
    def default_model(self) -> str:
        return "gpt-image-2"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(MODELS.get(model or "", model))

    def _generate(self, req: GenRequest) -> tuple[bytes, str]:
        key = require_key("OPENAI_API_KEY", "openai")
        size = (
            _snap_legacy(req.output.aspect_ratio)
            if self.model in LEGACY_MODELS
            else _exact_size(req.output)
        )

        quality = req.output.quality

        if req.reference_images:
            fields = {
                "model": self.model,
                "prompt": req.prompt,
                "size": size,
                "quality": quality,
                "n": "1",
            }
            files = [("image[]", Path(p)) for p in req.reference_images]
            body, content_type = _encode_multipart(fields, files)
            payload = _request(f"{API_BASE}/images/edits", body, content_type, key)
        else:
            body = json.dumps(
                {
                    "model": self.model,
                    "prompt": req.prompt,
                    "size": size,
                    "quality": quality,
                    "n": 1,
                }
            ).encode("utf-8")
            payload = _request(f"{API_BASE}/images/generations", body, "application/json", key)

        items = payload.get("data") or []
        if not items or not items[0].get("b64_json"):
            raise BackendError(f"OpenAI returned no image data: {json.dumps(payload)[:400]}")

        note = f"model={self.model} size={size} quality={quality}"
        usage = payload.get("usage") or {}
        if usage:
            self.last_usage = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "images": 1,
            }
            note += f" tokens={usage.get('total_tokens')}"
        return b64decode(items[0]["b64_json"]), note


def has_key() -> bool:
    """Cheap check used by the CLI to give a useful message before spending money."""
    from ..config import load_dotenv

    load_dotenv()
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())
