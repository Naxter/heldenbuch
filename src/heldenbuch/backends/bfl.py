"""Black Forest Labs (FLUX) image backend.

FLUX is asynchronous: you POST a job, get back a `polling_url`, and poll it
until the status flips to `Ready`. The finished image sits behind a signed URL
that expires after about ten minutes, so we download it straight away.

Reference images go in as base64 in `input_image` .. `input_image_8`, and the
prompt refers to them as "image 1", "image 2" and so on -- that numbering is
how FLUX.2 knows which reference is which.

Uses only the standard library, so no extra install is needed.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..config import require_key
from ..types import GenRequest
from .base import Backend, BackendError, explain_provider_error

API_BASE = "https://api.bfl.ai/v1"

# endpoint slug -> how many reference images it accepts
MODELS = {
    "flux-2-pro": 8,
    "flux-2-max": 8,
    "flux-2-flex": 8,
    "flux-2-klein-9b": 4,
    "flux-2-klein-4b": 4,
    "flux-kontext-pro": 1,
    "flux-kontext-max": 1,
}

# Statuses BFL can report. Anything not "Ready" or "Pending" is terminal.
TERMINAL_FAILURES = {
    "Error",
    "Request Moderated",
    "Content Moderated",
    "Task not found",
    "Task Failed",
}


def _post_json(url: str, body: dict, key: str, timeout: int = 60) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-key": key, "accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        # 4xx will not fix itself on retry; 5xx might.
        if 400 <= exc.code < 500:
            raise BackendError(f"FLUX rejected the request ({exc.code}): {detail}") from exc
        raise RuntimeError(f"FLUX server error ({exc.code}): {detail}") from exc


def _get_json(url: str, key: str, timeout: int = 30) -> dict:
    request = urllib.request.Request(url, headers={"x-key": key, "accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class BflBackend(Backend):
    name = "bfl"

    @property
    def default_model(self) -> str:
        return "flux-2-pro"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(model)
        if self.model not in MODELS:
            raise BackendError(
                f"unknown FLUX model {self.model!r}; valid: {', '.join(sorted(MODELS))}"
            )
        self.max_references = MODELS[self.model]
        self.honours_seed = True

    def _generate(self, req: GenRequest) -> tuple[bytes, str]:
        key = require_key("BFL_API_KEY", "bfl")
        width, height = req.output.pixel_size()

        body: dict[str, object] = {
            "prompt": req.prompt,
            "width": width,
            "height": height,
            "output_format": "png",
            "safety_tolerance": 2,
        }
        if req.output.seed is not None:
            body["seed"] = req.output.seed

        for index, path in enumerate(req.reference_images, start=1):
            field = "input_image" if index == 1 else f"input_image_{index}"
            body[field] = base64.b64encode(Path(path).read_bytes()).decode("ascii")

        submitted = _post_json(f"{API_BASE}/{self.model}", body, key)
        polling_url = submitted.get("polling_url")
        if not polling_url:
            raise BackendError(f"FLUX returned no polling_url: {submitted}")
        cost = submitted.get("cost")

        image_url = self._poll(polling_url, key)
        with urllib.request.urlopen(image_url, timeout=120) as response:
            data = response.read()

        note = f"model={self.model}"
        if cost is not None:
            # One BFL credit is one US cent.
            self.last_usage = {"credits": float(cost), "usd": float(cost) * 0.01, "images": 1}
            note += f" cost={cost} credits (~${float(cost) * 0.01:.3f})"
        else:
            self.last_usage = {"images": 1}
        return data, note

    def _poll(self, polling_url: str, key: str, timeout_s: int = 300) -> str:
        deadline = time.monotonic() + timeout_s
        delay = 1.0
        while time.monotonic() < deadline:
            payload = _get_json(polling_url, key)
            status = payload.get("status", "")
            if status == "Ready":
                sample = (payload.get("result") or {}).get("sample")
                if not sample:
                    raise BackendError(f"FLUX reported Ready but sent no image: {payload}")
                return sample
            if status in TERMINAL_FAILURES:
                hint = explain_provider_error(status)
                raise BackendError(
                    (f"{hint}\n\n" if hint else "")
                    + f"FLUX job ended as {status!r}: {payload.get('details') or payload}"
                )
            time.sleep(delay)
            delay = min(delay * 1.4, 5.0)  # back off, but keep checking
        raise BackendError(f"FLUX job did not finish within {timeout_s}s")
