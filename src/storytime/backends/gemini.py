"""Google Gemini image backend -- the models marketed as "Nano Banana".

Uses the Interactions API (`client.interactions.create`), which replaced
`generate_content` as the recommended entry point. Reference images are passed
inline as base64 in the `input` list; the model is told what each one is for in
the prompt text, because the API has no separate "this is the character" slot.
"""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path

from ..config import require_key
from ..types import GenRequest
from .base import Backend, BackendError, explain_provider_error

# Model ids, cheapest first. Pass `--model` to pick one explicitly.
MODELS = {
    "flash-lite": "gemini-3.1-flash-lite-image",
    "flash": "gemini-3.1-flash-image",
    "pro": "gemini-3-pro-image",
    "legacy": "gemini-2.5-flash-image",
}


class GeminiBackend(Backend):
    name = "gemini"
    # 3-pro-image takes up to 5 character-consistency references; we never
    # send more than 2, so this is a generous ceiling.
    max_references = 5

    @property
    def default_model(self) -> str:
        return MODELS["pro"]

    def __init__(self, model: str | None = None) -> None:
        # Allow the short aliases above as well as full model ids.
        super().__init__(MODELS.get(model or "", model))
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - depends on install
                raise BackendError(
                    "the gemini backend needs the google-genai SDK: "
                    "pip install 'google-genai>=2.3.0'"
                ) from exc
            self._client = genai.Client(api_key=require_key("GEMINI_API_KEY", "gemini"))
        return self._client

    def _generate(self, req: GenRequest) -> tuple[bytes, str]:
        client = self._get_client()

        payload: list[dict] = [{"type": "text", "text": req.prompt}]
        for path in req.reference_images:
            payload.append(
                {
                    "type": "image",
                    "data": base64.b64encode(Path(path).read_bytes()).decode("ascii"),
                    "mime_type": mimetypes.guess_type(str(path))[0] or "image/png",
                }
            )

        interaction = client.interactions.create(
            model=self.model,
            input=payload,
            response_format={
                "type": "image",
                "aspect_ratio": req.output.aspect_ratio,
                "image_size": req.output.image_size,
            },
        )

        image = getattr(interaction, "output_image", None)
        if image is None or not getattr(image, "data", None):
            text = getattr(interaction, "output_text", "") or ""
            hint = explain_provider_error(text) or explain_provider_error("safety filter")
            raise BackendError(
                (f"{hint}\n\n" if hint else "")
                + "Gemini returned no image"
                + (f"; it said: {text.strip()[:300]}" if text.strip() else "")
            )

        # Gemini bills per image at a list price, so the spend meter can count
        # it directly. 4K is the more expensive tier.
        from ..pricing import FLAT_IMAGE_USD

        draft_usd, print_usd = FLAT_IMAGE_USD.get(self.model, (0.0, 0.0))
        usd = print_usd if req.output.image_size == "4K" else draft_usd
        self.last_usage = {"images": 1, **({"usd": usd} if usd else {})}

        return base64.b64decode(image.data), f"model={self.model}"


# ------------------------------------------------------------------- batch

#: terminal states of a Gemini batch job
BATCH_DONE = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
              "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def build_inline_requests(requests: list[GenRequest], uri_for) -> list[dict]:
    """The batch payload: one generateContent-style request per image.

    `uri_for(path)` maps a reference image to an uploaded file URI -- inline
    batches are capped at 20 MB, and the character sheet would otherwise ride
    along in every single request. Via the Files API it is uploaded once and
    referenced sixteen times. Part order mirrors the interactive call: prompt
    first, then the references in the order the prompt numbers them.
    """
    inline = []
    for req in requests:
        parts: list[dict] = [{"text": req.prompt}]
        for path in req.reference_images:
            uri, mime = uri_for(Path(path))
            parts.append({"file_data": {"file_uri": uri, "mime_type": mime}})
        inline.append({
            "contents": [{"parts": parts, "role": "user"}],
            "config": {
                "response_modalities": ["TEXT", "IMAGE"],
                "image_config": {
                    "aspect_ratio": req.output.aspect_ratio,
                    "image_size": req.output.image_size,
                },
            },
        })
    return inline


def run_batch(
    model: str | None,
    requests: list[GenRequest],
    targets: list[Path],
    log=print,
    should_stop=None,
    on_progress=None,
    poll_s: int = 20,
) -> list[dict]:
    """Submit every image as one batch job and wait it out.

    Half the interactive price, in exchange for Google scheduling the work
    into spare capacity -- minutes to hours, guaranteed within 24 h. Returns
    one dict per request, aligned with the input order: either
    {"data": bytes, "usage": {...}} or {"error": str}.
    """
    from ..pricing import FLAT_IMAGE_USD

    stop = should_stop or (lambda: False)
    backend = GeminiBackend(model)
    client = backend._get_client()

    # Upload each unique reference once; sixteen pages share one sheet.
    uploaded: dict[Path, tuple[str, str]] = {}

    def uri_for(path: Path) -> tuple[str, str]:
        if path not in uploaded:
            handle = client.files.upload(file=str(path))
            uploaded[path] = (handle.uri, handle.mime_type or "image/png")
        return uploaded[path]

    unique_refs = {Path(p) for req in requests for p in req.reference_images}
    log(f"{len(requests)} Bilder werden als Batch eingereicht "
        f"({len(unique_refs)} Referenzbilder, je einmal hochgeladen) ...")
    job = client.batches.create(
        model=backend.model,
        src=build_inline_requests(requests, uri_for),
        config={"display_name": "storytime-batch"},
    )
    log(f"Batch angenommen: {job.name}")
    log("Google nimmt sich Zeit -- Minuten bis Stunden, dafuer der halbe "
        "Preis. Dieser Auftrag wartet und sammelt das Ergebnis ein.")

    last_state = ""
    while True:
        if stop():
            try:
                client.batches.cancel(name=job.name)
                log("Batch bei Google abgebrochen.")
            except Exception as exc:
                log(f"Abbrechen fehlgeschlagen: {exc}")
            return [{"error": "abgebrochen"} for _ in requests]
        job = client.batches.get(name=job.name)
        state = getattr(job.state, "name", str(job.state))
        if state != last_state:
            log(f"  Status: {state.replace('JOB_STATE_', '').lower()}")
            last_state = state
        stats = getattr(job, "batch_stats", None)
        if stats is not None and on_progress:
            done = int(getattr(stats, "succeeded_request_count", 0) or 0)                 + int(getattr(stats, "failed_request_count", 0) or 0)
            on_progress(done, len(requests))
        if state in BATCH_DONE:
            break
        time.sleep(poll_s)

    if state != "JOB_STATE_SUCCEEDED":
        raise BackendError(
            f"Der Batch endete als {state}. "
            + str(getattr(job, "error", "") or "")[:300]
        )

    # Batch bills at half the list price -- that is the whole point.
    draft_usd, print_usd = FLAT_IMAGE_USD.get(backend.model, (0.0, 0.0))
    results: list[dict] = []
    responses = list(getattr(job.dest, "inlined_responses", None) or [])
    for position, req in enumerate(requests):
        entry = responses[position] if position < len(responses) else None
        response = getattr(entry, "response", None) if entry else None
        if response is None:
            reason = str(getattr(entry, "error", "") or "keine Antwort")[:200]
            results.append({"error": reason})
            continue
        data = None
        try:
            for part in response.candidates[0].content.parts:
                blob = getattr(part, "inline_data", None)
                if blob is not None and getattr(blob, "data", None):
                    raw = blob.data
                    data = raw if isinstance(raw, bytes) else base64.b64decode(raw)
                    break
        except (AttributeError, IndexError):
            pass
        if data is None:
            results.append({"error": "die Antwort enthielt kein Bild "
                                     "(vermutlich Sicherheitsfilter)"})
            continue
        usd = (print_usd if req.output.image_size == "4K" else draft_usd) / 2
        results.append({"data": data, "usage": {
            "images": 1, "model": backend.model, "backend": "gemini",
            **({"usd": round(usd, 4)} if usd else {}),
        }})
    return results
