"""The interface every image backend implements.

A backend takes a `GenRequest` (prompt + zero or more reference images) and
writes one PNG/JPEG to disk. Everything model-specific -- auth, request shape,
polling -- stays inside the backend.
"""

from __future__ import annotations

import abc
import os
import time
from pathlib import Path

from ..types import GenRequest, GenResult


class BackendError(RuntimeError):
    """Raised when generation fails in a way retrying will not fix."""


def explain_provider_error(detail: str) -> str | None:
    """A provider error, said in words a person can act on.

    The raw payloads are written for developers; the person reading the job
    drawer is a parent whose page just failed. The child-safety refusal is
    the important one: image services are deliberately cautious around
    children and visibly inconsistent about it, so the honest advice is
    "try again, soften photo-language, or switch the service" -- not a JSON
    blob.
    """
    lowered = (detail or "").lower()
    if any(term in lowered for term in
           ("moderation", "safety system", "safety_", "content_policy",
            "content policy", "safety filter", "content moderated",
            "request moderated")):
        return (
            "Die Sicherheitsprüfung des Bilddienstes hat dieses Bild abgelehnt. "
            "Bei Kinderfiguren passiert das gelegentlich — die Prüfung ist "
            "bewusst vorsichtig und nicht immer konsistent. Was hilft, in "
            "dieser Reihenfolge: (1) einfach noch einmal zeichnen lassen, oft "
            "geht es beim zweiten Versuch durch; (2) foto-nahe Wörter aus "
            "Beschreibung und Stil nehmen („photo“, „photorealistic“, "
            "„realistic“) — eine klar gezeichnete Figur wird selten "
            "beanstandet; (3) für dieses Bild einen anderen Bilddienst wählen."
        )
    if "insufficient_quota" in lowered or "billing" in lowered or "exceeded your current quota" in lowered:
        return ("Das Guthaben bei diesem Dienst ist aufgebraucht oder die "
                "Zahlungsdaten fehlen — das löst sich nur im Konto des "
                "Anbieters, nicht hier.")
    if "rate limit" in lowered or "rate_limit" in lowered or "too many requests" in lowered:
        return ("Zu viele Anfragen kurz hintereinander. Eine Minute warten "
                "und noch einmal starten; bei großen Büchern hilft es, die "
                "Zahl gleichzeitiger Bilder zu senken.")
    return None


class Backend(abc.ABC):
    #: short id used on the command line and in output paths
    name: str = "base"
    #: how many reference images this backend accepts in one call
    max_references: int = 0

    def __init__(self, model: str | None = None) -> None:
        self.model = model or self.default_model

    @property
    @abc.abstractmethod
    def default_model(self) -> str: ...

    #: usage the last call reported, picked up by `generate` into the result
    last_usage: dict = {}

    @abc.abstractmethod
    def _generate(self, req: GenRequest) -> tuple[bytes, str]:
        """Return (image_bytes, cost_note). Raise BackendError on failure."""

    def generate(self, req: GenRequest, out_path: Path, retries: int = 2) -> GenResult:
        """Generate one image, with a short backoff on transient failures."""
        if len(req.reference_images) > self.max_references:
            raise BackendError(
                f"{self.name} accepts at most {self.max_references} reference "
                f"image(s), got {len(req.reference_images)}"
            )

        started = time.monotonic()
        last_error: Exception | None = None
        self.last_usage = {}

        for attempt in range(retries + 1):
            try:
                data, cost_note = self._generate(req)
            except BackendError:
                raise
            except Exception as exc:  # network hiccups, 5xx, timeouts
                last_error = exc
                if attempt < retries:
                    time.sleep(2 ** attempt * 2)
                    continue
                break
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                # Write beside the target and rename. A page file that exists
                # is treated as finished work and is never redrawn, so a render
                # interrupted mid-write would otherwise leave a truncated PNG
                # that the next run adopts as a completed page.
                tmp = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")
                try:
                    tmp.write_bytes(data)
                    os.replace(tmp, out_path)
                except BaseException:
                    tmp.unlink(missing_ok=True)
                    raise
                return GenResult(
                    image_path=out_path,
                    backend=self.name,
                    model=self.model,
                    prompt=req.prompt,
                    reference_images=[str(p) for p in req.reference_images],
                    latency_s=round(time.monotonic() - started, 2),
                    cost_note=cost_note,
                    usage={"model": self.model, "backend": self.name, **self.last_usage},
                )

        raise BackendError(f"{self.name} failed after {retries + 1} attempts: {last_error}")
