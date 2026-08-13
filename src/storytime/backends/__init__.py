"""Backend registry.

Backends are imported lazily so that a missing optional SDK only breaks the
backend that needs it, not the whole CLI.
"""

from __future__ import annotations

from .base import Backend, BackendError

BACKEND_NAMES = ("stub", "gemini", "bfl", "openai", "comfy")

#: env var each backend needs, for pre-flight checks in the CLI.
#: `comfy` needs no key -- it needs a running local ComfyUI server instead;
#: check that with `comfy_available()`.
REQUIRED_KEY = {
    "stub": None,
    "gemini": "GEMINI_API_KEY",
    "bfl": "BFL_API_KEY",
    "openai": "OPENAI_API_KEY",
    "comfy": None,
}


def comfy_available() -> bool:
    """Is a local ComfyUI server answering? Imported lazily so the check does
    not load anything unless someone actually asks."""
    from .comfy import is_available

    return is_available()


def get_backend(name: str, model: str | None = None) -> Backend:
    if name == "stub":
        from .stub import StubBackend

        return StubBackend(model)
    if name == "comfy":
        from .comfy import ComfyBackend

        return ComfyBackend(model)
    if name == "gemini":
        from .gemini import GeminiBackend

        return GeminiBackend(model)
    if name == "bfl":
        from .bfl import BflBackend

        return BflBackend(model)
    if name == "openai":
        from .openai import OpenAIBackend

        return OpenAIBackend(model)
    raise BackendError(f"unknown backend {name!r}; valid: {', '.join(BACKEND_NAMES)}")


__all__ = ["Backend", "BackendError", "BACKEND_NAMES", "REQUIRED_KEY",
           "comfy_available", "get_backend"]
