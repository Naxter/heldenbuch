"""Local ComfyUI backend -- your own GPU draws, and nothing leaves the machine.

Built for FLUX.2 klein on a 12 GB card (RTX 3060 class), but it actually runs
whatever workflow you give it: the backend does not hardcode a node graph,
because graphs differ between ComfyUI versions and model files. Instead it
fills placeholders into a workflow *you* exported from a working setup. That
way "it works in ComfyUI" is the only requirement.

Setup, once:

  1. Install ComfyUI and get FLUX.2 klein generating -- ideally from the
     official FLUX.2 template with one or more reference images (that is what
     keeps the character consistent).
  2. In ComfyUI: settings -> enable dev mode, then "Export (API)" and save the
     file as `comfy_workflow.json` in the StoryTime project root.
  3. Open that file and replace values with placeholders:
        the positive prompt text          -> "{PROMPT}"
        width / height numbers            -> "{WIDTH}" / "{HEIGHT}"
        the seed (or noise_seed)          -> "{SEED}"
        each LoadImage node's "image"     -> "{IMAGE_1}", "{IMAGE_2}", ...
     The number of {IMAGE_n} slots is how many reference images the backend
     will accept. Pages send 1-4 (hero sheet, cast, place).
  4. Optional: a second file `comfy_workflow_t2i.json` with no {IMAGE_n} slots
     is used for calls without references (a character sheet drawn from a
     description alone).

Environment (optional, `.env`):
  COMFY_URL       where ComfyUI listens; default http://127.0.0.1:8188
  COMFY_WORKFLOW  path to the workflow file; default <project>/comfy_workflow.json

ComfyUI caches identical jobs, so when no seed is pinned a random one is used
per call -- otherwise "neu zeichnen" would return the same picture forever.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, load_dotenv
from ..types import GenRequest
from .base import Backend, BackendError

DEFAULT_URL = "http://127.0.0.1:8188"
DEFAULT_WORKFLOW = PROJECT_ROOT / "comfy_workflow.json"
DEFAULT_WORKFLOW_T2I = PROJECT_ROOT / "comfy_workflow_t2i.json"

_IMAGE_SLOT = re.compile(r"\{IMAGE_(\d+)\}")

SETUP_HELP = (
    "The comfy backend needs a workflow exported from your own ComfyUI: enable "
    "dev mode, 'Export (API)', save as comfy_workflow.json in the project "
    "root, then replace the prompt with \"{PROMPT}\", width/height with "
    "\"{WIDTH}\"/\"{HEIGHT}\", the seed with \"{SEED}\" and each LoadImage's "
    "image with \"{IMAGE_1}\", \"{IMAGE_2}\", ... See backends/comfy.py."
)


def _base_url() -> str:
    load_dotenv()
    return (os.environ.get("COMFY_URL", "").strip() or DEFAULT_URL).rstrip("/")


def _workflow_path() -> Path:
    load_dotenv()
    custom = os.environ.get("COMFY_WORKFLOW", "").strip()
    return Path(custom) if custom else DEFAULT_WORKFLOW


def is_available(timeout: float = 1.5) -> bool:
    """Is a ComfyUI server answering? Cheap enough to ask in a status call."""
    try:
        with urllib.request.urlopen(f"{_base_url()}/system_stats", timeout=timeout):
            return True
    except Exception:
        return False


# ------------------------------------------------------------------ workflow


def load_workflow(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BackendError(f"workflow file not found: {path}\n{SETUP_HELP}")
    try:
        workflow = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BackendError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(workflow, dict):
        raise BackendError(f"{path} must contain the API-format workflow object")
    if '"{PROMPT}"' not in json.dumps(workflow):
        raise BackendError(f"{path} has no \"{{PROMPT}}\" placeholder.\n{SETUP_HELP}")
    return workflow


def image_slots(workflow: dict[str, Any]) -> int:
    """How many {IMAGE_n} placeholders the workflow carries."""
    found = _IMAGE_SLOT.findall(json.dumps(workflow))
    return max((int(n) for n in found), default=0)


def fill_workflow(
    workflow: dict[str, Any],
    prompt: str,
    width: int,
    height: int,
    seed: int,
    image_names: list[str],
) -> dict[str, Any]:
    """Substitute every placeholder. Unfilled {IMAGE_n} slots repeat the first
    reference (the character sheet) -- harmless for identity, and it keeps a
    four-slot workflow usable on a one-reference page."""

    def replace(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, str):
            if value == "{PROMPT}":
                return prompt
            if value == "{WIDTH}":
                return width
            if value == "{HEIGHT}":
                return height
            if value == "{SEED}":
                return seed
            match = _IMAGE_SLOT.fullmatch(value)
            if match:
                index = int(match.group(1)) - 1
                if index < len(image_names):
                    return image_names[index]
                if image_names:
                    return image_names[0]
                raise BackendError(
                    "this workflow expects reference images, but the call "
                    "brought none. Add a comfy_workflow_t2i.json without "
                    "{IMAGE_n} slots for reference-free drawing."
                )
        return value

    return replace(workflow)


# ------------------------------------------------------------------ http


def _get_json(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, body: dict, timeout: int = 60) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:600]
        raise BackendError(f"ComfyUI rejected the job ({exc.code}): {detail}") from exc


class ComfyBackend(Backend):
    name = "comfy"
    # Refined from the workflow's {IMAGE_n} slots when it loads; a generous
    # default so pre-flight checks pass before the file exists.
    max_references = 8

    @property
    def default_model(self) -> str:
        return "flux-2-klein"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(model)
        try:  # missing file only matters once something is actually drawn
            self.max_references = image_slots(load_workflow(_workflow_path()))
        except BackendError:
            pass

    # ---------------------------------------------------------------- pieces

    def _upload(self, path: Path) -> str:
        """Push one reference image into ComfyUI's input folder.

        Uploaded under a unique name: pages render four at a time, and two
        books both calling their sheet `sheet.png` must not overwrite each
        other mid-flight.
        """
        boundary = f"----storytime{uuid.uuid4().hex}"
        stored = f"storytime_{uuid.uuid4().hex[:10]}_{path.name}"
        body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
            f'filename="{stored}"\r\nContent-Type: image/png\r\n\r\n'
        ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()

        request = urllib.request.Request(
            f"{_base_url()}/upload/image", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        name = payload.get("name") or stored
        subfolder = payload.get("subfolder") or ""
        return f"{subfolder}/{name}" if subfolder else name

    def _poll(self, prompt_id: str, timeout_s: int = 1500) -> dict:
        """Wait for the job. A 3060 needs a minute per draft image and much
        longer at print size, so the ceiling is generous."""
        deadline = time.monotonic() + timeout_s
        delay = 1.0
        while time.monotonic() < deadline:
            history = _get_json(f"{_base_url()}/history/{prompt_id}")
            entry = history.get(prompt_id)
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    messages = json.dumps(status.get("messages", []))[:600]
                    raise BackendError(f"ComfyUI reported an error: {messages}")
                outputs = entry.get("outputs") or {}
                if outputs:
                    return outputs
            time.sleep(delay)
            delay = min(delay * 1.3, 4.0)
        raise BackendError(f"ComfyUI job did not finish within {timeout_s}s")

    def _fetch(self, image: dict) -> bytes:
        query = urllib.parse.urlencode({
            "filename": image.get("filename", ""),
            "subfolder": image.get("subfolder", ""),
            "type": image.get("type", "output"),
        })
        with urllib.request.urlopen(f"{_base_url()}/view?{query}", timeout=120) as response:
            return response.read()

    # -------------------------------------------------------------- generate

    def _generate(self, req: GenRequest) -> tuple[bytes, str]:
        if not is_available():
            raise BackendError(
                f"no ComfyUI server at {_base_url()} -- start ComfyUI, or set "
                "COMFY_URL in .env if it listens elsewhere."
            )

        # No references and a reference workflow do not mix; fall back to the
        # text-to-image workflow when one is provided.
        path = _workflow_path()
        if not req.reference_images and DEFAULT_WORKFLOW_T2I.is_file():
            path = DEFAULT_WORKFLOW_T2I
        workflow = load_workflow(path)

        if req.reference_images and not image_slots(workflow):
            raise BackendError(
                f"{path} has no {{IMAGE_n}} slots, but this call carries "
                f"{len(req.reference_images)} reference image(s). References "
                "are what keep the character consistent -- export a workflow "
                "that loads them."
            )

        names = [self._upload(Path(p)) for p in req.reference_images]
        width, height = req.output.pixel_size()
        # ComfyUI returns the cached result for an identical graph, which
        # would make every redraw a no-op -- so unpinned seeds are random.
        seed = req.output.seed if req.output.seed is not None else random.randrange(2**31)

        filled = fill_workflow(workflow, req.prompt, width, height, seed, names)
        submitted = _post_json(f"{_base_url()}/prompt",
                               {"prompt": filled, "client_id": f"storytime-{uuid.uuid4().hex[:8]}"})
        if submitted.get("node_errors"):
            raise BackendError(f"ComfyUI node errors: {json.dumps(submitted['node_errors'])[:600]}")
        prompt_id = submitted.get("prompt_id")
        if not prompt_id:
            raise BackendError(f"ComfyUI returned no prompt_id: {json.dumps(submitted)[:300]}")

        outputs = self._poll(prompt_id)
        for node_output in outputs.values():
            for image in node_output.get("images", []):
                if image.get("type") == "output":
                    self.last_usage = {"images": 1, "usd": 0.0}
                    return self._fetch(image), (
                        f"model={self.model} local seed={seed} {width}x{height}"
                    )
        raise BackendError(f"ComfyUI finished but produced no image: {json.dumps(outputs)[:400]}")
