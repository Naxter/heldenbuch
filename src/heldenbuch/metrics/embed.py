"""Optional embedding metric: DINOv2 cosine similarity between page and sheet.

DINOv2 is a self-supervised vision model whose embeddings capture "what is in
this picture and roughly how does it look" without being trained on labels. The
cosine similarity between two embeddings is a single number for how visually
alike two images are.

Caveat worth stating plainly: this compares the *whole image*, so a page with a
faithful character in a dark forest scores lower than a page with a wrong
character on a plain background. It is a useful second opinion, not a verdict.
Treat it as a style/continuity signal and let the VLM judge rule on identity.

Needs `pip install 'heldenbuch[embed]'` (torch + transformers, ~2.5 GB).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..imageutil import load_rgb

MODEL_ID = "facebook/dinov2-base"


class EmbedUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_model():
    try:
        import torch  # noqa: F401
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise EmbedUnavailable(
            "the embedding metric needs torch and transformers: "
            "pip install 'heldenbuch[embed]'"
        ) from exc
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval()
    return processor, model


def embed(path: Path | str):
    import torch

    processor, model = _load_model()
    image = load_rgb(path, max_edge=518)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # The CLS token is the pooled representation of the whole image.
    vector = outputs.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(vector, dim=-1)[0]


def similarity(page_path: Path | str, sheet_path: Path | str) -> float:
    import torch

    return float(torch.dot(embed(page_path), embed(sheet_path)))


def score_run(records, run_root: Path, log=print) -> None:
    """Attach `dino_cosine` to every record in place. Skips silently if unavailable."""
    try:
        _load_model()
    except EmbedUnavailable as exc:
        log(f"  embedding metric skipped: {exc}")
        return

    for record in records:
        if record.error:
            continue
        sheet = run_root / record.backend / "sheet.png"
        page = run_root / record.image_path
        if sheet.is_file() and page.is_file():
            record.metrics["dino_cosine"] = round(similarity(page, sheet), 4)
