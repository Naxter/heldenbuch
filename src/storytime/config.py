"""Loading the benchmark spec and the API keys.

`.env` parsing is done by hand rather than pulling in python-dotenv -- it is
twelve lines and one less dependency.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .types import BenchmarkSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = PROJECT_ROOT / "benchmark.yaml"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"
#: heroes, styles and finished books
DEFAULT_LIBRARY = PROJECT_ROOT / "library"


def load_dotenv(path: Path | None = None) -> None:
    """Read `.env` into os.environ. Real environment variables win."""
    path = path or (PROJECT_ROOT / ".env")
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def load_spec(path: Path | None = None) -> BenchmarkSpec:
    path = path or DEFAULT_SPEC
    if not path.is_file():
        raise FileNotFoundError(f"benchmark spec not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return BenchmarkSpec.from_dict(raw)


def require_key(name: str, backend: str) -> str:
    load_dotenv()
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set, which the '{backend}' backend needs.\n"
            f"Put it in {PROJECT_ROOT / '.env'} (see .env.example) or export it."
        )
    return value
