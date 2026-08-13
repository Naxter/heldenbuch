"""Plain data structures shared across the package.

Everything here is a dataclass with no behaviour beyond parsing, so that a run
can be serialised to JSON and inspected afterwards without the code that made
it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# The three ways we condition the model on the character. This is the
# independent variable of the experiment.
STRATEGIES = ("text_only", "sheet_ref", "sheet_plus_prev")


@dataclass(frozen=True)
class Character:
    name: str
    description: str
    sheet_prompt: str | None = None


@dataclass(frozen=True)
class Style:
    name: str
    description: str


@dataclass(frozen=True)
class OutputSpec:
    aspect_ratio: str = "4:3"
    image_size: str = "1K"
    seed: int | None = None
    # OpenAI only: low | medium | high. Drives both price and render time --
    # high is roughly four times the cost of medium per image.
    quality: str = "medium"
    # Explicit long edge in pixels, overriding `image_size`. Print needs exact
    # numbers -- a 21.6 cm page at 300 dpi is 2624 px, which no size label
    # happens to land on.
    long_edge_px: int | None = None

    def pixel_size(self) -> tuple[int, int]:
        """Turn `image_size` + `aspect_ratio` into concrete width/height.

        Gemini takes the size as a label ("1K"); FLUX and the GPT Image
        endpoints take explicit pixels, so we need both. Sizes are rounded to a
        multiple of 32, which satisfies every model here (GPT Image wants
        multiples of 16, FLUX wants 64-aligned-ish, 32 covers both).
        """
        long_edge = self.long_edge_px or {
            "512px": 512, "1K": 1024, "2K": 2048, "4K": 4096
        }.get(self.image_size, 1024)
        try:
            a, b = (int(x) for x in self.aspect_ratio.split(":"))
        except ValueError:
            a, b = 1, 1
        if a >= b:
            w, h = long_edge, round(long_edge * b / a)
        else:
            w, h = round(long_edge * a / b), long_edge
        snap = lambda v: max(64, int(round(v / 32)) * 32)  # noqa: E731
        return snap(w), snap(h)


@dataclass(frozen=True)
class Scene:
    id: str
    action: str
    stress: str = ""


@dataclass(frozen=True)
class JudgeSpec:
    provider: str = "gemini"  # gemini | anthropic | none
    model: str | None = None


@dataclass(frozen=True)
class Experiment:
    backends: list[str] = field(default_factory=lambda: ["stub"])
    strategies: list[str] = field(default_factory=lambda: list(STRATEGIES))


@dataclass(frozen=True)
class BenchmarkSpec:
    run_name: str
    character: Character
    style: Style
    output: OutputSpec
    scenes: list[Scene]
    experiment: Experiment
    judge: JudgeSpec

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> BenchmarkSpec:
        missing = [k for k in ("run_name", "character", "style", "scenes") if k not in raw]
        if missing:
            raise ValueError(f"benchmark spec is missing required keys: {', '.join(missing)}")

        char = raw["character"]
        if not char.get("name") or not char.get("description"):
            raise ValueError("character needs both a 'name' and a 'description'")

        scenes = [
            Scene(
                id=str(s["id"]),
                action=str(s["action"]).strip(),
                stress=str(s.get("stress", "")).strip(),
            )
            for s in raw["scenes"]
        ]
        if not scenes:
            raise ValueError("the spec contains no scenes")
        if len({s.id for s in scenes}) != len(scenes):
            raise ValueError("scene ids must be unique")

        exp_raw = raw.get("experiment", {}) or {}
        strategies = [str(s) for s in exp_raw.get("strategies", list(STRATEGIES))]
        unknown = [s for s in strategies if s not in STRATEGIES]
        if unknown:
            raise ValueError(
                f"unknown strategy {unknown!r}; valid: {', '.join(STRATEGIES)}"
            )

        judge_raw = raw.get("judge", {}) or {}
        out_raw = raw.get("output", {}) or {}

        return BenchmarkSpec(
            run_name=str(raw["run_name"]),
            character=Character(
                name=str(char["name"]),
                description=str(char["description"]).strip(),
                sheet_prompt=(str(char["sheet_prompt"]).strip() if char.get("sheet_prompt") else None),
            ),
            style=Style(
                name=str(raw["style"].get("name", "unnamed")),
                description=str(raw["style"]["description"]).strip(),
            ),
            output=OutputSpec(
                aspect_ratio=str(out_raw.get("aspect_ratio", "4:3")),
                image_size=str(out_raw.get("image_size", "1K")),
                seed=(int(out_raw["seed"]) if out_raw.get("seed") is not None else None),
                quality=str(out_raw.get("quality", "medium")),
                long_edge_px=(
                    int(out_raw["long_edge_px"]) if out_raw.get("long_edge_px") else None
                ),
            ),
            scenes=scenes,
            experiment=Experiment(
                backends=[str(b) for b in exp_raw.get("backends", ["stub"])],
                strategies=strategies,
            ),
            judge=JudgeSpec(
                provider=str(judge_raw.get("provider", "gemini")),
                model=(str(judge_raw["model"]) if judge_raw.get("model") else None),
            ),
        )


@dataclass
class GenRequest:
    """One call to an image backend."""

    prompt: str
    reference_images: list[Path] = field(default_factory=list)
    output: OutputSpec = field(default_factory=OutputSpec)
    # Free-text labels for what each reference image is, in the same order.
    # Backends that support named roles (Gemini) use these; others ignore them.
    reference_roles: list[str] = field(default_factory=list)
    # "sheet" or "page". Real backends ignore this -- the prompt already says
    # what to draw. The stub uses it to draw a plain-background sheet.
    kind: str = "page"


@dataclass
class GenResult:
    """What a backend gives back, plus enough metadata to audit the run."""

    image_path: Path
    backend: str
    model: str
    prompt: str
    reference_images: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    cost_note: str = ""
    #: raw usage the provider reported, for the spend meter
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class PageRecord:
    """One generated page, with every score attached to it."""

    scene_id: str
    backend: str
    strategy: str
    image_path: str
    prompt: str
    stress: str = ""
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
