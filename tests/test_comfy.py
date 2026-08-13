"""Tests for the local ComfyUI backend -- everything that runs without a server.

The placeholder substitution is the part that can silently ruin a render (a
"{SEED}" left as a string crashes the node; a wrong image name draws the wrong
character), so it is pinned here. The HTTP half needs a live ComfyUI and is
exercised the day one is running.
"""

from __future__ import annotations

import json

import pytest

from storytime.backends import BACKEND_NAMES, REQUIRED_KEY, get_backend
from storytime.backends.base import BackendError
from storytime.backends.comfy import (
    ComfyBackend,
    fill_workflow,
    image_slots,
    load_workflow,
)


def _workflow() -> dict:
    """A miniature API-format graph with every placeholder kind."""
    return {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "{PROMPT}", "clip": ["4", 0]}},
        "2": {"class_type": "EmptyLatentImage",
              "inputs": {"width": "{WIDTH}", "height": "{HEIGHT}", "batch_size": 1}},
        "3": {"class_type": "KSampler", "inputs": {"seed": "{SEED}", "steps": 4}},
        "5": {"class_type": "LoadImage", "inputs": {"image": "{IMAGE_1}"}},
        "6": {"class_type": "LoadImage", "inputs": {"image": "{IMAGE_2}"}},
    }


def test_fill_workflow_substitutes_every_placeholder():
    filled = fill_workflow(_workflow(), "a fox", 1024, 768, 42, ["sheet.png", "oma.png"])
    assert filled["1"]["inputs"]["text"] == "a fox"
    assert filled["2"]["inputs"]["width"] == 1024      # int, not "1024"
    assert filled["2"]["inputs"]["height"] == 768
    assert filled["3"]["inputs"]["seed"] == 42
    assert filled["5"]["inputs"]["image"] == "sheet.png"
    assert filled["6"]["inputs"]["image"] == "oma.png"


def test_unfilled_image_slots_repeat_the_character_sheet():
    """A four-slot workflow must stay usable on a one-reference page."""
    filled = fill_workflow(_workflow(), "p", 512, 512, 1, ["sheet.png"])
    assert filled["5"]["inputs"]["image"] == "sheet.png"
    assert filled["6"]["inputs"]["image"] == "sheet.png"


def test_image_slot_with_no_references_is_an_error():
    with pytest.raises(BackendError):
        fill_workflow(_workflow(), "p", 512, 512, 1, [])


def test_image_slots_counts_the_highest_slot():
    assert image_slots(_workflow()) == 2
    assert image_slots({"1": {"inputs": {"text": "{PROMPT}"}}}) == 0


def test_load_workflow_demands_the_prompt_placeholder(tmp_path):
    target = tmp_path / "wf.json"
    target.write_text(json.dumps({"1": {"inputs": {"text": "hardcoded"}}}), encoding="utf-8")
    with pytest.raises(BackendError, match="PROMPT"):
        load_workflow(target)


def test_load_workflow_explains_when_the_file_is_missing(tmp_path):
    with pytest.raises(BackendError, match="comfy_workflow.json"):
        load_workflow(tmp_path / "nope.json")


def test_comfy_is_registered():
    assert "comfy" in BACKEND_NAMES
    assert REQUIRED_KEY["comfy"] is None  # a running server, not a key
    backend = get_backend("comfy")
    assert isinstance(backend, ComfyBackend)
    assert backend.model == "flux-2-klein"


def test_max_references_follows_the_workflow(tmp_path, monkeypatch):
    target = tmp_path / "wf.json"
    target.write_text(json.dumps(_workflow()), encoding="utf-8")
    monkeypatch.setenv("COMFY_WORKFLOW", str(target))
    assert ComfyBackend().max_references == 2
