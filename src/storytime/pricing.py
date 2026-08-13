"""Turning reported usage into money.

Providers report tokens, not euros. The rates below are list prices in US
dollars per million tokens, taken from the published pricing pages. They are
the one thing here that goes stale on someone else's schedule, so they live in
one small table with the date they were checked, and every total the app shows
is labelled as an estimate.

BFL is the exception: it reports credits, and one credit is one cent, so that
number is exact.
"""

from __future__ import annotations

from typing import Any

#: checked against the published pricing pages on this date
RATES_CHECKED = "2026-08-11"

#: US dollars per million tokens
RATES: dict[str, dict[str, float]] = {
    # image models: billed on image tokens in and out
    "gpt-image-2": {"input": 8.00, "output": 30.00},
    "gpt-image-1.5": {"input": 8.00, "output": 30.00},
    "gpt-image-1": {"input": 10.00, "output": 40.00},
    "gpt-image-1-mini": {"input": 2.50, "output": 8.00},
    # text models
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00},
    "gpt-5.6-sol": {"input": 5.00, "output": 20.00},
    "gpt-5.6-luna": {"input": 0.20, "output": 1.60},
    # speech
    "gpt-4o-mini-tts": {"input": 0.60, "output": 12.00},
}

USD_PER_EUR = 1.08  # rough, only used to show a second number

#: measured 2026-08-12 with gpt-image-2 at 1024 px (see HANDOFF.md). Input
#: tokens were identical at every quality; the cost is almost entirely output
#: tokens, and those scale with pixel area.
MEASURED_IMAGE_USD = {"low": 0.0125, "medium": 0.0593, "high": 0.2174}

#: list prices for per-image-billed models as (draft, print) US dollars,
#: checked 2026-08-12. Gemini 3 Pro Image: $0.134 up to 2K, $0.24 at 4K
#: (batch mode halves both). FLUX.2 pro bills per megapixel and per reference
#: image; these are ballparks for a page with references -- BFL reports the
#: exact figure after every call, which `price` then uses instead.
#: The 3.1 flash tier is assumed at the 2.5-flash list price, unverified.
FLAT_IMAGE_USD: dict[str, tuple[float, float]] = {
    "gemini-3-pro-image": (0.134, 0.24),
    "gemini-3.1-flash-image": (0.039, 0.039),
    "flux-2-pro": (0.08, 0.20),
}


def image_estimate(quality: str, long_edge_px: int = 1024,
                   model: str = "gpt-image-2") -> float:
    """Rough US dollars for one image at this quality and size, *before* the
    call is made. The after-the-fact numbers come from `price`; this exists so
    a button can say what it is about to spend.

    Measured for gpt-image-2; other models are scaled by their output-token
    rate, which is what the cost of an image almost entirely consists of.
    """
    base = MEASURED_IMAGE_USD.get(quality, MEASURED_IMAGE_USD["medium"])
    reference = RATES["gpt-image-2"]["output"]
    rate = RATES.get(model, {}).get("output", reference)
    return round(base * (long_edge_px / 1024) ** 2 * (rate / reference), 4)


def price(usage: dict[str, Any]) -> float:
    """US dollars for one call, from whatever the provider reported."""
    if "usd" in usage:  # BFL tells us directly
        return float(usage["usd"])

    rate = RATES.get(str(usage.get("model", "")))
    if not rate:
        return 0.0
    cost = (
        float(usage.get("input_tokens", 0)) * rate["input"]
        + float(usage.get("output_tokens", 0)) * rate["output"]
    ) / 1_000_000
    return round(cost, 6)


def add(spend: dict[str, Any], usage: dict[str, Any], what: str) -> dict[str, Any]:
    """Fold one call into a running tally. Mutates and returns `spend`.

    Besides the running totals, every call leaves a ledger entry, so the
    header figure can be opened up into "what exactly was paid for". The
    totals are always the sum of the entries.
    """
    import time

    if not usage:
        return spend

    amount = price(usage)
    spend.setdefault("usd", 0.0)
    spend.setdefault("calls", 0)
    spend.setdefault("images", 0)
    spend.setdefault("by", {})

    spend["usd"] = round(spend["usd"] + amount, 6)
    spend["calls"] += 1
    spend["images"] += int(usage.get("images", 0))

    bucket = spend["by"].setdefault(what, {"usd": 0.0, "calls": 0})
    bucket["usd"] = round(bucket["usd"] + amount, 6)
    bucket["calls"] += 1

    spend.setdefault("entries", []).append({
        "what": what,
        "usd": amount,
        "images": int(usage.get("images", 0)),
        "model": str(usage.get("model", "")),
        "backend": str(usage.get("backend", "")),
        "at": int(time.time()),
    })
    return spend


def summary(spend: dict[str, Any]) -> dict[str, Any]:
    """What the UI shows."""
    usd = float(spend.get("usd", 0.0))
    return {
        "usd": round(usd, 2),
        "eur": round(usd / USD_PER_EUR, 2),
        "calls": int(spend.get("calls", 0)),
        "images": int(spend.get("images", 0)),
        "by": spend.get("by", {}),
        #: newest last; capped so a long-lived book stays a readable payload
        "entries": (spend.get("entries") or [])[-250:],
        "estimate": True,
        "rates_checked": RATES_CHECKED,
    }
