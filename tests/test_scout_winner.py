"""The scout must not name a winner it did not actually find.

The judge scores identity in whole numbers from 1 to 5 over two scenes, so
averages land on halves and ties are common. `max()` broke those ties by
dictionary order, which made "openai" look like a finding when it was really
just the first key.
"""

from heldenbuch.book.scout import WINNING_MARGIN, pick_winner


def _results(**scores):
    return {name: {"identity": value, "scenes": 2, "images": []}
            for name, value in scores.items()}


def test_a_clear_lead_wins():
    winner, runner_up, margin = pick_winner(_results(openai=4.5, bfl=3.0))
    assert winner == "openai"
    assert runner_up == "bfl"
    assert margin == 1.5


def test_a_dead_tie_has_no_winner():
    winner, _, margin = pick_winner(_results(openai=4.0, bfl=4.0, gemini=4.0))
    assert winner is None
    assert margin == 0.0


def test_a_lead_too_small_to_mean_anything_has_no_winner():
    winner, _, margin = pick_winner(_results(openai=4.0, bfl=3.75))
    assert winner is None
    assert margin == 0.25


def test_the_margin_is_inclusive_at_the_threshold():
    winner, _, _ = pick_winner(_results(openai=4.0, bfl=4.0 - WINNING_MARGIN))
    assert winner == "openai"


def test_the_tie_is_not_broken_by_ordering():
    """The bug: the same numbers gave a different answer per dict order."""
    first = pick_winner(_results(openai=4.0, bfl=4.0))[0]
    second = pick_winner(_results(bfl=4.0, openai=4.0))[0]
    assert first is second is None


def test_a_lone_service_is_the_winner():
    winner, runner_up, _ = pick_winner(_results(bfl=3.0))
    assert winner == "bfl"
    assert runner_up is None
