"""Tests for the job queue and the benchmark's colour metrics."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from PIL import Image

from storytime.metrics.cheap import score_page
from storytime.web.jobs import JobManager


def _wait(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestJobQueue:
    def test_a_second_job_queues_instead_of_being_refused(self):
        release = threading.Event()
        manager = JobManager({
            "slow": lambda job, log: release.wait(3),
            "quick": lambda job, log: log("done"),
        })

        first = manager.start("slow", {})
        assert _wait(lambda: first.status == "running")

        second = manager.start("quick", {})
        assert second.status == "queued"
        assert second.queued_behind == 1

        release.set()
        assert _wait(lambda: second.status == "done")
        assert first.status == "done"

    def test_jobs_run_one_at_a_time(self):
        concurrent, peak = 0, 0
        lock = threading.Lock()

        def worker(job, log):
            nonlocal concurrent, peak
            with lock:
                concurrent += 1
                peak = max(peak, concurrent)
            time.sleep(0.05)
            with lock:
                concurrent -= 1

        manager = JobManager({"w": worker})
        jobs = [manager.start("w", {}) for _ in range(4)]
        assert _wait(lambda: all(j.status == "done" for j in jobs), timeout=8)
        assert peak == 1

    def test_queue_drains_in_order(self):
        order: list[str] = []
        manager = JobManager({"w": lambda job, log: order.append(job.params["tag"])})
        jobs = [manager.start("w", {"tag": tag}) for tag in ("a", "b", "c")]
        assert _wait(lambda: all(j.status == "done" for j in jobs), timeout=8)
        assert order == ["a", "b", "c"]

    def test_cancelling_a_queued_job_removes_it_without_running_it(self):
        release = threading.Event()
        ran: list[str] = []
        manager = JobManager({
            "slow": lambda job, log: release.wait(3),
            "never": lambda job, log: ran.append("oops"),
        })

        first = manager.start("slow", {})
        assert _wait(lambda: first.status == "running")
        second = manager.start("never", {})

        assert manager.cancel(second.id) is True
        assert second.status == "cancelled"

        release.set()
        assert _wait(lambda: first.status == "done")
        assert ran == []

    def test_a_failing_job_does_not_block_the_queue(self):
        def boom(job, log):
            raise ValueError("kaputt")

        manager = JobManager({"boom": boom, "ok": lambda job, log: log("fine")})
        bad = manager.start("boom", {})
        good = manager.start("ok", {})

        assert _wait(lambda: good.status == "done", timeout=8)
        assert bad.status == "failed"
        assert "kaputt" in (bad.error or "")

    def test_unknown_action_is_rejected_up_front(self):
        with pytest.raises(ValueError):
            JobManager({}).start("nope", {})

    def test_photos_are_not_echoed_back_to_the_browser(self):
        manager = JobManager({"w": lambda job, log: None})
        job = manager.start("w", {"photos": ["data:image/png;base64,AAA"], "name": "Claudio"})
        assert "photos" not in job.public()["params"]
        assert job.public()["params"]["name"] == "Claudio"


class TestColourMetrics:
    """The consistency metric has to survive a small hue shift."""

    @staticmethod
    def _swatch(tmp_path, name, rgb):
        path = tmp_path / name
        array = np.zeros((128, 128, 3), dtype=np.uint8)
        array[:, :] = (250, 248, 243)          # paper
        array[32:96, 32:96] = rgb              # the character
        Image.fromarray(array).save(path)
        return path

    def test_identical_images_score_near_one(self, tmp_path):
        a = self._swatch(tmp_path, "a.png", (206, 106, 48))
        b = self._swatch(tmp_path, "b.png", (206, 106, 48))
        assert score_page(a, b)["palette_cosine"] > 0.95

    def test_a_small_hue_shift_does_not_collapse_the_score(self, tmp_path):
        """The bug that made near-identical pages score 0.02 against 0.61.

        Hard histogram bins meant a shift of one bin moved every pixel to a
        neighbouring bucket and the cosine fell off a cliff. The histogram is
        smoothed circularly across hue so a nudge stays a nudge.
        """
        sheet = self._swatch(tmp_path, "sheet.png", (206, 106, 48))
        nudged = self._swatch(tmp_path, "nudged.png", (206, 122, 48))
        assert score_page(nudged, sheet)["palette_cosine"] > 0.55

    def test_a_completely_different_colour_scores_low(self, tmp_path):
        sheet = self._swatch(tmp_path, "sheet.png", (206, 106, 48))   # orange
        wrong = self._swatch(tmp_path, "wrong.png", (60, 90, 200))    # blue
        assert score_page(wrong, sheet)["palette_cosine"] < 0.2

    def test_drift_is_ordered_correctly(self, tmp_path):
        sheet = self._swatch(tmp_path, "sheet.png", (206, 106, 48))
        near = self._swatch(tmp_path, "near.png", (206, 122, 48))
        far = self._swatch(tmp_path, "far.png", (120, 190, 60))
        assert (score_page(near, sheet)["palette_cosine"]
                > score_page(far, sheet)["palette_cosine"])
