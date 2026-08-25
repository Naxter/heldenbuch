"""Background jobs.

Anything that costs money or takes more than a moment runs here: drawing a
character sheet, writing a story, illustrating a book, exporting a PDF. Each
job runs on its own thread and writes its log into a buffer the browser polls.

One job runs at a time, globally — the book app and the benchmark share the
same image APIs and rate limits, so running two at once makes both slower and
the log unreadable. But asking for a second job *queues* it rather than
refusing it: a twenty-minute print render should not lock you out of pressing
anything else.
"""

from __future__ import annotations

import itertools
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Worker = Callable[["Job", Callable[..., None]], None]

#: Job parameters that carry an upload and must never be echoed back. These are
#: base64 images -- a photograph of a child, or a style reference -- and the
#: status endpoint is polled every second, so returning them would re-send the
#: picture continuously as well as put it somewhere it does not belong.
UPLOAD_PARAMS = frozenset({"photos", "photo", "reference"})

#: Anything longer than this is treated as an upload even under a new name, so
#: the next parameter is redacted by default rather than by being remembered.
_BULK_CHARS = 4096


def _safe_param(key: str, value: Any) -> Any:
    if key in UPLOAD_PARAMS:
        return "<upload>"
    if isinstance(value, str) and len(value) > _BULK_CHARS:
        return "<upload>"
    if isinstance(value, dict) and "data" in value:
        return "<upload>"
    if isinstance(value, list) and any(isinstance(v, dict) and "data" in v for v in value):
        return "<upload>"
    return value


@dataclass
class Job:
    id: str
    action: str
    params: dict[str, Any]
    status: str = "queued"  # queued | running | done | failed | cancelled
    lines: list[str] = field(default_factory=list)
    #: whatever the worker wants to hand back to the browser
    result: dict[str, Any] = field(default_factory=dict)
    started: float = field(default_factory=time.time)
    finished: float | None = None
    error: str | None = None
    #: (done, total) once the worker knows how much work lies ahead --
    #: the drawer turns this into a progress bar
    progress: tuple[int, int] | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    #: how many jobs are ahead of this one when it was accepted
    queued_behind: int = 0

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def public(self, since: int = 0) -> dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action,
            "params": {k: _safe_param(k, v) for k, v in self.params.items()},
            "status": self.status,
            "error": self.error,
            "result": self.result,
            "queued_behind": self.queued_behind,
            "progress": ({"done": self.progress[0], "total": self.progress[1]}
                         if self.progress else None),
            "started": self.started,
            "finished": self.finished,
            "elapsed": round((self.finished or time.time()) - self.started, 1),
            "lines": self.lines[since:],
            "total_lines": len(self.lines),
        }


class JobManager:
    def __init__(self, workers: dict[str, Worker] | None = None) -> None:
        self.workers: dict[str, Worker] = dict(workers or {})
        self._jobs: dict[str, Job] = {}
        self._queue: deque[Job] = deque()
        self._running: Job | None = None
        self._lock = threading.Lock()
        self._ids = itertools.count(1)

    def register(self, workers: dict[str, Worker]) -> None:
        self.workers.update(workers)

    # ---------------------------------------------------------------- access

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def active(self) -> Job | None:
        """The job currently running, or the next one waiting to."""
        with self._lock:
            if self._running and self._running.status == "running":
                return self._running
            return self._queue[0] if self._queue else None

    def pending(self) -> int:
        with self._lock:
            return len(self._queue) + (1 if self._running else 0)

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        # Snapshot under the lock: a status poll iterating this dict while a
        # POST inserts a new job raised "dictionary changed size" and turned
        # the whole status endpoint into a 500 at the worst moment.
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.started, reverse=True)
        return [{k: v for k, v in j.public().items() if k != "lines"} for j in jobs[:limit]]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status not in ("running", "queued"):
                return False
            job._cancel.set()
            if job.status == "queued":
                # Never started, so drop it outright.
                if job in self._queue:
                    self._queue.remove(job)
                job.status = "cancelled"
                job.finished = time.time()
                job.lines.append("— aus der Warteschlange genommen —")
                return True
        job.lines.append("— wird abgebrochen, das aktuelle Bild wird noch fertig —")
        return True

    def retry(self, job_id: str) -> Job:
        """Run a finished job again with the parameters it actually had.

        The browser cannot do this itself: `public()` redacts uploads, so a
        client-side resubmit sent `"<upload>"` where the photos belonged and
        a retried hero was drawn from no photos at all. Here the originals
        are still in hand.
        """
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"no job {job_id}")
        if job.status in ("running", "queued"):
            raise ValueError("dieser Auftrag läuft noch")
        return self.start(job.action, dict(job.params))

    # ---------------------------------------------------------------- launch

    def start(self, action: str, params: dict[str, Any]) -> Job:
        worker = self.workers.get(action)
        if worker is None:
            raise ValueError(
                f"unknown action {action!r}; valid: {', '.join(sorted(self.workers))}"
            )
        with self._lock:
            job = Job(id=str(next(self._ids)), action=action, params=params)
            job.queued_behind = len(self._queue) + (1 if self._running else 0)
            self._jobs[job.id] = job
            self._queue.append(job)
            if job.queued_behind:
                job.lines.append(
                    f"— in der Warteschlange, {job.queued_behind} vor dir —"
                )
            should_start = self._running is None
        if should_start:
            self._pump()
        return job

    def _pump(self) -> None:
        """Start the next queued job, if nothing is running."""
        with self._lock:
            if self._running is not None or not self._queue:
                return
            job = self._queue.popleft()
            self._running = job
            job.status = "running"
            job.started = time.time()
        try:
            threading.Thread(
                target=self._execute, args=(job, self.workers[job.action]), daemon=True
            ).start()
        except BaseException:
            # A thread that never started must not stay the "running" job, or
            # nothing queued after it would ever run again.
            with self._lock:
                self._running = None
                job.status = "failed"
                job.lines.append("Der Auftrag konnte nicht gestartet werden.")
            raise

    def _execute(self, job: Job, worker: Worker) -> None:
        def log(*args) -> None:
            text = " ".join(str(a) for a in args)
            job.lines.extend(text.split("\n") if text else [""])

        try:
            worker(job, log)
            job.status = "cancelled" if job.cancelled else "done"
        except Exception as exc:
            job.status = "failed"
            # Domain errors (a provider refusal, a validation message) are
            # written for the person reading the drawer; a traceback under
            # them is developer noise. Genuinely unexpected exceptions keep
            # theirs, because then the code is the suspect.
            expected = isinstance(exc, (ValueError, RuntimeError, FileNotFoundError))
            job.error = str(exc) if expected else f"{type(exc).__name__}: {exc}"
            log("")
            log(f"— fehlgeschlagen: {job.error} —")
            if not expected:
                for line in traceback.format_exc().splitlines()[-6:]:
                    log(f"   {line}")
        finally:
            job.finished = time.time()
            with self._lock:
                self._running = None
                self._evict_finished()
            self._pump()

    #: Finished jobs kept for the drawer's history. This is a long-lived local
    #: control panel: without a cap, every job ever run -- log lines included --
    #: stayed memory-resident for the life of the process.
    _KEEP_FINISHED = 50

    def _evict_finished(self) -> None:
        """Drop the oldest finished jobs beyond the cap. Caller holds the lock."""
        done = [j for j in self._jobs.values()
                if j.status not in ("running", "queued")]
        if len(done) <= self._KEEP_FINISHED:
            return
        done.sort(key=lambda j: j.finished or j.started)
        for stale in done[:len(done) - self._KEEP_FINISHED]:
            self._jobs.pop(stale.id, None)
