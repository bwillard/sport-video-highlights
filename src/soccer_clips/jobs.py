"""
In-process background jobs (clip exports, reel renders), run one at a time by
a single worker thread because encoding is CPU-bound. Jobs live in memory
only; restarting the server forgets them (finished outputs stay on disk).
"""

import logging
import queue
import secrets
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "done", "failed"]


class Job(BaseModel):
    id: str = Field(default_factory=lambda: secrets.token_hex(4))
    kind: Literal["export", "render"]
    game_id: str | None = None  # export
    clip_id: str | None = None  # export
    fast: bool = False  # export
    reel_id: str | None = None  # render
    preset: str | None = None  # render
    variant: str | None = None  # render
    status: JobStatus = "queued"
    progress: float = 0.0  # 0..1
    error: str | None = None
    output: str | None = None  # relative path of the result, set when done
    created_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())


class JobQueue:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue[tuple[Job, Callable[[Job], None]]] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="jobs", daemon=True)
        self._thread.start()

    def submit(self, job: Job, run: Callable[[Job], None]) -> Job:
        """Queue `run(job)`; it may update job.progress and set job.output."""
        self.jobs[job.id] = job
        self._queue.put((job, run))
        return job

    def active(self, **match) -> Job | None:
        """A queued or running job whose fields equal `match`, if any."""
        for job in self.jobs.values():
            if job.status in ("queued", "running") and all(getattr(job, k) == v for k, v in match.items()):
                return job
        return None

    def wait(self) -> None:
        """Block until every submitted job has finished (for tests and the CLI)."""
        self._queue.join()

    def _worker(self) -> None:
        while True:
            job, run = self._queue.get()
            try:
                job.status = "running"
                run(job)
                job.status = "done"
                job.progress = 1.0
            except Exception as e:
                logger.exception("%s job %s failed", job.kind, job.id)
                job.status = "failed"
                job.error = str(e)
            finally:
                self._queue.task_done()
