"""
Clip export: cut a clip from the game's original with ffmpeg, in a single
background worker (encoding is CPU-bound, so one at a time).

Accurate mode re-encodes and is frame-exact. Fast mode stream-copies, so it's
quick but the start snaps back to the previous keyframe (up to 2.5s on Veo).
"""

import logging
import os
import queue
import secrets
import subprocess
import threading
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from soccer_clips.models import Clip, Game
from soccer_clips.store import Store

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "done", "failed"]


class Job(BaseModel):
    id: str = Field(default_factory=lambda: secrets.token_hex(4))
    game_id: str
    clip_id: str
    fast: bool = False
    status: JobStatus = "queued"
    progress: float = 0.0  # 0..1
    error: str | None = None
    output: str | None = None  # relative to clips_dir, set when done
    created_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())


def ffmpeg_cut_args(game: Game, clip: Clip, fast: bool) -> tuple[float, float, list[str]]:
    """
    Returns (seek_s, duration_s, codec args). Times are snapped to whole frames:
    seeking to half a frame before the first frame means float rounding can't
    drop or add a frame, and the duration then covers exactly
    frame_at(end) - frame_at(start) frames.
    """
    first, last = game.frame_at(clip.start_s), game.frame_at(clip.end_s)
    frame_s = game.fps_den / game.fps_num
    seek_s = max(0.0, (first - 0.5) * frame_s)
    duration_s = (last - first) * frame_s
    if fast:
        codec = ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        codec = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "160k"]
    return seek_s, duration_s, codec


class ExportQueue:
    def __init__(self, store: Store):
        self.store = store
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="export", daemon=True)
        self._thread.start()

    def submit(self, game_id: str, clip_id: str, fast: bool = False) -> Job:
        job = Job(game_id=game_id, clip_id=clip_id, fast=fast)
        self.jobs[job.id] = job
        self._queue.put(job.id)
        return job

    def wait(self) -> None:
        """Block until every submitted job has finished (for tests and the CLI)."""
        self._queue.join()

    def _worker(self) -> None:
        while True:
            job = self.jobs[self._queue.get()]
            try:
                job.status = "running"
                self._run(job)
                job.status = "done"
                job.progress = 1.0
            except Exception as e:
                logger.exception("export %s failed", job.id)
                job.status = "failed"
                job.error = str(e)
            finally:
                self._queue.task_done()

    def _run(self, job: Job) -> None:
        settings = self.store.settings
        game = self.store.load_game(job.game_id)
        clip = game.clip(job.clip_id)
        rel = self.store.clip_export_rel(game, clip.id)
        out = settings.clips_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".part.mp4")

        seek_s, duration_s, codec = ffmpeg_cut_args(game, clip, job.fast)
        cmd = [
            settings.ffmpeg, "-y", "-v", "error", "-nostats", "-progress", "pipe:1",
            "-ss", f"{seek_s:.6f}", "-i", str(self.store.original_path(game)),
            "-t", f"{duration_s:.6f}",
            "-map", "0:v:0", "-map", "0:a:0?", *codec,
            "-movflags", "+faststart", str(tmp),
        ]
        logger.info("export %s: %s", job.id, " ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            if key == "out_time_us" and value.isdigit():
                job.progress = min(int(value) / 1e6 / duration_s, 0.99)
        stderr = proc.stderr.read()
        if proc.wait() != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg exited {proc.returncode}: {stderr.strip()[-500:]}")
        os.replace(tmp, out)
        job.output = rel

        # Only record the export if the clip still exists with the same range;
        # if it was edited meanwhile, the file is stale and needs a re-export.
        with self.store.edit_game(game.id) as current:
            try:
                now = current.clip(clip.id)
            except KeyError:  # deleted while exporting
                out.unlink(missing_ok=True)
                job.output = None
                return
            if (now.start_s, now.end_s) == (clip.start_s, clip.end_s):
                now.export = rel
