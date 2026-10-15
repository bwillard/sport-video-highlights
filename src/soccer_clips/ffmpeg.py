"""
Small ffmpeg helpers shared by clip export and reel rendering.
"""

import json
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path

from soccer_clips.config import Settings
from soccer_clips.models import Game

logger = logging.getLogger(__name__)


def cut_window(game: Game, start_s: float, end_s: float) -> tuple[float, float]:
    """
    (seek_s, duration_s) for cutting [start_s, end_s) out of a game, snapped to
    whole frames. Seeking to half a frame before the first frame means float
    rounding can't drop or add a frame, and the duration then covers exactly
    frame_at(end) - frame_at(start) frames.
    """
    first, last = game.frame_at(start_s), game.frame_at(end_s)
    frame_s = game.fps_den / game.fps_num
    return max(0.0, (first - 0.5) * frame_s), (last - first) * frame_s


def run(cmd: list[str], duration_s: float, on_progress: Callable[[float], None] | None = None) -> None:
    """
    Run ffmpeg, reporting progress as a 0..1 fraction of duration_s. The
    command must not already set -progress; it is added here.
    """
    cmd = [cmd[0], "-nostats", "-progress", "pipe:1", *cmd[1:]]
    logger.info("ffmpeg: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stdout:
        key, _, value = line.strip().partition("=")
        if on_progress and key == "out_time_us" and value.isdigit() and duration_s > 0:
            on_progress(min(int(value) / 1e6 / duration_s, 1.0))
    stderr = proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg exited {proc.returncode}: {stderr.strip()[-500:]}")


def stream_durations(settings: Settings, path: Path) -> dict[str, float]:
    """Duration of the first video and first audio stream, keyed "video"/"audio"."""
    out = subprocess.run(
        [settings.ffprobe, "-v", "error", "-show_entries", "stream=codec_type,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    durations: dict[str, float] = {}
    for s in json.loads(out)["streams"]:
        if s["codec_type"] in ("video", "audio") and s["codec_type"] not in durations and "duration" in s:
            durations[s["codec_type"]] = float(s["duration"])
    return durations
