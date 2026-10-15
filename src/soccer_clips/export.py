"""
Clip export: cut one clip from the game's original into its own mp4.

Accurate mode re-encodes and is frame-exact. Fast mode stream-copies, so it's
quick but the start snaps back to the previous keyframe (up to 2.5s on Veo).
"""

import os

from soccer_clips import ffmpeg
from soccer_clips.jobs import Job
from soccer_clips.store import Store

ACCURATE_CODEC = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                  "-c:a", "aac", "-b:a", "160k"]
FAST_CODEC = ["-c", "copy", "-avoid_negative_ts", "make_zero"]


def export_job(game_id: str, clip_id: str, fast: bool = False) -> Job:
    return Job(kind="export", game_id=game_id, clip_id=clip_id, fast=fast)


def run_export(store: Store, job: Job) -> None:
    settings = store.settings
    game = store.load_game(job.game_id)
    clip = game.clip(job.clip_id)
    rel = store.clip_export_rel(game, clip.id)
    out = settings.clips_dir / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.mp4")

    seek_s, duration_s = ffmpeg.cut_window(game, clip.start_s, clip.end_s)
    cmd = [
        settings.ffmpeg, "-y", "-v", "error",
        "-ss", f"{seek_s:.6f}", "-i", str(store.original_path(game)),
        "-t", f"{duration_s:.6f}",
        "-map", "0:v:0", "-map", "0:a:0?", *(FAST_CODEC if job.fast else ACCURATE_CODEC),
        "-movflags", "+faststart", str(tmp),
    ]
    try:
        ffmpeg.run(cmd, duration_s, lambda f: setattr(job, "progress", min(f, 0.99)))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out)
    job.output = rel

    # Only record the export if the clip still exists with the same range;
    # if it was edited meanwhile, the file is stale and needs a re-export.
    with store.edit_game(game.id) as current:
        try:
            now = current.clip(clip.id)
        except KeyError:  # deleted while exporting
            out.unlink(missing_ok=True)
            job.output = None
            return
        if (now.start_s, now.end_s) == (clip.start_s, clip.end_s):
            now.export = rel
