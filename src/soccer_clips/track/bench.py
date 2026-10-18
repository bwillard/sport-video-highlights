"""
Benchmark clip set for tracking experiments (Phase 3.2 / 4.1).

The set is a hand-editable list in <bench_dir>/clips.json; `cut` makes
<bench_dir>/clips/<name>.mp4 from it and recuts only entries whose game or
range changed, so tweaking a start/end and re-running is cheap.
"""

import json
import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from soccer_clips import ffmpeg
from soccer_clips.config import Settings
from soccer_clips.export import ACCURATE_CODEC
from soccer_clips.models import Callout
from soccer_clips.store import Store, atomic_write_json, atomic_write_text

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class BenchClip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str  # file name without .mp4
    game_id: str
    start_s: float = Field(ge=0)  # game time
    end_s: float
    why: str = ""  # what makes this clip a useful test
    from_clip: str | None = None  # tagged clip it was taken from
    # A callout copied from that clip: where a player is at one moment (game
    # time), so tracking can be checked without labelling anything.
    callout: Callout | None = None
    callout_player: int | None = None  # jersey number, when the clip was tagged with exactly one player

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not NAME_RE.match(v):
            raise ValueError("name must be lowercase letters, digits and dashes")
        return v


class BenchSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clips: list[BenchClip] = []


def bench_dir(settings: Settings) -> Path:
    return settings.bench_dir or settings.games_dir.parent / "bench"


def _set_path(settings: Settings) -> Path:
    return bench_dir(settings) / "clips.json"


def load_set(settings: Settings) -> BenchSet:
    path = _set_path(settings)
    return BenchSet.model_validate_json(path.read_text()) if path.exists() else BenchSet()


def save_set(settings: Settings, bench: BenchSet) -> None:
    atomic_write_text(_set_path(settings), bench.model_dump_json(indent=2, exclude_none=True) + "\n")


def add_clip(store: Store, game_id: str, clip_id: str, name: str, why: str = "") -> BenchClip:
    """Copy a tagged clip's range (and callout) into the set; replaces an entry of the same name."""
    clip = store.load_game(game_id).clip(clip_id)
    entry = BenchClip(
        name=name, game_id=game_id, start_s=clip.start_s, end_s=clip.end_s, why=why or (clip.note or ""),
        from_clip=clip.id, callout=clip.callout,
        callout_player=clip.players[0] if clip.callout and len(clip.players) == 1 else None,
    )
    bench = load_set(store.settings)
    bench.clips = [c for c in bench.clips if c.name != name] + [entry]
    save_set(store.settings, bench)
    return entry


def cut(store: Store, force: bool = False) -> list[str]:
    """Cut every clip in the set whose mp4 is missing or out of date; returns the names cut."""
    settings = store.settings
    out_dir = bench_dir(settings) / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamps_path = out_dir / ".cut.json"
    stamps = json.loads(stamps_path.read_text()) if stamps_path.exists() else {}
    done = []
    for c in load_set(settings).clips:
        out = out_dir / f"{c.name}.mp4"
        stamp = [c.game_id, c.start_s, c.end_s]
        if not force and out.exists() and stamps.get(c.name) == stamp:
            continue
        game = store.load_game(c.game_id)
        seek_s, duration_s = ffmpeg.cut_window(game, c.start_s, c.end_s)
        tmp = out.with_name(out.stem + ".part.mp4")
        cmd = [settings.ffmpeg, "-y", "-v", "error", "-ss", f"{seek_s:.6f}", "-i", str(store.original_path(game)),
               "-t", f"{duration_s:.6f}", "-map", "0:v:0", "-map", "0:a:0?", *ACCURATE_CODEC,
               "-movflags", "+faststart", str(tmp)]
        try:
            ffmpeg.run(cmd, duration_s)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, out)
        stamps[c.name] = stamp
        atomic_write_json(stamps_path, stamps)
        done.append(c.name)
    return done
