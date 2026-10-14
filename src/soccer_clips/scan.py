"""
Find full-game videos in <full_dir>/<year>/*.mp4 that don't have a game.json
yet, probe them, and register them in place (originals are never copied,
moved or modified).
"""

import json
import logging
import re
import subprocess
from datetime import date, datetime
from fractions import Fraction
from pathlib import Path

from soccer_clips.config import Settings
from soccer_clips.models import Game
from soccer_clips.store import Store, atomic_write_json

logger = logging.getLogger(__name__)

_DATE_SUFFIX = re.compile(r"(\d{4}-\d{2}-\d{2})$")
# Warn if the average frame rate differs from the nominal one by more than this
# fraction; Veo files are constant frame rate and differ by ~1e-6.
_VFR_TOLERANCE = 0.001


def parse_played_on(stem: str) -> date | None:
    """Filenames end in YYYY-MM-DD; nothing else about them is consistent."""
    match = _DATE_SUFFIX.search(stem)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def ffprobe(settings: Settings, path: Path) -> dict:
    out = subprocess.run(
        [settings.ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


def game_from_probe(probe: dict, video: Path, full_dir: Path) -> Game:
    streams = probe["streams"]
    v = next(s for s in streams if s["codec_type"] == "video")
    nominal = Fraction(v["r_frame_rate"])
    avg = Fraction(v["avg_frame_rate"]) if v.get("avg_frame_rate", "0/0") != "0/0" else nominal
    if abs(float(avg) - float(nominal)) / float(nominal) > _VFR_TOLERANCE:
        logger.warning(
            "%s may be variable frame rate (r=%s avg=%.4f); frame numbers may be off",
            video.name, nominal, float(avg),
        )
    return Game(
        id=video.stem,
        year=int(video.parent.name),
        played_on=parse_played_on(video.stem),
        original=str(video.relative_to(full_dir)),
        duration_s=float(probe["format"]["duration"]),
        fps_num=nominal.numerator,
        fps_den=nominal.denominator,
        width=int(v["width"]),
        height=int(v["height"]),
        has_audio=any(s["codec_type"] == "audio" for s in streams),
        created_at=datetime.now().astimezone(),
    )


def find_videos(full_dir: Path) -> list[Path]:
    return sorted(
        p
        for year_dir in full_dir.iterdir()
        if year_dir.is_dir() and re.fullmatch(r"\d{4}", year_dir.name)
        for p in year_dir.glob("*.mp4")
    )


def scan(settings: Settings, store: Store) -> list[Game]:
    """Register any new videos; returns the newly added games."""
    added = []
    for video in find_videos(settings.full_dir):
        year = int(video.parent.name)
        if store.has_game(year, video.stem):
            continue
        logger.info("registering %s", video)
        probe = ffprobe(settings, video)
        game = game_from_probe(probe, video, settings.full_dir)
        atomic_write_json(store.game_dir(year, game.id) / "probe.json", probe)
        store.save_game(game)
        added.append(game)
    return added
