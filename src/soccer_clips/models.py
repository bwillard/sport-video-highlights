"""
Pydantic models for everything stored as JSON. New fields must have defaults
so older files keep loading; bump schema_version only for a real migration.
"""

import secrets
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def _now() -> datetime:
    return datetime.now().astimezone()


class Tag(BaseModel):
    name: str
    hotkey: str | None = None  # "1".."9" in the tagger


DEFAULT_TAGS = [
    Tag(name=name, hotkey=str(i))
    for i, name in enumerate(
        ["goal", "shot", "assist", "pass", "dribble", "defense", "save", "celebration", "funny"],
        start=1,
    )
]


class Clip(BaseModel):
    # Random rather than sequential so reels can reference (game_id, clip_id)
    # and clips can be deleted without renumbering.
    id: str = Field(default_factory=lambda: secrets.token_hex(4))
    start_s: float = Field(ge=0)
    end_s: float
    tags: list[str] = []
    rating: int | None = Field(default=None, ge=1, le=5)
    involves_her: bool = False
    note: str | None = None
    export: str | None = None  # relative to clips_dir; cleared when start/end change
    created_at: datetime = Field(default_factory=_now)

    @model_validator(mode="after")
    def _check_range(self) -> "Clip":
        if self.end_s <= self.start_s:
            raise ValueError(f"end_s ({self.end_s}) must be after start_s ({self.start_s})")
        return self


class Game(BaseModel):
    schema_version: int = 1
    id: str  # video filename without .mp4; also the folder name under games_dir/<year>/
    year: int
    played_on: date | None = None
    opponent: str | None = None
    title: str | None = None
    original: str  # relative to full_dir, e.g. "2026/gu11-vs-saa-2026-10-10.mp4"
    proxy: str | None = None  # relative to proxy_dir; set if a keyframe-dense copy exists
    duration_s: float
    # Exact frame rate as a fraction (Veo is 30000/1001). Rounding 29.97 to 30
    # drifts by ~4s over a 70 min game.
    fps_num: int
    fps_den: int
    width: int
    height: int
    has_audio: bool = True
    kickoff_1h_s: float | None = None
    kickoff_2h_s: float | None = None
    created_at: datetime
    clips: list[Clip] = []

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den

    def frame_at(self, t_s: float) -> int:
        return round(t_s * self.fps_num / self.fps_den)

    def match_clock(self, t_s: float) -> str | None:
        """"23'" / "2H 7'" from the kickoff times (same rule as the web pages); None if unknown."""
        if self.kickoff_2h_s is not None and t_s >= self.kickoff_2h_s:
            return f"2H {int((t_s - self.kickoff_2h_s) // 60) + 1}'"
        if self.kickoff_1h_s is not None and t_s >= self.kickoff_1h_s:
            return f"{int((t_s - self.kickoff_1h_s) // 60) + 1}'"
        return None

    def time_label(self, t_s: float) -> str:
        """Match clock if kickoffs are set, otherwise the video time as m:ss."""
        if clock := self.match_clock(t_s):
            return clock
        m, s = divmod(int(t_s), 60)
        return f"{m}:{s:02d}"

    def clip(self, clip_id: str) -> Clip:
        for c in self.clips:
            if c.id == clip_id:
                return c
        raise KeyError(f"no clip {clip_id!r} in game {self.id!r}")


# ---------- reels (Phase 2) ----------

Preset = Literal["landscape_1080p", "preview_540p"]


class ReelItem(BaseModel):
    game_id: str
    clip_id: str


class Reel(BaseModel):
    schema_version: int = 1
    id: str = Field(default_factory=lambda: secrets.token_hex(4))
    title: str
    items: list[ReelItem] = []
    # Seconds for the opening title card and the card before each clip; 0 = no cards.
    card_s: float = Field(default=3.0, ge=0, le=15)
    created_at: datetime = Field(default_factory=_now)


class Segment(BaseModel):
    """Where one reel item sits in a render, and where it came from in its game."""

    game_id: str
    clip_id: str
    reel_start_s: float
    reel_end_s: float
    # Source times are recorded (not just the clip id) so jumping back to the
    # full game still works after the clip is edited or deleted.
    source_start_s: float
    source_end_s: float
    speed: float = 1.0
    # The card shown before this clip, if any: where it starts in the reel and its text.
    card_start_s: float | None = None
    card_lines: list[str] = []

    def source_time(self, reel_t: float) -> float:
        return self.source_start_s + (reel_t - self.reel_start_s) * self.speed


class Render(BaseModel):
    """Written next to each render as <preset>.render.json."""

    reel_id: str
    preset: Preset
    file: str  # relative to the reel's folder
    duration_s: float
    segments: list[Segment]
    skipped: list[ReelItem] = []  # items whose clip was missing at render time
    start_card_lines: list[str] = []  # empty if there was no opening card
    reel: Reel  # the definition this was rendered from, so it can be reproduced
    created_at: datetime = Field(default_factory=_now)
