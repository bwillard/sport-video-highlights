"""
Pydantic models for everything stored as JSON. New fields must have defaults
so older files keep loading; bump schema_version only for a real migration.
"""

import secrets
from datetime import date, datetime

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

    def clip(self, clip_id: str) -> Clip:
        for c in self.clips:
            if c.id == clip_id:
                return c
        raise KeyError(f"no clip {clip_id!r} in game {self.id!r}")
