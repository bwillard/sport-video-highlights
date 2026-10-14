"""
Pydantic models for everything stored as JSON. New fields must have defaults
so older files keep loading; bump schema_version only for a real migration.
"""

from datetime import date, datetime

from pydantic import BaseModel


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

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den

    def frame_at(self, t_s: float) -> int:
        return round(t_s * self.fps_num / self.fps_den)
