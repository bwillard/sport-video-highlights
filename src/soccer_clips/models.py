"""
Pydantic models for everything stored as JSON. New fields must have defaults
so older files keep loading; bump schema_version only for a real migration.
"""

import secrets
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


def _now() -> datetime:
    return datetime.now().astimezone()


class Tag(BaseModel):
    name: str
    hotkey: str | None = None  # "1".."9" in the tagger


DEFAULT_TAGS = [
    Tag(name=name, hotkey=str(i))
    for i, name in enumerate(
        ["goal", "shot", "assist", "pass", "dribble", "defense", "save"],
        start=1,
    )
]


class Callout(BaseModel):
    """A freeze-frame highlight of a player at one moment in a clip (Phase 3.1)."""

    t_s: float  # game time of the frozen frame; inside the clip's range
    # Position in the frame: x as a fraction of the width, y of the height.
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    r: float = Field(default=0.06, gt=0, le=0.5)  # ring radius, fraction of the frame height


Rating = Literal["happy", "sad"]


class Clip(BaseModel):
    # Random rather than sequential so reels can reference (game_id, clip_id)
    # and clips can be deleted without renumbering.
    id: str = Field(default_factory=lambda: secrets.token_hex(4))
    start_s: float = Field(ge=0)
    end_s: float
    tags: list[str] = []
    rating: Rating | None = None
    # Jersey numbers of the players the clip is noteworthy for, and whether
    # it's a noteworthy team play; either, both or neither. (Replaced the old
    # involves_her flag, which the store converts to the default player.)
    players: list[int] = []
    team: bool = False
    note: str | None = None
    callout: Callout | None = None  # dropped if start/end change so it falls outside the clip
    export: str | None = None  # relative to clips_dir; cleared when start/end change
    created_at: datetime = Field(default_factory=_now)

    @field_validator("players")
    @classmethod
    def _sorted_players(cls, v: list[int]) -> list[int]:
        if any(n < 0 or n > 999 for n in v):
            raise ValueError("player numbers must be 0-999")
        return sorted(set(v))

    @field_validator("rating", mode="before")
    @classmethod
    def _from_stars(cls, v):
        """Ratings used to be 1-5 stars: 4-5 became happy, 1-2 sad, 3 unrated."""
        if isinstance(v, int) and not isinstance(v, bool):
            return "happy" if v >= 4 else "sad" if v <= 2 else None
        return v

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
    description: str | None = None  # free text set in the tagger, e.g. "Home vs SAA, 3-1 win"
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
# What a render includes on top of the clips: everything ("all"), everything
# but the clip notes ("no_notes"), or neither notes nor callout freezes
# ("plain"). The opening card and game times are shown in all of them.
Variant = Literal["all", "no_notes", "plain"]
# How each clip's time + note is shown: a caption over the start of the clip
# ("overlay") or a separate title card before it ("card").
CardStyle = Literal["overlay", "card"]
# Callout freeze length when the reel doesn't set one. Longer with overlays,
# where the caption is also read during the freeze.
DEFAULT_FREEZE_S = {"overlay": 1.5, "card": 0.75}


class ReelItem(BaseModel):
    game_id: str
    clip_id: str


class Reel(BaseModel):
    schema_version: int = 1
    id: str = Field(default_factory=lambda: secrets.token_hex(4))
    title: str
    items: list[ReelItem] = []
    card_style: CardStyle = "overlay"
    # Shortest a title card or caption is shown, in seconds; ones with more
    # text stay up longer, long enough to read (see reel/cards.py). 0 = none.
    card_s: float = Field(default=2.0, ge=0, le=15)
    # Seconds to freeze on each clip's callout; 0 = ignore callouts; None =
    # the default for the card style (DEFAULT_FREEZE_S).
    callout_s: float | None = Field(default=None, ge=0, le=10)
    created_at: datetime = Field(default_factory=_now)

    @computed_field
    @property
    def freeze_s(self) -> float:
        """The callout freeze length actually used."""
        return self.callout_s if self.callout_s is not None else DEFAULT_FREEZE_S[self.card_style]


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
    # The caption over the start of this clip, if any (overlay style): its
    # text and how long it shows from the clip's start (it's also shown
    # during the callout freeze).
    caption_lines: list[str] = []
    caption_s: float = 0.0
    # The callout freeze inside this clip, if any: reel time it starts, its
    # length, and the callout it shows.
    freeze_at_s: float | None = None
    freeze_s: float = 0.0
    callout: Callout | None = None

    def source_time(self, reel_t: float) -> float:
        offset = max(0.0, reel_t - self.reel_start_s)
        if self.freeze_at_s is not None and reel_t >= self.freeze_at_s:
            # frozen on the callout frame, then the clip carries on from there
            offset = max(self.freeze_at_s - self.reel_start_s, offset - self.freeze_s)
        return self.source_start_s + offset * self.speed


class Render(BaseModel):
    """Written next to each render as <name>.render.json."""

    reel_id: str
    preset: Preset
    variant: Variant = "all"
    file: str  # relative to the reel's folder
    duration_s: float
    segments: list[Segment]
    skipped: list[ReelItem] = []  # items whose clip was missing at render time
    start_card_lines: list[str] = []  # empty if there was no opening card
    card_style: CardStyle = "card"  # renders made before overlays existed used cards
    reel: Reel  # the definition this was rendered from, so it can be reproduced
    created_at: datetime = Field(default_factory=_now)

    @computed_field
    @property
    def name(self) -> str:
        return render_name(self.preset, self.variant)


def render_name(preset: Preset, variant: Variant = "all") -> str:
    """File name (without extension) of a render: the preset, plus the variant unless it's "all"."""
    return preset if variant == "all" else f"{preset}-{variant}"
