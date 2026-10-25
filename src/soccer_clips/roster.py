"""
Our team's roster: shirt numbers, and names where given. Kept with the game data in
<SOCCER_GAMES_DIR>/roster.json (not in the repo: names of minors stay on this machine).

Numbers not on the roster can still turn up (guest players, number changes), so readers
treat the roster as the likely numbers, not the only ones.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from soccer_clips.config import Settings
from soccer_clips.store import atomic_write_text


class RosterPlayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=0, le=99)
    name: str | None = None


class Roster(BaseModel):
    model_config = ConfigDict(extra="forbid")

    players: list[RosterPlayer] = []

    def numbers(self) -> list[int]:
        return sorted(p.number for p in self.players)

    def name(self, number: int) -> str | None:
        return next((p.name for p in self.players if p.number == number), None)


def roster_path(settings: Settings) -> Path:
    return settings.games_dir / "roster.json"


def load_roster(settings: Settings) -> Roster:
    p = roster_path(settings)
    return Roster.model_validate_json(p.read_text()) if p.exists() else Roster()


def save_roster(settings: Settings, roster: Roster) -> None:
    atomic_write_text(roster_path(settings), roster.model_dump_json(indent=2, exclude_none=True) + "\n")
