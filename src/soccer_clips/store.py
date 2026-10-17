"""
JSON storage. Every read/write of state goes through here so that writes are
atomic (temp file + rename) and serialized per file (background jobs and the
UI can update the same game.json).
"""

import json
import os
import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import TypeAdapter

from soccer_clips.config import Settings
from soccer_clips.models import DEFAULT_TAGS, Game, Preset, Reel, Render, Tag

_tags_adapter = TypeAdapter(list[Tag])

_locks: defaultdict[Path, threading.Lock] = defaultdict(threading.Lock)
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _locks_guard:
        return _locks[path.resolve()]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: object) -> None:
    atomic_write_text(path, json.dumps(data, indent=2) + "\n")


class Store:
    def __init__(self, settings: Settings):
        self.settings = settings

    def game_dir(self, year: int, game_id: str) -> Path:
        return self.settings.games_dir / str(year) / game_id

    def _game_path(self, year: int, game_id: str) -> Path:
        return self.game_dir(year, game_id) / "game.json"

    def _find_game_path(self, game_id: str) -> Path:
        matches = list(self.settings.games_dir.glob(f"*/{game_id}/game.json"))
        if not matches:
            raise KeyError(f"no game {game_id!r}")
        if len(matches) > 1:
            raise KeyError(f"game id {game_id!r} exists in more than one year: {matches}")
        return matches[0]

    def has_game(self, year: int, game_id: str) -> bool:
        return self._game_path(year, game_id).exists()

    def _parse_game(self, text: str) -> Game:
        """
        Game from its JSON. Clips saved before players existed have an
        involves_her flag instead; that becomes the default player's number.
        """
        data = json.loads(text)
        for clip in data.get("clips", []):
            if clip.pop("involves_her", False) and "players" not in clip:
                if self.settings.default_player is None:
                    raise ValueError("clips flagged 'her' need SOCCER_DEFAULT_PLAYER set in .env to convert them")
                clip["players"] = [self.settings.default_player]
        return Game.model_validate(data)

    def load_game(self, game_id: str) -> Game:
        return self._parse_game(self._find_game_path(game_id).read_text())

    def save_game(self, game: Game) -> None:
        path = self._game_path(game.year, game.id)
        with _lock_for(path):
            atomic_write_text(path, game.model_dump_json(indent=2) + "\n")

    @contextmanager
    def edit_game(self, game_id: str) -> Iterator[Game]:
        """Load-modify-save under the file's lock: `with store.edit_game(id) as g: g.x = 1`."""
        path = self._find_game_path(game_id)
        with _lock_for(path):
            game = self._parse_game(path.read_text())
            yield game
            atomic_write_text(path, game.model_dump_json(indent=2) + "\n")

    def list_games(self) -> list[Game]:
        paths = sorted(self.settings.games_dir.glob("*/*/game.json"))
        games = [self._parse_game(p.read_text()) for p in paths]
        return sorted(games, key=lambda g: (g.played_on is None, g.played_on, g.id))

    @property
    def tags_path(self) -> Path:
        return self.settings.games_dir / "tags.json"

    def load_tags(self) -> list[Tag]:
        """The tag list (hand-editable); written with the defaults on first use."""
        path = self.tags_path
        with _lock_for(path):
            if not path.exists():
                atomic_write_text(path, _tags_adapter.dump_json(DEFAULT_TAGS, indent=2).decode() + "\n")
            return _tags_adapter.validate_json(path.read_text())

    def original_path(self, game: Game) -> Path:
        return self.settings.full_dir / game.original

    def playback_path(self, game: Game) -> Path:
        """Keyframe-dense copy if one exists, otherwise the original."""
        if game.proxy:
            proxy = self.settings.proxy_dir / game.proxy
            if proxy.exists():
                return proxy
        return self.original_path(game)

    def clip_export_rel(self, game: Game, clip_id: str) -> str:
        """Export path for a clip, relative to clips_dir."""
        return f"{game.year}/{game.id}/{clip_id}.mp4"

    # ---------- reels ----------

    def reel_dir(self, reel_id: str) -> Path:
        return self.settings.reels_dir / reel_id

    def _reel_path(self, reel_id: str) -> Path:
        return self.reel_dir(reel_id) / "reel.json"

    def load_reel(self, reel_id: str) -> Reel:
        path = self._reel_path(reel_id)
        if not path.exists():
            raise KeyError(f"no reel {reel_id!r}")
        return Reel.model_validate_json(path.read_text())

    def save_reel(self, reel: Reel) -> None:
        path = self._reel_path(reel.id)
        with _lock_for(path):
            atomic_write_text(path, reel.model_dump_json(indent=2) + "\n")

    @contextmanager
    def edit_reel(self, reel_id: str) -> Iterator[Reel]:
        path = self._reel_path(reel_id)
        with _lock_for(path):
            if not path.exists():
                raise KeyError(f"no reel {reel_id!r}")
            reel = Reel.model_validate_json(path.read_text())
            yield reel
            atomic_write_text(path, reel.model_dump_json(indent=2) + "\n")

    def list_reels(self) -> list[Reel]:
        reels = [Reel.model_validate_json(p.read_text()) for p in self.settings.reels_dir.glob("*/reel.json")]
        return sorted(reels, key=lambda r: r.created_at, reverse=True)

    def render_path(self, reel_id: str, preset: Preset) -> Path:
        return self.reel_dir(reel_id) / f"{preset}.render.json"

    def load_render(self, reel_id: str, preset: Preset) -> Render | None:
        path = self.render_path(reel_id, preset)
        return Render.model_validate_json(path.read_text()) if path.exists() else None

    def save_render(self, render: Render) -> None:
        atomic_write_text(self.render_path(render.reel_id, render.preset), render.model_dump_json(indent=2) + "\n")
