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

from soccer_clips.config import Settings
from soccer_clips.models import Game

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

    def load_game(self, game_id: str) -> Game:
        return Game.model_validate_json(self._find_game_path(game_id).read_text())

    def save_game(self, game: Game) -> None:
        path = self._game_path(game.year, game.id)
        with _lock_for(path):
            atomic_write_text(path, game.model_dump_json(indent=2) + "\n")

    @contextmanager
    def edit_game(self, game_id: str) -> Iterator[Game]:
        """Load-modify-save under the file's lock: `with store.edit_game(id) as g: g.x = 1`."""
        path = self._find_game_path(game_id)
        with _lock_for(path):
            game = Game.model_validate_json(path.read_text())
            yield game
            atomic_write_text(path, game.model_dump_json(indent=2) + "\n")

    def list_games(self) -> list[Game]:
        paths = sorted(self.settings.games_dir.glob("*/*/game.json"))
        games = [Game.model_validate_json(p.read_text()) for p in paths]
        return sorted(games, key=lambda g: (g.played_on is None, g.played_on, g.id))

    def original_path(self, game: Game) -> Path:
        return self.settings.full_dir / game.original

    def playback_path(self, game: Game) -> Path:
        """Keyframe-dense copy if one exists, otherwise the original."""
        if game.proxy:
            proxy = self.settings.proxy_dir / game.proxy
            if proxy.exists():
                return proxy
        return self.original_path(game)
