"""
FastAPI app. Phase 0: list games and stream video (with HTTP Range so the
browser can seek), plus a scrub-test page. Phase 1 grows this into the tagger.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from soccer_clips.config import get_settings
from soccer_clips.models import Game
from soccer_clips.scan import scan
from soccer_clips.store import Store

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


def create_app(store: Store | None = None) -> FastAPI:
    store = store or Store(get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        added = scan(store.settings, store)
        if added:
            logger.info("registered %d new game(s): %s", len(added), [g.id for g in added])
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/api/games")
    def list_games() -> list[Game]:
        return store.list_games()

    @app.get("/api/games/{game_id}")
    def get_game(game_id: str) -> Game:
        try:
            return store.load_game(game_id)
        except KeyError:
            raise HTTPException(404, f"no game {game_id}")

    @app.get("/video/{game_id}")
    def video(game_id: str) -> FileResponse:
        # FileResponse answers Range requests, which the <video> element needs to seek.
        game = get_game(game_id)
        return FileResponse(store.playback_path(game), media_type="video/mp4")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app
