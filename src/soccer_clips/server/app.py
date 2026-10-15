"""
FastAPI app: games, clips, tags and export jobs as a JSON API, video streaming
(with HTTP Range so the browser can seek), and the tagging page.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from soccer_clips.config import get_settings
from soccer_clips.export import ExportQueue, Job
from soccer_clips.models import Clip, Game, Tag
from soccer_clips.scan import scan
from soccer_clips.store import Store

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class GamePatch(BaseModel):
    opponent: str | None = None
    title: str | None = None
    kickoff_1h_s: float | None = None
    kickoff_2h_s: float | None = None


class ClipCreate(BaseModel):
    start_s: float
    end_s: float
    tags: list[str] = []
    rating: int | None = Field(default=None, ge=1, le=5)
    involves_her: bool = False
    note: str | None = None


class ClipPatch(BaseModel):
    start_s: float | None = None
    end_s: float | None = None
    tags: list[str] | None = None
    rating: int | None = Field(default=None, ge=1, le=5)
    involves_her: bool | None = None
    note: str | None = None


class ExportRequest(BaseModel):
    clip_ids: list[str]
    fast: bool = False


def create_app(store: Store | None = None) -> FastAPI:
    store = store or Store(get_settings())
    exports = ExportQueue(store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        added = scan(store.settings, store)
        if added:
            logger.info("registered %d new game(s): %s", len(added), [g.id for g in added])
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.exports = exports

    def load(game_id: str) -> Game:
        try:
            return store.load_game(game_id)
        except KeyError:
            raise HTTPException(404, f"no game {game_id}")

    def check_range(game: Game, clip: Clip) -> None:
        if clip.end_s > game.duration_s + 0.5:
            raise HTTPException(422, f"end_s {clip.end_s} is past the end of the video")

    @app.get("/api/games")
    def list_games() -> list[Game]:
        return store.list_games()

    @app.get("/api/games/{game_id}")
    def get_game(game_id: str) -> Game:
        return load(game_id)

    @app.patch("/api/games/{game_id}")
    def patch_game(game_id: str, patch: GamePatch) -> Game:
        load(game_id)
        with store.edit_game(game_id) as game:
            # exclude_unset so an explicit null clears a field (e.g. a kickoff).
            for key, value in patch.model_dump(exclude_unset=True).items():
                setattr(game, key, value)
        return game

    @app.get("/api/tags")
    def list_tags() -> list[Tag]:
        return store.load_tags()

    @app.get("/api/games/{game_id}/clips")
    def list_clips(game_id: str) -> list[Clip]:
        return sorted(load(game_id).clips, key=lambda c: c.start_s)

    @app.post("/api/games/{game_id}/clips", status_code=201)
    def create_clip(game_id: str, body: ClipCreate) -> Clip:
        game = load(game_id)
        try:
            clip = Clip(**body.model_dump())
        except ValueError as e:
            raise HTTPException(422, str(e))
        check_range(game, clip)
        with store.edit_game(game_id) as game:
            game.clips.append(clip)
        return clip

    @app.patch("/api/games/{game_id}/clips/{clip_id}")
    def patch_clip(game_id: str, clip_id: str, patch: ClipPatch) -> Clip:
        load(game_id)
        with store.edit_game(game_id) as game:
            try:
                old = game.clip(clip_id)
            except KeyError:
                raise HTTPException(404, f"no clip {clip_id}")
            changes = patch.model_dump(exclude_unset=True)
            try:
                new = Clip(**{**old.model_dump(), **changes})
            except ValueError as e:
                raise HTTPException(422, str(e))
            check_range(game, new)
            if (new.start_s, new.end_s) != (old.start_s, old.end_s):
                new.export = None  # the exported file no longer matches
            game.clips[game.clips.index(old)] = new
        return new

    @app.delete("/api/games/{game_id}/clips/{clip_id}", status_code=204)
    def delete_clip(game_id: str, clip_id: str) -> None:
        load(game_id)
        with store.edit_game(game_id) as game:
            try:
                clip = game.clip(clip_id)
            except KeyError:
                raise HTTPException(404, f"no clip {clip_id}")
            game.clips.remove(clip)
        (store.settings.clips_dir / store.clip_export_rel(game, clip_id)).unlink(missing_ok=True)

    @app.post("/api/games/{game_id}/export", status_code=202)
    def export_clips(game_id: str, body: ExportRequest) -> list[Job]:
        game = load(game_id)
        known = {c.id for c in game.clips}
        missing = [cid for cid in body.clip_ids if cid not in known]
        if missing:
            raise HTTPException(404, f"no clips {missing}")
        return [exports.submit(game_id, cid, body.fast) for cid in body.clip_ids]

    @app.get("/api/jobs")
    def list_jobs() -> list[Job]:
        return list(exports.jobs.values())

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> Job:
        if job_id not in exports.jobs:
            raise HTTPException(404, f"no job {job_id}")
        return exports.jobs[job_id]

    @app.get("/video/{game_id}")
    def video(game_id: str) -> FileResponse:
        # FileResponse answers Range requests, which the <video> element needs to seek.
        return FileResponse(store.playback_path(load(game_id)), media_type="video/mp4")

    @app.get("/clips/{game_id}/{clip_id}.mp4")
    def clip_file(game_id: str, clip_id: str) -> FileResponse:
        game = load(game_id)
        path = store.settings.clips_dir / store.clip_export_rel(game, clip_id)
        if not path.exists():
            raise HTTPException(404, "clip not exported")
        return FileResponse(path, media_type="video/mp4", filename=f"{game_id}-{clip_id}.mp4")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app
