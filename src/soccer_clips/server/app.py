"""
FastAPI app: games, clips, tags, reels and background jobs as a JSON API,
video streaming (with HTTP Range so the browser can seek), the tagging page
and the reel page.
"""

import logging
import re
import shutil
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import get_args

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from soccer_clips.config import get_settings
from soccer_clips.export import export_job, run_export
from soccer_clips.jobs import Job, JobQueue
from soccer_clips.models import Callout, Clip, Game, Preset, Reel, ReelItem, Render, Tag
from soccer_clips.reel.render import render_job, run_render, stale_reasons
from soccer_clips.scan import scan
from soccer_clips.store import Store

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class RequestBody(BaseModel):
    # Reject unknown fields instead of ignoring them, so a page that's newer
    # than the running server gets an error rather than a silent no-op save.
    model_config = ConfigDict(extra="forbid")


class GamePatch(RequestBody):
    played_on: date | None = None
    opponent: str | None = None
    description: str | None = None
    kickoff_1h_s: float | None = None
    kickoff_2h_s: float | None = None


class ClipCreate(RequestBody):
    start_s: float
    end_s: float
    tags: list[str] = []
    rating: int | None = Field(default=None, ge=1, le=5)
    involves_her: bool = False
    note: str | None = None


class ClipPatch(RequestBody):
    start_s: float | None = None
    end_s: float | None = None
    tags: list[str] | None = None
    rating: int | None = Field(default=None, ge=1, le=5)
    involves_her: bool | None = None
    note: str | None = None
    callout: Callout | None = None  # null removes it


class ExportRequest(RequestBody):
    clip_ids: list[str]
    fast: bool = False


class ReelCreate(RequestBody):
    title: str
    items: list[ReelItem] = []
    card_s: float = Field(default=2.0, ge=0, le=15)
    callout_s: float = Field(default=0.75, ge=0, le=10)


class ReelPatch(RequestBody):
    title: str | None = None
    items: list[ReelItem] | None = None
    card_s: float | None = Field(default=None, ge=0, le=15)
    callout_s: float | None = Field(default=None, ge=0, le=10)


class RenderRequest(RequestBody):
    preset: Preset = "landscape_1080p"


class ReelItemView(ReelItem):
    """A reel item plus what the reel page needs to show it."""

    missing: bool = False
    start_s: float | None = None
    end_s: float | None = None
    tags: list[str] = []
    involves_her: bool = False
    rating: int | None = None
    note: str | None = None
    has_callout: bool = False
    kickoff_1h_s: float | None = None
    kickoff_2h_s: float | None = None


class RenderView(Render):
    stale: list[str] = []  # reasons the render no longer matches; empty if current


class ReelView(BaseModel):
    reel: Reel
    items: list[ReelItemView]
    renders: list[RenderView]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "reel"


def create_app(store: Store | None = None) -> FastAPI:
    store = store or Store(get_settings())
    jobs = JobQueue()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        added = scan(store.settings, store)
        if added:
            logger.info("registered %d new game(s): %s", len(added), [g.id for g in added])
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.jobs = jobs

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
            if "callout" in changes and new.callout and not new.start_s <= new.callout.t_s < new.end_s:
                raise HTTPException(422, "the callout must be inside the clip")
            if (new.start_s, new.end_s) != (old.start_s, old.end_s):
                new.export = None  # the exported file no longer matches
                if new.callout and not new.start_s <= new.callout.t_s < new.end_s:
                    new.callout = None  # the clip no longer contains that moment
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
        return [
            jobs.submit(export_job(game_id, cid, body.fast), lambda job: run_export(store, job))
            for cid in body.clip_ids
        ]

    @app.get("/api/jobs")
    def list_jobs() -> list[Job]:
        return list(jobs.jobs.values())

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> Job:
        if job_id not in jobs.jobs:
            raise HTTPException(404, f"no job {job_id}")
        return jobs.jobs[job_id]

    # ---------- reels ----------

    def load_reel(reel_id: str) -> Reel:
        try:
            return store.load_reel(reel_id)
        except KeyError:
            raise HTTPException(404, f"no reel {reel_id}")

    def check_items(items: list[ReelItem]) -> None:
        for item in items:
            try:
                store.load_game(item.game_id).clip(item.clip_id)
            except KeyError:
                raise HTTPException(404, f"no clip {item.clip_id} in game {item.game_id}")

    def reel_view(reel: Reel) -> ReelView:
        games: dict[str, Game] = {}
        items = []
        for item in reel.items:
            try:
                game = games.get(item.game_id) or store.load_game(item.game_id)
                games[game.id] = game
                clip = game.clip(item.clip_id)
            except KeyError:
                items.append(ReelItemView(**item.model_dump(), missing=True))
                continue
            items.append(ReelItemView(
                **item.model_dump(), start_s=clip.start_s, end_s=clip.end_s, tags=clip.tags,
                involves_her=clip.involves_her, rating=clip.rating, note=clip.note, has_callout=clip.callout is not None,
                kickoff_1h_s=game.kickoff_1h_s, kickoff_2h_s=game.kickoff_2h_s,
            ))
        renders = []
        for preset in get_args(Preset):
            render = store.load_render(reel.id, preset)
            if render and (store.reel_dir(reel.id) / render.file).exists():
                renders.append(RenderView(**render.model_dump(), stale=stale_reasons(store, reel, render)))
        return ReelView(reel=reel, items=items, renders=renders)

    @app.get("/api/reels")
    def list_reels() -> list[ReelView]:
        return [reel_view(r) for r in store.list_reels()]

    @app.post("/api/reels", status_code=201)
    def create_reel(body: ReelCreate) -> ReelView:
        check_items(body.items)
        reel = Reel(title=body.title.strip() or "Untitled reel", items=body.items,
                    card_s=body.card_s, callout_s=body.callout_s)
        store.save_reel(reel)
        return reel_view(reel)

    @app.get("/api/reels/{reel_id}")
    def get_reel(reel_id: str) -> ReelView:
        return reel_view(load_reel(reel_id))

    @app.patch("/api/reels/{reel_id}")
    def patch_reel(reel_id: str, patch: ReelPatch) -> ReelView:
        load_reel(reel_id)
        if patch.items is not None:
            check_items(patch.items)
        with store.edit_reel(reel_id) as reel:
            if patch.title is not None:
                reel.title = patch.title.strip() or reel.title
            if patch.items is not None:
                reel.items = patch.items
            if patch.card_s is not None:
                reel.card_s = patch.card_s
            if patch.callout_s is not None:
                reel.callout_s = patch.callout_s
        return reel_view(reel)

    @app.delete("/api/reels/{reel_id}", status_code=204)
    def delete_reel(reel_id: str) -> None:
        load_reel(reel_id)
        if jobs.active(reel_id=reel_id):
            raise HTTPException(409, "reel is rendering; wait for it to finish")
        shutil.rmtree(store.reel_dir(reel_id))

    @app.post("/api/reels/{reel_id}/render", status_code=202)
    def render_reel(reel_id: str, body: RenderRequest) -> Job:
        reel = load_reel(reel_id)
        if not reel.items:
            raise HTTPException(422, "reel has no clips")
        existing = jobs.active(reel_id=reel_id, preset=body.preset)
        if existing:
            return existing
        return jobs.submit(render_job(reel_id, body.preset), lambda job: run_render(store, job))

    @app.get("/reels/{reel_id}/{preset}.mp4")
    def reel_file(reel_id: str, preset: Preset) -> FileResponse:
        reel = load_reel(reel_id)
        path = store.reel_dir(reel_id) / f"{preset}.mp4"
        if not path.exists():
            raise HTTPException(404, "not rendered")
        # inline so <video> can play it; the page's download link sets the save-as name
        return FileResponse(path, media_type="video/mp4", filename=f"{slug(reel.title)}-{preset}.mp4",
                            content_disposition_type="inline")

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

    @app.get("/reel")
    def reel_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "reel.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app
