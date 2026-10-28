"""
FastAPI app: games, clips, tags, reels and background jobs as a JSON API,
video streaming (with HTTP Range so the browser can seek), the tagging page
and the reel page.
"""

import json
import logging
import os
import re

import numpy as np
import shutil
import threading
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal, get_args

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from soccer_clips.config import get_settings
from soccer_clips.export import export_job, run_export
from soccer_clips.jobs import Job, JobQueue
from soccer_clips.models import (
    Callout, CardStyle, Clip, Game, Preset, Rating, Reel, ReelItem, Render, Tag, Variant, render_name,
)
from soccer_clips.reel.render import render_job, run_render, stale_reasons
from soccer_clips.scan import scan
from soccer_clips.store import Store
from soccer_clips.track import ball_labels, ball_others

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
    end_1h_s: float | None = None
    end_2h_s: float | None = None


class ClipCreate(RequestBody):
    start_s: float
    end_s: float
    tags: list[str] = []
    rating: Rating | None = None
    players: list[int] = []
    team: bool = False
    note: str | None = None


class ClipPatch(RequestBody):
    start_s: float | None = None
    end_s: float | None = None
    tags: list[str] | None = None
    rating: Rating | None = None
    players: list[int] | None = None
    team: bool | None = None
    note: str | None = None
    callout: Callout | None = None  # null removes it


class ExportRequest(RequestBody):
    clip_ids: list[str]
    fast: bool = False


class ReelCreate(RequestBody):
    title: str
    items: list[ReelItem] = []
    card_style: CardStyle = "overlay"
    card_s: float = Field(default=2.0, ge=0, le=15)
    callout_s: float | None = Field(default=None, ge=0, le=10)  # None = default for the card style


class ReelPatch(RequestBody):
    title: str | None = None
    items: list[ReelItem] | None = None
    card_style: CardStyle | None = None
    card_s: float | None = Field(default=None, ge=0, le=15)
    callout_s: float | None = Field(default=None, ge=0, le=10)  # explicit null = back to the style default


class RenderRequest(RequestBody):
    preset: Preset = "landscape_1080p"
    variant: Variant = "all"


class BallLabelIn(RequestBody):
    game_id: str
    frame: int = Field(ge=0)
    state: ball_labels.State | Literal["cleared"]
    x: float | None = Field(default=None, ge=0, le=1)
    y: float | None = Field(default=None, ge=0, le=1)
    r: float | None = Field(default=None, gt=0, le=0.1)


class OtherBallIn(RequestBody):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    r: float | None = Field(default=None, gt=0, le=0.1)


class BallFixIn(RequestBody):
    frame: int = Field(ge=0)
    action: Literal["ball", "none"]
    x: float | None = Field(default=None, ge=0, le=1)
    y: float | None = Field(default=None, ge=0, le=1)
    r: float | None = Field(default=None, gt=0, le=0.1)
    label_set: Literal["auto", "train", "bench"] | None = None  # also add the frame as a labelled ball frame


class OtherBallFixIn(OtherBallIn):
    frame: int = Field(ge=0)
    label_set: Literal["auto", "train", "bench"] | None = "auto"  # if the frame isn't labelled yet


class PickIn(RequestBody):
    frame: int = Field(ge=0)
    player: int = Field(ge=0)
    kind: Literal["is", "not", "no"]  # "no": a suggestion turned down
    x: float | None = Field(default=None, ge=0, le=1)  # "not" without a position: the marker's own box
    y: float | None = Field(default=None, ge=0, le=1)
    track: int | None = None  # the tracklet meant, when boxes overlap (suggestions know it)


class AnswerIn(RequestBody):
    track: int
    frame: int = Field(ge=0)  # a frame the tracklet has a box in
    start: int = Field(ge=0)  # where the stretch starts (finds a labelled stretch's pick)
    yes: bool
    labelled: bool = False  # a stretch already labelled as the player


class AnswersIn(RequestBody):
    player: int = Field(ge=0)
    batch: str = Field(min_length=1, max_length=64)  # an id for the page, so it can be taken back
    answers: list[AnswerIn]


class RosterPlayerIn(RequestBody):
    number: int = Field(ge=0, le=99)
    name: str | None = Field(default=None, max_length=60)


class RosterIn(RequestBody):
    players: list[RosterPlayerIn]


class StretchIn(RequestBody):
    game: str
    track: int
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    label: int | None = None  # who it's labelled as now (None: unlabelled)


class AssignIn(RequestBody):
    player: int | None = Field(default=None, ge=-1, le=99)  # None: unlabel; -1: not one of ours (people.NOT_OURS)
    batch: str = Field(min_length=1, max_length=64)  # so it can be taken back
    stretches: list[StretchIn] = Field(max_length=2000)


class RetrainIn(RequestBody):
    kind: Literal["reid", "ball"] = "reid"
    action: Literal["train", "retrack"] = "train"


class SplitIn(RequestBody):
    game: str
    track: int
    frames: list[int] = Field(min_length=1, max_length=40)  # where a different girl starts
    batch: str = Field(min_length=1, max_length=64)


class VerdictIn(RequestBody):
    verdict: ball_others.Verdict


class ReelItemView(ReelItem):
    """A reel item plus what the reel page needs to show it."""

    missing: bool = False
    start_s: float | None = None
    end_s: float | None = None
    tags: list[str] = []
    players: list[int] = []
    team: bool = False
    rating: Rating | None = None
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
        stop = threading.Event()
        if os.environ.get("RUNPOD_API_KEY"):
            threading.Thread(target=sweep_pods, args=(stop,), name="pod-sweep", daemon=True).start()
        yield
        stop.set()

    def sweep_pods(stop: threading.Event) -> None:
        """Every 10 minutes: delete our RunPod pods past their deadline (retrain.sweep), in case a pod's own
        switch failed."""
        from soccer_clips import retrain

        while not stop.is_set():
            try:
                retrain.sweep(store, log=logger.warning)
            except Exception as e:
                logger.warning("pod sweep failed: %s", e)
            stop.wait(600)

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
        if jobs.active(kind="cut", game_id=game_id):
            raise HTTPException(409, "the game is being cut to game time; try again when it's done")
        with store.edit_game(game_id) as game:
            # exclude_unset so an explicit null clears a field (e.g. a kickoff).
            for key, value in patch.model_dump(exclude_unset=True).items():
                setattr(game, key, value)
        return game

    @app.get("/api/tags")
    def list_tags() -> list[Tag]:
        return store.load_tags()

    @app.get("/api/settings")
    def ui_settings() -> dict:
        """What the pages need: the player the F key tags (.env), the roster's shirt numbers and first names (shown
        on the video only when asked), and the automatic labels' default threshold."""
        from soccer_clips.roster import load_roster
        from soccer_clips.track import reid

        roster = load_roster(store.settings)
        return {"default_player": store.settings.default_player, "roster": roster.numbers(),
                "first_names": {str(p.number): p.name.split()[0] for p in roster.players if p.name and p.name.split()},
                "auto_min": reid.AUTO_MIN}

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

    # ---- game time (gametime.py) ----

    @app.get("/api/games/{game_id}/gametime/suggest")
    def gametime_suggest(game_id: str) -> dict:
        """Suggested half start/end times from the player tracks (to check in the tagger)."""
        import dataclasses

        from soccer_clips import gametime

        return dataclasses.asdict(gametime.suggest(store, load(game_id)))

    @app.post("/api/games/{game_id}/gametime/cut", status_code=202)
    def gametime_cut(game_id: str) -> Job:
        """Cut the video to game time and move the game's data onto it (a background job)."""
        from soccer_clips import gametime

        game = load(game_id)
        if game.cut:
            raise HTTPException(409, "already cut to game time")
        try:
            gametime.periods(game)
        except ValueError as e:
            raise HTTPException(422, str(e))
        if existing := jobs.active(kind="cut", game_id=game_id):
            return existing

        def run(job: Job) -> None:
            report = gametime.cut_game(store, game_id, log=lambda m: logger.info(m))
            job.output = json.dumps(report)

        return jobs.submit(Job(kind="cut", game_id=game_id), run)

    @app.post("/api/games/{game_id}/gametime/restore")
    def gametime_restore(game_id: str) -> Game:
        """Undo a cut: everything back as it was (from the backup made before cutting)."""
        from soccer_clips import gametime

        try:
            gametime.restore(store, game_id, log=lambda m: logger.info(m))
        except ValueError as e:
            raise HTTPException(422, str(e))
        return load(game_id)

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
                players=clip.players, team=clip.team, rating=clip.rating, note=clip.note, has_callout=clip.callout is not None,
                kickoff_1h_s=game.kickoff_1h_s, kickoff_2h_s=game.kickoff_2h_s,
            ))
        renders = []
        for preset in get_args(Preset):
            for variant in get_args(Variant):
                render = store.load_render(reel.id, render_name(preset, variant))
                if render and (store.reel_dir(reel.id) / render.file).exists():
                    renders.append(RenderView(**render.model_dump(exclude={"name"}),
                                              stale=stale_reasons(store, reel, render)))
        return ReelView(reel=reel, items=items, renders=renders)

    @app.get("/api/reels")
    def list_reels() -> list[ReelView]:
        return [reel_view(r) for r in store.list_reels()]

    @app.post("/api/reels", status_code=201)
    def create_reel(body: ReelCreate) -> ReelView:
        check_items(body.items)
        reel = Reel(title=body.title.strip() or "Untitled reel", items=body.items,
                    card_style=body.card_style, card_s=body.card_s, callout_s=body.callout_s)
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
            if patch.card_style is not None:
                reel.card_style = patch.card_style
            if "callout_s" in patch.model_fields_set:
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
        existing = jobs.active(reel_id=reel_id, preset=body.preset, variant=body.variant)
        if existing:
            return existing
        return jobs.submit(render_job(reel_id, body.preset, body.variant), lambda job: run_render(store, job))

    render_names = {render_name(p, v) for p in get_args(Preset) for v in get_args(Variant)}

    @app.get("/reels/{reel_id}/{name}.mp4")
    def reel_file(reel_id: str, name: str) -> FileResponse:
        reel = load_reel(reel_id)
        path = store.reel_dir(reel_id) / f"{name}.mp4"
        if name not in render_names or not path.exists():
            raise HTTPException(404, "not rendered")
        # inline so <video> can play it; the page's download link sets the save-as name
        return FileResponse(path, media_type="video/mp4", filename=f"{slug(reel.title)}-{name}.mp4",
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

    # ---- ball labels (Phase 4.1) ----

    @app.get("/api/ball/sets")
    def ball_sets() -> list[dict]:
        return ball_labels.summary(store)

    @app.get("/api/ball/sets/{set_name}")
    def ball_set(set_name: str) -> list[dict]:
        if set_name not in ball_labels.load_tasks(store.settings).sets:
            raise HTTPException(404, f"no ball label set {set_name}")
        return ball_labels.tasks_with_labels(store, set_name)

    @app.post("/api/ball/labels", status_code=204)
    def ball_label(body: BallLabelIn) -> None:
        game = load(body.game_id)
        if body.frame >= game.frame_at(game.duration_s):
            raise HTTPException(422, f"frame {body.frame} is past the end of the video")
        if body.state in ("visible", "hidden") and (body.x is None or body.y is None):
            raise HTTPException(422, f"a {body.state} ball needs x and y")
        if body.state not in ("visible", "hidden"):
            body.x = body.y = body.r = None
        ball_labels.add_label(store.settings, ball_labels.BallLabel(**body.model_dump()))

    # Other real balls in a frame. Entries are only ever appended (removing one marks it
    # "not"), so an entry's index is a stable id even while a scan adds more.

    @app.get("/api/ball/others/review")
    def ball_others_review() -> list[dict]:
        return ball_others.review_list(store)

    @app.post("/api/ball/others/{game_id}/{frame}", status_code=201)
    def ball_other_add(game_id: str, frame: int, body: OtherBallIn) -> dict:
        load(game_id)
        k = ball_others.key(game_id, frame)
        others = ball_others.edit(store.settings, lambda o: o.frames.setdefault(k, []).append(
            ball_others.OtherBall(**body.model_dump(), verdict="ball")))
        return {"index": len(others.frames[k]) - 1}

    @app.put("/api/ball/others/{game_id}/{frame}/{index}", status_code=204)
    def ball_other_verdict(game_id: str, frame: int, index: int, body: VerdictIn) -> None:
        k = ball_others.key(game_id, frame)

        def set_verdict(o: ball_others.Others) -> None:
            entries = o.frames.get(k, [])
            if not 0 <= index < len(entries):
                raise HTTPException(404, f"no other ball {index} in {k}")
            entries[index].verdict = body.verdict

        ball_others.edit(store.settings, set_verdict)

    def ball_json(game: Game) -> dict:
        from soccer_clips.track import ball_game

        z = ball_game.load_path(store, game)
        if z is None:
            return {"ranges": [], "frame": [], "x": [], "y": [], "r": [], "kind": []}
        return {"ranges": z["ranges"].tolist(), "frame": z["frame"].tolist(),
                "x": np.round(z["x"], 4).tolist(), "y": np.round(z["y"], 4).tolist(),
                "r": np.round(z["r"], 4).tolist(), "kind": z["kind"].tolist()}

    @app.get("/api/games/{game_id}/ball")
    def game_ball(game_id: str) -> dict:
        """The game-ball path: parallel arrays, positions as fractions of the frame (empty if not tracked)."""
        return ball_json(load(game_id))

    @app.get("/api/games/{game_id}/ball/raw")
    def game_ball_raw(game_id: str) -> dict:
        """
        Every ball detection (the debug layer): parallel arrays sorted by frame, positions as
        fractions of the frame, plus the hand fixes (placed balls, suppressed detections).
        """
        from soccer_clips.track import ball_game

        from soccer_clips.track import world

        game = load(game_id)
        raw = ball_game.load_raw(store, game)
        order = np.argsort(raw["dets"][:, 0], kind="stable")
        d = raw["dets"][order]
        blocked = world.load_blocked(store, game, len(raw["dets"]))
        W, H = game.width, game.height
        fixes = ball_game.load_fixes(store, game)
        prefix = f"{game_id}:"  # ball_others.key
        others = [{"frame": int(k[len(prefix):]), "index": i, **b.model_dump(include={"x", "y", "r"})}
                  for k, bs in ball_others.load(store.settings).frames.items() if k.startswith(prefix)
                  for i, b in enumerate(bs) if b.verdict == "ball"]
        return {"ranges": raw["ranges"].tolist(), "frame": d[:, 0].astype(int).tolist(),
                "x": np.round((d[:, 1] + d[:, 3]) / 2 / W, 4).tolist(),
                "y": np.round((d[:, 2] + d[:, 4]) / 2 / H, 4).tolist(),
                "r": np.round(np.maximum(d[:, 3] - d[:, 1], d[:, 4] - d[:, 2]) / 2 / H, 4).tolist(),
                "conf": np.round(d[:, 5], 2).tolist(),
                "fix_balls": fixes["balls"], "suppressed": fixes["suppress"], "others": others,
                "fixed": [] if blocked is None else np.nonzero(blocked[order])[0].tolist()}  # indexes, on fixed objects

    @app.post("/api/games/{game_id}/ball/fix")
    def game_ball_fix(game_id: str, body: BallFixIn) -> dict:
        """Correct the ball path while watching; returns the updated path."""
        from soccer_clips.track import ball_game

        game = load(game_id)
        if body.action == "ball" and (body.x is None or body.y is None):
            raise HTTPException(422, "a ball fix needs x and y")
        try:
            used = ball_game.fix(store, game, body.frame, body.action, body.x, body.y, body.r, body.label_set)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {**ball_json(game), "labelled_as": used}

    @app.post("/api/games/{game_id}/ball/other", status_code=201)
    def game_ball_other(game_id: str, body: OtherBallFixIn) -> dict:
        """Mark another real ball (not the game ball) in a frame while watching, for training."""
        from soccer_clips.track import ball_game

        game = load(game_id)
        try:
            index, used = ball_game.other_ball(store, game, body.frame, body.x, body.y, body.r, body.label_set)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"index": index, "labelled_as": used}

    # ---- tracked players (Phase 3.2 / 3.3) ----

    PEOPLE_MAX_FRAMES = 1800  # one request covers at most a minute

    def people_labels(game: Game) -> dict:
        from soccer_clips.track import people

        t = people.tracks(store, game)
        picks = people.load_picks(store, game)
        return {"picks": picks, "segments": people.segments(t, picks) if t else []}

    @app.get("/api/games/{game_id}/people")
    def game_people(game_id: str) -> dict:
        """Where the game has player tracks (ranges of game frames), and the labelled stretches."""
        from soccer_clips.track import people

        game = load(game_id)
        t = people.tracks(store, game)
        got = people.roles_for(store, game)
        roles = got[0] if got else {}
        return {"ranges": t.ranges if t else [], "teams": {str(k): v for k, v in roles.items()}, **people_labels(game)}

    @app.get("/api/games/{game_id}/people/boxes")
    def game_people_boxes(game_id: str, start: int, end: int) -> Response:
        """
        Every tracked box in game frames [start, end), as little-endian float32 rows of frame,
        track, x1, y1, x2, y2 (fractions of the frame), team (0 ours, 1 opponents, 2 other, -1
        unknown; people.box_roles), sorted by frame. Binary because a minute is ~50k boxes.
        """
        from soccer_clips.track import people

        if not 0 <= start <= end or end - start > PEOPLE_MAX_FRAMES:
            raise HTTPException(422, f"ask for up to {PEOPLE_MAX_FRAMES} frames")
        game = load(game_id)
        t = people.tracks(store, game)
        if t is None:
            return Response(b"", media_type="application/octet-stream")
        a, b = t.window(start, end)
        team = people.roles_for(store, game)[1][a:b].astype(np.float32)
        rows = np.column_stack([t.frame[a:b], t.track[a:b], t.box[a:b], team]).astype("<f4")
        return Response(rows.tobytes(), media_type="application/octet-stream")

    @app.post("/api/games/{game_id}/people/picks")
    def game_people_pick(game_id: str, body: PickIn) -> dict:
        """'This is #N' / 'the marker is on the wrong person from here'; returns the labels."""
        from soccer_clips.track import people

        game = load(game_id)
        try:
            pick = people.add_pick(store, game, body.frame, body.player, body.kind, body.x, body.y, track=body.track)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"pick": pick, **people_labels(game)}

    @app.post("/api/games/{game_id}/people/answers")
    def game_people_answers(game_id: str, body: AnswersIn) -> dict:
        """A page of yes / no answers from the players review (people.answer_tracklets); returns the labels."""
        from soccer_clips.track import people

        game = load(game_id)
        try:
            n = people.answer_tracklets(store, game, body.player, [a.model_dump() for a in body.answers], body.batch)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"saved": n, **people_labels(game)}

    @app.delete("/api/games/{game_id}/people/answers/{batch}")
    def game_people_answers_undo(game_id: str, batch: str) -> dict:
        """Take back a page of answers; returns how many picks it touched and the labels."""
        from soccer_clips.track import people

        game = load(game_id)
        return {"undone": people.undo_answers(store, game, batch), **people_labels(game)}

    @app.get("/api/games/{game_id}/people/labelled")
    def game_people_labelled(game_id: str, player: int) -> dict:
        """The player's labelled stretches, least like them first (reid.labelled_scores)."""
        from soccer_clips.track import reid

        return {"stretches": reid.labelled_scores(store, load(game_id), player)}

    @app.get("/api/games/{game_id}/people/strip.jpg")
    def game_people_strip(game_id: str, track: int, start: int, end: int, n: int = 3) -> Response:
        """Crops of a tracklet across a stretch, side by side (reid.strip)."""
        from soccer_clips.track import reid

        jpg = reid.strip(store, load(game_id), track, start, end, n=min(max(n, 1), 6))
        if jpg is None:
            raise HTTPException(404, "no crops of this tracklet")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/games/{game_id}/people/suggest")
    def game_people_suggest(game_id: str, player: int, limit: int = 20) -> dict:
        """Unlabelled our-team tracklets that look most like #player (reid.suggest), best first."""
        from soccer_clips.track import reid

        return reid.suggest(store, load(game_id), player, limit=min(max(limit, 1), 200))

    @app.get("/api/games/{game_id}/people/guess")
    def game_people_guess(game_id: str, track: int, frame: int) -> dict:
        """Who the tracklet looks like at `frame` (reid.guess): {players: [{player, score, busy}]}, best first."""
        from soccer_clips.track import reid

        return reid.guess(store, load(game_id), track, frame)

    @app.get("/api/games/{game_id}/people/auto")
    def game_people_auto(game_id: str, min_score: float | None = None, margin: float | None = None) -> dict:
        """Automatic labels (reid.auto_all): {player: [{track, start, end, margin}]} for every labelled player;
        min_score / margin: how alike a stretch must look to be claimed (the tagger's setting), else the defaults."""
        from soccer_clips.track import reid

        ms = reid.AUTO_MIN if min_score is None else min(max(min_score, 0.5), 1.0)
        mg = reid.AUTO_MARGIN if margin is None else min(max(margin, 0.0), 0.5)
        return {str(p): cs for p, cs in reid.auto_all(store, load(game_id), ms, mg).items()}

    @app.delete("/api/games/{game_id}/people/picks/last")
    def game_people_undo(game_id: str) -> dict:
        """Take back the last pick; returns it (null if there were none) and the labels."""
        from soccer_clips.track import people

        game = load(game_id)
        return {"undone": people.undo_pick(store, game), **people_labels(game)}

    @app.get("/review-balls")
    def review_balls_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "review-balls.html")

    @app.get("/ball/frame/{game_id}/{frame}.jpg")
    def ball_frame(game_id: str, frame: int) -> FileResponse:
        game = load(game_id)
        if not 0 <= frame < game.frame_at(game.duration_s):
            raise HTTPException(404, f"no frame {frame}")
        return FileResponse(ball_labels.frame_image(store, game, frame), media_type="image/jpeg")

    @app.get("/roster")
    def roster_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "roster.html")

    @app.get("/players")
    def players_page() -> RedirectResponse:
        return RedirectResponse("/roster")

    @app.get("/api/roster")
    def roster_get() -> dict:
        """The roster (numbers and names) and, per number labelled anywhere, how much (gallery.summary)."""
        from soccer_clips.roster import load_roster
        from soccer_clips.track import gallery

        return {"roster": load_roster(store.settings).model_dump(exclude_none=True)["players"], **gallery.summary(store)}

    @app.put("/api/roster")
    def roster_put(body: RosterIn) -> dict:
        """Replace the roster (kept with the game data, not in the repo)."""
        from soccer_clips.roster import Roster, save_roster

        nums = [p.number for p in body.players]
        if len(set(nums)) != len(nums):
            raise HTTPException(422, "each number once")
        roster = Roster.model_validate({"players": [{"number": p.number, "name": (p.name or "").strip() or None}
                                                    for p in sorted(body.players, key=lambda p: p.number)]})
        save_roster(store.settings, roster)
        return {"roster": roster.model_dump(exclude_none=True)["players"]}

    @app.get("/api/retrain")
    def retrain_get(kind: Literal["reid", "ball"] = "reid") -> dict:
        """A model's state for its panel: in use since when, how it scores, what retraining costs, the last runs and
        the current or last job (reid: the recognition model; ball: the ball detector, ball_retrain.overview)."""
        from soccer_clips import retrain

        if kind == "ball":
            from soccer_clips import ball_retrain

            return ball_retrain.overview(store)
        return {"model": retrain.model_info(store), "estimate": retrain.estimate(store), "status": retrain.status(store),
                "history": retrain.history(store), "runpod": bool(os.environ.get("RUNPOD_API_KEY"))}

    @app.post("/api/retrain")
    def retrain_start(body: RetrainIn | None = None) -> dict:
        """Start a job on RunPod (detached from the server): retrain a model, or re-track the games' balls."""
        from soccer_clips import retrain

        body = body or RetrainIn()
        try:
            return {"status": retrain.start(store, body.kind, body.action)}
        except KeyError:
            raise HTTPException(422, f"no {body.action} for {body.kind}")
        except RuntimeError as e:  # one running already, or RunPod not set up (ConfigError)
            raise HTTPException(409, str(e))

    @app.get("/api/retrain/log")
    def retrain_log(kind: Literal["reid", "ball"] = "reid", lines: int = 40) -> dict:
        """The end of the current or last job's log."""
        from soccer_clips import retrain

        return {"lines": retrain.log_tail(store, kind, min(max(lines, 1), 400))}

    @app.get("/api/roster/pictures")
    def roster_pictures(view: Literal["player", "probable", "unassigned", "unsure"], player: int | None = None,
                        game: str | None = None, sort: str = "", offset: int = 0, limit: int = 60) -> dict:
        """A page of tracklet stretches for the roster page (gallery.pictures)."""
        from soccer_clips.track import gallery

        return gallery.pictures(store, view, player, game, sort, max(offset, 0), min(max(limit, 1), 500))

    @app.post("/api/roster/assign")
    def roster_assign(body: AssignIn) -> dict:
        """Give stretches to a player, unlabel them, or mark them not ours (gallery.assign)."""
        from soccer_clips.track import gallery

        try:
            n = gallery.assign(store, [x.model_dump() for x in body.stretches], body.player, body.batch)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(422, str(e))
        return {"changed": n}

    @app.post("/api/roster/split")
    def roster_split(body: SplitIn) -> dict:
        """A different girl from each of these frames on (gallery.split); undone like an assignment."""
        from soccer_clips.track import gallery

        try:
            return {"changed": gallery.split(store, body.game, body.track, body.frames, body.batch)}
        except (ValueError, KeyError) as e:
            raise HTTPException(422, str(e))

    @app.get("/api/roster/crops")
    def roster_crops(game: str, track: int, start: int, end: int, n: int = 16) -> dict:
        """Crop frames across a stretch, for choosing where to split it."""
        from soccer_clips.track import gallery

        return {"frames": gallery.crop_frames(store, game, track, start, end, min(max(n, 2), 40))}

    @app.delete("/api/roster/assign/{batch}")
    def roster_assign_undo(batch: str) -> dict:
        from soccer_clips.track import gallery

        return {"undone": gallery.undo(store, batch)}

    @app.get("/watch")
    def watch_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "watch.html")

    @app.get("/api/watch")
    def watch_runs(player: int, game: str | None = None, guesses: bool = True, min_score: float | None = None) -> dict:
        """The player view (watch.playlist): the runs of the game video where #player is on screen, in one game
        or (no game) all of them; guesses: include the stretches the appearance model claims, at min_score."""
        from soccer_clips.track import watch

        games = [load(game)] if game else sorted(store.list_games(), key=lambda g: (g.played_on is None, g.played_on, g.id))
        ms = None if min_score is None else min(max(min_score, 0.5), 1.0)
        return watch.playlist(store, games, player, guesses, ms)

    @app.get("/label-ball")
    def label_ball_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "label-ball.html")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:  # browsers ask for it even with the <link> on every page
        return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

    @app.get("/reel")
    def reel_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "reel.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # Pages and their scripts change together, so browsers must check for a
    # newer copy every time (a cheap 304 if unchanged). Without this, a
    # browser can reuse a cached common.js with a newer page, which breaks
    # the page (the match clock and kickoffs stopped updating that way).
    @app.middleware("http")
    async def revalidate_pages(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path in ("/", "/reel", "/label-ball", "/review-balls") or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    return app
