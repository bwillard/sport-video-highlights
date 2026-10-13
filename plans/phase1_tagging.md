# Phase 1 — Clip Tagging

**Status:** Not started. Depends on Phase 0.
**Goal:** Go through a full game quickly with the keyboard, mark clips with in/out points, tag and rate them, and export them.

## Scope
- FastAPI server + one HTML/JS page.
- Clip CRUD, tags, ratings, notes.
- Match-clock display (kickoff markers).
- Export individual clips from the original video.

**Stopgap:** until this exists, LosslessCut can do in/out marking and export with no code. An importer for its segment CSV is optional (see tasks).

## Server
- `soccer-clips serve` → FastAPI on `SOCCER_HOST:SOCCER_PORT` from `.env` (default `0.0.0.0:8000`) so the tagging machine on the LAN can reach it. There's no auth (single user, home network). The port may need opening in `ufw` if the firewall is on.
- Serve the video (keyframe copy if it exists, otherwise the original) with **HTTP Range support** (required for seeking). Verify Starlette's `FileResponse` handles Range in the installed version; otherwise write a small range-serving endpoint.
- JSON API:
  - `GET /api/games`, `GET /api/games/{id}`
  - `PATCH /api/games/{id}` (kickoff times)
  - `GET/POST /api/games/{id}/clips`, `PATCH/DELETE /api/clips/{id}`
  - `POST /api/clips/{id}/export` → background job; `GET /api/jobs/{id}`
- Static page at `/` (vanilla JS, no build step).

## UI
Layout: large video on the left; clip list on the right; timeline strip underneath showing clip ranges (and the thumbnail sprite if built).

### Keyboard map (draft)
| Key | Action |
|---|---|
| Space | play / pause |
| J / K / L | back ×2 / pause / forward ×2 (press repeatedly for faster) |
| ← / → | −5s / +5s |
| , / . | step one frame back / forward (while paused) |
| I / O | set in / out point |
| Enter | save clip (if no in point: in = now − 8s, out = now + 2s) |
| 1–9 | toggle tag 1–9 on current clip |
| Shift+1–5 | rating |
| N | note |
| [ / ] | previous / next clip |
| Delete | delete selected clip |
| H | set kickoff for current half |

"Quick clip" with Enter is the main workflow: watch at 2×, hit Enter right after something good happens, adjust the in/out points later.

### Tags (default set, editable)
goal, shot, assist, pass, dribble, tackle/defense, save, celebration, funny. Plus a **"her"** flag (is she involved?), which is the main filter for highlight reels.

### Match clock
Store `kickoff_1h_s` / `kickoff_2h_s`; display "23'" / "2H 7'" next to timestamps. This makes it easy to cross-reference what you remember ("her shot right before halftime").

## Model additions
Clips live inside their game's `game.json` (`Game.clips`):
```python
class Clip(BaseModel):
    id: str                       # short random id (e.g. 8 hex chars), unique across games
    start_s: float
    end_s: float
    tags: list[str] = []          # tag names, from tags.json
    rating: int | None = None     # 1..5
    involves_her: bool = False
    note: str | None = None
    callout: Callout | None = None   # Phase 3.1
    export: str | None = None     # relative to SOCCER_CLIPS_DIR, e.g. "2026/<game_id>/<id>.mp4"
    created_at: datetime
```
`tags.json` at the data root holds the tag list and hotkeys:
```json
[{"name": "goal", "hotkey": "1"}, {"name": "shot", "hotkey": "2"}]
```
- Times are stored in **seconds**, and frame numbers are derived as `round(t * fps)` when needed.
- Clip ids are random rather than sequential so reels can reference `(game_id, clip_id)` and clips can be deleted without renumbering.
- Renaming a tag means rewriting it across all `game.json` files. That's rare, so a small CLI command is enough.
- Every UI edit is a load-modify-save of the whole `game.json` through `store.py`. At ~100 clips per game that's a few KB, so it's instant.

## Export
- Frame-accurate re-encode from the **original**:
  ```
  ffmpeg -ss <start> -i <original> -t <dur> -c:v libx264 -preset medium -crf 18 \
    -c:a aac -b:a 160k -movflags +faststart <SOCCER_CLIPS_DIR>/<year>/<game_id>/<clip_id>.mp4
  ```
  (With modern ffmpeg, input-side `-ss` is both fast and accurate when re-encoding.)
- Optional "fast export" that stream-copies (`-c copy`) for quick sharing, accepting keyframe-aligned cuts.
- Exports run in a simple in-process job queue (one worker; CPU-bound).
- Also: "export all clips matching filter" → batch.

## Tasks
- [ ] FastAPI app skeleton + `serve` command
- [ ] Range-capable video serving; confirm seeking works
- [ ] `Clip` model, `tags.json`, clip API on top of `store.py`
- [ ] Player page: playback controls + keyboard map
- [ ] In/out marking, quick-clip, clip list, edit/delete
- [ ] Tags, rating, "her" flag, notes; filter the clip list
- [ ] Timeline strip with clip ranges
- [ ] Kickoff markers + match clock
- [ ] Export job (single + batch)
- [ ] *(Optional)* LosslessCut CSV import
- [ ] Tag one full real game; note friction points below

## Acceptance criteria
- Tag a full 60–90 min game in noticeably less time than on Veo's website.
- Exported clips start and end on the intended frames, with audio in sync.

## Open questions
- Is the default lead-in/lead-out for quick-clip right? (8s/2s is a guess.)
- ~~Multiple users?~~ Resolved: single user only.
- ~~Where do you tag from?~~ Resolved: another machine on the LAN.
- Should clips be able to overlap? Assume yes.

## Findings
_(fill in as we go — friction points, timing)_
