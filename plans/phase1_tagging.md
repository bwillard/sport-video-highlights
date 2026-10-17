# Phase 1 — Clip Tagging

**Status:** In progress — tagger built and browser-tested on synthetic video (2026-10-13). Next: tag a real game and note friction.
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
  - `GET /api/tags`
  - `GET/POST /api/games/{id}/clips`, `PATCH/DELETE /api/games/{id}/clips/{clip_id}` (nested under the game so no cross-game lookup is needed)
  - `POST /api/games/{id}/export` with `{clip_ids, fast}` (single or batch) → background jobs; `GET /api/jobs`, `GET /api/jobs/{id}`
  - `GET /clips/{game_id}/{clip_id}.mp4` downloads an exported clip
- Static page at `/` (vanilla JS, no build step).

## UI
Layout: large video on the left; clip list on the right; timeline strip underneath showing clip ranges (and the thumbnail sprite if built).

### Keyboard map (as built)
| Key | Action |
|---|---|
| Space | play / pause |
| M | sound on / off (starts muted) |
| [ / ] | speed −/+ 0.5× (0.5×–8×) |
| ← / → | −5s / +5s (Shift: ±60s) |
| , / . | step one frame back / forward (pauses) |
| I / O | mark in / out for a new clip |
| Shift+I / Shift+O | move the selected clip's start / end to now |
| Enter | save clip (if no in point: in = now − 8s, out = now + 2s); the new clip is selected |
| Esc | clear marks and deselect (in the note box: leave it, which saves) |
| 1–9 | toggle tag on selected clip (hotkeys from `tags.json`) |
| + / - | rate the clip 😊 / 🙁 (same key again clears it) |
| F | toggle the default player (`SOCCER_DEFAULT_PLAYER` in `.env`) |
| P | type another player's number, Enter to add it |
| T | toggle "team play" |
| N | edit note |
| ↑ / ↓ | previous / next clip (in the filtered list) and seek to it |
| R | replay selected clip, stopping at its end |
| E | export selected clip |
| Delete / Backspace | delete selected clip (asks first; also deletes its export) |
| H / Shift+H | set 1st / 2nd half kickoff to now (plain H 30+ minutes after the 1st half kickoff asks: OK sets the 2nd half, Cancel moves the 1st) |

`[`/`]` became speed controls during Phase 0 testing, so previous/next clip moved to ↑/↓. J/K/L were dropped (2026-10-14) as unneeded.

"Quick clip" with Enter is the main workflow: watch at 2×, hit Enter right after something good happens, adjust the in/out points later.

### Tags (default set, editable)
goal, shot, assist, pass, dribble, tackle/defense, save.

### Who (players and team)
Each clip has a list of **player numbers** it's noteworthy for, and a separate **team play** flag; a clip can have either, both or neither. This replaced the original "her" flag (2026-10-17) so other players and whole-team moments can be tagged too. The default player (`SOCCER_DEFAULT_PLAYER` in `.env`) is on `F`; other numbers are added with `P` or the `#` box, and every number already used in the game gets a button. The "who" filter (everyone / team plays / #N) narrows the clip list, so "Export shown" and "Add shown to reel" give a team reel or a reel for one player. Old `involves_her` flags are converted to the default player when a game is loaded.

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
    rating: "happy" | "sad" | None  # was 1–5 stars until 2026-10-17: 4–5 → happy, 1–2 → sad, 3 → none
    players: list[int] = []      # jersey numbers; sorted, no repeats (was involves_her: bool)
    team: bool = False           # a noteworthy team play
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
- [x] FastAPI app skeleton + `serve` command
- [x] Range-capable video serving; confirm seeking works
- [x] `Clip` model, `tags.json`, clip API on top of `store.py`
- [x] Player page: playback controls + keyboard map
- [x] In/out marking, quick-clip, clip list, edit/delete
- [x] Tags, rating, "her" flag, notes; filter the clip list (her flag → player numbers + team flag, 2026-10-17)
- [x] Timeline strip with clip ranges
- [x] Kickoff markers + match clock
- [x] Export job (single + batch)
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
- **2026-10-13** — Export snaps times to whole frames (seeks to half a frame before the first frame), so an accurate export has exactly `frame_at(end) − frame_at(start)` frames; tested (1.0–2.0s → 30 frames).
- **2026-10-13** — Editing a clip's start/end clears its `export` field (the file is stale and is overwritten on re-export). Deleting a clip deletes its exported file.
- **2026-10-13** — Match clock is minutes only ("23'", "2H 7'"); the clip list shows exact video time on hover, the clip editor shows both.
- **2026-10-14** — Exported clips now have a "⬇ mp4" download link in the clip list (before, the only link was in the clip editor, which only shows when a clip is selected). Clicking it doesn't select the row.
- **2026-10-14** — "Export" makes one file per clip; it doesn't join them. Joining clips into a reel, watching it in the app and jumping back to the full game are Phase 2 (see the "Watching reels in the app" section in [phase2_reel.md](phase2_reel.md)).
- **2026-10-15** — Game info box at the top of the tagger's right-hand panel: the game's date (prefilled from the filename) and a free-text description (e.g. "Home vs SAA, 3-1 win"). Saved on change/Enter via `PATCH /api/games/{id}` (`played_on`, `description`). The unused `Game.title` field was renamed `description` (it was empty in every game file). Used on reel opening cards, and as the suggested title for a new reel.
- **2026-10-15** — Sound starts muted on the tagger and reel pages; `M` or the "sound off/on" button in the header toggles it.
- **2026-10-15** — Bug report: the game description "saved" but then emptied. Cause: the server process predated the change (it still expected `title`), and the pages are served fresh from disk, so a new page talked to old server code that ignored the unknown `description` field. Fixes: request bodies now reject unknown fields (422 instead of a silent no-op), and the tagger checks the saved value came back before saying "saved", otherwise it says to restart the server. Lesson: restart `soccer-clips serve` after pulling or committing code changes.
- Not built yet: thumbnail sprite, tag-rename CLI, LosslessCut import.

_(add friction points and timing from tagging a real game)_
