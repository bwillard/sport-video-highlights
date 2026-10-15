# Soccer Clips — Project Plan

> Working doc for tracking this project across sessions with Claude.
> - Keep **Status** current — it's the first thing to read when resuming work.
> - Check off plan items as they're finished; don't delete finished items, they're useful history.
> - Log meaningful decisions/findings in **Decisions & Findings Log** (dated).
> - Each phase has its own detailed doc in [plans/](plans/) — refine those as we build, and keep this doc as the summary.

## Status (last updated 2026-10-15)
**Current phase:** Phase 3 in progress: 3.1 (freeze-frame callout) built. Phase 2 done. Reels: add clips from the tagger, render preview/full with hard cuts, watch on the reel page with chapters, jump to the full game (G) and back, download. 23 unit tests + headless-Chrome runs pass.
**Next action:** Try callouts on a real reel and decide on the style. Then 3.2 (tracked marker), which starts with benchmarking YOLO11 on a real clip. Still open from Phase 1: tag a full real game and note friction. Still open from Phase 1: tag a full real game and note friction. Always call the env's pip/python by full path: plain `python` resolves to forbeck_env even under `conda run`.
**Blocked on:** nothing, but forbeck_env needs cleanup: `soccer_clips` + fastapi/pydantic/uvicorn/httpx2 stack were accidentally pip-installed there and idna was upgraded 3.7 → 3.20 (see 2026-10-13 log).

## Goals
1. **Tag clips** from a full-game Veo MP4 quickly — much faster than Veo's website.
2. **Highlight my daughter** in a clip (callout circle at minimum, tracked marker ideally).
3. **Build a highlight reel** from selected clips (landscape; a phone-friendly vertical version is a future idea).
4. *(Stretch)* **Auto-find her** in the footage, using appearance (build, hair, cleats, socks, movement) rather than relying on her jersey number, which is often hidden.

## Constraints
- **CPU only** — no GPU. Any ML step (tracking, segmentation, re-ID) must be benchmarked on CPU and designed to run on short clips, in the background, or at reduced frame rate.
- **Follow-cam MP4 only** — no Veo panorama. The virtual camera pans and zooms, so trackers must cope with camera motion, players are small, and she may leave the frame.
- **Everything runs locally.** No uploading video of a minor to third-party services.
- Reuse existing tools wherever possible; write glue code, not frameworks.

## Architecture (high level)
- **Python package** (`soccer_clips`) in a dedicated **conda env** (`soccer-clips`, conda-forge channel), matching other projects. Defined in `environment.yml`; the package itself is installed into it with `pip install -e .`.
- **FastAPI** backend + a single plain HTML/JS page using the browser's `<video>` element. No frontend build step.
- **JSON files** for all state (no database): one `game.json` per game (game info + its clips), one `reel.json` per reel, per-clip track files, and JSONL for append-heavy data. **Pydantic** models are the schema. All reads/writes go through one `store.py` module (atomic writes, per-file locks), so swapping in SQLite later would stay cheap.
- **ffmpeg** (≥ 6, installed from conda-forge into the project env; the 4.2 in miniconda's base env is too old) for all cutting, encoding and reel rendering.
- **Ultralytics YOLO11 + BoT-SORT** for player detection/tracking; **supervision** for drawing overlays; **SAM 2 (tiny)** as a fallback tracker for hard clips.
- **Re-ID embeddings** (OSNet via BoxMOT, or DINOv2) for the stretch auto-tagging.

### Repo layout (proposed)
```
~/soccer-clips/
  plan.md                  # this doc
  plans/                   # per-phase detailed docs
  .env                     # machine-specific paths, NOT committed (gitignored)
  .env.example             # committed template with placeholder values
  environment.yml          # conda env (python, ffmpeg, heavy deps)
  pyproject.toml           # package metadata; installed with pip install -e .
  src/soccer_clips/
    config.py              # loads .env (paths, host/port), ffmpeg path, presets
    models.py              # Pydantic models (Game, Clip, Reel, Track, ...)
    store.py               # JSON load/save: atomic writes, per-file locks
    ingest.py              # Phase 0
    export.py              # Phase 1 clip export
    server/                # FastAPI app + static web page
    reel/                  # Phase 2 rendering
    track/                 # Phase 3 tracking + overlays
    autotag/               # Phase 4
  tests/
```

### Data layout
Data lives on the large data drive (20 TB, 14 TB free). **Every directory is configured in `.env`, which is gitignored. Real paths never go in committed files, including these docs.** Paths stored inside JSON files are relative to the matching configured directory, so the JSON never contains absolute paths.
```
<SOCCER_FULL_DIR>/<year>/<game_id>.mp4          # you drop the Veo download here; never modified
<SOCCER_PROXY_DIR>/<year>/<game_id>.mp4         # optional re-encode with dense keyframes (only if needed, see Phase 0)
<SOCCER_CLIPS_DIR>/<year>/<game_id>/<clip_id>.mp4   # exported clips
<SOCCER_GAMES_DIR>/
  tags.json                                      # tag list + hotkeys
  <year>/<game_id>/
    game.json                                    # game info + clips (tags, rating, her flag, callout)
    probe.json                                   # raw ffprobe output
    tracks/<clip_id>.json                        # Phase 3 tracks
<SOCCER_REELS_DIR>/<reel_id>/reel.json, *.mp4    # reel definitions + renders
<SOCCER_GALLERY_DIR>/index.jsonl, crops/         # Phase 4 labelled crops of her
```
- The `game_id` is the video's filename without `.mp4`, e.g. `gu11-vs-saa-2026-10-10`. The game date is parsed from the trailing `YYYY-MM-DD`.
- In `.env`, all six directories are siblings under one root folder on the data drive: `full/`, `proxy/`, `clips/`, `games/`, `reels/`, `player_gallery/`.

## Data & Code Reference
| Item | Location |
|---|---|
| Project repo | `~/soccer-clips` |
| Data directories | Set in `.env` (see `.env.example`) |
| Full-game videos | `<SOCCER_FULL_DIR>/<year>/*.mp4` — 5 games from 2026 so far |
| Phase docs | [plans/](plans/) |

## Plan
- [x] **Phase 0 — Setup & ingest** → [plans/phase0_ingest.md](plans/phase0_ingest.md)
  Environment, ffmpeg upgrade, import a game, make a scrub-friendly proxy.
- [ ] **Phase 1 — Clip tagging** (built; real-game test pending) → [plans/phase1_tagging.md](plans/phase1_tagging.md)
  Keyboard-driven web player: mark in/out, tag, rate, export clips.
- [x] **Phase 2 — Highlight reel** → [plans/phase2_reel.md](plans/phase2_reel.md)
  Order clips, render with hard cuts and a title card, landscape. Watch reels in the app, jump back to the full game, download. (Fades, music, vertical, speed/slow-mo: future ideas.)
- [ ] **Phase 3 — Highlight her** → [plans/phase3_highlight.md](plans/phase3_highlight.md)
  3.1 freeze-frame callout (one click) → 3.2 tracked marker (click + correct) → 3.3 follow-crop.
- [ ] **Phase 4 (stretch) — Auto-tag her** → [plans/phase4_autotag.md](plans/phase4_autotag.md)
  Learn her appearance from manual clicks; suggest where she appears; confirm/reject.

**Milestones**
- Phases 0–2 give a usable tool: tag a game and produce a shareable reel.
- Phase 3.1 is the highest payoff for the effort — do it before 3.2/3.3.
- Use the tool on a game or two before starting Phase 4; the manual clicks from Phase 3 are its training data.

---

## Decisions & Findings Log
- **2026-10-12** — Project lives at `~/soccer-clips`, separate from `forbeck`.
- **2026-10-12** — No GPU available. Primary tracker will be YOLO11 + BoT-SORT (cheap on CPU, and BoT-SORT's global motion compensation helps with the panning follow-cam). SAM 2 is the fallback for hard clips, not the default.
- **2026-10-12** — Only the follow-cam MP4 is available (no panorama). Callout-style highlighting (Phase 3.1) is prioritised because it holds up even when tracking struggles.
- **2026-10-12** — The installed ffmpeg is 4.2 (miniconda base), which lacks `xfade`. Install ffmpeg ≥ 6 from conda-forge into the project env.
- **2026-10-13** — JSON files instead of SQLite. Data is small (~100 clips/game, ~2k/season), single user, single process; JSON keeps each game self-contained in its folder, is hand-editable, and needs no migrations (Pydantic defaults handle new fields). Cross-game queries just load every `game.json`. Risks handled in `store.py`: concurrent writes from background jobs (per-file lock + write-temp-then-rename), and large append-heavy data uses JSONL.
- **2026-10-13** — Data lives in one root folder on the large data drive (full games in `full/<year>/`, clips in `clips/<year>/`). All paths go in a gitignored `.env`, never in committed files.
- **2026-10-13** — Originals stay where you drop them (`full/<year>/`) and are kept forever. Ingest registers them in place instead of copying or moving.
- **2026-10-13** — Veo download format (probed on all 5 files): one file per game, 1920×1080 H.264 High at 29.97 fps (30000/1001) constant frame rate, AAC stereo 48 kHz, ~4.7 Mbps, 60–84 min, 2.1–2.9 GB. There's a keyframe every 2.5s, and the `moov` atom is already at the front of the file, so the originals can stream to a browser as-is. Durations suggest warm-up time is included, so the kickoff markers matter.
- **2026-10-13** — Tagging happens from another machine on the home network, so the server must listen on the LAN (host/port set in `.env`). Single user only, so no auth or multi-user handling.
- **2026-10-13** — Use conda (not uv) for the environment, to stay consistent with other projects. With ffmpeg in the env, the active env's copy is used automatically; config still allows an override path.
- **2026-10-13** — Accidentally installed `soccer_clips` (editable) and its deps into forbeck_env, because `conda run -n soccer_clips_env pip` used forbeck_env's pip. Cleanup (needs user to run): `forbeck_env/bin/pip uninstall -y soccer_clips fastapi starlette uvicorn httpx2 httpcore2 pydantic pydantic_core annotated_types annotated_doc typing_inspection anyio h11 truststore opentelemetry_api typing_extensions idna` then `conda install -n forbeck_env -y --force-reinstall --no-deps idna=3.7`.
- **2026-10-13** — Starlette 1.8's test client wants `httpx2`, not `httpx`; it's in the `pip:` section of `environment.yml`.
- **2026-10-13** — `config.py` looks for ffmpeg/ffprobe in the running interpreter's env (`sys.prefix/bin`) before `CONDA_PREFIX`, so running the env's `soccer-clips` by full path works without activating the env (before, it found base's ffmpeg 4.2 and failed the version check).
- **2026-10-13** — LAN scrub test passed on the originals: seeking and stepping feel fine, so the proxy step is skipped. Added `[`/`]` to halve/double playback speed (¼×–8×), since `L` could only speed up.
- **2026-10-13** — Phase 1 keymap: `[`/`]` stay as speed controls, so prev/next clip is ↑/↓ and the "her" flag is `F`. Clip API is nested under the game (`/api/games/{id}/clips/{clip_id}`).
- **2026-10-14** — Exports are single-clip files; a joined reel is Phase 2. Planned for Phase 2: each render writes a segment map (`render.json`) so the in-app reel player can jump to the matching moment in the full game, and offers the reel for download. Also: `[`/`]` step speed by 0.5× (0.5×–8×), J/K/L removed, and the clip list links to each exported file.
- **2026-10-14** — Phase 2 first slice: hard-cut reels with a segment map for jump-back. Exports and renders share one background worker (`jobs.py`) so two CPU-heavy encodes never run at once. Rendered reels live in `<SOCCER_REELS_DIR>/<reel_id>/`.
- **2026-10-14** — Scope cut: fades, music, the vertical version, and speed/slow-mo moved from Phase 2 to "Future ideas" in [plans/phase2_reel.md](plans/phase2_reel.md). Phase 3.3 (follow-crop) depends on the vertical preset, so it waits too.
- **2026-10-14** — Title cards: an opening card with the reel title, and a card before each clip with the time in the game and the clip's note. Dark background, white text, length set per reel (default 3s, 0 = off). Pillow added to the env.
- **2026-10-15** — Phase 2 closed: per-clip trim, lower thirds and OTIO export moved to its Future ideas. Phase 3.1 built: callout set with `C` + click in the tagger, rendered as a 1.5s spotlight freeze (configurable per reel). YOLO snap deferred (needs the ML stack).
- **2026-10-15** — Game date + description editable in the tagger; reel opening cards show them (one game) or the game count + date range (several games). `Game.title` renamed `description`.
- **2026-10-15** — Video starts muted in the app (`M` toggles). Measured: removing audio from reels would save ~2% file size and ~1% render time, so reels keep their audio.
- **2026-10-15** — Title cards are timed by their text: 1s + 15 characters/second, with a 2s minimum per reel (`Reel.card_s`, now a minimum) and a 15s cap.
- **2026-10-15** — API request bodies reject unknown fields, so a page newer than the running server fails loudly (found when a stale server silently dropped game descriptions).
- **2026-10-15** — Reel page: game date + description under the title, clip notes in both lists, ▶ preview of any clip straight from its game.
- **2026-10-15** — Callout freeze default shortened to 0.5s (was 1.5s). Reels created earlier keep their stored value.
