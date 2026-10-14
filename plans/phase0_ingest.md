# Phase 0 — Setup & Ingest

**Status:** Done — all 5 games registered; originals scrub fine over the LAN, so no proxies needed.
**Goal:** Register a Veo game that's already sitting in `<SOCCER_FULL_DIR>/<year>/`, record its metadata, and make sure it scrubs well in a browser (re-encoding a copy only if needed).

## Scope
- Project environment and tooling.
- `soccer-clips scan` command: find new videos and register them in place. Optional `soccer-clips proxy <game_id>` to make a keyframe-dense copy.
- Data directory layout, Pydantic models, and the JSON `store.py`.

Out of scope: any UI (Phase 1).

## Environment
- **Python:** conda env `soccer-clips` defined in `environment.yml` (conda-forge channel), consistent with other projects. Python 3.12 is the safer pick for torch/ultralytics/SAM 2 in later phases. Install our package into the env with `pip install -e .` (metadata in `pyproject.toml`). Rule of thumb: get binary/heavy deps (ffmpeg, opencv, pytorch CPU) from conda-forge, and anything missing there via the `pip:` section of `environment.yml`.
- **ffmpeg:** `ffmpeg>=6` from conda-forge in `environment.yml`. When the env is active, its ffmpeg shadows the old 4.2 in miniconda's base env. `config.py` resolves ffmpeg/ffprobe from the active env (`$CONDA_PREFIX/bin`), allows an env-var override, and checks the version at startup so a wrong env fails loudly.
- **Config:** `config.py` built on `pydantic-settings`, reading `.env` from the repo root. Fails with a clear message if a required directory is missing.
- **`.env` (gitignored)** holds all machine-specific paths. `.env.example` (committed) has the same keys with placeholder values:
  ```
  SOCCER_FULL_DIR=/path/to/soccer/full
  SOCCER_PROXY_DIR=/path/to/soccer/proxy
  SOCCER_CLIPS_DIR=/path/to/soccer/clips
  SOCCER_GAMES_DIR=/path/to/soccer/games
  SOCCER_REELS_DIR=/path/to/soccer/reels
  SOCCER_GALLERY_DIR=/path/to/soccer/player_gallery
  SOCCER_HOST=0.0.0.0        # listen on the LAN (tagging from another machine)
  SOCCER_PORT=8000
  # FFMPEG=/path/to/ffmpeg   # optional override
  ```
- **`.gitignore`** must include `.env` before the first commit.
- **Phase 0 dependencies:** `typer` (CLI), `pydantic`, `pydantic-settings`. Nothing heavy.

## Ingest flow
You copy the Veo download into `<SOCCER_FULL_DIR>/<year>/`. Then `soccer-clips scan` (also run automatically when the server starts) does, for each video without a `game.json`:

1. `game_id` = filename without `.mp4` (e.g. `gu11-vs-saa-2026-10-10`). Date parsed from the trailing `YYYY-MM-DD`. The opponent and title are optional and can be edited later in the UI. Filenames aren't fully consistent (`gu11black-vs-…` vs `gu11-black-vs-…`), so don't try to parse more than the date.
2. The original is **referenced in place**, never copied, moved or modified.
3. `ffprobe -show_streams -show_format -of json` → `<SOCCER_GAMES_DIR>/<year>/<game_id>/probe.json`.
4. Write `game.json` with key fields from the probe and an empty `clips` list.
5. *(Optional)* thumbnail sprite sheet (one frame every 10s) for a scrub-preview strip in Phase 1.

### Do we need a re-encoded copy?
Possibly not. The originals already have a keyframe every 2.5s and the `moov` atom at the front, so browsers can stream and seek them directly over the LAN (~4.7 Mbps is easy for a home network). **Test this first:** serve an original, then try seeking, J/K/L and frame-stepping backwards from the tagging machine.

If seeking or stepping backwards feels sluggish, `soccer-clips proxy <game_id>` makes a keyframe-dense copy in `<SOCCER_PROXY_DIR>/<year>/<game_id>.mp4`:
```
ffmpeg -i <original> -c:v libx264 -preset veryfast -crf 23 -g 15 -keyint_min 15 -sc_threshold 0 \
  -c:a copy -movflags +faststart <proxy>
```
- Keep full 1080p (the tagging machine is on the LAN, so bandwidth isn't a concern) unless decoding on the tagging machine struggles; then scale to 720p.
- **Invariant: the copy has exactly the same timeline as the original** (no trim, same start time), so timestamps in `game.json` are valid for both. The player uses the copy if it exists; **exports always cut from the original.**
- Runs as a background job with progress output (ffmpeg `-progress pipe:1`). Expect it to take a while on CPU for a 60–84 min game; record the time below.

## Storage: JSON + Pydantic (no database)
`models.py` (initial):
```python
class Game(BaseModel):
    schema_version: int = 1
    id: str                          # slug, also the folder name
    played_on: date
    opponent: str | None = None
    title: str | None = None
    year: int
    original: str                    # relative to SOCCER_FULL_DIR, e.g. "2026/gu11-vs-saa-2026-10-10.mp4"
    proxy: str | None = None         # relative to SOCCER_PROXY_DIR, set if a keyframe copy exists
    duration_s: float
    fps_num: int                     # exact frame rate as a fraction: 30000/1001 for Veo
    fps_den: int
    width: int
    height: int
    kickoff_1h_s: float | None = None   # set in Phase 1, for match-minute display
    kickoff_2h_s: float | None = None
    created_at: datetime
    clips: list[Clip] = []              # Phase 1
```
`store.py`:
- `load_game(id)`, `save_game(game)`, `list_games()` (scans `<SOCCER_GAMES_DIR>/*/*/game.json`).
- **Atomic writes:** write to `game.json.tmp`, `fsync`, then `os.replace` onto `game.json`.
- **Per-file lock** (an in-process `threading.Lock` per path) around load-modify-save, because background jobs (proxy encode, exports) update the same file the UI edits.
- Paths inside JSON are **relative to their configured directory** (`SOCCER_FULL_DIR`, `SOCCER_CLIPS_DIR`, …), so the data can move by editing `.env` alone.
- Frame numbers are `round(t_s * fps_num / fps_den)`. Store the exact fraction, because rounding 29.97 to 30 drifts by ~4s over 70 min.
- New fields always get defaults, so old files keep loading. `schema_version` is there in case a real migration is ever needed.

## Tasks
- [x] `environment.yml` (python 3.12, ffmpeg>=6, typer, pydantic) + `conda env create`; verify `xfade` is available (`ffmpeg -filters | grep xfade`)
- [x] `pyproject.toml`, package skeleton, `pip install -e .`
- [x] `.gitignore` (with `.env`), `.env.example`, local `.env`; `config.py` with `pydantic-settings` + ffmpeg version check
- [x] `models.py` (`Game`) + `store.py` (atomic save, locks, list games)
- [x] `scan` command: find new videos, probe → `probe.json`, write `game.json`
- [x] Serve an original over the LAN and test scrubbing from the tagging machine (decides whether proxies are needed)
- [x] ~~*(If needed)* `proxy` command~~ not needed with progress output
- [x] Scan all 5 existing 2026 games

## Acceptance criteria
- All 5 existing games registered; a game plays on the tagging machine over the LAN and seeking anywhere feels instant (original or keyframe copy).
- `probe.json` and `game.json` are correct, and `game.json` round-trips through the Pydantic model.

## Open questions
- ~~Keep originals forever?~~ Resolved: yes, keep them.
- ~~Data root~~: resolved, set in `.env`.
- ~~Copy vs move the original~~: resolved, referenced in place.
- ~~Veo format~~: resolved, see below.

## Benchmarks / findings
_(fill in as we go)_
| Measure | Value |
|---|---|
| Veo MP4 format | 1920×1080, H.264 High, yuv420p, 29.97 fps (30000/1001) constant, AAC LC stereo 48 kHz. Encoder `nvh264encoder`. |
| Bitrate | ~4.5 Mbps video, 128 kbps audio |
| Keyframe interval | 2.5s (75 frames) |
| `moov` atom | at the front (already "faststart") |
| Game length / size | 60–84 min, 2.1–2.9 GB (5 games, 2026) |
| Originals scrub OK in browser over LAN? | Yes (tested 2026-10-13) — no proxy needed |
| Proxy encode time (CPU), if needed | |
| Proxy file size, if needed | |
