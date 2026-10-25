# Soccer Clips — Project Plan

> Working doc for tracking this project across sessions with Claude.
> - Keep **Status** current — it's the first thing to read when resuming work.
> - Check off plan items as they're finished; don't delete finished items, they're useful history.
> - Log meaningful decisions/findings in **Decisions & Findings Log** (dated).
> - Each phase has its own detailed doc in [plans/](plans/) — refine those as we build, and keep this doc as the summary.

## Status (last updated 2026-10-24)
**Current phase:** Phase 3.2 (player tracking, option B) has whole-game data: every game has person boxes (YOLO11m at 1920, RunPod) and tracklets made here (`tracks/people_yolo11m.npz`, 21–25k tracklets a game, ~6 s each on average). At the 83 callouts on #38 a box is under her 99% of the time and her tracklet covers a median 75% of the clip. Phase 3.3 first slice built: follow a player in the tagger (`V`, `L`, `Shift+L`, `U`). Phase 4.2 built: kit colour per box keeps tracks on their own kit, cuts and joins tracklets, and names teams. Phase 4.1 (ball tracking) built and in use; Phases 0–2 done, 3.1 done. Every game has a whole-game ball track too, shown as a ring in the tagger (`B`), with fixes while watching (`K`) that become training labels; game ball found 66%, wrong 10%, missed 24% (`soccer-clips ball eval-games`). 98 unit tests pass.
**Next action:** Watch a game with `V` on: boxes now show teams (white ours, blue opponents, grey dashed refs/keepers/spectators) and #38's tracks hold longer and swap less. Keep labelling her (`L`, `Shift+L`): each label sharpens the yardstick. Next candidates: appearance re-ID to tell teammates apart (the remaining swaps and wrong joins), the reel-page overlay, clips where she's near the ball (4.4). Meanwhile: keep fixing ball mistakes; fine-tune the ball model again when enough fixes have piled up (RunPod, needs your go). Always call the env's pip/python by full path: plain `python` resolves to forbeck_env even under `conda run`.
**Decision pending:** none. ML runs on rented RunPod GPUs, each launch with your approval ([plans/hardware.md](plans/hardware.md)).
**Blocked on:** nothing.

## Goals
1. **Tag clips** from a full-game Veo MP4 quickly — much faster than Veo's website.
2. **Highlight a player** (my daughter by default, or any teammate): a callout circle at minimum, and a marker that follows them during playback and in reels.
3. **Build a highlight reel** from selected clips (landscape; a phone-friendly vertical version is a future idea).
4. **Track the ball and every player on our team** through each game, and name the players, to power (a) highlighting any chosen player and (b) pulling together clips where that player is close to the ball. Identify players by appearance (build, hair, cleats, socks) as well as jersey numbers, which are often hidden.

## Constraints
- **CPU for ML:** a 4-core Ryzen 3 3200G. Its Vega iGPU encodes/decodes video (VA-API) but can't run the models (no CUDA, no usable ROCm). Heavy ML (fine-tuning, detecting the ball in whole games) runs on rented RunPod GPUs; lighter steps (the ball filter, fixed-object finding, labelling) run here. See [plans/hardware.md](plans/hardware.md).
- **Follow-cam MP4 only** — no Veo panorama. The virtual camera pans and zooms, so trackers must cope with camera motion, players are small, and she may leave the frame.
- **Everything runs locally by default.** Video of a minor goes to a third-party service only with your explicit OK for that upload. So far approved: the bench clips, the ball training data, and the game videos and ball model on the RunPod network volume (US-KS-2, until `soccer-clips gpu volume delete --yes`).
- Reuse existing tools wherever possible; write glue code, not frameworks.

## Architecture (high level)
- **Python package** (`soccer_clips`) in a dedicated **conda env** (`soccer-clips`, conda-forge channel), matching other projects. Defined in `environment.yml`; the package itself is installed into it with `pip install -e .`.
- **FastAPI** backend + a single plain HTML/JS page using the browser's `<video>` element. No frontend build step.
- **JSON files** for all state (no database): one `game.json` per game (game info + its clips), one `reel.json` per reel, per-game track data (boxes in a compact `.npz` array file, labels in JSON), and JSONL for append-heavy data. **Pydantic** models are the schema. All reads/writes go through one `store.py` module (atomic writes, per-file locks), so swapping in SQLite later would stay cheap.
- **ffmpeg** (≥ 6, installed from conda-forge into the project env; the 4.2 in miniconda's base env is too old) for all cutting, encoding and reel rendering.
- **Players (Phase 3.2):** stock YOLO at full 1920 resolution on a rented GPU saves raw person boxes once per model (`tracks/people_raw_<model>.npz`); BoT-SORT runs here on the saved boxes with our own camera steps (from `ball_raw.npz`) in place of its image-based motion step, so tracking can be re-tuned for free (`tracks/people_<model>.npz`). The ball uses a **YOLO26x fine-tuned on our labels**, run on the centre crop on a rented GPU, then a **temporal filter** here (camera motion cancelled, tracklets, best chain by dynamic programming, still balls ignored unless kicked) and **fixed-object blocking** (frames registered to keyframes, so a spot that never moves all game can be found despite the panning).
- **Re-ID embeddings** (OSNet via BoxMOT, or DINOv2) + jersey-number OCR to name players.
- **Live highlights** are drawn in the browser on a `<canvas>` over the `<video>`, synced per frame with `requestVideoFrameCallback`; reels burn the same marker in.

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
    game.json                                    # game info + clips (tags, rating, players, team, callout)
    probe.json                                   # raw ffprobe output
    tracks/run.json, boxes.npz, tracklets.json, ball.npz   # Phase 3/4 whole-game tracking
<SOCCER_REELS_DIR>/<reel_id>/reel.json, *.mp4    # reel definitions + renders
<SOCCER_GAMES_DIR>/roster.json                   # Phase 4 team roster (number → name)
<SOCCER_GALLERY_DIR>/index.jsonl, crops/         # Phase 4 labelled crops of each player
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
| Hardware options for ML | [plans/hardware.md](plans/hardware.md) |

## Plan
- [x] **Phase 0 — Setup & ingest** → [plans/phase0_ingest.md](plans/phase0_ingest.md)
  Environment, ffmpeg upgrade, import a game, make a scrub-friendly proxy.
- [ ] **Phase 1 — Clip tagging** (built; real-game test pending) → [plans/phase1_tagging.md](plans/phase1_tagging.md)
  Keyboard-driven web player: mark in/out, tag, rate, export clips.
- [x] **Phase 2 — Highlight reel** → [plans/phase2_reel.md](plans/phase2_reel.md)
  Order clips, render with hard cuts and a title card, landscape. Watch reels in the app, jump back to the full game, download. (Fades, music, vertical, speed/slow-mo: future ideas.)
- [ ] **Phase 3 — Track & highlight players** → [plans/phase3_highlight.md](plans/phase3_highlight.md)
  3.1 freeze-frame callout (done) → 3.2 whole-game tracking job (done) → 3.3 live highlight (first slice built) during playback (click a player, correct by clicking) → 3.4 marker burned into reels → 3.5 follow-crop (later).
- [ ] **Phase 4 — Identify players & track the ball** → [plans/phase4_identify.md](plans/phase4_identify.md)
  4.1 ball tracking (built: whole-game tracks for every game, ring + fixes in the tagger) → 4.2 team split by kit colour (built; also used in tracking) → 4.3 name our players (re-ID + jersey OCR + confirm/reject) → 4.4 suggested clips where a chosen player is near the ball.

**Milestones**
- Phases 0–2 give a usable tool: tag a game and produce a shareable reel.
- Phase 3.1 is the highest payoff for the effort — do it before 3.2/3.3.
- The Phase 3.2 benchmark measures the ball too, since ball detection is the biggest unknown for 4.4.
- 3.3 works without Phase 4 (click a player to follow them); Phase 4.3 then makes it pick-by-name. Correction clicks from 3.3 are 4.3's training data, so use 3.3 on a game or two first.
- 4.4 (near-the-ball clips) needs 4.1 and 4.3.

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
- **2026-10-15** — After watching a real reel: freeze default 0.75s; title cards a little shorter (0.75s + 17 characters/second, 2s minimum). Renders made with the old timing are flagged "title card timing changed".
- **2026-10-15** — Clip descriptions can be a caption overlaid on the start of the clip (new default, per reel) instead of a title card. The freeze default follows the style: 1.5s with overlays, 0.75s with cards.
- **2026-10-17** — Ratings are 😊/🙁 instead of 1–5 (4–5 → 😊, 1–2 → 🙁, 3 → none). The "her" flag became player numbers plus a team-play flag, so other players and team moments can be tagged and filtered; old flags converted to `SOCCER_DEFAULT_PLAYER` (38). Tags "celebration" and "funny" dropped.
- **2026-10-17** — Pages and static files are served with `Cache-Control: no-cache`: a browser reused an old cached `common.js` with the new tagger page, which stopped the timer, match clock and kickoff display.
- **2026-10-17** — Reels can be rendered without the clip notes, or without notes and callouts, alongside the full version (`Render.variant`), to share a clean copy while keeping the annotated one.
- **2026-10-18** — Full reel renders use the GPU's H.264 encoder (VA-API) when it works, falling back to x264; quality and encoder are `.env` settings. Every piece is encoded once at final quality and the join copies the video, and clips are cut into captioned/plain/freeze pieces so only the captioned bits use the CPU. A real 3.5-minute reel went from 533 s to 98 s. Previews stay on x264 because this GPU's driver hangs at 960×540.
- **2026-10-18** — Re-scoped Phases 3–4 from "highlight her in a clip" to "track the ball and every player on our team through the whole game", to power highlighting any chosen player and finding clips where a player is near the ball. Tracking runs once per game (not per clip) and is stored per game: boxes in `tracks/boxes.npz` (too many for JSON: up to ~2M per game), labels in `tracks/tracklets.json`. Live highlights are a canvas over the video synced with `requestVideoFrameCallback`, interpolating between samples, so changing the highlighted player needs no re-render; reels burn in the same marker. "Near the ball" is image-space distance divided by the player's box height (no pitch calibration). SAM 2 fallback dropped (too slow on this CPU for whole games). Phase 4 doc renamed to `phase4_identify.md`. Hardware: Ryzen 3 3200G, 4 cores; the iGPU can't run the models.
- **2026-10-18** — Hardware for ML is undecided: (1) keep the 3200G (players every 3rd frame, ~1–1.5 h/game; weak ball tracking; ball-model training takes days), (2) add a GPU such as an RTX 5060 Ti 16GB (~20–60 min/game every frame at full resolution, fast training, NVENC; ~$590–710 mid-2026), or (3) rent (under $1/game, but breaks the local-only rule for video of a minor). Leaning towards (2) if local-only stays. Details in [plans/hardware.md](plans/hardware.md).
- **2026-10-18** — Benchmark clip set (7 clips, 87s, all 5 games) in `bench/` with `soccer-clips bench add/cut/run/draw`. First CPU benchmark of YOLO11 + BoT-SORT: n640 8.8 fps, n1280 4.8, s1280 2.1 (ONNX ~15% faster). At 1280 it finds her in 5/5 callout clips (58–73 px tall), but track IDs break often (her track covers ~10–50% of a clip), and sideline spectators are detected too. The stock ball detector is mostly right when it fires (~85% at conf ≥ 0.25) and resolution matters most: the s model on a native-resolution centre crop finds a ball in up to ~half the frames, but costs ~18 h per game on this CPU. Ultralytics' analytics sync is turned off; model weights live in `models/` next to the data. Details in the Phase 3/4 findings and [plans/hardware.md](plans/hardware.md).
- **2026-10-18** — Ran the benchmark on a rented NVIDIA L4 (RunPod Secure Cloud, ~45 min, under $1), with your OK to upload the 7 bench clips; they were deleted from the pod afterwards. Results are identical to the CPU for the same variants; 19–64 fps depending on variant vs 2–9 fps on the 3200G. Two bigger variants stand out: YOLO11m at full 1920 resolution keeps her track for most or all of a clip (median 100% vs ~10–50% for the CPU-sized variants), and YOLO11m on the centre crop finds the ball most often. Full-frame detection also finds spare balls by the camera, so the ball filter must drop stationary balls. Details: [plans/hardware.md](plans/hardware.md), Phase 3/4 findings. Results in `bench/results-l4/`; `bench draw --results` draws them.
- **2026-10-18** — Ball labelling page (`/label-ball`): click / Shift+click (hidden) / drag to size, `N` no ball, `S` can't tell, magnifier, `Q`/`E` peek ±3 frames. Two sets kept apart: `bench` (149 frames over the bench clips, only for scoring) and `train` (random frames from the games, at least 10s from any bench clip). Labels are keyed by game frame, append-only in `labels/ball.jsonl`. `soccer-clips ball score` turns them into recall/precision per benchmark variant. Better ball models (fine-tuned YOLO, then multi-frame heatmap models like WASB/TrackNet) all need these labels.
- **2026-10-19** — Bench ball labels done (149 frames) and scored. Best stock detector finds ~57% of visible balls with ~79% of its detections real (YOLO11m on the centre crop, conf ≥ 0.1); the s model on the crop gets 50% / 76%. 640-pixel inputs are useless for the ball (0–6%). Full-frame inputs lose precision to spare balls near the camera. Hidden balls are almost never found. Too many gaps to rely on detection alone: next are the temporal filter and fine-tuning on our own labels. Table in [plans/phase4_identify.md](plans/phase4_identify.md).
- **2026-10-19** — Fine-tuning chosen over the temporal filter as the next ball step. Train set of 500 frames created (~100 per game, away from the bench clips); labelling is sped up by a guess mode (YOLO11m-centre's best guess, Enter to accept), never shown on bench frames so the test set stays independent. The 149 bench labels stay test-only. Plan for the GPU session: ~1–1.5 h on a 4090, stock YOLO11l/x and YOLO26m/l/x on the bench first, then fine-tune the best base and YOLO11m.
- **2026-10-19** — Cloud GPU runs are automated: `soccer-clips gpu run ball-train` starts a RunPod pod through the API, uploads only what the job needs (~311 MB: training crops, bench frames, labels, game.json), runs, copies results back, then wipes and deletes the pod whatever happens, with a dead-man's switch on the pod as backup. Launches need your approval (Claude Code "ask" rule). See [plans/hardware.md](plans/hardware.md#cloud-jobs-2026-10-19).
- **2026-10-19** — First automated RunPod run (RTX 4090, 35 min, $0.52, pod deleted automatically). Stock YOLO26x already finds 75% of visible balls (vs 57% for YOLO11m); fine-tuned on our 500 labels it finds 81% (conf 0.1) / 75% with 92% of detections real (conf 0.4). The kept checkpoints were from epochs 2 and 6 because ultralytics selects on box-fit over a tiny validation split, so the next run will pick checkpoints by our own score. Models in `models/ball_yolo26x.pt`, `models/ball_yolo11m.pt`. Details in [plans/phase4_identify.md](plans/phase4_identify.md).
- **2026-10-19** — Spare balls and balls on other fields count as real balls: the detector should find them, and the temporal filter will pick out the game ball. Added other-ball labels (model candidates + Y/N review, or by hand), boxed them in training (they were being taught as background), and changed scoring to found / real (any real ball) / junk per frame. Also fixed checkpoint selection for the next fine-tune (see Phase 4 findings).
- **2026-10-21** — Fine-tune run 2 (YOLO26x, other balls boxed, checkpoint picked on our own score): 76% of visible game balls found at conf 0.4 with 92% of detections real and spare balls found again (18 vs 2 for run 1). ~$0.58, plus ~$0.07 for two attempts that ran out of GPU memory at batch 8 (now batch 4). Details in [plans/phase4_identify.md](plans/phase4_identify.md).
- **2026-10-21** — Game-ball temporal filter v1: camera motion cancelled with optical flow, detections linked into tracklets, best chain chosen by dynamic programming, gaps up to 1 s filled. Stock YOLO11m centre detections go from 52% to 72% of visible game balls found, wrong positions from 7% to 3%. Remaining misses are mostly frames with no detection on the ball. Review videos: `soccer-clips ball track <variant> --draw`.
- **2026-10-21** — The filter with the fine-tuned YOLO26x finds 82% of visible game balls on the bench (5% wrong, 12% missed). Lower detection thresholds don't help (the hard misses have no detection at all); balls parked 3 s+ are now never used, which halved wrong positions once the detector started finding spare balls.
- **2026-10-22** — Game-ball tracks per game (`tracks/ball_raw.npz` detections + camera motion, `tracks/ball.npz` path) and a ball ring in the tagger (`B`). Made by `soccer-clips gpu run ball-games` (tagged clips cut into segments for upload, or whole games) since the fine-tuned detector runs at 0.3 fps on this CPU. Bench-clip stretches are already filled from the bench detections.
- **2026-10-22** — Whole-game ball track for ebu (fine-tuned YOLO26x; 65 min, $1.28 on a US 5090; ball on 59% of frames). Pods now start in US data centres; long ranges are detected in parallel chunks (one CPU core was the limit). Ball fixes in the tagger (`K` / `Shift+K`) correct the path in ~2 s and become labelled frames, auto-assigned to training (most) or the benchmark (one moment in five, and anything near a bench clip).
- **2026-10-22** — Whole-game run for the other five games failed after its 68-min upload ($1.03): six detection workers ran a 24 GB 4090 out of memory and the first failure stopped the job. Fixed (workers sized to free GPU memory, failed chunks redone, games independent, partial results kept); detection stays in full precision by choice. To stop paying for uploads, a 50 GB network volume in US-KS-2 holds the game videos and ball model (~$3.50/month), filled from a CPU-only pod; GPU runs fall back to another US data centre with a full upload if Kansas has nothing free for 20 min.
- **2026-10-23** — Network volume filled over RunPod's S3 API (16.7 GB in 54 min, no pod; parts had to drop to 25 MB because RunPod's front end cuts requests at ~100 s). The other five games then ran from it on a Kansas A100 (227 min, ~$6.77, 216 KB uploaded): every game now has a whole-game ball track, ball on 68% of frames overall. Full precision on the A100 ran ~53 frames/s; next time prefer a 4090/5090 when free.
- **2026-10-23** — Tagger: a debug layer of every ball detection (`D`), other balls marked while watching (`K` then `O`, saved with the frame's label for training), tooltips on every control, a fixed-height header so the video never shifts, a soccer-ball favicon.
- **2026-10-23** — Ball filter over whole games: the chained camera estimate drifts during pans, so still objects far away (a flag behind a goal, a chair) seemed to move and were taken for the ball. Stillness is now judged over 1/3 s, set pieces are only credited when the camera kept the ball in place on screen, still pieces can't fill gaps, and tracklets are cut at kicks. New yardstick `soccer-clips ball eval-games` (whole-game paths at every labelled game frame, without hand fixes): found 55% → 66%, wrong 24% → 13%; bench unchanged.
- **2026-10-23** — Fixed objects: a Veo camera doesn't move (the follow-cam is a virtual view in a fixed recording), so frames are registered directly to keyframes (ORB + homography, `track/world.py`; one global flat plane failed on wide pans). A spot detected nearly every second it's in view, for a minute or more, is a fixed object and its detections are dropped (`soccer-clips ball fixed-objects`). Wrong rings 13.7% → 10.7%.
- **2026-10-24** — Fixed objects tuned on cases found while watching: 35 px spots (a Veo view is two lenses stitched, so placements wobble ~30 px near the edges) and keyframes that show the same place linked directly (they were only linked when the camera moved between them). A stale tagger server re-ran the old filter around every fix, so `soccer-clips serve` now restarts itself on code changes, once no export or render is running. All games: found 66%, wrong 10%; 5 of 451 labelled real-ball detections blocked (labels to check in fc-1 at 43:59, 56:14, 66:20).
- **2026-10-24** — Player tracking: option B chosen (detect on RunPod once, track on this CPU from saved boxes, like the ball). `track/people.py`: full-frame detection at 1920 per model, resumable in parallel chunks (`soccer-clips people detect`, GPU job `people-games`); tracking with ultralytics' BoT-SORT fed saved boxes and our camera steps (ByteTrack without camera motion as the baseline), ~2 ms a frame here; yardstick `soccer-clips people eval` = at each callout on a one-player clip, is there a box under it and how much of the clip its track covers (coverage can't see ID switches onto someone else; check by eye). Smoke test on the L4 m1920 bench boxes: our camera steps match BoT-SORT's own on her coverage with ~20% fewer track ids; ByteTrack loses her in the panning saa clip (8% vs 90%). Bench plan: the 102 callout stretches (~40k frames, 22 min of video) with YOLO11m, YOLO26m/l/x, then the best on whole games (815k frames). Pitch filter and team split come after.
- **2026-10-24** — Player bench run (RunPod L40, 129 min, $1.77): the 102 callout stretches (~40k frames) with YOLO11m and YOLO26m/l/x at 1920. At the 83 callouts on #38-only clips, all four find a box under her 99–100% of the time; her track's share of the clip is a tie (mean 65–68%; YOLO26l beats YOLO11m on 21 callouts and loses on 22; median 85% vs 75% is noise). YOLO11m and YOLO26l make the fewest tracks (~450 a minute vs ~540 for 26m/x, which also report ~15% more boxes). A 3 s lost-track buffer changes nothing. So YOLO11m (the cheapest) for whole games. Her track covers under 25% of the clip in 16 of 83 callouts, broken after ~3 s, at the same player size as the good ones (~62 px): crowds and occlusion, for tracklet linking and correction clicks to fix. Tracking here runs ~4 ms a frame with ~45 boxes a frame.
- **2026-10-25** — Whole-game person detection: YOLO11m on all six games (~776k frames, RunPod A100 at $1.79/h since Kansas had nothing cheaper, ~69 frames/s with 4 workers, 205 min, $6.12). Tracking here: 52 min for all six games (~9 min a game), 2.9–5.0M boxes and 21–25k tracklets a game in `tracks/people_<model>.npz` (50–87 MB). `people eval --saved` scores the saved tracklets without tracking again: at the callouts the same as the bench (99% found, coverage 75% median / 68% mean), her unbroken run around a callout now 24 s median since tracks aren't cut at the clip edges; ~310 new tracklets a minute over whole games.
- **2026-10-25** — Phase 3.3 first slice: follow a player in the tagger. `V` shows every tracked box and an orange ring under the followed player (picker: default player + numbers tagged on clips); `L` + click labels the tracklet under the click, `Shift+L` unlabels the marker's tracklet from that frame on, `U` undoes; an orange timeline strip shows where they're labelled. Boxes stream in 15 s binary chunks (~350 KB) from a per-game in-memory cache. Picks are stored as clicks (frame, position) in `tracks/people_picks.json` and matched to tracklets each time, so they survive re-tracking. On fc-1 one click at a callout labelled 80 s of her.
- **2026-10-25** — Kit colour (Phase 4.2, pulled forward because tracks broke or swapped where she crossed opponents and the ref: 85 of 128 of her labelled stretches ended overlapping another box). Torso colour histogram per detected box (`people kit`, 39 min for all six games here), used three ways in tracking: a gate stops a track taking a box in a different kit (in both matching passes), tracklets are cut where their kit changes, and gaps up to 1 s are joined in the same kit near where the player would be. New yardstick from your #38 labels (`people eval-labels`): her tracklets hold 23.1 s (median) vs 18.2, wrong swaps 5/29 left vs 15/29, conflicts 45 vs 60. Teams: tracklet kits clustered per game (6 clusters); the ones under her labels are 'ours', the biggest other 'opponents', the rest 'other'; boxes coloured by it in the tagger.
