# Phase 3 — Track & Highlight Players

**Status:** 3.1 built (2026-10-15): freeze-frame callout set with `C` in the tagger, rendered as a spotlight + ring + arrow freeze in reels. YOLO snap not done (optional). Re-scoped 2026-10-18 from "a marker on her in one clip" to "track every player through the whole game, then highlight any of them". 3.2 started 2026-10-24 (code built, see "As built" under 3.2); 3.3–3.5 not started. The game ball is tracked and drawn in the tagger already (Phase 4.1, `B`), which is a first version of the canvas overlay 3.3 needs: synced to the video per frame, solid/dashed for seen/filled.
**Goal:** Make it obvious who to watch, for any player on the team, with as few clicks as possible, on a 4-core CPU. Naming the players and tracking the ball is [Phase 4](phase4_identify.md); this phase builds the tracking data and everything that draws from it.

## Constraints that shape this phase
- **Follow-cam only:** the virtual camera pans and zooms, so image-space motion is player motion plus camera motion. Players leave the frame often.
- **Small players:** at 1080p, players may be well under 100 px tall (~70 px measured in one midfield frame). Small objects mean weaker detection and re-ID.
- **CPU for ML (for now; see [hardware options](hardware.md)):** a Ryzen 3 3200G (4 cores, no SMT). The Vega iGPU does video decode/encode (VA-API) but can't run the models (no CUDA, no usable ROCm). A 60–84 min game is 108k–151k frames, so whole-game tracking is an overnight background job, possibly on every 2nd or 3rd frame. Benchmark before choosing.
- **One track store per game:** tracking runs once per game, not per clip. Clips, reels and clip suggestions all read from the same per-game data, so a clip added later doesn't need any new tracking.

## 3.1 — Freeze-frame callout (do this first)
One click, no tracking — the standard technique in recruiting/highlight videos.

**Interaction:** In the player, pause at the moment you want → press `C` → click on her. Stored as `Clip.callout` in the game's `game.json`:
```json
{"t_s": 3.2, "x": 0.41, "y": 0.63, "r": 0.03}   // x, y, r normalised to frame size
```
**Optional snap:** run YOLO11n person detection on that single frame (fast on CPU) and snap the click to the nearest person box, so the ellipse is sized and placed under her feet automatically.

**Render** (in Phase 2's per-clip normalise step):
1. Play clip up to `t`.
2. Freeze frame for ~1.0–1.5s: darken/vignette everything except her, draw a circle or ellipse plus an arrow, optional name/number label, optional slight zoom-in (`zoompan` or crop+scale animation).
3. Resume playback.

Overlays are drawn with Pillow/OpenCV onto the frozen frame (or with supervision's annotators for a consistent look with 3.2).

**As built (2026-10-15)**
- **Placing it:** select a clip, pause inside it, press `C` (or "set" in the clip editor), click on her. Drag instead of clicking to size the ring. `Esc` cancels; `Shift+C` (or "remove") deletes it. One callout per clip.
- **Storage:** `Clip.callout = {t_s, x, y, r}`, with x as a fraction of the frame width, y and r as fractions of the height; default r = 0.06 (about 130 px across at 1080p, which frames one player). Moving the clip's start/end so the callout falls outside the clip removes it.
- **Preview:** when the playhead is on the callout frame (or passing it), the tagger draws the same overlay on top of the video. The clip list shows ◎ and the timeline has a white dot at the callout moment.
- **Look (default style):** everything outside a soft-edged spotlight dimmed to 45%, a white ring, and a white arrow pointing down at her (flipped below the ring if she's near the top). Drawn with Pillow in `soccer_clips/reel/callout.py`; the tagger's canvas preview mirrors it.
- **Render:** the clip's cached intermediate is split at the callout frame and the frozen, overlaid frame is held for `Reel.callout_s` seconds (default 1.5s with caption overlays and 0.75s with title cards, as of 2026-10-15; set in the "freeze … s" box on the reel page; 0 turns freezes off) with silence, then the clip carries on. The spliced version is cached separately, so moving a callout only redoes the splice.
- **Time map:** `Segment.freeze_at_s`, `freeze_s` and `callout`. `G` during the freeze goes to the callout moment in the game; after it, game time is offset by the freeze length. The reel page header shows "(callout)" during the freeze, the timeline draws it white, and chapters with a callout show ◎.
- **Out of date:** a render is flagged when a callout is added, moved or removed, or the freeze length changes.
- **Phase 4:** no crops are saved yet. Callout clicks are stored on each clip, so crops of her can be generated from them whenever Phase 4 needs them.

**Tasks**
- [x] Callout storage + `C` click interaction in the player
- [ ] Single-frame YOLO snap (optional, behind a flag) — not done; needs ultralytics/torch in the env
- [x] Freeze + overlay + resume in the render pipeline
- [x] Pick a visual style (ellipse vs spotlight vs arrow) — spotlight + ring + arrow by default; change after seeing it on real reels

## 3.2 — Whole-game tracking job
Detect and track every person (and the ball, see [Phase 4.1](phase4_identify.md#41--ball-tracking)) through the game once, in the background.

1. Run `ultralytics` `model.track(tracker="botsort.yaml")` from the 1st-half kickoff to the end of the game (warm-up skipped using the kickoff markers). BoT-SORT's **global motion compensation** accounts for the camera panning.
2. Decode with ffmpeg (VA-API decode if it helps) and feed frames to the model. If every frame is too slow, run on every 2nd/3rd frame and interpolate; the benchmark decides.
3. Output: **tracklets**, i.e. stretches where the tracker kept one ID on one person. Expect many short tracklets per player (occlusion, leaving the frame, ID switches). Joining them into "player #38 for the whole game" is Phase 4.3, or manual clicks in 3.3.
4. Post-process: interpolate short gaps (< ~0.5s) inside a tracklet, and smooth box centres (Savitzky–Golay or One-Euro) so markers don't jitter.

**Detection tuning to benchmark:** model size (n vs s), `imgsz` (640 vs 1280 for small players), every frame vs every 2nd/3rd frame + interpolation, PyTorch vs ONNX Runtime export on this CPU. SAHI-style tiling helps small objects but costs several times more; only if needed.

**Rough budget (to replace with measurements):** if YOLO11n @640 runs ~10 fps here, a 75-min game every frame is ~3.5–4 hours; every 3rd frame is ~1.3 hours. @1280 is roughly 3–4× slower. Overnight is fine.

**Storage** (proposed; per game, replaces the earlier per-clip `tracks/<clip_id>.json`). A full game is a few hundred thousand to ~2M boxes, too big for one JSON file and too slow to load on each request, so boxes go in a compact array file and the editable labels stay JSON:
```
<SOCCER_GAMES_DIR>/<year>/<game_id>/tracks/
  run.json         # model, imgsz, frame step, tracker settings, time range, created, timings
  boxes.npz        # columns: frame, tracklet_id, cls (person/ball), x1, y1, x2, y2 (normalised), conf
  tracklets.json   # tracklet_id -> {first, last, team, player, source, confirmed}; edited by the UI and Phase 4
  ball.npz         # filtered ball path from Phase 4.1: frame, x, y, conf, interpolated
```
`tracklets.json` goes through `store.py` like every other JSON file; the `.npz` files are written once by the job (temp file + rename) and only read afterwards.

**As built (2026-10-24): option B, detect on a GPU, track here** (`track/people.py`)
- **Detect** (RunPod, once per model): stock YOLO on the full frame at 1920, conf ≥ 0.1, batch 4, in parallel chunks; saved to `tracks/people_raw_<model>.npz` (`ranges`, `dets`: frame, x1, y1, x2, y2, conf in pixels) after each chunk, so a stopped run resumes. Modes: `callouts` (clips with a callout ±2 s, for the bench), `clips`, `all`.
- **Track** (this CPU, ~2 ms a frame): ultralytics' BoT-SORT on the saved boxes, with our camera step into each frame (from `ball_raw.npz`) applied to its predictions in place of its own image-based motion step; no re-ID. ByteTrack (no camera motion) is the baseline. Output `tracks/people_<model>.npz`: frame, track, x1, y1, x2, y2 (fractions), conf, ranges; track ids unique per game.
- **Yardstick:** `people eval` checks each callout on a clip tagged with one player: box under it (or nearest centre within the ring), the share of the clip its track covers, and the run around the callout with gaps ≤ 1 s. Also tracks per minute and boxes per frame. Coverage can't tell a track that stayed on her from one that switched to a neighbour.
- This replaces the earlier single-pass plan (`model.track()` on the pod), which would need a paid run for every tracker change, and BoT-SORT's own motion step was single-threaded and the bottleneck on a GPU.

**Tasks**
- [x] Benchmark clip set: 7 clips (87s) from all 5 games in `bench/clips.json`, cut with `soccer-clips bench cut` (2026-10-18)
- [x] Benchmark YOLO11n/s × imgsz 640/1280 (+ centre crop) on the benchmark clips, for people *and* the ball: `soccer-clips bench run`, results in `bench/results/` (2026-10-18; see Findings)
- [x] Detection job (2026-10-24): `soccer-clips people detect <game> --mode callouts|clips|all --model M`, GPU job `people-games`; resumable, parallel chunks
- [x] Tracking from saved boxes (2026-10-24): `soccer-clips people track`, BoT-SORT + our camera steps
- [x] Yardstick (2026-10-24): `soccer-clips people eval` at the callouts
- [ ] Bench run on RunPod: callout stretches × YOLO11m / YOLO26m / l / x; pick a detector and tracker settings
- [ ] Whole-game detection with the chosen model
- [ ] Pitch filter (drop sideline spectators) — after looking at real tracks; team split (4.2) may do most of it
- [ ] Gap interpolation + smoothing
- [ ] Track store (`run.json`, `boxes.npz`, `tracklets.json`) + API returning boxes for a time window and player/tracklet

## 3.3 — Live highlight during playback
Draw a marker that follows the chosen player while the video plays, in the tagger and on the reel page. Nothing is re-encoded, so changing who's highlighted, or the style, is instant.

**How it works**
- A `<canvas>` sits over the `<video>` (the 3.1 callout preview already does this). `video.requestVideoFrameCallback` fires for every frame the browser shows and gives its exact media time, so the overlay stays locked to the picture during play, pause, stepping and seeking.
- Frame index = media time × 30000/1001 (Veo files are constant frame rate). The page looks up the player's box at that frame, or interpolates linearly between the two nearest samples when tracking ran every Nth frame, and draws the marker: ring under the feet, spotlight, arrow or name tag, matching the callout style.
- **Data:** the page fetches the boxes for the chosen player over the visible range (a clip, or a few minutes around the playhead in the full game) from the 3.2 API. One player for a whole game at 10 samples/s is ~50k boxes, about 1 MB; a clip is a few KB. Drawing one marker per frame costs nothing.
- **Picking who:** click a player on the paused video → the tracklet whose box contains the click is highlighted. Once Phase 4.3 has named tracklets, pick by name/number (default: `SOCCER_DEFAULT_PLAYER`) and every tracklet labelled with that player lights up.
- **Gaps:** when the player has no box (off screen or lost), the marker fades out; it doesn't jump. Short gaps are already interpolated in 3.2.
- **Corrections:** when the marker drops or jumps to someone else, pause and click the right player → that tracklet is labelled as this player from that frame on (this also gives Phase 4.3 labelled examples). A timeline strip shows where the player is present, missing, or uncertain.

**Tasks**
- [ ] Canvas overlay driven by `requestVideoFrameCallback`, with interpolation and fade-out on gaps
- [ ] Click-to-pick a tracklet; player picker once tracklets are named
- [ ] Correction clicks (relabel a tracklet from a frame on) + present/missing timeline strip
- [ ] Same overlay on the reel page (reel time → game time via the segment map)

**Ball overlay (2026-10-22):** the same canvas-over-video idea is built for the game ball: `B` in the tagger draws a ring on the tracked ball (`/api/games/{id}/ball`), as a first step towards 3.3's player marker.

## 3.4 — Marker burned into reels
For downloads and sharing, the marker has to be in the pixels.

**Render:** for clips that have a highlighted player, decode → draw the marker per frame (Pillow/OpenCV, same style as 3.3; or supervision's `EllipseAnnotator`) → pipe raw frames to ffmpeg → mux the original audio back in. These pieces can't use the GPU-only path from the 2026-10-18 render work (frames go through the CPU to be drawn), but they can still be encoded with VA-API after uploading. A render is flagged out of date when the highlighted player's track labels change.

**Per reel:** the player to highlight (default: the reel's player filter) and the marker style, with the existing "with" menu able to turn it off for a clean copy.

**Tasks**
- [ ] Annotated render piece (decode → draw → ffmpeg), cached like the other pieces
- [ ] Reel setting for highlighted player + style; out-of-date detection

## 3.5 — Follow-crop / punch-in (later)
Use the smoothed track to drive a crop window:
- **Vertical 9:16** export that keeps the player in frame. Needs the vertical preset, which moved to Phase 2's [Future ideas](phase2_reel.md#future-ideas) on 2026-10-14; the landscape punch-in below doesn't.
- Optional **punch-in** zoom on landscape clips (e.g. 1.3–1.5×) to make the player bigger.
- Heavy smoothing + clamping to frame edges; keep the crop still when the player is briefly lost rather than jumping.
- Quality limit: cropping already-small follow-cam footage gets soft quickly; cap the zoom factor.

**Tasks**
- [ ] Crop-path generator from track (smoothing, clamping, max zoom)
- [ ] Vertical preset uses crop path
- [ ] Optional landscape punch-in

## Dropped from the earlier 3.2 plan
- **SAM 2 tiny as a fallback tracker** for hard clips. It is far too slow on this CPU for whole games; reconsider only if a few important clips can't be fixed with correction clicks.

## Acceptance criteria
- 3.1: adding a callout to a clip takes one click; it renders cleanly in the reel.
- 3.2: a full game tracks overnight without supervision; resuming after a stop doesn't redo finished work.
- 3.3: the marker stays on the player during normal playback and seeking with no visible lag; a typical 15s clip needs ≤ 3 correction clicks.
- 3.4: the burned-in marker matches the live one.
- 3.5: vertical export keeps the player in frame without nauseating motion.

## Open questions
- Visual style for the moving marker: ring under the feet, spotlight, or both? Name tag?
- When tracking loses the player, fade out, or hold the last spot briefly?
- How small are players in typical follow-cam frames? (Determines imgsz/tiling.)
- Frame step for the whole-game job: is every 3rd frame + interpolation smooth enough for the marker? (Probably yes for players, not for the ball.)

## Benchmarks / findings
| Measure | Value |
|---|---|
| Her typical height in px (1080p) | 58–73 px at the callout frame across 5 bench clips |
| YOLO11n @640, fps on CPU | 8.8 (detect 91 ms + track 20 ms per frame) |
| YOLO11s @640, fps on CPU | 6.3 |
| YOLO11n @1280, fps on CPU | 4.8 |
| YOLO11s @1280, fps on CPU | 2.1 (detect 464 ms); ONNX Runtime 2.6 |
| Full 75-min game (135k frames), hours on CPU | n640: 4.3 every frame, 1.4 every 3rd; n1280: 7.8 / 2.6; s1280: 18 / 6 |
| YOLO11m @1920 (full frame), fps on L4 GPU | 19.1 (~2 h per game); her track coverage 17–100%, median 100% |
| Tracking from saved boxes, CPU | ~2 ms/frame (BoT-SORT + our camera steps, 3200G) |
| Avg correction clicks / clip | |

## Findings
- **2026-10-18** — First benchmark (`bench/results/`, 7 clips, 2,605 frames, Ryzen 3 3200G). Speeds in the table above. Detection is almost all of the time; decode is 2 ms/frame and BoT-SORT 12–20 ms/frame, so on a GPU the tracker (its camera-motion step) would become the limit at ~50–80 fps unless it runs on a smaller image. ONNX Runtime is only ~15% faster than PyTorch on this CPU.
  - **Finding her:** at the callout frame, the box under the click was found in 5/5 clips at 1280 (either model), 4/5 with s640, 3/5 with n640. Small, distant players need 1280 (or the s model).
  - **Keeping her:** track IDs break a lot. 40–190 track IDs per clip for ~15–25 people in view, and the track at her callout covered anywhere from 1% to 100% of its clip (typically 10–50%). Joining tracklets (correction clicks in 3.3, linking in Phase 4.3) is the main work, as expected.
  - **Not just players:** 15–30 "people" per frame at 1280, including parents, coaches and subs on the sidelines. Tracking needs a pitch filter (e.g. drop boxes above the far touchline / outside the field area) before team split.
  - **On an L4 GPU (cloud, 2026-10-18; `bench/results-l4/`):** the same variants give identical results (detection doesn't depend on the hardware), and three bigger ones were added. **YOLO11m at full resolution (m1920) kept her track for 100%, 100%, 100%, 17% and 89% of the 5 callout clips**, against roughly 10–50% for every variant the CPU can run. Steadier, more confident boxes mean far fewer ID breaks. It runs at 19 fps on the L4 (~2 h per game) and would take days per game on the 3200G. Speeds in [hardware.md](hardware.md).
  - Annotated videos: `soccer-clips bench draw <variant> [clips]` writes `bench/results/<variant>/<clip>.mp4` (her track thick yellow, balls red with confidence). Drawn for s1280-centre.
- **2026-10-15** — 3.1 tested: 27 unit tests (callout API rules, exact 45-frame freeze, A/V sync, the held frame has the ring and is dimmed outside it, jump-back before/during/after the freeze, out-of-date detection) and a headless-Chrome run of placing (click, drag to size, Esc, Shift+C) and the reel freeze + `G`.
