# Phase 3 — Highlight Her

**Status:** 3.1 built (2026-10-15): freeze-frame callout set with `C` in the tagger, rendered as a spotlight + ring + arrow freeze in reels. YOLO snap not done (optional). 3.2 and 3.3 not started.
**Goal:** Make it obvious who to watch in each clip, with as few clicks as possible, on a CPU-only machine.

## Constraints that shape this phase
- **Follow-cam only:** the virtual camera pans and zooms, so image-space motion is player motion plus camera motion. She can leave the frame entirely.
- **Small players:** at 1080p, players may be well under 100 px tall — measure on real footage. Small objects mean weaker detection and re-ID.
- **CPU only:** everything is per-clip (10–20s), run as background jobs, and must be benchmarked before committing to an approach.

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
- **Render:** the clip's cached intermediate is split at the callout frame and the frozen, overlaid frame is held for `Reel.callout_s` seconds (default 0.5s since 2026-10-15, was 1.5s; set in the "freeze … s" box on the reel page; 0 turns freezes off) with silence, then the clip carries on. The spliced version is cached separately, so moving a callout only redoes the splice.
- **Time map:** `Segment.freeze_at_s`, `freeze_s` and `callout`. `G` during the freeze goes to the callout moment in the game; after it, game time is offset by the freeze length. The reel page header shows "(callout)" during the freeze, the timeline draws it white, and chapters with a callout show ◎.
- **Out of date:** a render is flagged when a callout is added, moved or removed, or the freeze length changes.
- **Phase 4:** no crops are saved yet. Callout clicks are stored on each clip, so crops of her can be generated from them whenever Phase 4 needs them.

**Tasks**
- [x] Callout storage + `C` click interaction in the player
- [ ] Single-frame YOLO snap (optional, behind a flag) — not done; needs ultralytics/torch in the env
- [x] Freeze + overlay + resume in the render pipeline
- [x] Pick a visual style (ellipse vs spotlight vs arrow) — spotlight + ring + arrow by default; change after seeing it on real reels

## 3.2 — Tracked marker (click + correct)
Keep a marker on her for the whole clip.

**Primary approach (CPU-friendly): YOLO11 + BoT-SORT**
1. Run `ultralytics` `model.track(tracker="botsort.yaml")` over the clip. BoT-SORT includes **global motion compensation**, which accounts for camera panning — important for follow-cam footage.
2. In the UI, you click her on any frame → we pick the track ID whose box contains the click.
3. Track IDs break (occlusion, leaving frame, ID switches). The UI shows a timeline bar of where "her" track is present, lost, or uncertain. You scrub to a gap and click her again → that track ID is joined onto her track from that frame.
4. Post-process: interpolate short gaps (< ~0.5s), smooth box centres (Savitzky–Golay or One-Euro filter) so the marker doesn't jitter.

**Detection tuning to benchmark:** model size (n vs s), `imgsz` (640 vs 1280 for small players), every frame vs every 2nd frame + interpolation. SAHI-style tiling helps small objects but is expensive on CPU — only if needed.

**Fallback for hard clips: SAM 2 (tiny / `sam2.1_hiera_tiny`)**
Click once → SAM 2 propagates a mask through the video. It handles crowding better than detection + IDs, but is much slower on CPU. Run it at reduced resolution/frame rate and only when YOLO+BoT-SORT needs too many corrections. Correction clicks work the same way.

**Review UI:** draw the track box/ellipse on a `<canvas>` over the `<video>`, synced by frame, so you can check before rendering. Corrections are just more clicks.

**Render:** OpenCV reads frames → supervision `EllipseAnnotator` / `TriangleAnnotator` (or a custom spotlight) → pipe raw frames to ffmpeg stdin → mux the original audio back in.

**Track storage:** `tracks/<clip_id>.json`
```json
{"fps": 30, "source": "yolo11s+botsort",
 "clicks": [{"frame": 12, "x": 0.41, "y": 0.63}],
 "frames": {"12": [x1, y1, x2, y2, conf], "13": [...]}}
```

**Tasks**
- [ ] Benchmark YOLO11n/s × imgsz 640/1280 on a real 15s clip (speed + does it find her?)
- [ ] Tracking job + track-ID selection from a click
- [ ] Re-click stitching + gap interpolation + smoothing
- [ ] Canvas overlay review UI with lost/present timeline
- [ ] Annotated render pipe (OpenCV → supervision → ffmpeg)
- [ ] SAM 2 tiny fallback + benchmark
- [ ] Save every confirmed "her" box/crop for Phase 4 (see below)

## 3.3 — Follow-crop / punch-in
Use the smoothed track to drive a crop window:
- **Vertical 9:16** export that keeps her in frame. Needs the vertical preset, which moved to Phase 2's [Future ideas](phase2_reel.md#future-ideas) on 2026-10-14; the landscape punch-in below doesn't.
- Optional **punch-in** zoom on landscape clips (e.g. 1.3–1.5×) to make her bigger.
- Heavy smoothing + clamping to frame edges; keep the crop still when she's briefly lost rather than jumping.
- Quality limit: cropping already-small follow-cam footage gets soft quickly — cap the zoom factor.

**Tasks**
- [ ] Crop-path generator from track (smoothing, clamping, max zoom)
- [ ] Vertical preset uses crop path
- [ ] Optional landscape punch-in

## Feeding Phase 4
Every confirmed position of her — callout clicks, track-ID selections, corrections — is a labelled example. From the start of 3.1/3.2, save:
- a crop of her (with game, frame, box), and
- crops of **other players in the same frame** as negatives, especially teammates (same kit = hard negatives).
Crops are saved as image files in `player_gallery/`, and each one gets a line appended to `player_gallery/index.jsonl` (game, clip, frame, box, label = her / teammate / opponent, source = callout / track / correction). It's append-only JSONL so it can grow to thousands of entries without rewriting a big file.

## Acceptance criteria
- 3.1: adding a callout to a clip takes one click; it renders cleanly in the reel.
- 3.2: a typical 15s clip needs ≤ 3 correction clicks and tracking finishes in under ~2 min on CPU (target, to be validated).
- 3.3: vertical export keeps her in frame without nauseating motion.

## Open questions
- Visual style for the marker — 3.1 uses spotlight + ring + arrow; revisit after real reels. Name tag?
- Is it OK for the tracked marker to just disappear when tracking loses her, or should it fall back to the last known spot briefly?
- How small is she in typical follow-cam frames? (Determines imgsz/tiling choices.)

## Benchmarks / findings
| Measure | Value |
|---|---|
| Her typical height in px (1080p) | ~70 px for players in one midfield frame of the SAA game (rough, one sample) |
| YOLO11n @640, fps on CPU | |
| YOLO11s @1280, fps on CPU | |
| Avg correction clicks / clip | |
| SAM 2 tiny, sec per clip-second | |

## Findings
- **2026-10-15** — 3.1 tested: 27 unit tests (callout API rules, exact 45-frame freeze, A/V sync, the held frame has the ring and is dimmed outside it, jump-back before/during/after the freeze, out-of-date detection) and a headless-Chrome run of placing (click, drag to size, Esc, Shift+C) and the reel freeze + `G`.
