# Phase 2 — Highlight Reel

**Status:** In progress. Built first (2026-10-14): reels from tagged clips, hard-cut render (preview 540p / full 1080p) with cached intermediates and a segment map, and the reel page (watch, chapters, jump to full game and back, download, reorder, out-of-date warning). Next: transitions, title card, music, vertical preset, per-item trim/speed/slow-mo.
**Goal:** Pick clips, order them, and render a polished reel — landscape for TV/YouTube and vertical for phones.

## Scope
- Reel builder UI (select, reorder, per-clip tweaks).
- Render pipeline with ffmpeg: normalise → transitions → titles → music.
- Output presets: 16:9 and 9:16.
- Optional timeline export for finishing in an NLE (DaVinci Resolve / Kdenlive).

## Reel definition
Stored as `reels/<reel_id>/reel.json` (Pydantic `Reel` model). Each render also saves a copy of the `reel.json` it used next to the output, so any render can be reproduced:
```json
{
  "title": "Fall 2026 — vs Rovers",
  "subtitle": "#14",
  "preset": "landscape_1080p",
  "transition": {"type": "fade", "duration_s": 0.5},
  "music": {"path": "...", "volume": 0.6, "game_audio_volume": 0.3},
  "items": [
    {"game_id": "2026-10-11-rovers", "clip_id": "a3f91c02", "trim_start_s": 0.0, "trim_end_s": 0.0,
     "speed": 1.0, "slowmo_replay": {"start_s": 4.0, "end_s": 6.0, "speed": 0.5},
     "lower_third": "23' — Goal", "use_callout": true}
  ]
}
```
`use_callout` is a hook for Phase 3: the callout itself is stored on the clip, and the reel decides whether to show it. If a referenced clip has been deleted, the reel builder flags that item instead of failing the render.

## Render pipeline
1. **Normalise each item** to an intermediate file: same resolution, fps, pixel format, audio sample rate/channels. Apply trim, speed, slow-mo replay, lower-third here. Doing this per clip keeps the final filter graph simple and lets us cache intermediate files.
2. **Title card**: render a PNG with Pillow (avoids fighting with fonts in `drawtext`), turn it into a 3s clip with a fade.
3. **Join**: generate an `xfade` + `acrossfade` filter chain over the intermediates. Fallback: concat demuxer (hard cuts) if long reels make the filter graph unwieldy.
4. **Music**: `amix` the music with the game audio at reduced volume; fade the music out at the end.
5. **Encode** final: H.264 `crf 20`, AAC 160k, `+faststart`.

MoviePy 2.x is an alternative if ffmpeg filter graphs become painful. It's slower, but easier to reason about. Start with plain ffmpeg.

## Presets
| Preset | Output | Notes |
|---|---|---|
| `landscape_1080p` | 1920×1080 | Default |
| `vertical_1080x1920` | 1080×1920 | Phase 2: centre crop (or blurred-background fill). Phase 3.3: crop follows her. |
| `preview_540p` | 960×540, `ultrafast` | Quick check before the full render |

**Vertical caveat:** a 9:16 crop of 1080p follow-cam footage is about 608 px wide, upscaled to 1080. It'll be soft, and players are already small. The **blurred-background fill** option (full 16:9 frame in the middle, blurred zoomed copy behind) may look better than a hard crop. Try both.

## UI
- From the clip list (Phase 1): filter (e.g. `her` + rating ≥ 4) → "add to reel".
- Reel page: drag to reorder, per-item trim/speed/slow-mo/lower-third, total running time, preview render, final render with progress.
- Clips can come from **multiple games** (season reel).

## As built (first slice, 2026-10-14)
- **Making a reel:** in the tagger, filter the clip list, pick "new reel…" (or an existing reel) and press **Add shown to reel**. Clips already in the reel aren't added twice. Clips can come from several games.
- **Storage:** `<SOCCER_REELS_DIR>/<reel_id>/reel.json` holds `{id, title, items: [{game_id, clip_id}]}`. Per-item options (trim, speed, slow-mo, lower third) and reel options (transition, music, title card) aren't in the model yet; they'll be added with defaults so existing reels keep loading.
- **Render** (`soccer_clips/reel/render.py`), run on the shared single-worker job queue (`jobs.py`) with exports:
  1. Each clip is cut frame-exactly from the original and normalised to the preset size (letterboxed if the aspect differs), 29.97 fps, yuv420p, 48 kHz stereo AAC; games with no audio get silence. These intermediates are cached in `<reel_id>/cache/<preset>-<hash>.mp4`, keyed on the source and the frame-snapped range, so re-rendering after reordering only redoes the final join. Unused cache files for that preset are deleted after each render.
  2. Each intermediate is measured with ffprobe and the segment map is built from those lengths.
  3. The intermediates are joined with ffmpeg's `concat` filter (hard cuts) and encoded once more.
  - Output: `<reel_id>/<preset>.mp4` plus `<preset>.render.json` (the `Render` model: segments, skipped clips, and a copy of the reel definition it was made from). One render per preset; rendering again replaces it.
- **Presets:** `preview_540p` (960×540, ultrafast) and `landscape_1080p` (1920×1080, x264 medium, crf 20). Vertical comes later.
- **Deleted clips** are shown as "clip deleted" in the reel order list and skipped at render time (listed in `skipped`); the render doesn't fail.
- **API:** `GET/POST /api/reels`, `GET/PATCH/DELETE /api/reels/{id}` (PATCH takes `title` and/or the full `items` list), `POST /api/reels/{id}/render` with `{preset}` (returns the running job if one is already going), `GET /reels/{id}/{preset}.mp4` (served inline with a `<slug-of-title>-<preset>.mp4` filename).
- **Reel page** (`/reel`, `/reel#reel=<id>&preset=<p>&t=<s>`): a reel list, then for a reel: player, timeline coloured by clip (orange = her), "Clips in this video" chapter list (current one highlighted), "Reel order" list with ↑/↓/✕ and a link to each clip in the tagger, render buttons with progress, download link per version, and the out-of-date warning. Keys: Space, `[`/`]` speed, ←/→ ±5s, ↑/↓ previous/next clip, **G** go to the full game.
- **Jumping back:** G (or "game ↗" on a chapter) opens the tagger at `#game=<id>&t=<source time>&clip=<id>&back=<reel url>`. The tagger seeks there, selects the clip and shows "← back to reel"; the browser's Back button also returns to the same spot in the reel.
- Shared page code lives in `static/common.js` / `common.css`.

## Watching reels in the app
Requested 2026-10-14: watch a rendered reel in the tagger app, jump from any moment of the reel back to that moment in the full game, and download the reel. (Built in the first slice above; this is the design.)

- **Segment map.** Each render writes `render.json` next to its mp4: for every item, its span in the reel (`reel_start_s`, `reel_end_s`) and where it came from (`game_id`, `clip_id`, `source_start_s`, `speed`). It records source **times**, not just the clip id, so jumping back still works if the clip is later edited or deleted. Spans are measured from the actual normalised intermediates (ffprobe), not computed from the reel definition, because transitions overlap neighbouring items and speed changes and slow-mo replays stretch them. Inside a slow-mo replay, map back with `source_start_s + (t − reel_start_s) × speed`. Title cards map to nothing.
- **Reel page** (`/reel.html#reel=<id>`): a player for the render, with a chapter list (one row per item: match clock, tags, game). The current chapter is highlighted as it plays, and clicking a row jumps there. The landscape/vertical toggle appears when both have been rendered.
- **Jump to full game:** a button (and key `G`) opens the tagger at the matching source time, selecting the clip if it still exists. The tagger takes `#game=<id>&t=<seconds>&clip=<id>` in the URL hash and shows a "← back to reel" link that returns to the same reel position. Because it's a URL change, the browser's Back button works too.
- **Download:** a link for each rendered file (`/reels/<id>/<file>.mp4`, sent with `Content-Disposition` so the browser saves it with a readable name such as `fall-2026-vs-rovers-landscape.mp4`).
- **Reels list:** the tagger header gets a "Reels" menu listing each reel's renders (date, length, preset).
- **Stale renders:** if a referenced clip has been edited or deleted since the render, the reel page shows "clips changed since this render" with a re-render button. The jump-back map still points at the old source times, which is what was actually rendered.

## Optional: NLE export
Export the reel as OpenTimelineIO (`opentimelineio`) or CMX EDL pointing at the original files, for finishing by hand in DaVinci Resolve / Kdenlive.

## Tasks
- [x] `Reel` model + `reel.json` storage + API
- [x] Per-clip normalise step with caching (trim, speed, lower-third still to do)
- [ ] Title card generation (Pillow)
- [ ] `xfade`/`acrossfade` chain builder (concat with hard cuts is done)
- [ ] Music mixing
- [ ] Presets incl. vertical (crop vs blurred fill — compare)
- [x] Reel builder UI, basic: add from the tagger, reorder, remove (per-item options still to do)
- [x] `render.json` segment map written by each render (measured from the intermediates)
- [x] Reel page: player, chapter list, jump to full game (`G`), download links, stale-render warning
- [x] Tagger accepts `#game=&t=&clip=` and shows "← back to reel"
- [ ] Slow-mo replay
- [ ] *(Optional)* OTIO export
- [ ] Render a real reel; record timings below

## Acceptance criteria
- A 2–3 minute reel from 10–20 clips renders on CPU in a reasonable time (target: < 10 min for the final render, < 2 min for preview) with no audio drift.
- The vertical version is watchable on a phone.

## Open questions
- Music source — your own files? (Note: platforms may mute copyrighted music.)
- Lower-third style/branding: name + number, team colours?
- Transition defaults: fades only, or also hard cuts for fast action?

## Benchmarks / findings
| Measure | Value |
|---|---|
| Preview render (540p), 15 × 10s clips = 2:30 reel | 29s (target < 2 min) — 96 MB, ultrafast is bulky but it's only for checking |
| Full render (1080p), same reel, nothing cached | 215s ≈ 3.6 min (target < 10 min) — 154 MB |
| Full render again with cached intermediates (e.g. after reordering) | 143s (final encode only) |
| Normalise time per 10s clip (1080p, veryfast crf 16) | ≈ 5s (from the above) |

Measured 2026-10-14 on `gu11-vs-saa-2026-10-10` (real 1080p Veo footage), CPU only.

## Findings
- **2026-10-14** — Segment lengths are max(video, audio) of each intermediate, which is a few ms longer than the video, because the concat filter starts the next clip after the longest stream ends. The reel page allows 50 ms of slack at chapter boundaries; without it, seeking to a chapter start landed in the previous chapter.
- **2026-10-14** — Tested: 23 unit tests (including frame counts, A/V sync, segment mapping, cache reuse, out-of-date detection, a game with no audio) and a headless-Chrome run of the whole flow (tagger → new reel → render → ↓ → G → tagger at the right frame with the clip selected → Back → same reel position).
