# Phase 2 — Highlight Reel

**Status:** Not started. Depends on Phase 1.
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

## Optional: NLE export
Export the reel as OpenTimelineIO (`opentimelineio`) or CMX EDL pointing at the original files, for finishing by hand in DaVinci Resolve / Kdenlive.

## Tasks
- [ ] `Reel` model + `reel.json` storage + API
- [ ] Per-clip normalise step (trim, speed, lower-third), with caching
- [ ] Title card generation (Pillow)
- [ ] `xfade`/`acrossfade` chain builder + concat fallback
- [ ] Music mixing
- [ ] Presets incl. vertical (crop vs blurred fill — compare)
- [ ] Reel builder UI
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
| Normalise time per 10s clip | |
| Final render time, 3 min reel | |
