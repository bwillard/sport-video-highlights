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

## Watching reels in the app
Requested 2026-10-14: watch a rendered reel in the tagger app, jump from any moment of the reel back to that moment in the full game, and download the reel.

- **Segment map.** Each render writes `render.json` next to its mp4: for every item, its span in the reel (`reel_start_s`, `reel_end_s`) and where it came from (`game_id`, `clip_id`, `source_start_s`, `speed`). It records source **times**, not just the clip id, so jumping back still works if the clip is later edited or deleted. Spans are measured from the actual normalised intermediates (ffprobe), not computed from the reel definition, because transitions overlap neighbouring items and speed changes and slow-mo replays stretch them. Inside a slow-mo replay, map back with `source_start_s + (t − reel_start_s) × speed`. Title cards map to nothing.
- **Reel page** (`/reel.html#reel=<id>`): a player for the render, with a chapter list (one row per item: match clock, tags, game). The current chapter is highlighted as it plays, and clicking a row jumps there. The landscape/vertical toggle appears when both have been rendered.
- **Jump to full game:** a button (and key `G`) opens the tagger at the matching source time, selecting the clip if it still exists. The tagger takes `#game=<id>&t=<seconds>&clip=<id>` in the URL hash and shows a "← back to reel" link that returns to the same reel position. Because it's a URL change, the browser's Back button works too.
- **Download:** a link for each rendered file (`/reels/<id>/<file>.mp4`, sent with `Content-Disposition` so the browser saves it with a readable name such as `fall-2026-vs-rovers-landscape.mp4`).
- **Reels list:** the tagger header gets a "Reels" menu listing each reel's renders (date, length, preset).
- **Stale renders:** if a referenced clip has been edited or deleted since the render, the reel page shows "clips changed since this render" with a re-render button. The jump-back map still points at the old source times, which is what was actually rendered.

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
- [ ] `render.json` segment map written by each render (measured from the intermediates)
- [ ] Reel page: player, chapter list, jump to full game (`G`), download links, stale-render warning
- [ ] Tagger accepts `#game=&t=&clip=` and shows "← back to reel"
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
