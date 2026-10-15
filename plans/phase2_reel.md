# Phase 2 — Highlight Reel

**Status:** Done (2026-10-15). Built 2026-10-14: reels from tagged clips, hard-cut render (preview 540p / full 1080p) with cached intermediates and a segment map, and the reel page (watch, chapters, jump to full game and back, download, reorder, out-of-date warning). Title cards added the same day. A real reel was rendered and looked good. Fades, music, vertical, speed/slow-mo, per-clip trim, lower thirds and OTIO export are in [Future ideas](#future-ideas).
**Goal:** Pick clips, order them, and render a reel (landscape, for TV/YouTube and sharing) with hard cuts.

## Scope
- Reel builder UI (select, reorder, remove).
- Render pipeline with ffmpeg: normalise → title card → join with hard cuts.
- Output presets: 16:9 full and a quick 16:9 preview.
- Optional timeline export for finishing in an NLE (DaVinci Resolve / Kdenlive).

## Reel definition
Stored as `reels/<reel_id>/reel.json` (Pydantic `Reel` model). Each render also saves a copy of the `reel.json` it used next to the output, so any render can be reproduced:
```json
{
  "title": "Fall 2026 — vs Rovers",
  "subtitle": "#14",
  "items": [
    {"game_id": "2026-10-11-rovers", "clip_id": "a3f91c02", "trim_start_s": 0.0, "trim_end_s": 0.0,
     "lower_third": "23' — Goal", "use_callout": true}
  ]
}
```
`use_callout` is a hook for Phase 3: the callout itself is stored on the clip, and the reel decides whether to show it. If a referenced clip has been deleted, the reel builder flags that item instead of failing the render.

## Render pipeline
1. **Normalise each item** to an intermediate file: same resolution, fps, pixel format, audio sample rate/channels. Apply trim and lower-third here. Doing this per clip keeps the final filter graph simple and lets us cache intermediate files.
2. **Title cards**: render PNGs with Pillow (avoids fighting with fonts in `drawtext`), turn each into a short silent clip. See "Title cards" below.
3. **Join** with hard cuts (ffmpeg `concat` filter).
4. **Encode** final: H.264 `crf 20`, AAC 160k, `+faststart`.

MoviePy 2.x is an alternative if ffmpeg filter graphs become painful. It's slower, but easier to reason about. Start with plain ffmpeg.

## Presets
| Preset | Output | Notes |
|---|---|---|
| `landscape_1080p` | 1920×1080 | Default |
| `preview_540p` | 960×540, `ultrafast` | Quick check before the full render |

A vertical (9:16) preset is a [future idea](#future-ideas).

## UI
- From the clip list (Phase 1): filter (e.g. `her` + rating ≥ 4) → "add to reel".
- Reel page: reorder, per-item trim/lower-third, total running time, preview render, final render with progress.
- Clips can come from **multiple games** (season reel).

## As built (first slice, 2026-10-14)
- **Making a reel:** in the tagger, filter the clip list, pick "new reel…" (or an existing reel) and press **Add shown to reel**. Clips already in the reel aren't added twice. Clips can come from several games.
- **Storage:** `<SOCCER_REELS_DIR>/<reel_id>/reel.json` holds `{id, title, items: [{game_id, clip_id}]}`. New options get defaults so existing reels keep loading (e.g. `card_s` was added this way).
- **Render** (`soccer_clips/reel/render.py`), run on the shared single-worker job queue (`jobs.py`) with exports:
  1. Each clip is cut frame-exactly from the original and normalised to the preset size (letterboxed if the aspect differs), 29.97 fps, yuv420p, 48 kHz stereo AAC; games with no audio get silence. These intermediates are cached in `<reel_id>/cache/<preset>-<hash>.mp4`, keyed on the source and the frame-snapped range, so re-rendering after reordering only redoes the final join. Unused cache files for that preset are deleted after each render.
  2. Each intermediate is measured with ffprobe and the segment map is built from those lengths.
  3. The intermediates are joined with ffmpeg's `concat` filter (hard cuts) and encoded once more.
  - Output: `<reel_id>/<preset>.mp4` plus `<preset>.render.json` (the `Render` model: segments, skipped clips, and a copy of the reel definition it was made from). One render per preset; rendering again replaces it.
- **Presets:** `preview_540p` (960×540, ultrafast) and `landscape_1080p` (1920×1080, x264 medium, crf 20).
- **Deleted clips** are shown as "clip deleted" in the reel order list and skipped at render time (listed in `skipped`); the render doesn't fail.
- **API:** `GET/POST /api/reels`, `GET/PATCH/DELETE /api/reels/{id}` (PATCH takes `title` and/or the full `items` list), `POST /api/reels/{id}/render` with `{preset}` (returns the running job if one is already going), `GET /reels/{id}/{preset}.mp4` (served inline with a `<slug-of-title>-<preset>.mp4` filename).
- **Reel page** (`/reel`, `/reel#reel=<id>&preset=<p>&t=<s>`): a reel list, then for a reel: player, timeline coloured by clip (orange = her), "Clips in this video" chapter list (current one highlighted), "Reel order" list with ↑/↓/✕ and a link to each clip in the tagger, render buttons with progress, download link per version, and the out-of-date warning. Keys: Space, `[`/`]` speed, ←/→ ±5s, ↑/↓ previous/next clip, **G** go to the full game.
- **Jumping back:** G (or "game ↗" on a chapter) opens the tagger at `#game=<id>&t=<source time>&clip=<id>&back=<reel url>`. The tagger seeks there, selects the clip and shows "← back to reel"; the browser's Back button also returns to the same spot in the reel.
- Shared page code lives in `static/common.js` / `common.css`.

## Title cards (2026-10-14)
- **What's on them:** white text on a dark background (`#111`), centred, DejaVu Sans Bold (Pillow's built-in font if that's missing).
  - **Opening card:** the reel title.
  - **Card before each clip:** the time in the game ("23'", "2H 7'"; the video time as m:ss if that game has no kickoff set) and, underneath, the clip's note if it has one (wrapped, at most 4 lines).
- **Length:** `Reel.card_s`, default 3s, set in the "cards … s" box next to the render buttons on the reel page; 0 turns all cards off. Cards are exactly `round(card_s × 29.97)` frames, with silent audio.
- **Code:** `soccer_clips/reel/cards.py` draws them; the renderer caches each card like a clip (keyed on its text, length, preset and font).
- **Time map:** each `Segment` records `card_start_s` and `card_lines`; the `Render` records `start_card_lines`. On the reel page a clip's chapter starts at its card: ↑/↓ and the chapter list land on the card, the header shows "(card)", the timeline draws cards in grey, and `G` on a card goes to the start of the clip in the game.
- **Out of date:** a render is flagged if the card length, the title card, or a clip card's text (note or kickoff edited) changed since it was made.
- Possible later tweaks: separate lengths for the opening card and clip cards; show the opponent/date on clip cards in multi-game reels (the time alone is ambiguous there).

## Watching reels in the app
Requested 2026-10-14: watch a rendered reel in the tagger app, jump from any moment of the reel back to that moment in the full game, and download the reel. (Built in the first slice above; this is the design.)

- **Segment map.** Each render writes `render.json` next to its mp4: for every item, its span in the reel (`reel_start_s`, `reel_end_s`) and where it came from (`game_id`, `clip_id`, `source_start_s`, `speed`). It records source **times**, not just the clip id, so jumping back still works if the clip is later edited or deleted. Spans are measured from the actual normalised intermediates (ffprobe), not computed from the reel definition, so they stay right if fades or speed changes are added later (see Future ideas). `Segment.speed` is already in the model for that; it's always 1.0 for now. Title cards map to nothing.
- **Reel page** (`/reel.html#reel=<id>`): a player for the render, with a chapter list (one row per item: match clock, tags, game). The current chapter is highlighted as it plays, and clicking a row jumps there. A version toggle appears when both the preview and the full render exist.
- **Jump to full game:** a button (and key `G`) opens the tagger at the matching source time, selecting the clip if it still exists. The tagger takes `#game=<id>&t=<seconds>&clip=<id>` in the URL hash and shows a "← back to reel" link that returns to the same reel position. Because it's a URL change, the browser's Back button works too.
- **Download:** a link for each rendered file (`/reels/<id>/<file>.mp4`, sent with `Content-Disposition` so the browser saves it with a readable name such as `fall-2026-vs-rovers-landscape.mp4`).
- **Reels list:** the tagger header gets a "Reels" menu listing each reel's renders (date, length, preset).
- **Stale renders:** if a referenced clip has been edited or deleted since the render, the reel page shows "clips changed since this render" with a re-render button. The jump-back map still points at the old source times, which is what was actually rendered.

## Tasks
- [x] `Reel` model + `reel.json` storage + API
- [x] Per-clip normalise step with caching
- [x] Title cards (Pillow): opening card + a card before each clip, length set per reel
- [x] Join with hard cuts (concat filter)
- [x] Reel builder UI, basic: add from the tagger, reorder, remove
- [x] `render.json` segment map written by each render (measured from the intermediates)
- [x] Reel page: player, chapter list, jump to full game (`G`), download links, stale-render warning
- [x] Tagger accepts `#game=&t=&clip=` and shows "← back to reel"
- [x] Render a real reel; record timings below

## Acceptance criteria
- A 2–3 minute reel from 10–20 clips renders on CPU in a reasonable time (target: < 10 min for the final render, < 2 min for preview) with no audio drift. Met 2026-10-14 (see benchmarks).

## Future ideas
Moved out of Phase 2 on 2026-10-14/15 to keep it small. The design notes are kept here for when we come back to them.

- **Fades / transitions.** `xfade` + `acrossfade` chain over the intermediates, falling back to hard cuts if long reels make the filter graph unwieldy. Fades overlap neighbouring clips, so segment spans must still come from measurement (the renderer already does this), and the reel page's chapter boundaries would sit mid-fade. Open question: fades only, or hard cuts for fast action?
- **Music.** `amix` the music with the game audio at reduced volume (e.g. music 0.6, game 0.3) and fade it out at the end. Reel option like `"music": {"path": "...", "volume": 0.6, "game_audio_volume": 0.3}`. Open question: music source (your own files?). Platforms may mute copyrighted music.
- **Vertical version (9:16) for phones.** Preset `vertical_1080x1920`. A 9:16 crop of 1080p follow-cam footage is about 608 px wide, upscaled to 1080, so it'll be soft and players are already small. A **blurred-background fill** (full 16:9 frame in the middle, blurred zoomed copy behind) may look better than a centre crop; try both. Phase 3.3 (crop that follows her) builds on this preset, so 3.3 waits for it too.
- **Per-clip trim.** Shave the start/end of a clip for one reel without changing the tagged clip (`trim_start_s` / `trim_end_s` on the reel item). Low priority: real clip lengths have been fine, and the clip itself can be fixed via `G` → `Shift+I`/`O`.
- **Lower thirds.** A caption over the bottom of the video while a clip plays (e.g. "23' — Goal"). Mostly covered by the clip cards. Open question if revived: style/branding (name + number, team colours?).
- **NLE export (OTIO / EDL).** Export the reel as OpenTimelineIO (`opentimelineio`) or CMX EDL pointing at the original files, for finishing by hand in DaVinci Resolve / Kdenlive.
- **Speed changes and slow-mo replays.** Per-item `"speed": 1.0` and `"slowmo_replay": {"start_s": 4.0, "end_s": 6.0, "speed": 0.5}`, applied in the normalise step (`setpts` / `atempo`). Jump-back inside a slowed segment maps with `source_start_s + (t − reel_start_s) × speed`; `Segment.speed` is already in the model for this.

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
- **2026-10-14** — First preview reel from real footage: clip lengths felt right, so per-clip trim is lower priority.
- **2026-10-14** — Seeking exactly to a chapter boundary could show the previous clip's last frame, so chapter jumps land half a frame in.
- **2026-10-14** — Tested: 25 unit tests (adds card layout, frame counts with cards, card text, out-of-date on note/title/length change, long-note wrapping) and a headless-Chrome run of the card behaviour.
