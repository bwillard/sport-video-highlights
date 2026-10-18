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
- From the clip list (Phase 1): filter (e.g. `her` + 😊) → "add to reel".
- Reel page: reorder, per-item trim/lower-third, total running time, preview render, final render with progress.
- Clips can come from **multiple games** (season reel).

## As built (first slice, 2026-10-14)
- **Making a reel:** in the tagger, filter the clip list, pick "new reel…" (or an existing reel) and press **Add shown to reel**. Clips already in the reel aren't added twice. Clips can come from several games.
- **Storage:** `<SOCCER_REELS_DIR>/<reel_id>/reel.json` holds `{id, title, items: [{game_id, clip_id}]}`. New options get defaults so existing reels keep loading (e.g. `card_s` was added this way).
- **Render** (`soccer_clips/reel/render.py`), run on the shared single-worker job queue (`jobs.py`) with exports:
  1. Each clip is cut frame-exactly from the original and normalised to the preset size (letterboxed if the aspect differs), 29.97 fps, yuv420p, 48 kHz stereo; games with no audio get silence. A clip is made as up to four **pieces** (`clip_spans`): the part with the caption drawn over it, the plain rest, and the callout freeze as a still. Only the captioned pieces need the CPU; the plain ones stay on the GPU, and a note edit only redoes the captioned piece. Pieces are cached in `<reel_id>/cache/<preset>-<hash>.mov`, keyed on the source, the frame range, what's drawn on it and the encoder settings, so re-rendering after reordering only redoes the join. Two pieces are made at a time (`PIECE_WORKERS`). Pieces keep uncompressed (PCM) audio, which ends exactly with the video, so they join without gaps. After each render, cache files that no version of that preset uses are deleted; each version lists what it used in `cache/<name>.used.json`, since versions share pieces.
  2. Each piece is measured with ffprobe and the segment map is built from those lengths.
  3. The pieces are joined with ffmpeg's concat demuxer (hard cuts). The video is **copied**, not encoded again: every piece (clips, cards, callout freezes) is encoded once, at the final quality, with the same encoder and settings (`soccer_clips/encode.py`). Only the audio is encoded at the join, to AAC. Every piece gets the same colour tags (bt709, limited range), written into the stream with the `h264_metadata` bitstream filter, so the copied pieces share one set of stream parameters.
- **Presets:** `preview_540p` (960×540, always x264 ultrafast crf 26) and `landscape_1080p` (1920×1080, encoder and quality from `.env`).
- **Encoder** (`.env`: `SOCCER_ENCODER`, `SOCCER_GPU_QP`, `SOCCER_CPU_CRF`, `SOCCER_CPU_SPEED`; measured examples in `.env.example`): `auto` uses the GPU's H.264 encoder through VA-API when a quick test encode works at the preset's size, otherwise x264 on the CPU; `gpu` fails instead of falling back. Previews always use x264: this machine's GPU (Radeon Vega, Mesa driver) hangs encoding 960×540 and 640×360, while 1080p and 720p work. Plain pieces from a 1920×1080, 29.97 fps source (Veo) are decoded on the GPU and never leave it (`scale_vaapi`). Pieces with a caption come back to the CPU for the overlay, because `overlay_vaapi` isn't supported on this GPU. Title cards and callout stills are drawn on the CPU and uploaded. Each render records the encoder it used (`Render.encoder`, e.g. "GPU qp 22"), shown in the version button's tooltip.
- **Deleted clips** are shown as "clip deleted" in the reel order list and skipped at render time (listed in `skipped`); the render doesn't fail.
- **API:** `GET/POST /api/reels`, `GET/PATCH/DELETE /api/reels/{id}` (PATCH takes `title` and/or the full `items` list), `POST /api/reels/{id}/render` with `{preset, variant}` (returns the running job if one is already going), `GET /reels/{id}/{name}.mp4` (served inline with a `<slug-of-title>-<name>.mp4` filename).
- **Reel page** (`/reel`, `/reel#reel=<id>&preset=<render name>&t=<s>`): a reel list, then for a reel: player, timeline coloured by clip (orange = her), "Clips in this video" chapter list (current one highlighted), "Reel order" list with ↑/↓/✕ and a link to each clip in the tagger, render buttons with progress and a "with" menu for the variant, a button and download link per version (switching keeps the place in the current clip, since versions can differ in length), and the out-of-date warning. Keys: Space, `[`/`]` speed, ←/→ ±5s, ↑/↓ previous/next clip, **G** go to the full game.
- **Jumping back:** G (or "game ↗" on a chapter) opens the tagger at `#game=<id>&t=<source time>&clip=<id>&back=<reel url>`. The tagger seeks there, selects the clip and shows "← back to reel"; the browser's Back button also returns to the same spot in the reel.
- Shared page code lives in `static/common.js` / `common.css`.
- **Added 2026-10-15:** under the reel title, each game in the reel with its date and description; clip notes on their own line in both "Clips in this video" and "Reel order"; ◎ on clips with a callout; a ▶ preview button on each clip in "Reel order" that plays it straight from the full-game video (no render needed), stops at the clip's end (Space replays), and returns to the same spot in the reel with "✕ back to reel" or `Esc` (`G` during a preview opens the tagger there). Changing `#reel=` in the URL now switches reels without a reload.

## Title cards (2026-10-14)
- **What's on them:** white text on a dark background (`#111`), centred, DejaVu Sans Bold (Pillow's built-in font if that's missing).
  - **Opening card:** the reel title, then underneath: for a one-game reel, the game's description (if set and not the same as the title) and its date, e.g. "Saturday, October 10, 2026"; for a multi-game reel, "N games · September 19 – October 10, 2026". Date and description are set in the tagger (added 2026-10-15).
  - **Card before each clip:** the time in the game ("23'", "2H 7'"; the video time as m:ss if that game has no kickoff set) and, underneath, the clip's note if it has one (wrapped, at most 4 lines at the normal size; longer notes are drawn smaller, down to half size, so they fit in the same space instead of being cut off).
- **Length (changed 2026-10-15):** each card stays up long enough to read: 0.75s to take it in, plus its text (all lines) at 17 characters per second, capped at 15s. (First tried 1s + 15 cps; on a real reel that felt a little long.) `Reel.card_s` is the minimum (default 2s), set in the "cards ≥ … s" box next to the render buttons; 0 turns all cards off. A card with only a time ("23'") gets the minimum; one with a 40-character note gets about 3.3s. Cards are a whole number of frames, with silent audio.
  - Why 17 characters/second: subtitle guides allow 17–20 for adults (Netflix's English guide is reported as 20 for adult and 17 for children's programmes; secondary sources) and silent reading of prose averages about 238 words/minute (Brysbaert 2019), which is roughly 20 characters/second. Cards are read cold, by young viewers too, alongside a glance at the time, so the children's rate plus 0.75s. Tune `READING_CPS` / `ORIENT_S` in `reel/cards.py`.
  - Reels made before this change keep their stored minimum (3s then); set the box to 2 for the new default.
  - Changing the timing rule marks existing renders "title card timing changed" (the measured card lengths in the time map no longer match).
- **Code:** `soccer_clips/reel/cards.py` draws them; the renderer caches each card like a clip (keyed on its text, length, preset and font).
- **Time map:** each `Segment` records `card_start_s` and `card_lines`; the `Render` records `start_card_lines`. On the reel page a clip's chapter starts at its card: ↑/↓ and the chapter list land on the card, the header shows "(card)", the timeline draws cards in grey, and `G` on a card goes to the start of the clip in the game.
- **Out of date:** a render is flagged if the card length, the title card, or a clip card's text (note or kickoff edited) changed since it was made.
- Possible later tweaks: separate lengths for the opening card and clip cards; show the game's description/date on clip cards in multi-game reels (the time alone is ambiguous there).

## Caption overlays (2026-10-15)
Per-reel setting **Descriptions** (`Reel.card_style`): `"overlay"` (default) or `"card"` (the clip cards above). The reel's opening card is a full card either way.
- **Overlay:** the clip's card text (game time + note) is drawn in a rounded, ~65%-opaque black box near the bottom centre of the video (time bold on top, note wrapped below, up to 3 lines, with longer notes drawn smaller (down to half size) so the box stays about the same height; `draw_caption` in `reel/cards.py`). It shows from the start of the clip for the same reading time a card would get (0.75s + 17 cps, at least `card_s`), and again during the callout freeze if the clip has one.
- **Render:** one more cached step after the clip and the callout splice: ffmpeg `overlay` with `enable=between(t,…)` windows. Each step is cached separately, so a note edit only redoes the caption pass. The time map records `caption_lines` / `caption_s`; `Render.card_style` records which style was used (renders from before this default to `"card"`).
- **Freeze default depends on the style:** `Reel.callout_s` is now optional; empty means the style's default (`DEFAULT_FREEZE_S`: 1.5s with overlays, where the caption is read during the freeze; 0.75s with cards). A typed value sticks across styles; clearing the box goes back to the default. The reel page shows the default as the box's placeholder.
- **Out of date:** switching style, editing a note/kickoff shown in a caption, or changing the timing rule flags the render.
- Existing reels have no `card_style` saved, so they now use overlays; their old renders are flagged "descriptions switched between overlay and title cards" until re-rendered.

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

**Does dropping audio help? (2026-10-15, same 15 × 10s reel, back-to-back runs)**
| Step | With audio | Video only |
|---|---|---|
| Final 1080p join | 265s, 154.0 MB | 263s, 150.8 MB |
| Normalise one 10s clip | 9.2s | 8.9s |

Audio is ~2% of the file (160 kbps AAC) and ~1% of the time (AAC is cheap next to x264). Not worth removing; keep it. (Absolute times were about twice the 2026-10-14 numbers because the machine was busier; only the with/without comparison matters here.)

**Render speed (2026-10-18), 18-clip reel from `gu11-vs-fc-stars-2026-09-27` (3:35, 1080p, overlay captions on every clip, 15 callouts), nothing cached**
| Pipeline | Time | Per minute of reel | File |
|---|---|---|---|
| Before: x264 veryfast intermediates, caption and callout steps each re-encoding the clip, final x264 medium re-encode | 533 s | 149 s | 190 MB |
| Pieces + copy join, x264 medium crf 20 | 361 s | 101 s | 230 MB |
| Pieces + copy join, GPU qp 22 | 98 s | 27 s | 296 MB |

Along the way: the same pipeline with the caption, callout and clip each re-encoding the one before took 231 s on the GPU; one ffmpeg pass per clip took 120 s; pieces one at a time took 133 s. Three pieces at a time gave 90 s, not worth oversubscribing the CPU path.

## Findings
- **2026-10-14** — Segment lengths are max(video, audio) of each intermediate, which is a few ms longer than the video, because the concat filter starts the next clip after the longest stream ends. The reel page allows 50 ms of slack at chapter boundaries; without it, seeking to a chapter start landed in the previous chapter.
- **2026-10-14** — Tested: 23 unit tests (including frame counts, A/V sync, segment mapping, cache reuse, out-of-date detection, a game with no audio) and a headless-Chrome run of the whole flow (tagger → new reel → render → ↓ → G → tagger at the right frame with the clip selected → Back → same reel position).
- **2026-10-14** — First preview reel from real footage: clip lengths felt right, so per-clip trim is lower priority.
- **2026-10-14** — Seeking exactly to a chapter boundary could show the previous clip's last frame, so chapter jumps land half a frame in.
- **2026-10-14** — Tested: 25 unit tests (adds card layout, frame counts with cards, card text, out-of-date on note/title/length change, long-note wrapping) and a headless-Chrome run of the card behaviour.
