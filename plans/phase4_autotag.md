# Phase 4 (Stretch) — Auto-tag Her

**Status:** Not started. Depends on Phase 3 having collected labelled examples from a few games.
**Goal:** Automatically suggest where she appears, and pre-select her in clips, without depending on her jersey number. Suggestions with confirm/reject, not fully automatic.

## Why this can beat Veo
Veo's player tagging appears to lean heavily on jersey numbers, which are often hidden. We can use **everything else**: build, hair (colour, ponytail), cleats/socks, skin tone, movement, and the context that she's on our team. And we train specifically on her, from your own clicks, so it gets better every game.

## Data
From Phase 3 (`player_gallery/`):
- **Positives:** crops of her from callouts, track selections, and every frame of confirmed tracks (sampled so one clip doesn't dominate).
- **Hard negatives:** teammates in the same frames (same kit).
- **Easy negatives:** opponents, referees.
Track per-game counts; aim for examples from several games and lighting conditions before judging accuracy.

## Pipeline
1. **Detect + track the whole game** (or a time range) with YOLO11 + BoT-SORT at a reduced sample rate (e.g. 5 fps). Output: tracklets (track ID → boxes over time). A 90-min game at 5 fps is ~27k frames — background job; benchmark it.
2. **Team filter:** cluster torso colours (k-means on the upper half of each box) to keep only our team's tracklets. This removes most candidates cheaply.
3. **Embed** several crops per tracklet with a re-ID model:
   - **OSNet** (via BoxMOT / torchreid) — built for person re-ID, small, CPU-friendly.
   - **DINOv2-small** — general-purpose features; may capture hair/build better. Compare both.
4. **Score** each tracklet vs her gallery: max / top-k mean cosine similarity, or a small classifier (logistic regression / linear probe) trained on her positives vs teammate negatives.
5. **Optional extra signal: jersey number OCR** (e.g. PARSeq on the torso/back crop) when the number is visible — a strong positive when it fires, ignored when it doesn't.
6. **Combine** into a per-tracklet "is her" score; smooth across time.

## Outputs / UI integration
- **"Her on screen" timeline** on the game page (Phase 1) — jump to stretches where she's visible.
- **Clip suggestions:** clips where she's visible, ranked by score. Confirm/reject in one key press; each answer is added to the gallery (active learning).
- **Auto-select in Phase 3:** when tracking a clip, pre-pick the most likely track ID as her. You only click if it's wrong.
- **Auto-tag existing clips:** set/suggest her number (the default player) on clips.

## Even-more-stretch: event suggestions
- Ball detection (YOLO's `sports ball` class is weak on small, distant balls; may need a fine-tuned model) → moments when the ball is near her = candidate clips.
- Audio peaks (cheering) as a cheap "something happened" signal.

## Evaluation
- Hold out one fully hand-labelled game.
- Tracklet-level precision/recall for "is her", and how many correction clicks Phase 3 needs with vs without auto-select.
- Track results per game in the table below so we can see whether more data helps.

## Tasks
- [ ] Gallery index format + sampling (written during Phase 3)
- [ ] Full-game tracking job at reduced fps; benchmark
- [ ] Team colour clustering filter
- [ ] OSNet embeddings; DINOv2 embeddings; compare
- [ ] Scoring / linear probe
- [ ] "Her on screen" timeline UI
- [ ] Confirm/reject suggestion flow feeding the gallery
- [ ] Auto-select her track in Phase 3
- [ ] *(Optional)* jersey OCR signal
- [ ] Hold-out evaluation

## Open questions
- How many labelled games before this is worth trying? (Guess: 2–3.)
- Does her appearance change much between games (kits, hair, cleats)? Home/away kits mean the team colour filter needs per-game settings.
- Acceptable full-game processing time on CPU (overnight is fine?).

## Benchmarks / findings
| Game | Positives | Tracklet precision | Tracklet recall | Notes |
|---|---|---|---|---|
| | | | | |
