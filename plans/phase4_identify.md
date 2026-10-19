# Phase 4 — Identify Players & Track the Ball

**Status:** Not started. Re-scoped 2026-10-18 from "auto-find her" to "name every player on our team and track the ball", to power two features: highlighting any chosen player ([Phase 3.3/3.4](phase3_highlight.md)) and finding clips where a player is near the ball (4.4).
**Depends on:** the Phase 3.2 whole-game tracking job (tracklets per game).

## What this phase produces
For each game, on top of Phase 3.2's tracklets:
- `ball.npz`: the ball's position per frame (with confidence, and gaps where it's unknown).
- `tracklets.json` labels: team (ours / opponent / referee) and, for our players, which player.
- From those: clip suggestions where a chosen player is close to the ball.

Everything is **suggest, then confirm**: the app proposes labels and clips, you accept or fix them with one key or click, and each answer becomes training data for the next game.

## Roster
A team roster in `$SOCCER_GAMES_DIR/roster.json`: number, name (or nickname), optional notes (hair, cleats). Per game, who played (default: everyone on the roster). Numbers already tag clips (`F`, `P`), so a roster entry is just a name for a number. Kids sometimes change numbers between seasons, so entries carry a season.

## 4.1 — Ball tracking
The ball is hard to *find* (5–15 px, motion blur, hidden by legs, look-alikes: heads, cones, line marks, balls on the next field) but easy to *identify* (there's one). So the work is in detection and filtering.

1. **Stock detector first:** YOLO11's COCO `sports ball` class, in the same pass as players (Phase 3.2), at the highest imgsz the CPU budget allows. Measure recall on hand-checked frames. Expect it to be weak on distant balls.
2. **Cheaper high-res option:** the Veo camera follows the play, so the ball is usually near the centre. Run a second, higher-resolution pass only on a centre crop (e.g. the middle 50% of the frame at 1280), instead of the whole frame.
3. **Fine-tune if needed:** label ball positions on a few hundred frames from our own games (a click-the-ball labelling page in the app, same as callouts), fine-tune a small YOLO on 1–2 classes. Training on this CPU is slow but feasible for a nano model over a few hours; or label here and train once on a rented/borrowed GPU (training frames would then leave the house, which conflicts with "everything runs locally", so ask first).
4. **Temporal filter:** a Kalman filter / simple physics (max speed, mostly smooth paths, sudden direction changes only near a player) keeps the one most plausible ball per frame, throws out look-alikes, and fills short gaps. Mark filled frames as interpolated.
5. **Possession hint:** when the ball is hidden but was last seen at a player's feet, it's probably still with that player; good enough for "near the ball".

**Tasks**
- [ ] Measure stock `sports ball` recall/precision on hand-checked frames (part of the Phase 3.2 benchmark)
- [ ] Centre-crop high-res pass; compare
- [x] Ball labelling page (2026-10-18): `/label-ball`, sets `bench` (149 frames over the bench clips, for scoring) and `train` (random game frames away from the bench clips); `soccer-clips ball sample/score`. Labels in `labels/ball.jsonl`.
- [ ] Label the bench set and score the variants (`soccer-clips ball score --results results-l4`)
- [ ] Fine-tune a nano ball model (if stock isn't good enough)
- [ ] Temporal filter + gap filling → `ball.npz`
- [ ] Ball overlay in the tagger to check it by eye

## 4.2 — Team split
Cluster torso colours (k-means on the upper half of each box, in a colour space that handles shade, e.g. Lab) per game into our team / opponents / referees. Cheap, and it removes most candidates before identification. Home/away kits differ, so the clusters are per game, with one click to say "this cluster is us".

**Tasks**
- [ ] Torso colour clustering per tracklet, per game
- [ ] "Which one is us" confirm in the UI

## 4.3 — Player identity
Turn many short, anonymous tracklets of our team into named players.

**Signals**
- **Jersey number OCR** (e.g. PARSeq on the back/torso crop) when the number is visible: a strong label when it fires, ignored when it doesn't. Numbers are often hidden or too small, so this alone isn't enough.
- **Appearance re-ID embeddings:** OSNet (via BoxMOT / torchreid; made for person re-ID, small, CPU-friendly) and DINOv2-small (general features; may capture hair/build better). Compare both. Teammates wear the same kit, so this relies on build, hair, skin, socks, cleats.
- **Constraints:** two tracklets that overlap in time can't be the same player; one player can't be in two places; the number of our players on the field is at most 11 (or the format's number). These clean up a lot of mistakes.
- **Your labels:** correction clicks from Phase 3.3, callouts on clips tagged with a number, and confirm/reject answers here.

**Pipeline**
1. Embed several crops per tracklet (our team only, from 4.2).
2. Link tracklets within a game into longer chains using embedding similarity + time/position constraints.
3. Name chains: OCR hits, your labels, and similarity to each player's gallery from earlier games (a small per-player classifier / linear probe once there are enough examples).
4. Score each label; low-confidence ones go to a review queue.

**Review UI:** a grid of tracklet thumbnails per player, "is this #38?" yes/no with one key, sorted by least confident first. Each answer is saved to the gallery.

**Gallery** (`player_gallery/`): crops saved as image files, each with a line in `player_gallery/index.jsonl` (game, frame, box, player or team label, source = callout / correction / review / ocr). Teammates are hard negatives for each other. Append-only JSONL so it can grow to many thousands of entries.

**Tasks**
- [ ] Roster file + per-game "who played"
- [ ] Gallery index + crop saving (from callouts, corrections, reviews)
- [ ] OSNet embeddings; DINOv2 embeddings; compare
- [ ] Tracklet linking within a game (similarity + constraints)
- [ ] Jersey OCR signal
- [ ] Per-player scoring / linear probe across games
- [ ] Review queue UI

## 4.4 — Clips where a player is near the ball
For a chosen player, find the stretches of a game where they're involved with the ball and offer them as clips.

1. Per frame where both are known: distance from the ball to the player's feet (bottom-centre of the box), **divided by the player's box height**. Box height is a stand-in for distance from the camera, so "within 1.5 heights" means roughly the same real distance near and far. Image-space only; no pitch calibration needed.
2. "Near" = under a threshold (start ~1.5 heights, tune on real games) for at least ~0.5s; join runs separated by short gaps; pad by a few seconds before (the build-up) and after.
3. Score each run (time near the ball, closest distance, ball speed change near them = a touch/kick) and list them in the tagger as **suggested clips**: watch, then accept (becomes a normal clip tagged with the player's number) or reject with one key.
4. Also: suggest the player's number on existing clips they appear near the ball in.

**Tasks**
- [ ] Proximity per frame + run detection + scoring
- [ ] Suggested-clips list in the tagger (accept / reject)
- [ ] Suggest player tags on existing clips

## Even-more-stretch
- Event types from the same data: shots (ball speed + direction toward goal), passes (ball moves from one of our players to another).
- Audio peaks (cheering) as a cheap "something happened" signal to rank suggestions.

## Evaluation
- Hold out one fully hand-checked game.
- Ball: % of frames with the ball found within a few px, false positives per minute.
- Identity: tracklet-level precision/recall per player, and how many review answers a game needs.
- Proximity clips: of the clips you'd have tagged by hand for that player, how many are suggested, and how many suggestions you reject.
- Track results per game in the table below so we can see whether more data helps.

## Open questions
- How many labelled games before identity is worth trying? (Guess: 2–3.)
- Does appearance change much between games (kits, hair, cleats)?
- Is training a ball model on this CPU fast enough, or is it worth labelling here and training elsewhere?
- Acceptable full-game processing time (overnight is fine?).

## Findings
- **2026-10-18** — Stock YOLO11 `sports ball` on the 7 bench clips (Phase 3.2 benchmark; details in `bench/results/*/summary.json`):
  - **Resolution matters most.** Share of frames with a ball detection at conf ≥ 0.25, excluding elite-shot (a spare ball sits by the camera): n640 0–2%; s640 0–23%; n1280 5–46%; s1280 2–38%; s1280 on the centre 1280×720 crop at native resolution 1–53% (best on 5 of 6 clips). The crowded corner (ebu-corner) is the worst for every 1280 variant (1–24%).
  - **Detections are mostly real.** Checked by eye on random crops from s1280-centre: ~35/40 real at conf ≥ 0.25, ~32/42 real at 0.1–0.25. False hits: corner flags, bags, a white cleat, blonde ponytails/heads, a water bottle, a spare ball on the sideline (stationary, so a temporal filter can drop it). Weak detections are worth keeping for the temporal filter.
  - **Recall is unknown** without ground truth: frames with no detection may have the ball out of the crop, hidden, in the air, or just missed. Labelling the ball on ~150 frames would measure it.
  - **Cost on CPU:** s1280-centre is 2.1 fps, so ~18 h per game every frame, ~6 h every 3rd frame (too sparse for a kicked ball). Not practical on this CPU; see [hardware.md](hardware.md).
  - Untested at first: full frame at native resolution, bigger models. Tested on an L4 GPU the same day (below).
- **2026-10-18, L4 GPU** (`bench/results-l4/`): added s1920, m1280-centre, m1920.
  - **m1280-centre is the best ball variant:** frames with a ball at conf ≥ 0.25 per clip: fc1-passing 52%, fc1-far-switch 34% (vs 16% for s1280-centre), ebu-goal-kick 56%, stars-defense 39%, saa-far-hustle 13%, ebu-corner 1%. Its weak detections (0.1–0.25) were ~38/42 real by eye (misses: a head, a ponytail, a bottle).
  - **Full frame (s1920/m1920) doesn't beat the crop for the game ball**, except in the crowded corner (9% vs 1%, so the ball was probably outside the crop there). It also finds **spare balls lying near the camera** (ebu-corner, elite-shot) with high confidence; they don't move, so the temporal filter must drop stationary balls.
  - Still the case: recall unknown without labelled frames; the crowded corner is the weak spot for every variant. Likely setup: players on the full frame (m1920) and the ball on the centre crop (m1280-centre), plus the full frame's ball detections as a fallback near the edges.

## Benchmarks / findings
| Game | Ball recall | Ball false pos/min | Identity precision | Identity recall | Review answers | Notes |
|---|---|---|---|---|---|---|
