# Phase 4 — Identify Players & Track the Ball

**Status (2026-10-24):** 4.1 ball tracking built and in use: every game has a whole-game ball track (fine-tuned YOLO26x detections from RunPod, then the temporal filter and fixed-object blocking here), shown and corrected in the tagger. On the labelled game frames: found 66%, wrong 10%, missed 24%. 4.2 team split built 2026-10-25 (kit colour per box, used in tracking and to name teams); 4.3–4.4 not started. Re-scoped 2026-10-18 from "auto-find her" to "name every player on our team and track the ball", to power two features: highlighting any chosen player ([Phase 3.3/3.4](phase3_highlight.md)) and finding clips where a player is near the ball (4.4).
**Depends on:** 4.2–4.4 need the Phase 3.2 whole-game player tracking (tracklets per game); 4.1 turned out not to (the ball runs its own pass).

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
- [x] Measure stock `sports ball` recall/precision on hand-checked frames (2026-10-19: best stock 57% found, YOLO11m on the centre crop)
- [x] Centre-crop high-res pass; compare (the 1280×720 native-resolution centre crop is what the ball model runs on)
- [x] Ball labelling page (2026-10-18): `/label-ball`, sets `bench` (149 frames over the bench clips, for scoring) and `train` (random game frames away from the bench clips); `soccer-clips ball sample/score`. Labels in `labels/ball.jsonl`.
- [x] Label the bench set and score the variants (`soccer-clips ball score --results results-l4`): done 2026-10-19, see Findings
- [x] Guess mode for labelling (2026-10-19): `soccer-clips ball guess train` stores YOLO11m-centre's best guess per frame; Enter accepts it. Never shown for bench.
- [x] Train set: 500 frames (seed 1), ~100 per game, half from tagged clips, none within 10s of a bench clip (2026-10-19)
- [x] Label the train set (2026-10-19): 318 visible, 70 hidden, 71 none, 41 can't tell; of 237 model guesses, 169 accepted and 68 wrong (spare balls, heads, etc.: useful hard negatives)
- [x] Training pipeline (2026-10-19): `soccer-clips ball export/train/eval`. Export: 1280×720 native crops (centre + one random crop around the ball), boxes from your sizes or a fitted size-vs-height line, the ball kept as COCO class 32 so the pretrained head carries over; 700 train / 39 val images from separate frames. Eval on the bench frame images reproduces the clip-based score (stock YOLO11m centre: 57% / 80%).
- [x] Fine-tune a ball model on a rented GPU, score on the bench labels (2026-10-19/21: YOLO26x, run 2 with other balls boxed: 76% of visible game balls, 92% of detections real)
- [x] Temporal filter + gap filling (2026-10-21): `track/camera.py` (camera motion), `track/ball_track.py` (tracklets + DP chain + fill), `soccer-clips ball track`.
- [x] Whole-game ball tracks (2026-10-22/23): `soccer-clips gpu run ball-games` (RunPod, videos on the network volume) into each game's `tracks/ball_raw.npz` + `ball.npz`
- [x] Ball overlay in the tagger to check it by eye (2026-10-22): `B`; debug layer of every detection `D` (2026-10-23)
- [x] Fixes while watching (2026-10-22/23): `K` / `Shift+K` / `K` then `O`; each fix also a labelled frame, auto-assigned to train or bench
- [x] Whole-game filter fixes and yardstick (2026-10-23): stillness over short spans, set pieces only when watched, `soccer-clips ball eval-games`
- [x] Fixed objects blocked (2026-10-23/24): `track/world.py`, `soccer-clips ball fixed-objects`
- [ ] Fine-tune again once enough fixes have become labels; re-run ball-games with the new model (needs a go for the GPU run)
- [ ] Possession hint (5 above): not done; may come with 4.4

## 4.2 — Team split
Cluster torso colours (k-means on the upper half of each box, in a colour space that handles shade, e.g. Lab) per game into our team / opponents / referees. Cheap, and it removes most candidates before identification. Home/away kits differ, so the clusters are per game, with one click to say "this cluster is us".

**As built (2026-10-25)** — kit colour also steers the tracker. That's why it was done now: tracks were lost or swapped where she crossed an opponent or the ref (85 of 128 of her labelled stretches ended with another box overlapping hers).
- **Kit colour per box** (`track/kit.py`, `soccer-clips people kit`): two colour histograms per detected box, torso (middle half of the width, 20–50% of the height) and shorts (30–70%, 52–68%), each 35 bins: 8 hues × (tinted, saturated, vivid bright, vivid dark) plus dark/grey/light, leaving out pixels near that moment's grass colour; square-rooted (Hellinger), uint8, in `tracks/people_kit_<model>.npz` row for row with the raw boxes. Decoded at half size; ~7–9 min a game on the 4 cores. One box's kit is noisy, so the steps below use running or windowed means. (First version, 2026-10-25 morning: torso only, 19 bins with one saturation level. Revised the same day after PRSC, below.)
- **In the tracker** (`Params.kit_gate`, 0.65 with the shirt + shorts kit; 0.45 with the first one): a track can't take a box whose kit is further than the gate from the track's running kit, in both of BoT-SORT's matching passes (the second, for weak boxes, is where crossings happen). Ultralytics' own re-ID only ever lowers costs, never blocks, so it's our own cost function.
- **Cut on a kit change** (`split_kit`, 0.45; 0.35 with the first kit): a tracklet whose mean kit before and after some frame differs by more than this is cut there.
- **Join across gaps** (`link_gap_s` 1 s, `link_kit` 0.2, `link_heights` 1): a tracklet that starts within 1 s of another ending, near where that one would be by then (camera motion cancelled), in the same kit, continues it. Closest first, one-to-one. The kit threshold drives the trade-off: looser joins more (longer holds) but also joins teammates (more conflicts).
- **Teams** (revised 2026-10-25 after PRSC came out wrong: their royal blue players drawn as ours). Each kit shows up as several colour clusters depending on the light: our white kit is white in sun, blue-tinted in shade, and from behind largely its big blue number; their blue is vivid in sun, dark in shade. So:
  - tracklet colour *shares* (weighted towards big boxes: a 30 px player counts a quarter) clustered per game into 10 clusters (Euclidean k-means, so the main colours decide), saved with the tracklets; `soccer-clips people teams` redoes it in seconds;
  - every box gets the cluster nearest its tracklet's colour over the half second around it (`kit.box_teams`), so a tracklet that swapped players changes cluster where it swapped;
  - clusters sharing tracklets (a player running from sun into shade) are taken as the same kit (`kit.affinity`, ≥ 0.15);
  - **ours** = clusters where the default player's labelled boxes (every box of the stretches her picks label, thousands, in every light) make up at least a quarter of the share they make up in her top cluster. She's one of ~9 in our kit, so every one of our clusters holds a steady share of her; the opponents' hold only a stray swap's worth (PRSC: 1.1% and 0.6% in our two clusters, 0.08% in their sunlit one). With fewer than 200 labelled boxes, the clusters under her labelled points;
  - **opponents** = the biggest other cluster and its same-kit clusters; **other** = the rest (refs, keepers, spectators). Without labels, the two biggest kit groups are "team";
  - ours needs ≥ 12% of the top cluster's share (PRSC: ours ≥ 17%, theirs ≤ 7%; fc-1: ours ≥ 97%, theirs ≤ 3%), and opponents are the biggest group of the remaining clusters linked at ≥ 0.2 (at 0.15 groups chained through mixed clusters and swallowed one of ours and the refs);
  - finally each box takes the most common team among its tracklet's boxes within ±5 s (`people.box_roles`), so a white shirt that looks blue in deep shade for a moment doesn't flicker, while a swap that lasts does. The tagger draws that (white / blue / grey dashed).
  - What it took on PRSC: the vivid saturation level (their blue pixels: median saturation 228; our white kit's blue-cast pixels: 98, 90% below 172), the shorts (white vs blue), share-based clustering (square roots made a 0.4 share look like a 0.2 one), and labelled boxes instead of points (several of her labelled points sat on an opponent's overlapping box). At 36:30.9 and 25:00 every player on the pitch then came out right.
- **Yardstick from the labels** (`soccer-clips people eval-labels`): known points = her "is" picks + #38 callouts (215); hold = how long the tracklet under each lasts; conflicts = a tracklet under one point passes another of her points without covering it (swapped or wrongly joined); swaps left = "not" picks (29) still on a tracklet that was hers earlier.

**Tasks**
- [x] Torso colour per box (2026-10-25)
- [x] Kit gate, cut and join in tracking, tuned on the labels (2026-10-25)
- [x] Torso colour clustering per tracklet, per game; "ours" from the labels (2026-10-25)
- [ ] "Which one is us" confirm in the UI (only needed for games without labels; there the two biggest clusters are "team")
- [ ] Use teams to drop spectators / refs from the follow and near-the-ball features

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
- **2026-10-25** — Kit colour in tracking, tuned against the #38 labels (215 known points, 29 swaps; `people eval-labels`). Tracking cached per gate, cut/join scored on top:

  | Setting | Hold median / mean (s) | Conflicts | Swaps left |
  |---|---|---|---|
  | none (before) | 18.2 / 26.4 | 60 | 15/29 |
  | gate 0.45 | 16.1 / 25.2 | 34 | 6/29 |
  | gate + cut 0.35 | 14.4 / 21.7 | 27 | 5/29 |
  | gate + cut + join (kit 0.15) | 17.4 / 25.1 | 36 | 5/29 |
  | **gate + cut + join (kit 0.2), chosen** | **23.2 / 30.1** | **45** | **5/29** |
  | gate + cut + join (kit 0.3) | 28.0 / 36.4 | 50 | 5/29 |

  Re-tuned with the shirt + shorts kit (same labels; baseline without kit 18.2 / 26.4 s, 60 conflicts, 15/29 swaps):

  | Setting | Hold median / mean (s) | Conflicts | Swaps left |
  |---|---|---|---|
  | gate 0.55, cut 0.35, join 0.2 | 21.0 / 27.6 | 50 | 7/29 |
  | gate 0.65, cut 0.35, join 0.2 | 19.7 / 26.4 | 42 | 7/29 |
  | **gate 0.65, cut 0.45, join 0.2, chosen** | **21.9 / 28.8** | **45** | **8/29** |
  | gate 0.65, cut 0.45, join 0.3 | 28.1 / 36.4 | 60 | 8/29 |

  A little behind the torso-only kit for tracking (23.2 s, 5/29 swaps; the shorts are noisier frame to frame), but that kit couldn't tell our white from PRSC's blue. Saved tracks: 21.7 / 28.8 s, 45 conflicts, 8/29; callouts 99% found, coverage 72% / 66%. If tracking needs the last bit back, the gate could use the shirt half only.

  The gate alone halves conflicts and swaps; joining buys back length at some cost in conflicts (joins onto teammates in the same kit, which only appearance re-ID can tell apart). Join distance (0.5 vs 1 box height) hardly matters. Saved tracks with the chosen settings: hold 23.1 / 30.1 s, 45 conflicts, 5/29 swaps. Callout coverage went from 75% / 68% to 71% / 63% (median / mean; the old figure counted swapped tracks as coverage). Re-tracking all six games takes 29 min with 4 processes.
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

- **2026-10-19 — Scored against hand labels** (149 bench frames: ball visible in 120, hidden in 17, no ball in 9, can't tell 3). Recall = share of visible-ball frames with a detection within 10 px (or 2 ball radii) of the label; precision = share of detections that are on a labelled ball. Labels and detections line up (median 1.5 px when matched), so these are real numbers.

  | Variant | Recall @0.1 | Precision @0.1 | Recall @0.25 | Precision @0.25 |
  |---|---|---|---|---|
  | n640 | 0% | 0% | 0% | 0% |
  | s640 | 6% | 17% | 2% | 8% |
  | n1280 | 16% | 15% | 8% | 14% |
  | s1280 | 39% | 48% | 24% | 49% |
  | s1920 (full, native) | 45% | 50% | 28% | 52% |
  | m1920 (full, native) | 42% | 51% | 21% | 52% |
  | s1280-centre | 50% | 76% | 41% | 78% |
  | **m1280-centre** | **57%** | **79%** | 38% | 75% |

  - The **centre crop wins on both counts**; the full frame loses precision mainly to spare balls lying near the camera (elite-shot, fc1-passing bottom-left). Those don't move, so a temporal filter can remove them.
  - **Best stock result: about 57% of visible balls found** (m1280-centre at conf ≥ 0.1, ~4 in 5 detections real). The s model on the crop is close (50%), and runs on the CPU at 2.1 fps.
  - **Hidden balls are almost never found** (1 of 17), as expected; that's the temporal filter's job (possession hint, gap filling).
  - Low thresholds pay off: going from 0.25 to 0.1 adds ~10–20 points of recall at similar precision.
  - What this means: a stock detector misses ~4 in 10 visible balls even at its best, too many gaps for "near the ball" to be reliable from detection alone. Next: the temporal filter on the current detections (cheap, shows how far gaps can be filled), and fine-tuning YOLO on our own ball labels (the train set).

- **2026-10-19 — First cloud fine-tune** (RunPod RTX 4090, 35 min, ~$0.52; `gpu/20261019-214518-ball-train/`). Scored on the 149 bench frames (centre crop, 1280):

  | Model | Found @0.1 | Real @0.1 | Found @0.25 | Real @0.25 | Found @0.4 | Real @0.4 |
  |---|---|---|---|---|---|---|
  | stock YOLO11m (previous best) | 57% | 80% | 42% | 80% | 30% | 78% |
  | stock YOLO11l | 68% | 79% | 62% | 86% | 53% | 87% |
  | stock YOLO11x | 66% | 78% | 56% | 80% | 42% | 80% |
  | stock YOLO26m | 71% | 70% | 62% | 82% | 56% | 83% |
  | stock YOLO26l | 64% | 78% | 61% | 84% | 52% | 83% |
  | stock YOLO26x | 75% | 75% | 71% | 84% | 64% | 83% |
  | fine-tuned YOLO11m | 62% | 95% | 51% | 100% | 41% | 100% |
  | **fine-tuned YOLO26x** | **81%** | 66% | **78%** | 80% | **75%** | **92%** |

  - **YOLO26x is the base to use**: best stock (75%) and best after fine-tuning. At conf ≥ 0.4 the fine-tuned model finds 3 in 4 visible balls with ~92% of detections real, vs 30% / 78% for stock YOLO11m at the same threshold.
  - Fine-tuning mostly made YOLO11m *more precise* (95–100% real) and YOLO26x *find more* at the higher thresholds.
  - **The kept checkpoints are undercooked**: ultralytics keeps the epoch with the best box-fit "fitness" (0.9 × mAP50-95) on a 39-image / 31-ball validation split, which picked epoch 2 (YOLO11m) and epoch 6 (YOLO26x), and early stopping (patience 30) ended training at epochs 32 / 36. Validation mAP50 is noisy (0.6–0.82) and peaks around epochs 10–22. Next run: a bigger validation split, no early stopping on box fit, checkpoints every few epochs, and pick the checkpoint by our own found/real score on the validation frames (never the bench).
  - Not yet measured: YOLO26x speed on a whole game (it's the biggest model; on a GPU it should still be fine), and how much the temporal filter adds on top.

- **2026-10-19 — Other real balls.** Labels marked only the game ball, so spare balls were negatives in training (teaching the model that real balls aren't balls) and mistakes in scoring. Now: `labels/ball_others.json` holds other real balls per frame (from `soccer-clips ball others find` + Y/N review at `/review-balls`, or added by hand with `O` on the labelling page). Training boxes them; scoring counts detections on them as fine and reports junk (non-ball detections) per frame separately. Checkpoint selection fixed for the next run: 20% validation split, no early stopping, a checkpoint every 5 epochs, keep the best by harmonic mean of found/real at conf 0.25 on the validation frames.

- **2026-10-20 — Re-scored with other balls counted as real** (bench: 80 other balls confirmed, 72 of them added by hand; train: 270; only 30 of the model's 141 candidates were real balls). At conf 0.25 on the bench frames:

  | Model | Game ball found | Real (any ball) | Junk / frame | Other balls found |
  |---|---|---|---|---|
  | stock YOLO11m | 42% | 96% | 0.02 | 11 |
  | fine-tuned YOLO11m (run 1) | 50% | 98% | 0.01 | 0 |
  | stock YOLO26x | 71% | 98% | 0.01 | 15 |
  | fine-tuned YOLO26x (run 1) | 77% | 83% | 0.14 | 5 |

  - Most "false" detections before were real balls: full-frame YOLO11m goes from 51% to 85% real.
  - Run 1's fine-tuning taught the models to ignore spare balls (other balls found dropped from 11 → 0 and 15 → 5), and YOLO26x traded junk for found balls. Run 2 trains with spare balls boxed and picks its checkpoint on found/real.

- **2026-10-21 — Run 2: YOLO26x fine-tuned with other balls boxed** (RunPod RTX 4090, 39 min, ~$0.58; `gpu/20261021-143240-ball-train/`; two earlier attempts died in ~2 min each, ~$0.07 total: YOLO26x at 1280 runs out of memory at batch 8 on 24 GB and ultralytics' in-process retry at a smaller batch left CUDA broken; now batch 4). Checkpoint chosen on the 78 validation frames: epoch 45 (validation found 85%, real 96%); validation scores plateau from epoch 15 to 50 (0.86–0.905), and ultralytics' own "best" was an early epoch again. Bench, at each confidence:

  | Model | conf | Found | Real | Junk / frame | Other balls found |
  |---|---|---|---|---|---|
  | stock YOLO26x | 0.25 | 71% | 98% | 0.01 | 15 |
  | run 1 | 0.25 | 77% | 83% | 0.14 | 5 |
  | **run 2** | 0.25 | 77% | 88% | 0.10 | 19 |
  | stock YOLO26x | 0.4 | 64% | 98% | 0.01 | 14 |
  | run 1 | 0.4 | 74% | 93% | 0.05 | 2 |
  | **run 2** | **0.4** | **76%** | **92%** | 0.06 | 18 |

  - Run 2 finds spare balls again (18–19, more than stock) instead of ignoring them, with the same or better game-ball rate than run 1 and less junk.
  - Against stock YOLO26x, fine-tuning buys +6 (conf 0.25) to +12 (conf 0.4) points of game balls found, for ~0.05–0.1 junk detections per frame, which the temporal filter should absorb.
  - Validation (85–88% found) beats the bench (76–78%): the bench clips include the hardest cases (crowded corner, far side by a dark tree line).
  - `models/ball_yolo26x.pt` is now run 2; run 1's model is kept in its run folder.

- **2026-10-21 — Temporal filter v1.** Camera motion from optical flow on background corners (RANSAC similarity fit, Veo logo masked); checked by warping later frames onto the first (lines, poles, goal line up). Detections are linked into tracklets, a chain is chosen by dynamic programming (confidence, missing frames, implausible jumps, parked balls penalised), gaps up to 1 s filled. On the bench, % of visible game-ball frames:

  | Detector | Method | Found | Wrong | Missed |
  |---|---|---|---|---|
  | stock YOLO11m centre | top detection | 52% | 7% | 41% |
  | stock YOLO11m centre | **filter** | **72%** | 3% | 24% |
  | stock YOLO11s centre | top detection | 45% | 6% | 49% |
  | stock YOLO11s centre | filter | 57–60% | 3% | 37–39% |
  | stock YOLO11m full frame | top detection | 31% | 22% | 47% |
  | stock YOLO11m full frame | filter | 47% | 12% | 41% |

  About 30% of the found frames are filled gaps; a quarter of the hidden balls are found by interpolation. Of the misses left (YOLO11m centre), 31 of 41 had no detection on the ball at all (17 on the far side by the tree line, 9 in the crowded corner): a detector problem, which the fine-tuned model should help. Fine-tuned YOLO26x detections on every bench frame are being computed on the CPU (`bench run ft26x-centre`, ~4 h).

- **2026-10-21 — Filter with the fine-tuned detector** (YOLO26x run 2 on every bench frame, CPU, 2.6 h). % of visible game-ball frames:

  | Detector | Method | Found | Wrong | Missed |
  |---|---|---|---|---|
  | stock YOLO11m centre | filter | 72% | 3% | 24% |
  | fine-tuned YOLO26x centre | top detection | 58% | 20% | 22% |
  | **fine-tuned YOLO26x centre** | **filter** | **82%** | **5%** | **12%** |

  - The far-side clip (saa-far-hustle) went from 10 to 17 of 25 labelled frames found; its path covers 63% of the clip (was 28%).
  - With the fine-tuned model also finding spare balls, the filter first fell back on a spare when the game ball went undetected (5 of 12 wrong positions, mostly in the crowded corner). Balls still for 3 s or more are now never used, and a chain worth less than "no ball" isn't chosen: wrong 10% → 5% with no loss of found frames.
  - Lowering the detection threshold doesn't help: in the long empty stretches the stock model sees nothing near the ball even at 0.005, and the filter at ≥ 0.05 found less (68%) with more wrong (6%). A 2x zoomed tile let the fine-tuned model see 4 of 9 missed balls in the crowded far-goal corner (0 at normal size), but not on the far side (6 vs 9 of 15). Options: a zoomed pass over the far band of the field, targeted labels (far side, crowded goalmouths), manual anchors in clips that matter.

- **2026-10-22 — Set pieces.** The filter decides a whole clip at once, so it can look back in time: a ball parked 3 s+ is now credited when the chosen path leaves it with a kick (within 0.5 s, even if the kicker hides the ball for a few frames) or arrives at it as the ball comes to rest. The kicked/arriving piece must actually travel (≥ 40 px), otherwise a spare ball briefly hidden and seen again looked like a kick (wrong went back to 10% until that check). Bench unchanged at 82% / 5% / 12%; synthetic tests cover a 5 s free-kick wait with the kick hidden 0/4/8 frames, a ball rolling to a stop, and a spare ball blinking.

- **2026-10-22 — Whole-game tracks and fixes while watching.** First whole game (ebu, 60 min) on a US RTX 5090: 65 min, $1.28; ball on 59% of frames (44% seen, 15% filled), mostly 50–90% per minute in open play. The GPU sat idle: one CPU core (decoding + camera motion) was the limit, so long ranges now run as parallel chunks. In the tagger, `K` / `Shift+K` fix the ball (placed ball wins its frame; a wrong ring's run of detections is suppressed; path redone ±2 min in ~2 s) and each fix becomes a labelled frame, assigned automatically: training, except one 30-s moment in five and anything within 10 s of a bench clip go to the benchmark. As the benchmark grows with hard frames, re-score older models on it before comparing.

- **2026-10-23 — All six games tracked end to end** (7.6 h of video; fine-tuned YOLO26x, full precision). Ball on 60–76% of frames per game (43–57% seen, 10–24% filled), 68% overall, counting out-of-play time and stretches with no ball in view. Fixes made while watching (`K`) feed the next fine-tune.

- **2026-10-23 — Still objects through a drifting camera estimate.** In PRSC (22–24 min) the path kept landing on a white flag/towel lying behind the far goal line (conf 0.6–0.8, above the real ball's). It never moves, but over minutes the chained camera estimate slips during pans and zooms, so in the reference it seemed to wander 100–180 px and failed the "still within 12 px" test. New yardstick: `soccer-clips ball eval-games` scores the whole-game paths, without hand fixes, at every labelled frame in the games (580 visible, 94 no-ball). Bench clips are short, so drift never builds up there: bench unchanged at 82 / 5 / 12 throughout.
  - A tracklet is also parked if it moves at most 30 px (or 2 radii) per second for 75%+ of its life. 12 px was too tight: the same failure also hid in fc-1, where at 22 px the path sat 20 s on a still object by the far tree line; 25–40 px all scored alike.
  - A parked tracklet costs at least what "no ball" would, so it can't fill gaps for free.
  - Set-piece credit only if the camera kept the parked ball within 300 px on screen: Veo follows the game ball, so a set piece waiting to be taken stays put; the 28 set pieces credited before moved ≤ 227 px, the flag ~1000.
  - Tracklets are cut where a ball still for 3 s moves 40 px away within 20 frames (a kick, or a ball coming to rest), so a free kick taken without the ball ever being lost is credited too. With no time limit, drift "kicked" the flag every 3 s.
  - Tried and dropped: cutting out every still stretch (lost balls held at a player's feet), "the spot is empty after the kick" (no gain), rejoining pieces seen again in place (cost found frames).

  | Whole-game labels, no fixes | Found | Wrong | Missed | Ball claimed when none |
  |---|---|---|---|---|
  | before | 55.5% | 24.3% | 20.2% | 35% |
  | **now** | **66.9%** | **12.6%** | **20.5%** | **32%** |

  PRSC 22–24 min without fixes: 10 of the 18 hand-placed balls matched (was 1); with the fixes, 0 of 513 frames after the last fix on the flag (was all). `detections()` scanned every detection per frame; one sort makes whole-game retracks ~15 s instead of minutes.

- **2026-10-23 — Shorter stillness span.** From 30:05 in PRSC the path sat 25 s on a white chair far to the side: during fast pans the camera estimate slips enough that over 1 s it seemed to move more than 30 px (60% still, under the 75% bar). Slip grows with the span, so stillness is now judged over 1/3 s (≤ 20 px): the chair moves a median 7 px per 1/3 s, the game ball 50. Path off the chair (0 of 737 frames, was all), still off the flag; game labels within noise (found 65.8% → 65.5%, wrong 13.2% → 12.7%, now 591 labelled frames); bench unchanged.

- **2026-10-23 — Fixed objects, in drift-free coordinates.** PRSC 53:45 had the path on a small item up the hill, there all game but detected only in short, weak pieces, so no per-tracklet rule could catch it. A Veo camera doesn't move (the follow-cam is a virtual view inside a fixed wide recording), so frames showing the same area are related by a homography. `world.py` matches every 10th frame directly to a keyframe (ORB + RANSAC homography; candidates: the last keyframe, its linked neighbours, the most similar thumbnails), ~10 min a game on the CPU, 100% of samples placed, 50-200 keyframes a game. A first version mapping everything onto one flat world plane fell apart: wide pans stretched far views into extreme perspective (24% matched).
  - A detection is placed as (keyframe, position in it); for each spot: seconds something was detected there / seconds the spot was inside the detector's search area. Fixed objects ~1.0 (PRSC flag 374 s, hill item 144 s, chair 101 s); the game ball at labelled frames at most 0.18 in five games (fc-1: 3 of 95 at ~1.0, a ball lying out of play 113 s and one among the bench spares). Blocked: share >= 0.6 over >= 60 s.
  - Blocks 1-40% of a game's detections (PRSC 40%, fc-1 30%: lots of spare balls, bags, chairs). Whole-game labels (598 visible): found 64.7% -> 66.1%, wrong 13.7% -> 10.7%, ball claimed when none 33.7% -> 31.6%; no game found less (fc-1 wrong 23% -> 12%, PRSC 43% -> 35%).

- **2026-10-24 — Faint fixed objects.** PRSC 22:12: a mark up the hill the detector sees only weakly (0.1-0.3), so only 153 of the 351 s its spot was searched. Also blocked now: share >= 0.25 over >= 150 s (the game ball's busiest spot at any label: 117 s). Real-ball detections blocked unchanged (3 of 443); whole-game wrong 10.9% -> 10.6%, PRSC wrong 18 -> 16 of 51. Also: a tagger server started before a filter change keeps the old filter for every fix made in it (it re-runs the path around each fix): restart it after changes.

- **2026-10-24 — Bigger spots for fixed objects.** fc-1 56:10: a ball lying still all game wasn't blocked: as the camera panned it was placed in several keyframes, and within one keyframe its positions wobbled by ~30 px near the edge of the view (a Veo view is two lenses stitched, so one homography per frame is only approximate). At 15 px cells it split into pieces under a minute. Cells are now 35 px: that ball, the PRSC flag, chair, hill item and the faint hill mark all score ~1.0 (the faint-object tier is no longer needed and was dropped: with bigger cells it caught the real ball in a busy midfield spot at 0.30). Following links 3 hops instead of 1 was worse (errors add up). Real-ball detections at labels blocked: 5 of 449, all fc-1, at least three on that still ball (labels to check: 43:59, 56:14, 66:20). Whole-game labels about unchanged (found 65%, wrong 11%).

- **2026-10-24 — Keyframes linked directly.** fc-1 51:04: the still ball again, now in a view (keyframe 240) two or three links from the views where it piled up its time: registration only links a keyframe to the one the camera came from. Composing links (3 hops) adds up errors, and chaining spots view to view (union-find) merged whole stretches of field into one group (the PRSC flag, chair and hill item all lost). What works: re-grab each keyframe's frame (one seek each) and match it directly to its 12 most similar keyframes, adding a link wherever >= 60 inliers (370-600 links a game became 1,000-3,100; ~4 min a game). fc-1: found 65% -> 69%, wrong 12% -> 7%; all games found 66%, wrong 10%; real-ball detections blocked unchanged (5 of 451, fc-1's still ball and a bench spare). The real ball's highest share at a label rose to 0.49 (fc-stars), so the 0.6 bar has less margin than before.

## Benchmarks / findings
| Game | Ball recall | Ball false pos/min | Identity precision | Identity recall | Review answers | Notes |
|---|---|---|---|---|---|---|
