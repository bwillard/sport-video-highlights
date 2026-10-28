# Phase 5 — Pitch Map (where everyone is on the field)

**Status (2026-10-28):** planned, nothing built.
**Goal:** For any moment of a game, know where each tracked player (and the ball) is on the field in metres, and show it as a top-down 2D map next to the video, synced frame by frame, in the tagger and the Player view.
**Depends on:** Phase 3.2 player tracks, 4.2 teams, 4.3 names (all built), and the frame registration in `track/world.py` (built for the ball's fixed-object finder; six of seven games have it, fcu still needs `soccer-clips ball fixed-objects`).

## The key idea: one calibration per game, not per frame
A Veo camera doesn't move. The follow-cam MP4 is a virtual camera panned and zoomed inside one fixed wide recording, so every frame is the same scene seen in a different direction. `world.py` already uses this: each game is a graph of keyframes (128–341 per game), with an accurate homography from every 10th frame to a keyframe, homographies between keyframes that show the same place (974–3124 links per game, every sampled frame placed), and camera steps for the frames in between (`world.place`).

So the work is to tie that graph to the pitch once per game: find a homography from each keyframe's pixels to pitch metres. Then any pixel in any frame maps to the pitch:

```
frame pixel --(camera steps + world.H)--> keyframe pixel --(H_pitch[keyframe])--> pitch (x, y) in metres
```

A player's position is the bottom-centre of their box (their feet), which is on the ground plane, which is exactly what a homography maps correctly.

Why this works where "one flat world plane" failed (2026-10-23): that attempt used one keyframe's *image* as the shared plane, and wide pans stretched far views into extreme perspective. Here the shared plane is the *ground* itself, which is the plane the homographies are physically right for. Points above the horizon (trees, sky) have no pitch position, and we never need them to.

## Why calibration starts with clicks, not automatic line detection
Automatic field calibration (the SoccerNet camera-calibration models, e.g. PnLCalib or No Bells Just Whistles) learns from professional broadcast pitches with standard 105×68 m markings. Our fields don't look like that (checked on frames from two games):
- **fc-1:** grass with worn, faint lines. Midfield views are almost all plain grass.
- **PRSC:** turf with two overlapping line sets (white for our field, yellow for another), so a line detector would match the wrong field half the time.
- Youth field sizes vary by venue, and some fields are marked partly with cones.

Clicking ~10–20 landmarks per game (2–3 minutes) is reliable, and one calibration covers the whole game because of the keyframe graph. Snapping clicks to nearby painted lines, or suggesting clicks automatically, can come later once there's a working baseline to measure against.

## What this phase produces
Per game, in `tracks/`:
- `pitch_points.json`: calibration clicks as (game frame, pixel x, y, landmark or line id), like `people_picks.json`, so they survive re-registration.
- `pitch_cal.npz`: `H` (keyframes, 3, 3) keyframe pixels → pitch metres (NaN for keyframes the solve couldn't reach), plus fit residuals.
- `pitch.npz`: pitch x, y for every row of `people_<model>.npz` (NaN where unplaced) and for every frame of the ball path, the camera's view outline on the pitch per sampled frame, and the counts it was made from (like `fixed.npz`'s `n_dets`) so it's rebuilt when tracks or calibration change.

Per game in `game.json` (or a venue it points to): the field's length and width, and which markings it has.

## 5.0 — Prerequisites
- Register fcu (`soccer-clips ball fixed-objects gu11-black-vs-fcu-2026-10-25`, CPU, here).
- Check that `world.place` holds up for player feet across the whole frame, not just the detector's centre crop the ball uses. Registration masks only the Veo logo, so it should.

## 5.1 — Pitch model
`track/pitch.py`: the field in metres, origin at the centre spot, x along the length (toward the right-hand goal as the camera sees it), y across.
- **Size:** length × width, entered per game (yards or metres). Saved as a named venue so a home field is entered once. Sources: the league's posted size, or Google Maps' measure tool. A wrong size stretches the map but keeps everyone's relative positions right.
- **Landmarks** (points whose position follows from the size): 4 corners, 2 halfway-line ends, centre spot, goal posts (9v9 goal 18.5 ft wide). Penalty and goal area corners and penalty spots only if their sizes are entered, since they vary.
- **Lines** (for "this click is somewhere on this line"): touchlines, goal lines, halfway line, centre circle (radius entered or the 9v9 default). Line clicks matter because many views show a stretch of line and no corner.
- Drawing data for the map: the same lines, as polylines.

## 5.2 — Solving the calibration
Unknowns: `H_pitch[k]` for every keyframe k (8 numbers each; ~300 keyframes means ~2,400 unknowns). Residuals, solved together with `scipy.optimize.least_squares` (sparse Jacobian, robust loss):
1. **Clicks:** each click is mapped into its keyframe with `world.place`, then through `H_pitch[k]`. For a point landmark the residual is the distance to the landmark; for a line it's the distance to the line or circle (metres).
2. **Link consistency:** for each keyframe link (a, b, `link_H`), take a grid of pitch points that both keyframes show, put each into a's pixels (`H_pitch[a]⁻¹`) and on through `link_H` into b's, and compare with where b says they are (`H_pitch[b]⁻¹`), in pixels. Using pitch points (always on the ground) instead of image points avoids anything above the horizon. This is what spreads a few clicks across hundreds of keyframes.

**Start point:** a direct fit (DLT) on the keyframe with the most clicks, then outward along links, breadth-first, chaining `link_H`. Keyframes no link path reaches stay uncalibrated, so their frames get no positions, and each game's coverage is reported.

**Fallback if far views come out wobbly:** model the camera physically, as one fixed point with each keyframe's pan, tilt and zoom (4 numbers per keyframe instead of 8, plus the camera's height and position). Fewer unknowns means fewer clicks and less freedom to bend. It isn't the first choice because a Veo view isn't an exact pinhole (two lenses stitched; still objects wobble ~30 px near a view's edge, 2026-10-24), and free homographies can absorb some of that.

Tests: synthetic keyframes made from a known pitch and known camera rotations, plus noisy clicks. The solve should recover positions to within the noise, and keep uncalibrated keyframes NaN.

## 5.3 — Calibration page (`/pitch`)
- Pick a game. The video plays as in the tagger, with a top-down pitch diagram beside it.
- Pause on a frame showing a landmark. Click it in the video, then click the matching spot on the diagram (a landmark snaps to the nearest one; clicking a line means "somewhere on this line"). The page suggests frames with wide views toward each end and the middle (keyframes that between them cover the most of the field), so you don't have to hunt for them.
- After 4+ points it solves in the background (seconds) and **draws the pitch lines over the video** for whatever frame is showing. Scrub around: where the drawn lines miss the painted ones, add a click there. Each click shows its residual, and you can delete it.
- Same nav row as the other pages. Undo as on the roster page.

This overlay is also the main check on the calibration by eye.

## 5.4 — Positions (`soccer-clips pitch place [games]`)
- Every player box: bottom-centre → pitch metres, smoothed along its tracklet over ~0.5 s. Box bottoms jitter, and feet hidden behind another player push the box bottom up, which reads as "further away".
- Ball: the path's position → pitch. A ball in the air projects beyond where it really is (no height is known), so its dot is approximate while airborne.
- **View outline:** the frame's bottom corners and its top edge (clipped where it nears the horizon) → a quadrilateral on the pitch, per sampled frame.
- **Off-pitch flag:** boxes more than ~3 m outside the touchlines (parents, coaches, subs) are marked. This is the "pitch filter" the people tracking plan left for later.
- CPU only, here: `world.place` per frame for ~100–150k frames per game plus a homography per box. Expect minutes per game. Run automatically after re-tracking, like teams.

## 5.5 — The map in the app
A canvas panel, drawn in the same `requestVideoFrameCallback` loop that draws boxes now:
- The pitch, drawn from 5.1's lines (later: a real top-down picture of the field stitched from the video with the calibration, as the background).
- The camera's view as a faint quadrilateral, so it's obvious which part of the field the dots cover.
- A dot per player in view, coloured by team as the boxes are, our players' numbers on their dots, the followed player highlighted (the Player view's marker colour), and the ball.
- **Players who leave the view fade out** at their last known position over ~5–10 s, then disappear.
- Data: the tagger already streams boxes in 15 s binary chunks (`/api/games/{id}/people/boxes`); add pitch x, y (and the off-pitch flag) as extra columns, plus a small per-chunk array for the ball and view outline. The Player view gets the same columns on `/api/watch/boxes` (the followed player) and fetches everyone else's chunks.
- Where: in the Player view first (it's where "where is she playing" matters most), then a toggle in the tagger. Beside the video on a wide screen, a corner overlay otherwise.

**What the map can't show:** the follow-cam typically shows a third to half of the field, so players far from the ball are often off camera, and the map only knows where they were last seen. If Veo's panorama ever becomes available as a download, the same pipeline would place everyone at once.

## 5.6 — Uses once positions exist (later, each its own item)
- **Near the ball in metres** (4.4's ⚽ filter is in box heights today, which shrink and grow with the zoom).
- **Per player per game:** heatmap, average position, share of time in each third. Distance run only while on camera, so present it with that caveat or not at all.
- **Spectators out of tracking and the roster page:** drop off-pitch boxes before naming, so there are fewer junk pictures to reject.
- **Tracklet joining by pitch speed** (a player covers at most ~7 m/s), which is steadier than box heights across zoom changes.
- **Ball out of play**, as another signal for game-time suggestions.

## Yardsticks
- **Held-out clicks:** leave each calibration click out in turn, and measure its error in metres. Report the near half and the far half of the field separately, since one pixel covers far more ground on the far side.
- **Seam jumps:** positions jump when a frame switches from one keyframe to another. Report the median jump at switches for players in tracklets that span one.
- **Overlay by eye:** pitch lines drawn on 10 random frames per game.
- **Still things stay still:** goals and corner flags (fixed spots `world.py` already finds) should map to the same pitch spot all game.
- Rough targets to revisit after the first game: median ≤ 1 m near side, ≤ 3 m far side; seam jumps ≤ 1 m.

## Risks
- **Veo stitching wobble** (~30 px near view edges) could be metres on the far side. Mitigations: per-keyframe homographies absorb some of it, more clicks where the overlay misses, then the physical model or a smooth correction per keyframe.
- **Far side at grazing angles:** 1 px of box bottom may be 0.2–0.5 m of depth. Smoothing helps; precision across the field's width will always be worse than along its length.
- **Keyframes no link path reaches from the clicked ones:** rare given the dense links, but reported per game, and fixable with a click in that view.
- **Field size unknown or wrong:** the map is stretched, but relative positions hold.

## Tasks
- [ ] 5.0 Register fcu; spot-check `world.place` on player feet near frame edges
- [ ] 5.1 Pitch model + venues (size, landmarks, lines), with tests
- [ ] 5.2 Solver (clicks + link consistency, BFS start, coverage report), with synthetic tests
- [ ] 5.3 `/pitch` calibration page with the live line overlay; calibrate one game end to end
- [ ] 5.4 `pitch place`: player / ball positions, smoothing, view outline, off-pitch flag; yardsticks on that game
- [ ] 5.5 Map panel in the Player view, then the tagger
- [ ] Calibrate the remaining games; record results in Findings
- [ ] 5.6 items as separate follow-ups

## Open questions
- Field sizes: do the leagues post them, or measure each venue on a map?
- Map placement: beside the video, or a corner overlay on it?
- 9v9 markings at every venue, or are some games on fields marked for something else (as at PRSC)? Affects which landmarks to offer.

## Findings
- **2026-10-28** — Planning check: six of seven games have `world.npz` (128–341 keyframes, 974–3124 links, every sampled frame placed); fcu has none yet. Sample frames: fc-1's lines are faint on grass, and PRSC's turf has white and yellow line sets on top of each other. That's why calibration starts with clicks.
