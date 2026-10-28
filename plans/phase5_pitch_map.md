# Phase 5 — Pitch Map (where everyone is on the field)

**Status (2026-10-28):** 5.1–5.3 built and tried on PRSC: calibration clicks in the tagger (`Q`), the solve (`track/pitch.py`), the field lines layer over the video, and the map below the video with the camera's view, players and the ball (placed in the browser from the boxes; 5.4's smoothing and stored positions not built). The map is in the Player view too (a "field map" checkbox), with the player you're watching ringed, and players who leave the view fade out where they were last seen. All seven games are registered (5.0). Next: calibrate every game, then 5.4.
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

Per game in `game.json`: the field's length and width as the solve estimated them (or as entered, if you ever override them).

## 5.0 — Prerequisites
- Register fcu (`soccer-clips ball fixed-objects gu11-black-vs-fcu-2026-10-25`, CPU, here).
- Check that `world.place` holds up for player feet across the whole frame, not just the detector's centre crop the ball uses. Registration masks only the Veo logo, so it should.

## 5.1 — Pitch model
`track/pitch.py`: the field in metres, origin at the centre spot, x along the length (toward the right-hand goal as the camera sees it), y across.
- **Size is solved, not entered.** Sizes aren't posted and vary by venue, so length and width are unknowns of the calibration (5.2), starting from the middle of the 9v9 range (70–80 × 45–55 yd) with a soft pull toward it. What makes them solvable is the markings whose size *doesn't* change with the field. On 9v9 those are the centre circle's radius and the goal's width (18.5 ft), and probably the penalty and goal areas (sizes from US Soccer's 9v9 guidelines, checked on the first field). Each of those is a parameter too, starting at its standard value with a tight pull, so a field painted slightly off still fits.
  - Corner and line clicks pin down where the field's edges are, and the circle and goals tell the solve how many metres that is. A circle has to come out round, which also fixes the length-to-width ratio.
  - If a game shows neither the circle nor a goal clearly, the size stays near the 9v9 middle and the map is a little stretched. Everyone's positions relative to the lines are still right, which is what the map and heatmaps need. Only metre-based uses (near the ball in metres, distance run) are off, by the same few percent.
  - The estimate (with how sure it is) is shown in the tagger and saved per game. A later game at the same field can start from it.
- **Landmarks** (points the solve knows the position of, given the sizes): 4 corners, 2 halfway-line ends, centre spot, goal posts, penalty and goal area corners, penalty spots.
- **Lines** (for "this click is somewhere on this line"): touchlines, goal lines, halfway line, centre circle, penalty and goal area edges. Line clicks matter because many views show a stretch of line and no corner.
- Drawing data for the map: the same lines, as polylines, at the solved sizes.

## 5.2 — Solving the calibration
*As built (2026-10-28), after trying it on PRSC (see Findings): links are compared at fixed pixels of one view, not at pitch points, each link weighs about as much as one click, and the start can also come from the centre circle alone.*

Unknowns: `H_pitch[k]` for every keyframe k (8 numbers each; ~300 keyframes means ~2,400 unknowns), plus the field's length and width and the marking sizes (5.1). Residuals, solved together with `scipy.optimize.least_squares` (sparse Jacobian, robust loss):
1. **Clicks:** each click is mapped into its keyframe with `world.place`, then through `H_pitch[k]`. For a point landmark the residual is the distance to the landmark; for a line it's the distance to the line or circle (metres). Landmark positions are functions of the sizes, so the clicks also pull the sizes into place.
2. **Size priors:** a gentle pull of length and width toward the 9v9 middle, and a tight pull of each marking size toward its standard value.
3. **Link consistency:** for each keyframe link (a, b, `link_H`), take a grid of pitch points that both keyframes show, put each into a's pixels (`H_pitch[a]⁻¹`) and on through `link_H` into b's, and compare with where b says they are (`H_pitch[b]⁻¹`), in pixels. Using pitch points (always on the ground) instead of image points avoids anything above the horizon. This is what spreads a few clicks across hundreds of keyframes.

**Start point:** a direct fit (DLT) on the keyframe with the most clicks, then outward along links, breadth-first, chaining `link_H`. Keyframes no link path reaches stay uncalibrated, so their frames get no positions, and each game's coverage is reported.

**Fallback if far views come out wobbly:** model the camera physically, as one fixed point with each keyframe's pan, tilt and zoom (4 numbers per keyframe instead of 8, plus the camera's height and position). Fewer unknowns means fewer clicks and less freedom to bend. It isn't the first choice because a Veo view isn't an exact pinhole (two lenses stitched; still objects wobble ~30 px near a view's edge, 2026-10-24), and free homographies can absorb some of that.

**Clicks on the wrong set of lines** (PRSC's yellow field under our white one) disagree with the rest, so the robust loss mostly ignores them and they show with a large residual, flagged red, for you to delete.

Tests: synthetic keyframes made from a known pitch and known camera rotations, plus noisy clicks. The solve should recover positions and the field's size to within the noise, keep uncalibrated keyframes NaN, and flag a click placed on a second, offset set of lines.

## 5.3 — Calibrating and correcting in the tagger
No separate page: the map below the video (5.5) doubles as the diagram you click on, so calibrating and fixing are the same action, done where you already watch.
- **Field lines layer** (a checkbox in 🔍 layers, alongside the ball detections): the solved field geometry drawn over the video for whatever frame is showing. This includes all the lines, the circle, the areas and the goal posts, in a colour that stands out from both white and yellow paint. Where the calibration is right the drawn lines sit on the painted ones; where it's off you see it straight away. Your clicks show as small markers on frames near where they were made.
- **`Q` then click a spot in the video, then click where it is on the map below.** Clicking near a landmark on the map snaps to it; clicking a line means "somewhere on this line". It re-solves (seconds) and the overlay and map redraw. `Q` again or `Esc` stops. Clicking one of your own markers removes it, and `U` takes back the last click while in this mode.
- **First calibration of a game:** the same, from scratch. The tagger offers "next wide view" jumps to a few frames that between them show most of the field (toward each end and the middle), so you don't have to hunt for corners. After 4+ clicks the overlay appears.
- The field's estimated size (and how sure it is) shows next to the layer's checkbox.

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
- Where: **below the video**, in the Player view (where "where is she playing" matters most) and in the tagger (where it's also the calibration diagram, 5.3). It gets a show/hide toggle, and its height is kept fixed so the video never shifts.

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
- **Field size poorly constrained** (no circle or goal clicked): the map is a little stretched, but positions relative to the lines hold. The tagger shows how sure the size is, so you know to add a circle or goal-post click.
- **Double-marked fields:** a click on the wrong set is flagged by its residual; the overlay shows which set the solve followed.

## Tasks
- [x] 5.0 Register fcu (2026-10-28: 452 keyframes, 2020 links, every sampled frame placed; 802 ball detections on fixed objects blocked and its ball path redone). Player feet near frame edges: the PRSC overlay lines up to the edges by eye; measured spot checks still to do
- [x] 5.1 Pitch model (9v9 landmarks and lines as functions of the sizes), with tests. Standard marking sizes from secondary sources, not yet checked against a field
- [x] 5.2 Solver (clicks + size priors + link consistency, start from 4 landmarks or the centre circle, coverage, wrong-line flag, how sure each size is), with synthetic tests
- [x] 5.3 + 5.5 Map below the tagger's video, the field lines layer, and `Q` clicks (tried end to end on PRSC in headless Chrome); calibrate one game fully (PRSC has the circle, halfway line and left goal so far)
- [ ] 5.4 `pitch place`: player / ball positions, smoothing, view outline, off-pitch flag; yardsticks on that game
- [x] 5.5 Players and ball on the map in the tagger (placed in the browser from the box chunks; spectators > 3 m outside the lines left off)
- [x] 5.5 The map below the Player view's video (tried on PRSC #38); players fading out over 6 s at their last place when they leave the view (only if that place is now out of view, so a broken tracklet leaves no ghost)
- [ ] Bench players and subs sitting at the touchline show on the map (inside the 3 m margin); filter them in 5.4 (tracklets that stay put by the line)
- [ ] Calibrate the remaining games; record results in Findings
- [ ] 5.6 items as separate follow-ups

## Decided (2026-10-28)
- **Field sizes** aren't posted and vary, so they're solved from the fixed-size markings (centre circle, goals, areas), with the 9v9 middle as the fallback.
- **Map** goes below the video. The derived field geometry is drawn over the video as a layer in the tagger's 🔍 layers menu, and corrected in place with `Q`.
- **Markings:** all games are on 9v9 fields, some double-marked. Clicks plus the residual flag handle the second set.

## Open questions
- Confirm the 9v9 penalty / goal area sizes and centre circle radius the leagues use (on the first calibrated field, the solve's estimate will show whether they match).

## Findings
- **2026-10-28** — Planning check: six of seven games have `world.npz` (128–341 keyframes, 974–3124 links, every sampled frame placed); fcu has none yet. Sample frames: fc-1's lines are faint on grass, and PRSC's turf has white and yellow line sets on top of each other. That's why calibration starts with clicks.
- **2026-10-28** — First real-data run (PRSC), with 20 clicks around the centre circle in one kickoff view (eyeballed by Claude from frames, left in the game as a starting point): all 141 keyframes calibrated, 100% of sampled frames placed, and the drawn circle and halfway line sit on the painted white ones in views 15–35 minutes later, ignoring the yellow set. Adding the left goal's posts through the tagger moved the length from the 9v9 middle (68.6 m) to 62.7 ± 2.9 m; the width stays unmeasured (45.7 ± 4.6) until a touchline or the right goal is clicked.
- **2026-10-28** — Keyframe links disagree on ground points by a median 5 px (90th percentile 23 px, max 217 px), the same one link from the clicks as four: each link's own error, not a build-up along chains (ORB matches are mostly trees and stands, and a Veo view isn't an exact rotation). At 2 px per link point they outvoted the clicks and bent the solve (clicks 200 px off); now each link weighs about one click at 6 px.
- **2026-10-28** — Links were first compared at pitch points in metres; the solve then grew the centre circle to 9.5 m (standard 7.3 ± 0.5), because rescaling the field moves those points onto ground where the links agree better. Compared at fixed pixels instead (`P_b · P_a⁻¹ · x` vs `H_ab · x`), the links don't care about scale and the circle stays at 7.3–7.9 m.
- **2026-10-28** — Kickoff views often show only the centre circle and halfway line (three landmarks on one line), so the solve can also start from the centre spot, the two halfway-line crossings and 3+ circle points: the centre's harmonic conjugate gives the halfway line's vanishing point, whose polar is the circle's other diameter.
- **2026-10-28** — Solve time for PRSC's 141 keyframes: 7–10 s while fcu was registering (load ~12 on 4 cores), 4 s with only the test suite alongside (load ~11); to measure on an idle machine, and on fcu (452 keyframes, the most). Stopping early (1e-3 of the cost) left positions a median 1.6 m off across the field, so it runs to 1e-6; starting from the previous calibration saved nothing and was taken back out.
