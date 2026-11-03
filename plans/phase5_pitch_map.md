# Phase 5 — Pitch Map (where everyone is on the field)

**Status (2026-11-02):** 5.0–5.5 built, and all seven games calibrated and placed: calibration clicks in the tagger (`Q`, with a list of the clicks to jump to or remove, start over, and a solve that can't be a field refused), the solve (`track/pitch.py`), the field lines layer over the video, stored positions (`track/positions.py`, `soccer-clips pitch place`: every box's feet smoothed along its tracklet, off the field flagged by the Games page's rule, the ball), and the map below the video with the camera's view, players and the ball. The map is in the Player view too (a "field map" checkbox), with the player you're watching ringed, and players who leave the view fade out where they were last seen. 5.6's first use is built: labels followed on by place (`track/bridge.py`). The Games page uses the field to count only players on it. Next: Phase 6 (game events) builds on the positions; the held-out-click yardstick is still to do.
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
- **The clicks** (a list under the map): each click's time (jump there; on its own frame its × is where it was clicked), how far off the solve puts it, and ✕ to remove it; removals queue during a solve like clicks. **start over…** removes them all (kept in `pitch_points.before-reset.json`).
- **A solve that can't be a field** (`pitch.not_a_field`: a size negative or more than 3× off 9v9, the goal area not inside the penalty area, that not inside the field) isn't used: the last good one is kept, every click is measured against it, and the tagger and the Games page say so.
- **First calibration of a game:** the same, from scratch. The tagger offers "next wide view" jumps to a few frames that between them show most of the field (toward each end and the middle), so you don't have to hunt for corners. After 4+ clicks the overlay appears.
- The field's estimated size (and how sure it is) shows next to the layer's checkbox.

This overlay is also the main check on the calibration by eye.

## 5.4 — Positions (`soccer-clips pitch place [games]`)
*As built (2026-11-02), `track/positions.py`: `tracks/pitch.npz` holds per tracklet row the feet in metres (running median over 15 rows, ~0.5 s) and an off-the-field flag, and the ball path in metres with an "in the air" flag (> 10 m outside the lines); keyed on the tracklets, calibration, registration and ball files, so it's made again when any changes. 16–29 s a game. The view outline isn't stored: the map draws it from the per-frame homographies already, and anything later can do the same. The app's box chunks carry the positions; a game whose positions are out of date is placed in the background and the map places boxes in the browser meanwhile. The Games page has a Positions step.*

- Every player box: bottom-centre → pitch metres, smoothed along its tracklet over ~0.5 s. Box bottoms jitter, and feet hidden behind another player push the box bottom up, which reads as "further away".
- Ball: the path's position → pitch. A ball in the air projects beyond where it really is (no height is known), so its dot is approximate while airborne.
- **View outline:** the frame's bottom corners and its top edge (clipped where it nears the horizon) → a quadrilateral on the pitch, per sampled frame.
- **Off-pitch flag:** boxes outside the lines (parents, coaches, subs) are marked. This is the "pitch filter" the people tracking plan left for later. Not a plain 3 m margin: the crowd stands 1–3 m behind a touchline, and on the far side reads up to 4 m inside it. The Games page's rule (`status.still_by_line`: 1 m margin, and anyone still for 10 s within 4 m of a touchline left out) is the starting point.
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
- **Tracklet joining by pitch speed** (a player covers at most ~7 m/s), which is steadier than box heights across zoom changes. First form built (2026-10-29): `track/bridge.py` carries labels and guesses on across holes in the tracking, as guesses (see Findings).
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
- [x] 5.4 `pitch place`: player / ball positions, smoothing, off-pitch flag, seam jumps (2026-11-02; view outline not stored, drawn from the homographies)
- [ ] 5.4 Yardsticks still to do: held-out clicks, still things stay still
- [x] 5.5 Players and ball on the map in the tagger (placed in the browser from the box chunks; spectators > 3 m outside the lines left off)
- [x] 5.5 The map below the Player view's video (tried on PRSC #38); players fading out over 6 s at their last place when they leave the view (only if that place is now out of view, so a broken tracklet leaves no ghost)
- [x] Bench players and subs sitting at the touchline show on the map (inside the 3 m margin); filter them in 5.4 (tracklets that stay put by the line). Done for the Games page's count (`status.still_by_line`, 2026-11-02), and on the map through 5.4's off flag (2026-11-02)
- [x] Calibrate the remaining games (all seven by 2026-10-29); record results in Findings
- [x] The clicks list, queued removals and start over (2026-10-29); a solve that can't be a field refused, the last good one kept (2026-11-02)
- [x] 5.6 Following a player on by place (`track/bridge.py`, guesses `by: "place"`; `soccer-clips reid place-eval`)
- [ ] 5.6 other items as separate follow-ups

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
- **2026-10-28** — Usability (first use): `Q` showed the field lines as well as starting calibration, so it looked like an overlay toggle, and nothing said clicks on the video now placed points instead of pausing. Split into 📐 lines (`Shift+Q`, just the overlay) and ✏️ calibrate (`Q`), each a button in the top bar, with a magenta banner over the video while calibrating.
- **2026-10-28** — fcu lost its goal areas (depth solved to −0.1 m): a box's side line had the same id as its corner in the field (`ga_lf`), the map labelled it as the side, and the solve read it as the corner, so a click on the side near the goal line pulled the corner onto the goal line. The sides are now `ga_lf_side` etc. Saved clicks (pitch_points.json version 1) are read as the side, which holds whether the corner or the side was meant; fcu then had too few landmarks in its kickoff view (three on the halfway line plus one), which the start used to accept and solved into a degenerate field (2 min, every click at x = 0). It now wants 4 landmarks with no 3 on one line. The map also let a landmark within 12 map pixels win over a line, so the middle of a goal area's side (~20 px long) snapped to a corner: landmarks win only within 7 px now.
- **2026-10-28** — Clicks waited for the solve: the × only appeared when it finished, and video clicks during it were dropped, so the crosshair didn't come. Clicks now show at once and queue during a solve; queued ones are saved without solving (`solve: false`) and solved once together. fcu's solve with 15 clicks: ~30 s (452 keyframes, load ~6), against PRSC's 4 s.
- **2026-10-28** — The ball was on the map already but easy to miss: a 3 px dot, only in frames the ball path has (62% of PRSC's), and in the tagger only with the ⚽ ball layer on. Now it loads with the map, is drawn bigger on top of everyone, and holds its last place (hollow) for up to 1.5 s, which puts it on the map in ~90% of frames; a ball more than 10 m outside the lines is taken to be in the air and left off.
- **2026-10-28** — On the map, our players show as their number where it's known (labelled, or guessed by appearance at the tagger's "auto-label at" threshold, drawn fainter), a dot where not; opponents stay dots. The tagger now loads labels and guesses whenever the map is on, not only with 👥 players.
- **2026-10-29** — Following players on by place (`track/bridge.py`). fcu at 28:05–28:19: #15 is labelled until 28:07.3 in the right goal area, guessed (appearance) to 28:13.7, then lost until her next label at 28:18.8, though no teammate came near: she stood behind the yellow referee, and her half-hidden box read as "other" kit, with a second undetected either side. Links between our-team tracklet ends and starts on the field: cost = distance / (2 m + 3 m/s x gap), plus 0.25 per half second the straight way between them was in plain view with nobody's box there (beyond 0.7 s; out of the camera's view or behind a box is free), greedy most plausible first, taken only when no other open end or start is within 3x, except ones that could be the same player again later (a player's scraps otherwise rival each other). Of links with labels at both ends, 88 joined the same player and 2 didn't; by time, following from half of each player's labels landed 1836 s on their hidden half and 189 s on others' labels (one 83 s tracklet after a tracker kit cut at the same spot, same frame). An appearance veto (a reached tracklet that looks 0.05 more like another labelled teammate, by its own crops, against looks built without that game) took the wrong time to 0 and kept all 1836 s. On top of the appearance guesses it adds ~2000 s of unlabelled time over the two games (30–130 s a player a game), little of the labelled time (the appearance guesses already have 83% of it). Gaps are filled by unlinked tracklets on the way between (ours scraps, or misread "other" kit; not opponents). Cost: ~8 s a game once to place the tracklet ends (kept on disk), ~1.3 s to link, ~0.5 s after each relabel.
- **2026-10-29** — `reid place-eval` over all seven games (now all calibrated): 3075 s right, 63 s wrong (precision 0.98), 2480 s on unlabelled time. The wrong time is one link in FC Stars: #15's tracklet ends and one labelled #14 starts 0.3 s later half a metre away (an opponent 1.7 m off, a teammate 6 m), and that tracklet looks a shade more like #15 (0.841) than #14 (0.829), so the veto can't fire, and the #14 label may be the mistake (video 1:06:57.7). FC Stars' calibration is also shaky: a click 130 px off and a 57 m width.
- **2026-10-29** — FC-1's solve collapsed (33 m wide, centre circle radius −10.5 m, goal area 0.6 m deep, 15 of 21 clicks placed at about one spot). Each red × was drawn where the solve put its click, so they sat on top of each other, and a removal re-solved for ~40 s while refusing other clicks. Fixed in the tagger: on its own frame a click's × is where it was clicked, a list of the clicks, queued removals, start over. FC-1 was re-clicked from scratch (16 clicks, 1 off).
- **2026-11-02** — Re-solving FC-1's 21 old clicks reproduces the collapse; measured against the re-clicked field, those clicks were only 10–90 px off (the worst the goal-area lines at 2:02.8), so the solve lost its way more than the clicks were wrong. Such a solve is now refused (`pitch.not_a_field`) and the last good one kept. The check is loose on purpose: Elite Strikers, FC Stars and FCU solve goal areas ~7–8 m wide, half the 9v9 14.6 m, and pass.
- **2026-11-02** — Every game has a crowd at a touchline (bench, subs, parents): 1–3 m behind it at FC Stars, SAA, Elite Strikers and FCU, and at PRSC and FC-1 read 1–4 m inside it (a few pixels at the far side's feet are metres). With a 3 m margin they added ~1 "our player" a frame on the Games page. Left out there now by `status.still_by_line`; the map still shows them.
- **2026-11-02** — Positions (5.4) for all seven games, 16–29 s each. Seam jumps (median position change from one frame to the next where the keyframe changes, raw, before smoothing): near half 0.29–0.35 m, far half 0.69–1.53 m (SAA worst, then FC-1 1.09 m), against 0.05–0.06 m / 0.08–0.12 m for ordinary steps; the 1 m target holds except SAA's and FC-1's far side. Smoothing moves a box a median 0.07–0.10 m. Off the field: 18–55% of all boxes (spectators, the bench, other fields), and of our kit in game time 11–34%, the same as the Games page's own count (one rule). People kept on the field per frame in the first half: median 16–18 in five games (9v9 plus a referee), 22–23 at PRSC and FC-1, whose crowds read inside the line in short tracklets that the still-by-the-line rule can't catch. SAA's "opponents" are 88% off the field: that game's opponent kit cluster is mostly the crowd (a teams problem, not a positions one).
