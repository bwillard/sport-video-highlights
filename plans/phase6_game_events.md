# Phase 6 — Game Events (what happened, not just where everyone was)

**Status (2026-11-02):** 6.0 built (`track/match.py`, `soccer-clips pitch match`): restarts, which goal we attack each half (sure in all seven games, and every rated goal clip agrees), goalies, out-of-play stretches; the tagger's Game time box sets the direction by hand if needed and the map says which way we attack. 6.1's labelling is built (Shift+E in the tagger, full windows in its Events box; `track/events.py`). Next: label a few full windows (you), then 6.2.
**Goal:** For each game, a log of what happened with the ball: who had it and for how long, every pass (who to whom, completed or not), chains of passes, shots and who took them, saves, goals, and who won or lost the ball. Built on that: possession time per team and per third of the field, per-player numbers, and a first rating of good and bad plays.
**Depends on:** Phase 4.1 ball path, 4.2 teams, 4.3 names, 5.2 calibration (all built) and 5.4 positions (built 2026-11-02).

## What we want out of it
- **Game log:** `12:34  #38 → #15  pass, 14 m forward, mid → final third`, `12:37  #15  shot (saved)`, `13:02  #7  won the ball (interception)`, and so on, each a link to that moment of the video.
- **Per game:** possession share per team, per half, and with the ball in each third (thirds counted from our attacking direction); passes attempted / completed per team; longest chains; shots, saves and goals.
- **Per player (our team only):** touches, time on the ball, passes attempted / completed / into danger and to whom, shots, balls won and lost. The other team is counted as a team, never per player, so their players needn't be told apart.
- **Plays:** moments rated good or bad (6.8), which become suggested clips (4.4's accept / reject idea).

## The approach: rules on the tracks we have, measured against labelled events
Everything is derived from data that exists (ball path, player boxes with team and name, field calibration), as layers that each read the one below:

```
ball path + player positions (5.4)
  → ball in metres, in play / out of play, airborne stretches   (6.2)
  → who is in control of the ball, frame by frame, by team        (6.2)
  → touches: who played the ball when                             (6.3)
  → passes, chains, shots, saves, goals, balls won / lost         (6.4–6.6)
  → game log, stats, ratings                                      (6.7–6.8)
```

Each layer is recomputed cheaply from the one below (like following by place, `track/bridge.py`), so a new name label or a ball fix updates the log without re-running anything on a GPU.

**Why rules first, not a learned action spotter.** There is published work on spotting ball actions in video (SoccerNet's ball action spotting task: passes, drives, shots, tackles and so on), but it is trained on professional broadcast footage, needs hundreds of labelled examples per kind of event, and would tell us *that* a pass happened, not who passed to whom. Rules over our tracks give the who and the where directly, can be explained when they're wrong, and work with the few labels we have. A learned spotter stays an option (6.9) for whichever events the rules can't see, typically touches hidden in a crowd of feet.

**Why the follow-cam helps here.** The camera follows the ball, so the ball and the players near it are almost always in view. Events happen where the camera is looking. What the camera misses is the far-off players a pass might be aimed at, who appear only as the ball arrives.

## 6.0 — Prerequisites
- **5.4 positions** (`pitch place`): every box's feet and the ball in metres, smoothed along tracklets, with the off-pitch filter (bench, parents). Events need positions stored per box, not placed in the browser as the map does now.
- **Attacking direction per half:** which goal we attack in each half, for "forward", thirds, and which goal a shot is on. Worked out from where our team stands at each kickoff (each team kicks off from its own half), stored per game with an override in the tagger.
- **Keepers:** each team's keeper wears a different kit, so the kit clustering (4.2) most likely puts them in 'other' with the referee. A keeper is the 'other' player who spends most of a half inside one penalty area. Needed for saves and to count keepers as part of their team. Our keeper isn't assumed to be the same player every game (or every half): found per half, named if their tracklets are labelled, and otherwise "our goalie" in the log. The other team's is always "their goalie".
- **Ball out of play:** the ball crossing a touchline or goal line (from 5.6's list), and restarts: the ball filter already finds a parked ball that gets kicked (`ball_track.kicked`); where it was parked says what the restart was (corner, goal kick, throw-in from the touchline, free kick, kickoff from the centre spot).

*As built (2026-11-02), `track/match.py` → `tracks/match.npz`, rebuilt when the positions, labels, half times, the direction setting or rated goal clips change:*
- *Restarts: the ball still on the ground 1 s (within 1 m) then kicked (3 m within 1 s), classed by place: kickoff, corner, goal kick, penalty, throw-in, the goalie's ball (still in a penalty area), free kick. 137–271 a game.*
- *Kickoffs are recognised by place and, where a calibration puts the centre a few metres out (FC-1, SAA), by the teams standing split either side of the ball. Kickoffs away from the half starts are goals.*
- *Direction: our kit's side of the ball at each clear kickoff votes for its half (the second half reversed), and so does every goal clip you've rated (😊 ours, 🙁 theirs; where the ball ends). Settable in the tagger.*
- *Goalies: a tracklet that stays in one penalty area in neither team's kit, or deep (median within 6 m of the goal line) and not in the attacking team's kit; the deepest per end per frame. Marked in the tagger (Shift+G, or 🧤 in the click box; `tracks/goalies.json`): the marked tracklet is that team's goalie, and each mark's kit cluster finds more of them (in that kit, 1 s or more, 60% of it in the penalty area that team defends). Found goalies show GK on their boxes, and their map dots take their team's colour; the Game time box shows how often each goalie is found, and the marks.*
- *Out of play: from the ball crossing a line (looking back from a restart on a line or a kickoff, never past the half's start), else from when it stopped, to the restart kick.*

## 6.1 — Event labels and yardsticks (build early)
*As built (2026-11-02): **Shift+E** in the tagger (plain E exports, and every other letter was taken) starts labelling until Esc: T touch, P pass, S shot, V save, G goal, W won the ball, U takes back; space, arrows and , . still move the video. A pass, shot or save is a touch too, labelled once with its own key. Each label is the frame, the kind and where the mouse was (the hovered box's centre when players are shown), matched to the tracklet under it whenever read; the list in the Events box shows who (name, guess, our / their goalie, team). Full windows: ⏺ / ⏹ in the Events box. Labels show on the video as the kind's letter on the player for half a second either side, and on the timeline as ticks (windows as a band).*
Everything after this needs something to be measured against, so labelling comes first.
- **Labelling in the tagger:** an event mode (`E`, then a key) that marks the moment: touch by the player under the mouse, pass, shot, save, goal, ball won. Stored like other picks, as clicks (frame, position) in `tracks/event_labels.json`, matched to tracklets each time so they survive re-tracking.
- **Dense windows:** a few 3–5 minute stretches per game where *every* touch is labelled. These measure touches and possession completely (precision and recall). Six windows across three games, about 20 minutes of play, is a starting target (roughly 150–250 touches).
- **Sparse checks from what's already tagged:** 198 clips carry tags (35 goal, 10 shot, 44 pass, 61 defense, 2 save). Each goal clip should contain a shot and a goal, each shot clip a shot. That is a recall check for shots and goals for free, though only a loose one (clips span seconds, not the moment).
- **Corrections become labels:** fixing an event in the app (6.7) adds a label, like ball fixes do.
- `soccer-clips events eval`: per kind of event, found / wrong / missed against the labels, with a time tolerance (±0.5 s for touches, ±1 s for shots).

## 6.2 — Ball state and control by team
The robust base layer, useful on its own for possession time even before touches are reliable.
- **Ball in metres** from 5.4, with its speed and direction, smoothed over the ball path's gaps where it is short and straight.
- **Airborne:** with one camera there is no height. A ball in the air projects further away than it is, so its ground speed looks too high and it can appear outside the field and come back. Mark stretches where the projected speed or path is impossible for a ball on the ground; positions there are rough, and events in them wait for the ball to come down.
- **In play / out of play:** out from crossing a line until the restart kick (6.0).
- **Control by team, per frame:** for each frame in play, the nearest player of each team to the ball and their distances. Our team controls the ball when one of ours is within reach (~1.5 m, more on the far side where a pixel is a metre) and no opponent is; contested when both are; loose when neither is. Short gaps are bridged when the same team controls the ball on both sides.
- **Team possession:** runs of control by one team, from when they gain it to when the other team gains it or the ball goes out. Loose and contested moments belong to the run in progress.
- Output: `tracks/ball_state.npz` (per frame: metres, speed, airborne, in play, nearest of each team, controlling team) and the possession runs.
- **Possession time:** per team, per half, and with the ball in each third. Frames with no ball path (about a third of frames) count as unknown, reported as such rather than guessed.

## 6.3 — Touches
A touch is a moment the ball's motion changes because a player played it: a kick (it speeds up), a trap (it slows or stops), a turn or deflection (its direction changes).
- **Candidates** from the ball in metres: sharp changes in speed or direction, plus kicks of a parked ball (restarts).
- **Who:** the player whose feet are nearest the ball at that moment, within reach. For our team, their name if they're labelled or guessed, otherwise "our #?"; for theirs, just "opponent" (or "their goalie"). Ties within a few tenths of a metre stay ambiguous rather than guessed.
- **Dribbling:** a player keeping the ball close over several touches is one possession with many touches, not many events.
- The weak spot: crowds around the ball, where several feet are within reach and the ball is hidden. Those touches get a lower confidence and are the first candidates for a learned spotter (6.9).
- Output: `tracks/events.json` with each touch's frame, place, tracklet, team, player, kind and confidence.

## 6.4 — Passes, bad passes and chains
From the touches:
- **A player's possession** runs from their first touch to their last before someone else touches the ball.
- **A pass** is a possession ending in a kick, with the next touch by someone else. The pass is from the kicker to that player.
  - **Completed:** the receiver is on the same team.
  - **Bad pass (lost):** the next touch is an opponent's (interception), or the ball goes out of play off our kick (lost out). A long ball into space that an opponent collects is a bad pass too; one that a teammate runs onto is completed.
  - **Pass into danger:** completed, but the receiver loses the ball before holding it long enough. "Long enough" is a setting to tune against labelled passes, starting at about 2 s of the receiver's possession (or a second touch, whichever is the better rule once measured). Counted as completed in the completion numbers and listed separately, so a player's numbers show both.
  - **Possibly a shot, a cross or a clearance**, not a pass: decided by direction and place (a kick at goal is a shot, 6.5; a long kick out of our own third under pressure with no teammate near its end is a clearance).
- **Properties of a pass:** length, direction relative to our attacking direction (forward, sideways, back), start and end third, and whether it's in the air.
- **Chains:** consecutive completed passes within one team possession. Stats: number of chains of 2, 3, 4+ passes, the longest, and what each ended in (shot, lost, out).
- **Receiver unknown:** when the ball path drops out between the kick and the next touch, the pass is logged with an unknown receiver (and its outcome from who has the ball next), not dropped.

## 6.5 — Shots, goals and saves
Few per game and high value, so these should be near-perfect, and the game log offers them for a quick check by eye.
- **Shot:** a kick in the attacking half whose direction, extended, passes between the goal posts (from the calibration) or close to them, fast enough to be a shot rather than a pass. Who took it is the touch's player.
- **What happened next** (using restarts, which are strong signals):
  - **Goal:** the ball crosses the goal line between the posts, and the next restart is a kickoff from the centre spot. The kickoff settles the ambiguity of a ball in the air over the bar, which looks like a goal on the ground.
  - **Save:** the next touch is the defending keeper's (6.0), in or near their area. Caught, parried and pushed out for a corner all count; a corner restart right after also means saved or deflected.
  - **Blocked:** the next touch is a defending field player near the shooter.
  - **Wide / over:** the ball crosses the goal line outside the posts, or over the bar with no kickoff after, and play restarts with a goal kick.
- **Yardstick:** the 35 goal and 10 shot clips, plus each game's final score, entered in the tagger and stored in `game.json` (goals for and against: a cheap, exact check).

## 6.6 — Defence: balls won and lost
From possession changes between teams, by how they happened:
- **Interception:** the other team's pass ends at our player.
- **Tackle:** the other team's player had the ball in their own possession (touched it at least once), and our player took it from close range.
- **Recovery:** a loose ball (after a clearance, a deflection, a 50-50) picked up by us.
- **Lost** is the same seen from the other side: our bad passes, being tackled, or a loose ball to them.
- Where it happened (third) matters most here: winning the ball in their half and losing it in ours are the two numbers coaches usually want.

## 6.7 — Game log, stats and corrections in the app
- **Events in the tagger:** an event list beside the video (time, players, kind; click to jump), events marked on the timeline strip, and a filter by player or kind. Fixing an event (wrong passer, not a shot, missed save) is a click in the list or the video, saved as an event label (6.1), and the layers above recompute.
- **Game page** (`/events/<game>` or a section of the Games page): the stats above, possession bars per half and per third, pass map (lines on the field for one player's or the team's passes, completed and lost in different colours), shot map.
- **Player view:** a filter for one player's events (her passes, her shots, balls she won), played like the ⚽ filter. This is where per-player numbers become clips.
- `soccer-clips events build [games]` and `events stats [games]`, and run automatically after re-tracking or labelling, like following by place.

## 6.8 — Good and bad plays
Two steps, the second only if the first proves too blunt.
- **Rules first**, so every rating can be explained:
  - Good: a completed forward pass that moves the ball into the final third or past opponents near the ball; a chain of 4+ passes; a shot on target; a save; winning the ball in their half.
  - Bad: a pass intercepted in our own third; a pass into danger; losing the ball in our own third; a shot well wide from far out.
- **Then a value of each action** from where it moves the ball: expected threat (xT), a published grid of how likely a team is to score from each zone of the field, learned from professional event data. The value of a pass is the grid at its end minus the grid at its start; losing the ball costs the grid's value at that place for the other team. A 9v9 field can be mapped onto the grid in field-relative terms (fractions of length and width), not metres. It rates moves no rule lists, and it's the same measure for every player. It is professional, not youth, soccer, so treat its numbers as a ranking, not as truth.
- **Output:** each event's rating feeds suggested clips: the best and worst moments of a player's game, offered for accept / reject into normal clips (4.4's plan).

## 6.9 — Later: a learned spotter for what rules miss
If touches in crowds or airborne passes stay weak, fine-tune an action-spotting model on our labelled events (RunPod, with your OK), using its output as touch candidates for 6.3. It needs a few hundred labels per kind of event, which the corrections from 6.7 build up over a season.

## Yardsticks
- **Touches (dense windows):** found / wrong / missed at ±0.5 s; of the found, the right player. Rough first targets: 70% found, 80% of found with the right player where the player is named.
- **Team possession (dense windows):** share of frames with the right controlling team. Target: 90%+ of frames with a ball path.
- **Passes:** completed vs lost right for 85%+ of labelled passes.
- **Shots and goals:** every goal clip has a goal event; goals per game match the score, if recorded.
- **Possession by third:** checked by eye against the map for a few stretches; no ground truth beyond that.

## Risks
- **Ball path gaps:** no ball position on about a third of frames, often exactly at crowds and tackles. Bridging short gaps helps; longer ones leave events unknown. More ball labels and retraining (4.1) help directly.
- **Ball in the air:** no height means lofted passes, crosses and shots over the bar are hard. Restarts (kickoff, goal kick, corner) settle the outcome for shots; for passes, the receiver's position does.
- **Far side precision:** a pixel at the far touchline is metres, so "within reach" there is loose. Reach grows with distance from the camera.
- **Unnamed players:** 45% of our on-field time is named today, so many events read "our #?". Team-level stats don't care; per-player numbers get better as labelling goes on, and recompute without re-running anything.
- **Keepers and the referee** share 'other' kit; misreading the referee as a keeper would invent saves. Keepers are found by staying in one area, which the referee doesn't.
- **Errors compound:** a missed touch can turn one pass into another, or a completed pass into a lost one. Each layer gets its own yardstick so problems are found where they start.

## Tasks
- [x] 6.0 5.4 positions stored per box and for the ball, with the off-pitch filter (Phase 5; 2026-11-02, `track/positions.py`)
- [x] 6.0 Attacking direction per half (from kickoffs and rated goal clips, override in the tagger's Game time box; 2026-11-02)
- [x] 6.0 Keepers found per team per half (90–100% of the time the ball is near their goal in most game-halves; weak spots in Findings)
- [x] 6.0 Ball out of play and restart kinds
- [x] 6.0 Mark a goalie in the tagger (Shift+G ours, or 🧤 our / their goalie in the click box; 2026-11-03): the tracklet, and anyone in its kit around that team's goal; GK on found goalies' boxes
- [ ] 6.0 Goalie gaps: mark FC-1's our goalie and PRSC's theirs (a mark in each half)
- [x] 6.1 Event labelling mode in the tagger (Shift+E: T touch, P pass, S shot, V save, G goal, W won), `event_labels.json`, full windows in the Events box (2026-11-02)
- [ ] 6.1 `events eval`: comes with the first events found (6.2 / 6.3)
- [ ] 6.1 Label dense windows (six windows, three games)
- [ ] 6.2 Ball state and control by team; possession time per team, half and third
- [ ] 6.3 Touches, with player and confidence; measured on the dense windows
- [ ] 6.4 Passes (completed / into danger / lost / out), properties, chains; tune how long the receiver must keep the ball
- [ ] 6.5 Final score per game: entry in the tagger, stored in `game.json`
- [ ] 6.5 Shots and outcomes (goal, save, blocked, wide)
- [ ] 6.6 Balls won and lost, by kind and third
- [ ] 6.7 Event list and timeline in the tagger, corrections, stats page, Player view event filter
- [ ] 6.8 Good / bad rules, then expected threat; suggested clips
- [ ] 6.9 (later) learned spotter if the rules plateau

## Decided (2026-11-02)
- **Passes into danger** are their own category: completed, but the receiver loses the ball too soon. How long they must keep it is a setting to tune against labels.
- **The other team** is counted as a team only (possession, passes, shots); no per-player numbers for them.
- **Final scores:** you'll enter each game's score in the tagger; it's the exact check on goals.
- **Keepers:** our keeper isn't assumed to be one player (it is this season). Found per half; "our goalie" / "their goalie" in the log when not named.

## Open questions
- None yet.

## Findings
- **2026-11-02** — 6.0 on all seven games (3–6 s a game). **Direction:** the kickoffs alone were sure in six games; SAA's opponent kit is mostly the crowd, so too few kickoffs were clear, and it came out wrong. Your rated goal clips settle it: 😊 goal clips are ours and 🙁 theirs (by their notes), and where the ball ends says which goal each was at. Where the kickoffs were sure, 25 of 25 rated goal clips agree with them; added as votes, all seven games are sure, SAA attacking right in the 1st half. **Kickoffs:** 22 of 35 goal clips have a kickoff found within about a minute after; SAA's calibration puts the centre spot 3–4 m off and FC-1's views put kickoffs up to 9 m off, hence the box around the centre and the split-formation test. Kickoffs found per game (5–7 goals) don't match the goal clips one for one (they include the other team's goals, which aren't all clipped); the scores, once entered, will measure them. **Goalies:** found 90–100% of the time the ball is within 20 m of their goal in 19 of 28 game-halves; weak: FC-1 ours (18–23%: its calibration near the goals reads the goal at x ≈ 23 m of a 30 m half-length, so the goalie reads 7 m out), PRSC theirs (11–26%), FC Stars 2H theirs (35%), Elite Strikers 1H theirs (56%). Goalies' kits are often clustered in with a team's, which is why depth counts as well as kit. **EBU's** stored 2nd-half start (17.5 min) is wrong, as noted when the times were suggested; set it in the tagger.
- **2026-11-03** — Goalie marks. Simulated with one mark per half on the longest goalie tracklet the rule already found (your marks on the real goalie should do at least as well): FC-1 ours 23% / 18% → 40% / 62% (1H / 2H), PRSC theirs 11% / 26% → 53% / 63%, FC Stars theirs 88% / 35% → 98% / 63%. One mark helps mostly the half it's in: the goalie's kit falls in different clusters in different light, so each mark adds its own cluster.
