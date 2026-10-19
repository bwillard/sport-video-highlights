# Soccer Clips

Tag clips from full-game Veo videos, highlight a player, and build highlight reels. See [plan.md](plan.md).

# Setup
 - `conda env create -f environment.yml` (creates `soccer_clips_env`; includes ffmpeg >= 6 from conda-forge)
 - `conda activate soccer_clips_env`
 - `pip install -e .`
 - `cp .env.example .env` and fill in the data directories. `.env` is gitignored; never commit real paths.

# Dev Instructions
 - `conda activate soccer_clips_env` (the ffmpeg in miniconda's base env is too old; config checks for >= 6)
 - Run tests: `pytest`

# Usage
 - Drop new Veo downloads into `$SOCCER_FULL_DIR/<year>/`. The filename (minus `.mp4`) becomes the game id; it should end in `YYYY-MM-DD`.
 - `soccer-clips scan` registers new videos (also runs when the server starts).
 - `soccer-clips list` lists registered games.
 - `soccer-clips serve` runs the web server on `SOCCER_HOST:SOCCER_PORT` from `.env`. Restart it after code changes: the pages reload from disk, but the server keeps running the code it started with.
 - Open `http://<server>:<port>/` to tag clips; the keyboard shortcuts are listed under the video. Tags and their 1–9 hotkeys are in `$SOCCER_GAMES_DIR/tags.json` (created with defaults on first run; edit by hand). `F` tags the player number in `SOCCER_DEFAULT_PLAYER` (`.env`), `P` adds any other number, `T` marks a team play; the "who" filter picks a player or team plays for exporting or reels.
 - Exported clips go to `$SOCCER_CLIPS_DIR/<year>/<game_id>/<clip_id>.mp4`.

 - Reels: in the tagger, filter the clip list and press "Add shown to reel". Open `/reel` to render (preview or full), watch, jump back to the full game with `G`, and download. Each render can include your notes and callouts, callouts only, or neither (the "with" menu), so you can keep an annotated version and share a clean one. Renders go to `$SOCCER_REELS_DIR/<reel_id>/<preset>[-<variant>].mp4`. Full renders use the GPU's video encoder when there is one (about 5× faster here); speed and quality settings are in `.env` (see `.env.example`).
 - Title cards: each reel gets an opening card (its title) and a card before each clip (time in the game + the clip's note). Each card stays up long enough to read its text (0.75s + 17 characters/second); the "cards ≥" box on the reel page sets the minimum (default 2s; 0 turns cards off).
 - Callouts: in the tagger, select a clip, pause on a moment inside it, press `C` and click on her (drag to size the ring). Reels freeze there with her spotlighted (1.5s with caption overlays, 0.75s with title cards); set the length in the "freeze" box on the reel page (0 turns it off).
 - Game info: set each game's date and a short description (e.g. "Home vs SAA, 3-1 win") in the box at the top of the tagger's right-hand panel. Reel opening cards show them under the reel title.
 - Descriptions: each clip's game time and note are shown as a caption over the start of the clip (default) or as a title card before it; pick per reel with the "descriptions" menu on the reel page.
 - Tracking benchmark clips: a small fixed set of test clips in `bench/` next to the games folder (or `SOCCER_BENCH_DIR`). The list is `bench/clips.json`: edit a clip's `start_s`/`end_s` (game time), or delete an entry, then run `soccer-clips bench cut` to recut only what changed (`--force` recuts all). `soccer-clips bench add <game_id> <clip_id> <name> [--why "..."]` adds a tagged clip, with its callout if it has one; `soccer-clips bench` lists the set. `soccer-clips bench run [variants]` runs the detection/tracking benchmark into `bench/results/<variant>/` (needs the tracking packages from `environment.yml`), and `soccer-clips bench draw <variant> [clips] [--results results-l4]` draws its boxes onto the clips for watching (`--results` picks another results folder, e.g. one copied back from a GPU machine).
 - Ball labels: open `/label-ball` (linked from the tagger) to mark where the ball is, frame by frame. Click the ball (Shift+click if it's hidden behind someone, drag to size it), `N` for no ball, `S` if you can't tell; a magnifier follows the cursor, and holding `Q`/`E` shows 3 frames earlier/later. Frames come in sets kept apart so models are never scored on frames they trained on: `soccer-clips ball sample bench` (150 frames over the benchmark clips, for scoring) and `soccer-clips ball sample train -n 200` (random frames from the games, for training). `soccer-clips ball` shows progress; `soccer-clips ball score [--results results-l4]` scores each benchmark variant's ball detections against the bench labels. Labels live in `labels/` next to the games folder (`ball.jsonl`, latest answer wins).
