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
 - `soccer-clips serve` runs the web server on `SOCCER_HOST:SOCCER_PORT` from `.env`.
 - Open `http://<server>:<port>/` to tag clips; the keyboard shortcuts are listed under the video. Tags and their 1–9 hotkeys are in `$SOCCER_GAMES_DIR/tags.json` (created with defaults on first run; edit by hand).
 - Exported clips go to `$SOCCER_CLIPS_DIR/<year>/<game_id>/<clip_id>.mp4`.

 - Reels: in the tagger, filter the clip list and press "Add shown to reel". Open `/reel` to render (preview or full), watch, jump back to the full game with `G`, and download. Renders go to `$SOCCER_REELS_DIR/<reel_id>/<preset>.mp4`.
 - Title cards: each reel gets an opening card (its title) and a card before each clip (time in the game + the clip's note). Set their length in the "cards" box on the reel page before rendering (0 turns them off).
 - Callouts: in the tagger, select a clip, pause on a moment inside it, press `C` and click on her (drag to size the ring). Reels freeze there for 1.5s with her spotlighted; set the length in the "freeze" box on the reel page (0 turns it off).
 - Game info: set each game's date and a short description (e.g. "Home vs SAA, 3-1 win") in the box at the top of the tagger's right-hand panel. Reel opening cards show them under the reel title.
