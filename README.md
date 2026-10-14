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
