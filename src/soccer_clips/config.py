"""
Settings loaded from the repo's .env (gitignored) plus the process environment.
All machine-specific paths live in .env; see .env.example for the keys.
"""

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
MIN_FFMPEG_MAJOR = 6


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    full_dir: Path
    proxy_dir: Path
    clips_dir: Path
    games_dir: Path
    reels_dir: Path
    gallery_dir: Path
    host: str
    port: int
    ffmpeg: str
    ffprobe: str
    default_player: int | None = None  # jersey number the F key tags; old "her" flags become this
    # Video encoding for full-size reel renders (see .env.example and encode.py).
    encoder: str = "auto"  # auto | gpu | cpu
    gpu_device: str = "/dev/dri/renderD128"
    gpu_qp: int = 22  # GPU quality: lower = better picture, bigger file
    cpu_crf: int = 20  # CPU quality: same scale
    cpu_speed: str = "medium"  # x264 preset: slower = smaller file at the same quality


def _required_dir(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"{name} is not set; copy .env.example to .env and fill it in")
    return Path(value)


def _find_tool(name: str, override_var: str) -> str:
    """
    Prefer an explicit override, then the env this interpreter belongs to, then
    the active conda env, then PATH. The env checks matter because miniconda's
    base env ships an ffmpeg 4.2 that is too old (no xfade), and running the
    env's python by full path leaves CONDA_PREFIX pointing at base.
    """
    override = os.environ.get(override_var)
    if override:
        return override
    for prefix in (sys.prefix, os.environ.get("CONDA_PREFIX")):
        if prefix:
            candidate = Path(prefix) / "bin" / name
            if candidate.exists():
                return str(candidate)
    found = shutil.which(name)
    if not found:
        raise ConfigError(f"{name} not found; activate soccer_clips_env or set {override_var}")
    return found


def check_ffmpeg_version(ffmpeg: str) -> int:
    out = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, check=True).stdout
    match = re.search(r"ffmpeg version n?(\d+)", out)
    if not match:
        raise ConfigError(f"could not parse ffmpeg version from: {out.splitlines()[0]}")
    major = int(match.group(1))
    if major < MIN_FFMPEG_MAJOR:
        raise ConfigError(
            f"{ffmpeg} is version {major}; need >= {MIN_FFMPEG_MAJOR} (is soccer_clips_env active?)"
        )
    return major


ENCODERS = ("auto", "gpu", "cpu")
X264_SPEEDS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow")


def _int(name: str, default: int, lo: int, hi: int) -> int:
    value = os.environ.get(name) or str(default)
    if not value.isdigit() or not lo <= int(value) <= hi:
        raise ConfigError(f"{name} must be a whole number from {lo} to {hi}, not {value!r}")
    return int(value)


def _choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    value = os.environ.get(name) or default
    if value not in choices:
        raise ConfigError(f"{name} must be one of {', '.join(choices)}, not {value!r}")
    return value


@cache
def get_settings() -> Settings:
    # Real environment variables win over .env, so tests and one-offs can override.
    load_dotenv(REPO_ROOT / ".env", override=False)
    ffmpeg = _find_tool("ffmpeg", "FFMPEG")
    check_ffmpeg_version(ffmpeg)
    return Settings(
        full_dir=_required_dir("SOCCER_FULL_DIR"),
        proxy_dir=_required_dir("SOCCER_PROXY_DIR"),
        clips_dir=_required_dir("SOCCER_CLIPS_DIR"),
        games_dir=_required_dir("SOCCER_GAMES_DIR"),
        reels_dir=_required_dir("SOCCER_REELS_DIR"),
        gallery_dir=_required_dir("SOCCER_GALLERY_DIR"),
        host=os.environ.get("SOCCER_HOST", "127.0.0.1"),
        port=int(os.environ.get("SOCCER_PORT", "8000")),
        ffmpeg=ffmpeg,
        ffprobe=_find_tool("ffprobe", "FFPROBE"),
        default_player=int(os.environ["SOCCER_DEFAULT_PLAYER"]) if os.environ.get("SOCCER_DEFAULT_PLAYER") else None,
        encoder=_choice("SOCCER_ENCODER", "auto", ENCODERS),
        gpu_device=os.environ.get("SOCCER_GPU_DEVICE") or "/dev/dri/renderD128",
        gpu_qp=_int("SOCCER_GPU_QP", 22, 1, 51),
        cpu_crf=_int("SOCCER_CPU_CRF", 20, 0, 51),
        cpu_speed=_choice("SOCCER_CPU_SPEED", "medium", X264_SPEEDS),
    )
