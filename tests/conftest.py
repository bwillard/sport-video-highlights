import subprocess
from pathlib import Path

import pytest

from soccer_clips.config import Settings, _find_tool


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    dirs = {name: tmp_path / name for name in ["full", "proxy", "clips", "games", "reels", "gallery"]}
    for d in dirs.values():
        d.mkdir()
    return Settings(
        full_dir=dirs["full"],
        proxy_dir=dirs["proxy"],
        clips_dir=dirs["clips"],
        games_dir=dirs["games"],
        reels_dir=dirs["reels"],
        gallery_dir=dirs["gallery"],
        host="127.0.0.1",
        port=0,
        ffmpeg=_find_tool("ffmpeg", "FFMPEG"),
        ffprobe=_find_tool("ffprobe", "FFPROBE"),
    )


def make_video(settings: Settings, rel: str) -> Path:
    """3s 29.97 fps test pattern with audio, laid out like a Veo download."""
    out = settings.full_dir / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [settings.ffmpeg, "-v", "error",
         "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30000/1001:duration=3",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(out)],
        check=True,
    )
    return out
