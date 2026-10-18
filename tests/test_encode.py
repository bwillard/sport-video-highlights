import dataclasses

import pytest
from fastapi.testclient import TestClient

from conftest import make_video
from soccer_clips.config import ConfigError, _choice, _find_tool, _int
from soccer_clips.encode import choose_encoder, gpu_works
from soccer_clips.server.app import create_app
from soccer_clips.store import Store
from test_reels import decode_errors, probe

GAME_HD = "hd-2026-02-01"  # 1080p like a Veo download: decoded on the GPU
GAME_SMALL = "small-2026-02-08"  # scaled and letterboxed on the CPU, then uploaded


def test_env_values_are_checked(monkeypatch):
    monkeypatch.setenv("SOCCER_GPU_QP", "60")
    with pytest.raises(ConfigError, match="SOCCER_GPU_QP"):
        _int("SOCCER_GPU_QP", 22, 1, 51)
    monkeypatch.setenv("SOCCER_GPU_QP", "18")
    assert _int("SOCCER_GPU_QP", 22, 1, 51) == 18
    monkeypatch.delenv("SOCCER_GPU_QP")
    assert _int("SOCCER_GPU_QP", 22, 1, 51) == 22
    monkeypatch.setenv("SOCCER_ENCODER", "fast")
    with pytest.raises(ConfigError, match="auto, gpu, cpu"):
        _choice("SOCCER_ENCODER", "auto", ("auto", "gpu", "cpu"))


def test_choose_encoder(settings):
    preview = choose_encoder(dataclasses.replace(settings, encoder="gpu"), 960, 540, preview=True)
    assert not preview.gpu and preview.cpu_speed == "ultrafast"
    cpu = choose_encoder(dataclasses.replace(settings, encoder="cpu", cpu_crf=18, cpu_speed="slow"), 1920, 1080, False)
    assert (cpu.gpu, cpu.quality, cpu.cpu_speed) == (False, 18, "slow")
    # no GPU: auto falls back to the CPU, gpu is an error
    missing = dataclasses.replace(settings, gpu_device="/dev/dri/no-such-device")
    assert not choose_encoder(missing, 1920, 1080, False).gpu
    with pytest.raises(RuntimeError, match="SOCCER_ENCODER=gpu"):
        choose_encoder(dataclasses.replace(missing, encoder="gpu"), 1920, 1080, False)


@pytest.mark.skipif(not gpu_works(_find_tool("ffmpeg", "FFMPEG"), "/dev/dri/renderD128", 1920, 1080),
                    reason="no working GPU encoder")
def test_gpu_render_matches_cpu_render(settings):
    """Same reel at 1080p on the GPU and the CPU: same frames and timings, clean files."""
    make_video(settings, f"2026/{GAME_HD}.mp4", duration_s=4, size="1920x1080")
    make_video(settings, f"2026/{GAME_SMALL}.mp4", duration_s=4, audio=False)
    results = {}
    for kind in ("cpu", "gpu"):
        app = create_app(Store(dataclasses.replace(settings, encoder=kind)))
        with TestClient(app) as c:
            if kind == "cpu":
                hd = c.post(f"/api/games/{GAME_HD}/clips", json={"start_s": 0.5, "end_s": 2.0}).json()["id"]
                c.patch(f"/api/games/{GAME_HD}/clips/{hd}", json={
                    "note": "Wins it", "callout": {"t_s": 1.0, "x": 0.5, "y": 0.5, "r": 0.1}})
                small = c.post(f"/api/games/{GAME_SMALL}/clips", json={"start_s": 1.0, "end_s": 2.5}).json()["id"]
                reel = c.post("/api/reels", json={"title": "GPU test", "card_s": 1, "items": [
                    {"game_id": GAME_HD, "clip_id": hd}, {"game_id": GAME_SMALL, "clip_id": small}]}).json()["reel"]
            job = c.post(f"/api/reels/{reel['id']}/render", json={"preset": "landscape_1080p"}).json()
            app.state.jobs.wait()
            job = c.get(f"/api/jobs/{job['id']}").json()
            assert job["status"] == "done", job["error"]
            [r] = c.get(f"/api/reels/{reel['id']}").json()["renders"]
        out = settings.reels_dir / reel["id"] / "landscape_1080p.mp4"
        streams = probe(settings, out)
        assert decode_errors(settings, out) == ""
        assert (streams["video"]["width"], streams["video"]["height"]) == (1920, 1080)
        assert float(streams["audio"]["duration"]) == pytest.approx(float(streams["video"]["duration"]), abs=0.05)
        results[kind] = (r, int(streams["video"]["nb_read_frames"]))

    (cpu, cpu_frames), (gpu, gpu_frames) = results["cpu"], results["gpu"]
    assert cpu["encoder"].startswith("CPU") and gpu["encoder"].startswith("GPU")
    assert gpu_frames == cpu_frames
    for a, b in zip(cpu["segments"], gpu["segments"]):
        assert a["reel_start_s"] == pytest.approx(b["reel_start_s"], abs=0.01)
        assert a["freeze_at_s"] == pytest.approx(b["freeze_at_s"], abs=0.01)
