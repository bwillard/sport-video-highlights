import pytest

from conftest import make_video
from soccer_clips import gpu
from soccer_clips.models import Clip
from soccer_clips.scan import scan
from soccer_clips.store import Store
from soccer_clips.track import ball_labels, bench

GAME = "gu11-test-2026-10-10"


@pytest.fixture
def store(settings):
    make_video(settings, f"2026/{GAME}.mp4", duration_s=4)
    store = Store(settings)
    scan(settings, store)
    with store.edit_game(GAME) as g:
        g.clips.append(Clip(id="c1", start_s=1.0, end_s=2.0))
    bench.add_clip(store, GAME, "c1", "one")
    ball_labels.add_tasks(store, "bench", ball_labels.sample_bench(store, n=2))
    ball_labels.add_tasks(store, "train", [ball_labels.BallTask(game_id=GAME, frame=100)])
    ball_labels.frame_image(store, store.load_game(GAME), 100)  # a cached train frame: must not be uploaded
    ball_labels.add_label(settings, ball_labels.BallLabel(game_id=GAME, frame=100, state="none"))
    return store


def test_ball_train_uploads_only_what_it_needs(store):
    with pytest.raises(gpu.ConfigError, match="ball export"):
        gpu.ball_train_job(store, stock=["yolo11m"], train_bases=["yolo11m"], epochs=1)
    data = ball_labels.labels_dir(store.settings) / "dataset" / "data.yaml"
    data.parent.mkdir(parents=True)
    data.write_text("path: x\n")
    job = gpu.ball_train_job(store, stock=["yolo11m"], train_bases=["yolo11m"], epochs=1)
    frames = [u for u in job.uploads if u.startswith("labels/frames/")]
    bench_frames = {t.frame for t in ball_labels.load_tasks(store.settings).sets["bench"]}
    assert sorted(frames) == sorted(f"labels/frames/{GAME}/{f}.jpg" for f in bench_frames)
    assert f"games/2026/{GAME}/game.json" in job.uploads
    assert not any(u.startswith(("full", "clips", "bench", "models")) or u.endswith("probe.json") for u in job.uploads)
    root = gpu.data_root(store)
    assert all((root / u).exists() for u in job.uploads)


def test_pod_deleted_when_a_step_fails(store, monkeypatch):
    calls = []

    def fake_api(method, path, body=None):
        calls.append((method, path))
        if method == "POST":
            return {"id": "pod1", "costPerHr": 0.7}
        return []

    monkeypatch.setattr(gpu, "api", fake_api)
    monkeypatch.setattr(gpu.Runner, "wait_for_ssh", lambda self: (_ for _ in ()).throw(TimeoutError("no ssh")))
    job = gpu.Job(name="t", uploads=[], script="true", downloads=[])
    runner = gpu.Runner(store, job, gpu_types=["X"], max_hours=1, log=lambda m: None)
    with pytest.raises(TimeoutError):
        runner.run()
    assert ("DELETE", "/pods/pod1") in calls
    log = (runner.dir / "log.txt").read_text()
    assert "FAILED" in log and "pod pod1 deleted" in log
