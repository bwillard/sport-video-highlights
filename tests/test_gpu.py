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
    ball_labels.add_tasks(store, "train", [ball_labels.BallTask(game_id=GAME, frame=110)])
    ball_labels.frame_image(store, store.load_game(GAME), 110)  # a cached train frame: must not be uploaded
    ball_labels.add_label(settings, ball_labels.BallLabel(game_id=GAME, frame=100, state="none"))
    return store


def test_ball_train_uploads_only_what_it_needs(store):
    with pytest.raises(gpu.ConfigError, match="ball export"):
        gpu.ball_train_job(store, stock=["yolo11m"], train_bases=["yolo11m"], epochs=1)
    data = ball_labels.labels_dir(store.settings) / "dataset" / "data.yaml"
    data.parent.mkdir(parents=True)
    data.write_text("path: x\n")
    (data.parent / "val_frames.json").write_text(f'[["{GAME}", 100]]')  # a train frame used for validation
    job = gpu.ball_train_job(store, stock=["yolo11m"], train_bases=["yolo11m"], epochs=1)
    frames = [u for u in job.uploads if u.startswith("labels/frames/")]
    bench_frames = {t.frame for t in ball_labels.load_tasks(store.settings).sets["bench"]}
    assert sorted(frames) == sorted(f"labels/frames/{GAME}/{f}.jpg" for f in bench_frames | {100})
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


def test_partial_results_downloaded_when_the_job_fails(store, monkeypatch):
    calls = []
    monkeypatch.setattr(gpu, "api", lambda m, p, b=None: {"id": "pod1", "costPerHr": 0.9} if m == "POST" else [])
    monkeypatch.setattr(gpu.Runner, "wait_for_ssh", lambda self: setattr(self, "ssh_args", ["x"]))
    for step in ("arm_dead_mans_switch", "upload"):
        monkeypatch.setattr(gpu.Runner, step, lambda self: None)

    def boom(self):
        raise RuntimeError("job script failed")

    monkeypatch.setattr(gpu.Runner, "run_script", boom)
    monkeypatch.setattr(gpu.Runner, "download", lambda self: calls.append("download"))
    monkeypatch.setattr(gpu.Runner, "teardown", lambda self: calls.append("teardown"))
    runner = gpu.Runner(store, gpu.Job(name="t", uploads=[], script="", downloads=[]), ["X"], 1, lambda m: None)
    with pytest.raises(RuntimeError, match="job script failed"):
        runner.run()
    assert calls == ["download", "teardown"]


def test_volume_plan(tmp_path):
    (tmp_path / "full" / "2026").mkdir(parents=True)
    (tmp_path / "full" / "2026" / "a.mp4").write_bytes(b"x" * 10)
    (tmp_path / "full" / "2026" / "b.mp4").write_bytes(b"x" * 20)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m.pt").write_bytes(b"x" * 5)
    listing = "full/2026/a.mp4\t10\nfull/2026/b.mp4\t7\n"  # b was cut short last time
    files, missing = gpu.volume_plan(tmp_path, ["full", "models/m.pt"], listing)
    assert files == ["full/2026/a.mp4", "full/2026/b.mp4", "models/m.pt"]
    assert missing == ["full/2026/b.mp4", "models/m.pt"]


def test_pod_uses_the_volume_and_falls_back(store, monkeypatch):
    import json as _json

    vol = {"id": "vol1", "dataCenterId": "US-KS-2", "size": 50}
    path = gpu.data_root(store) / "gpu" / "volume.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps(vol))
    bodies = []

    def fake_api(method, p, body=None):
        if method == "POST":
            bodies.append(dict(body))  # the runner reuses the dict when it falls back
            if "networkVolumeId" in body:
                raise RuntimeError('RunPod API POST /pods: 500 {"error":"There are no instances currently available"}')
            return {"id": "pod1", "costPerHr": 0.9}
        return {}

    monkeypatch.setattr(gpu, "api", fake_api)
    monkeypatch.setattr(gpu.time, "sleep", lambda s: None)
    job = gpu.Job(name="t", uploads=[], script="", downloads=[], volume_files=["full/a.mp4"])
    assert gpu.Runner(store, gpu.Job(name="t", uploads=["x"], script="", downloads=[]), ["4090"], 1,
                      lambda m: None).volume is None  # nothing on the volume: any data centre
    runner = gpu.Runner(store, job, ["4090"], 1, lambda m: None)
    assert runner.volume == vol
    runner.start_pod(wait_s=0)
    first, last = bodies[0], bodies[-1]
    assert first["networkVolumeId"] == "vol1" and first["dataCenterIds"] == ["US-KS-2"] and first["volumeMountPath"] == "/vol"
    assert "NVIDIA RTX 6000 Ada Generation" in first["gpuTypeIds"] and "countryCodes" not in first
    assert "networkVolumeId" not in last and last["countryCodes"] == ["US"] and last["gpuTypeIds"] == ["4090"]
    assert runner.volume is None  # so everything gets uploaded
    bodies.clear()
    with pytest.raises(RuntimeError, match="no instances"):
        gpu.Runner(store, job, ["4090"], 1, lambda m: None, fallback=False).start_pod(wait_s=0)


def test_s3_sync_uploads_only_what_is_missing(store, monkeypatch):
    import json as _json

    root = gpu.data_root(store)
    (root / "gpu").mkdir(exist_ok=True)
    (root / "gpu" / "volume.json").write_text(_json.dumps({"id": "vol1", "dataCenterId": "US-KS-2"}))
    (root / "vids").mkdir(parents=True, exist_ok=True)
    for name, size in (("a.mp4", 10), ("b.mp4", 20), ("c.mp4", 30)):
        (root / "vids" / name).write_bytes(b"x" * size)
    uploaded = []

    class FakeS3:
        def get_paginator(self, _):
            class P:
                def paginate(self, Bucket, Prefix):
                    assert Bucket == "vol1" and Prefix == "data/"
                    # a is there, b was cut short, c is missing
                    return [{"Contents": [{"Key": "data/vids/a.mp4", "Size": 10},
                                          {"Key": "data/vids/b.mp4", "Size": 3}]}]
            return P()

        def upload_file(self, path, bucket, key, Config, Callback):
            uploaded.append((key, Config.multipart_chunksize))
            Callback(1)

    monkeypatch.setattr(gpu, "s3_client", lambda store: FakeS3())
    files = gpu.s3_sync(store, ["vids"], log=lambda m: None)
    assert files == ["vids/a.mp4", "vids/b.mp4", "vids/c.mp4"]
    assert uploaded == [("data/vids/b.mp4", gpu.S3_PART_BYTES), ("data/vids/c.mp4", gpu.S3_PART_BYTES)]
