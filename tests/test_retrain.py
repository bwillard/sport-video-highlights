import json
import subprocess
import time
from pathlib import Path

import pytest

from soccer_clips import gpu, retrain
from soccer_clips.store import Store


def fake_run(store, name, minutes, dollars, failed=False):
    d = gpu.data_root(store) / "gpu" / f"20261027-0{len(list((gpu.data_root(store) / 'gpu').glob('*')))}0000-{name}"
    d.mkdir(parents=True)
    (d / "log.txt").write_text(("FAILED: boom\n" if failed else "") + f"08:45:22 total {minutes} min, about ${dollars}\n")
    return d


def test_estimate_and_model_info_from_old_runs(settings):
    store = Store(settings)
    (gpu.data_root(store) / "gpu").mkdir(parents=True)
    assert retrain.estimate(store)["dollars"] == pytest.approx(1.26)  # no runs yet: the defaults
    run = fake_run(store, "reid-train", 52, 1.03)
    fake_run(store, "reid-sigs", 10, 0.15)
    fake_run(store, "reid-train", 1, 0.01, failed=True)  # a failed run isn't the yardstick
    assert retrain.estimate(store) == {"minutes": 68, "dollars": 1.3, "max_dollars": 3.75}
    assert retrain.model_info(store) is None  # no model yet
    from soccer_clips.track import reid
    from soccer_clips.track.benchmark import models_dir

    (run / "results" / "out").mkdir(parents=True)
    old_style = {"g": {"stretches": 30, "her_top1": 0.7, "chance": 0.1, "mates_n": 10, "mates_top1": 0.9, "mates_chance": 0.2}}
    (run / "results" / "out" / "reid_results.json").write_text(json.dumps({"stock": old_style, "tuned": old_style}))
    (gpu.data_root(store) / "reid").mkdir()
    (gpu.data_root(store) / "reid" / "index.json").write_text(json.dumps({"g": {"fps": 30, "named": [[1, 0, 1800, 38], [2, 0, 900, 5]]}}))
    models_dir(store).mkdir(parents=True, exist_ok=True)
    (models_dir(store) / reid.TUNED).write_bytes(b"x")
    info = retrain.model_info(store)
    assert info["run"] == run.name and info["scores"]["all_top1"] == pytest.approx(0.75) and info["train_dollars"] == 1.03
    hist = retrain.history(store)
    assert [h["run"] for h in hist] == [run.name] and hist[0]["in_use"] and hist[0]["labelled_minutes"] == 2
    assert info["games"] == ["g"] and info["labelled_minutes"] == {"5": 0.5, "38": 1.0}


def test_keeps_the_old_model_only_if_the_new_one_is_clearly_worse():
    stock = {"all_top1": 0.3, "player_top1": 0.3}
    old = {"all_top1": 0.73}  # an older run: no per-player score, so every labelled moment is compared
    assert retrain._better_enough({"all_top1": 0.70, "player_top1": 0.8}, stock, old)[0]
    assert not retrain._better_enough({"all_top1": 0.60, "player_top1": 0.9}, stock, old)[0]
    old = {"all_top1": 0.73, "player_top1": 0.85}
    ok, why = retrain._better_enough({"all_top1": 0.60, "player_top1": 0.83}, stock, old)
    assert ok and why == "players picked out 85% → 83% (stock 30%)"
    assert not retrain._better_enough({"all_top1": 0.3, "player_top1": 0.25}, stock, None)[0]  # no better than stock
    assert retrain._better_enough({"all_top1": 0.5}, None, None)[0] and not retrain._better_enough(None, stock, old)[0]


def test_status_notices_a_dead_process(settings):
    store = Store(settings)
    retrain._save_status(store, state="running", pid=99999999, step="training")
    assert retrain.status(store)["state"] == "failed"
    proc = subprocess.Popen(["sleep", "30"])
    try:
        retrain._save_status(store, pid=proc.pid)
        assert retrain.status(store)["state"] == "running"
        with pytest.raises(RuntimeError, match="already running"):
            retrain.start(store)
    finally:
        proc.kill()


def test_sweep_deletes_only_overdue_pods(settings, monkeypatch):
    store = Store(settings)
    for run, deadline in (("a-reid-train", time.time() - 3600), ("b-reid-sigs", time.time() + 3600)):
        (gpu.data_root(store) / "gpu" / run).mkdir(parents=True)
        (gpu.data_root(store) / "gpu" / run / "deadline").write_text(str(deadline))
    pods = [{"id": "p1", "name": gpu.POD_PREFIX + "a-reid-train"}, {"id": "p2", "name": gpu.POD_PREFIX + "b-reid-sigs"},
            {"id": "p3", "name": gpu.POD_PREFIX + "made-by-hand"}]
    deleted = []
    monkeypatch.setattr(gpu, "list_pods", lambda: pods)
    monkeypatch.setattr(gpu, "delete_pod", deleted.append)
    assert retrain.sweep(store, log=lambda m: None) == ["p1"] and deleted == ["p1"]


def test_the_pod_deletes_itself(settings, monkeypatch, tmp_path):
    store = Store(settings)
    monkeypatch.setenv("RUNPOD_API_KEY", "k3y")
    runner = gpu.Runner(store, gpu.Job("t", [], "true", []), gpu.GPU_TYPES, max_hours=0.5, log=lambda m: None)
    runner.pod = {"id": "pod123"}
    sent = {}
    monkeypatch.setattr(subprocess, "run", lambda cmd, input=None, **kw: sent.update(cmd=cmd, script=input.decode()))
    runner.arm_dead_mans_switch()
    script = sent["script"]
    assert script.startswith("sleep 1800\n") and "k3y" not in " ".join(sent["cmd"])  # the key isn't on a command line
    assert float((runner.dir / "deadline").read_text()) == pytest.approx(time.time() + 1800, abs=5)
    f = tmp_path / "s.sh"
    f.write_text(script)
    assert subprocess.call(["bash", "-n", str(f)]) == 0
    py = script.split("<<'PY'\n")[1].split("\nPY\n")[0]
    compile(py, "deadline", "exec")
    assert '"https://rest.runpod.io/v1/pods/pod123", method="DELETE"' in py and "Bearer k3y" in py


def test_run_installs_a_good_model_and_resigns(settings, monkeypatch):
    from soccer_clips.track import reid
    from soccer_clips.track.benchmark import models_dir

    store = Store(settings)
    models_dir(store).mkdir(parents=True, exist_ok=True)
    (models_dir(store) / reid.TUNED).write_bytes(b"old")
    ran = []

    class FakeRunner:
        def __init__(self, store, job, gpu_types, max_hours, log):
            self.run_id = f"20261028-0{len(ran)}0000-{job.name}"
            self.dir = gpu.data_root(store) / "gpu" / self.run_id
            self.job = job
            ran.append((job.name, max_hours))

        def run(self):
            out = self.dir / "results" / "out"
            out.mkdir(parents=True)
            (self.dir / "log.txt").write_text("total 50 min, about $1.00\n")
            if self.job.name == "reid-train":
                (out / reid.TUNED).write_bytes(b"new")
                per = lambda h: {"g": {"players": {"38": [h, 10, 1.0], "5": [h, 10, 1.0]}, "same_top1": 0.9, "same_n": 5}}
                (out / "reid_results.json").write_text(json.dumps({"stock": per(3), "tuned": per(9)}))
            return self.dir / "results"

    def train_job(store, ids):
        (gpu.data_root(store) / "reid").mkdir(exist_ok=True)
        (gpu.data_root(store) / "reid" / "index.json").write_text(json.dumps({"g": {"fps": 30, "named": [[1, 0, 1800, 38]]}}))
        return gpu.Job("reid-train", [], "", [])

    monkeypatch.setattr(gpu, "Runner", FakeRunner)
    monkeypatch.setattr(gpu, "reid_train_job", train_job)
    monkeypatch.setattr(gpu, "reid_sigs_job", lambda store, ids: gpu.Job("reid-sigs", [], "", []))
    retrain.run(store, log=lambda m: None)
    assert ran == [("reid-train", retrain.TRAIN_HOURS), ("reid-sigs", retrain.SIGS_HOURS)]
    assert (models_dir(store) / reid.TUNED).read_bytes() == b"new"
    assert (models_dir(store) / "reid_osnet_ain.prev.pt").read_bytes() == b"old"
    st = retrain.status(store)
    assert st["state"] == "done" and st["dollars"] == 2.0
    assert retrain.model_info(store)["scores"]["player_top1"] == 0.9
    h = retrain.history(store)[0]
    assert h["installed"] is True and h["in_use"] and h["labelled_minutes"] == 1 and h["date"] == "2026-10-28T00:00"
