"""
Retraining models from the app on RunPod: the player recognition model here ("reid", roster page), the ball
detector in ball_retrain.py ("ball", ball labels page); the running, status and safety parts are shared.

The player recognition model: fine-tune on RunPod (gpu.reid_train_job),
keep the new model only if it scores about as well as the one in use, then recompute every game's appearance
signatures with it (gpu.reid_sigs_job).

It runs as its own process (`soccer-clips gpu retrain-reid`), started detached from the server, so a server
crash or restart doesn't stop it; progress is in gpu/retrain.json. Each pod deletes itself after its hours
are up (gpu.Runner.arm_dead_mans_switch) and the server deletes any of our pods past their deadline (sweep),
so nothing runs on unless both the pod and this machine fail.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from soccer_clips import gpu
from soccer_clips.store import Store, atomic_write_text

TRAIN_HOURS = 2.0  # hard limit on the training pod (runs so far: 52-54 min)
SIGS_HOURS = 1.0  # and on the signatures pod (9-10 min)
WORSE_BY = 0.05  # keep the old model if the new one picks out players this much less often (held-out games)
HISTORY = 3  # training runs shown side by side
SWEEP_GRACE_S = 600  # a pod this long past its deadline is deleted by the server


def _gpu_dir(store: Store) -> Path:
    return gpu.data_root(store) / "gpu"


STATUS_FILES = {"reid": "retrain.json", "ball": "ball-retrain.json"}  # one job of each kind at a time
COMMANDS = {("reid", "train"): ["retrain-reid"], ("ball", "train"): ["retrain-ball"], ("ball", "retrack"): ["retrack-balls"]}


def status_path(store: Store, kind: str = "reid") -> Path:
    return _gpu_dir(store) / STATUS_FILES[kind]


def _model_info_path(store: Store) -> Path:
    from soccer_clips.track import reid
    from soccer_clips.track.benchmark import models_dir

    return models_dir(store) / (Path(reid.TUNED).stem + ".json")


def _run_cost(run_dir: Path) -> tuple[float, float] | None:
    """(minutes, dollars) from a run's log ("total 52 min, about $1.03"), or None."""
    log = run_dir / "log.txt"
    if not log.exists():
        return None
    m = re.findall(r"total (\d+) min, about \$([\d.]+)", log.read_text())
    return (float(m[-1][0]), float(m[-1][1])) if m else None


def _last_run(store: Store, name: str, ok_only: bool = True) -> Path | None:
    """The latest run folder of a job that got far enough to cost something (and didn't fail, if ok_only)."""
    for d in sorted(_gpu_dir(store).glob(f"*-{name}"), reverse=True):
        log = (d / "log.txt").read_text() if (d / "log.txt").exists() else ""
        if _run_cost(d) and (not ok_only or "FAILED" not in log):
            return d
    return None


def _labelled_minutes(index: dict) -> dict[str, float]:
    """Minutes labelled per player in a bundle index ({game: {fps, named: [(track, start, end, player)]}})."""
    out: dict[str, float] = {}
    for v in index.values():
        for _, a, b, p in v["named"]:
            out[str(p)] = out.get(str(p), 0) + (b - a) / v["fps"] / 60
    return {p: round(m, 1) for p, m in sorted(out.items(), key=lambda x: int(x[0]))}


def _scores(run_dir: Path) -> dict | None:
    """A training run's held-out scores, {"stock": ..., "tuned": ...} (reid.pooled: not about any one player)."""
    from soccer_clips.track import reid

    f = run_dir / "results" / "out" / "reid_results.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    return {m: reid.pooled(r.get(m, {})) for m in ("stock", "tuned")}


def model_info(store: Store) -> dict | None:
    """When the model in use was trained, on what, how it scored and what it cost. Written when a retrain installs
    a model; for an older model, worked out from its run folder."""
    from soccer_clips.track import reid
    from soccer_clips.track.benchmark import models_dir

    p = _model_info_path(store)
    if p.exists():
        return json.loads(p.read_text())
    model = models_dir(store) / reid.TUNED
    run = _last_run(store, "reid-train")
    if not model.exists() or run is None:
        return None
    info = {"trained_at": datetime.fromtimestamp(model.stat().st_mtime).isoformat(timespec="minutes"), "run": run.name}
    sc = _scores(run)
    if sc:
        info["scores"], info["stock_scores"] = sc["tuned"], sc["stock"]
    idx = gpu.data_root(store) / "reid" / "index.json"  # the bundle it was trained on (until the next one)
    if idx.exists() and idx.stat().st_mtime <= model.stat().st_mtime:
        index = json.loads(idx.read_text())
        info.update(games=sorted(index), labelled_minutes=_labelled_minutes(index))
    cost = _run_cost(run)
    if cost:
        info["train_minutes"], info["train_dollars"] = cost
    return info


def estimate(store: Store) -> dict:
    """Roughly how long and how much a retrain takes: the last training and signatures runs, plus 10% (more games
    and labels take a little longer)."""
    mins = dollars = 0.0
    for name, default in (("reid-train", (55.0, 1.0)), ("reid-sigs", (10.0, 0.15))):
        run = _last_run(store, name)
        m, d = (_run_cost(run) if run else None) or default
        mins += m
        dollars += d
    return {"minutes": round(mins * 1.1), "dollars": round(dollars * 1.1, 2),
            "max_dollars": round((TRAIN_HOURS + SIGS_HOURS) * 1.25, 2)}  # the hour limits at ~$1.19/h, the dearest so far


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    # a zombie (finished, not yet reaped by the server that started it) isn't running
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[1][0] != "Z"
    except OSError:
        return True


def status(store: Store, kind: str = "reid") -> dict | None:
    """The current or last job of a kind (gpu/retrain.json, ...); a "running" one whose process has gone is "failed"."""
    p = status_path(store, kind)
    if not p.exists():
        return None
    st = json.loads(p.read_text())
    if st.get("state") == "running" and not _alive(st.get("pid")):
        st.update(state="failed", message=st.get("message") or "the retrain process stopped unexpectedly (see its log)")
    return st


def _save_status(store: Store, kind: str = "reid", **fields) -> dict:
    p = status_path(store, kind)
    st = json.loads(p.read_text()) if p.exists() else {}
    st.update(fields, updated=datetime.now().isoformat(timespec="seconds"))
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(p, json.dumps(st, indent=1) + "\n")
    return st


def start(store: Store, kind: str = "reid", action: str = "train") -> dict:
    """Start a job in its own process (detached: it outlives the server). Raises RuntimeError if one of the kind
    is running, or KeyError for an unknown kind / action."""
    cmd = COMMANDS[(kind, action)]
    st = status(store, kind)
    if st and st.get("state") == "running":
        raise RuntimeError("one is already running")
    gpu._api_key()  # fail now, not in the background, if RunPod isn't set up
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = _gpu_dir(store) / f"{cmd[0]}-{stamp}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(status_path(store, kind), json.dumps({"state": "running", "action": action, "step": "starting",
                                                            "started": datetime.now().isoformat(timespec="seconds"), "log": log.name}, indent=1) + "\n")
    with open(log, "ab") as out:
        proc = subprocess.Popen([sys.executable, "-m", "soccer_clips.cli", "gpu", *cmd, "--log-name", log.name],
                                stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                                cwd=str(Path(__file__).resolve().parents[2]))
    return _save_status(store, kind, pid=proc.pid)


def log_tail(store: Store, kind: str, lines: int) -> list[str]:
    """The end of the current or last job's log."""
    st = status(store, kind) or {}
    p = _gpu_dir(store) / st.get("log", "-")
    return p.read_text(errors="replace").splitlines()[-lines:] if st.get("log") and p.exists() else []


def _better_enough(new: dict | None, stock: dict | None, old: dict | None) -> tuple[bool, str]:
    """Keep the new model unless it's no better than stock, or picks out players WORSE_BY less often than the
    model in use (each player counting the same where both runs have that; else every labelled moment)."""
    if not new or "all_top1" not in new:
        return False, "the run brought back no scores"
    key = "player_top1" if old and "player_top1" in old and "player_top1" in new else "all_top1"
    msg = f"players picked out {new[key]:.0%} (stock {(stock or {}).get(key, 0):.0%})"
    if stock and new[key] <= stock.get(key, 0):
        return False, msg + ": no better than the stock model"
    if not old or key not in old:
        return True, msg
    msg = f"players picked out {old[key]:.0%} → {new[key]:.0%} (stock {(stock or {}).get(key, 0):.0%})"
    return new[key] >= old[key] - WORSE_BY, msg


def history(store: Store, n: int = HISTORY) -> list[dict]:
    """The last n training runs that brought back scores, newest first: when, games, labels, held-out scores of the
    stock and tuned models, whether its model was installed (and is in use), and what it cost."""
    use = (model_info(store) or {}).get("run")
    out = []
    for d in sorted(_gpu_dir(store).glob("*-reid-train"), reverse=True):
        sc = _scores(d)
        if not sc:
            continue
        r = json.loads((d / "results" / "out" / "reid_results.json").read_text())
        try:
            date = datetime.strptime(d.name[:15], "%Y%m%d-%H%M%S").isoformat(timespec="minutes")
        except ValueError:  # not a run folder name
            date = None
        row = {"run": d.name, "date": date,
               "games": len(r.get("tuned", {})), **sc, "in_use": d.name == use}
        idx = d / "bundle_index.json"
        if idx.exists():
            row["labelled_minutes"] = round(sum(_labelled_minutes(json.loads(idx.read_text())).values()))
        elif d.name == use and (model_info(store) or {}).get("labelled_minutes"):
            row["labelled_minutes"] = round(sum(model_info(store)["labelled_minutes"].values()))
        v = d / "verdict.json"
        if v.exists():
            row.update(json.loads(v.read_text()))
        cost = _run_cost(d)
        if cost:
            row["dollars"] = cost[1]
        out.append(row)
        if len(out) >= n:
            break
    return out


def run(store: Store, log=print, log_name: str | None = None) -> None:
    """The whole retrain (the detached process): train, compare, install, re-sign. Progress in gpu/retrain.json."""
    from soccer_clips.track import reid
    from soccer_clips.track.benchmark import models_dir

    _save_status(store, state="running", pid=os.getpid(), step="training", log=log_name, message=None)
    spent = 0.0
    try:
        old = model_info(store)
        job = gpu.reid_train_job(store, None)
        index = json.loads((gpu.data_root(store) / "reid" / "index.json").read_text())
        runner = gpu.Runner(store, job, gpu_types=gpu.GPU_TYPES, max_hours=TRAIN_HOURS, log=log)
        _save_status(store, train_run=runner.run_id)
        atomic_write_text(runner.dir / "bundle_index.json", json.dumps(index) + "\n")  # what it learned from
        try:
            results = runner.run()
        finally:
            spent += (_run_cost(runner.dir) or (0, 0))[1]
            _save_status(store, dollars=round(spent, 2))
        scores = _scores(runner.dir) or {}
        ok, why = _better_enough(scores.get("tuned"), scores.get("stock"), (old or {}).get("scores"))
        new_model = results / "out" / reid.TUNED
        ok = ok and new_model.exists()
        atomic_write_text(runner.dir / "verdict.json", json.dumps({"installed": ok, "why": why}) + "\n")
        if not ok:
            _save_status(store, state="done", step="kept the old model", message=f"kept the model in use: {why}",
                         finished=datetime.now().isoformat(timespec="seconds"), scores=scores)
            log(f"kept the old model: {why}")
            return
        dest = models_dir(store) / reid.TUNED
        if dest.exists():  # one step back, by hand if need be
            shutil.copy(dest, dest.with_suffix(".prev.pt"))
            if _model_info_path(store).exists():
                shutil.copy(_model_info_path(store), _model_info_path(store).with_suffix(".prev.json"))
            elif old:
                atomic_write_text(_model_info_path(store).with_suffix(".prev.json"), json.dumps(old, indent=1) + "\n")
        shutil.copy(new_model, dest)
        info = {"trained_at": datetime.now().isoformat(timespec="minutes"), "run": runner.run_id, "games": sorted(index),
                "labelled_minutes": _labelled_minutes(index), "scores": scores.get("tuned"), "stock_scores": scores.get("stock")}
        cost = _run_cost(runner.dir)
        if cost:
            info["train_minutes"], info["train_dollars"] = cost
        atomic_write_text(_model_info_path(store), json.dumps(info, indent=1) + "\n")
        log(f"installed the new model: {why}")

        _save_status(store, step="signatures", message=f"new model in use ({why}); recomputing every game's signatures")
        runner = gpu.Runner(store, gpu.reid_sigs_job(store, None), gpu_types=gpu.GPU_TYPES, max_hours=SIGS_HOURS, log=log)
        _save_status(store, sigs_run=runner.run_id)
        try:
            runner.run()
        finally:
            spent += (_run_cost(runner.dir) or (0, 0))[1]
            if (runner.dir / "results").exists():
                gpu.install_tracks_files(store, runner.dir / "results", "reid_sigs.npz", log=log)
        _save_status(store, state="done", step="done", dollars=round(spent, 2), scores=scores,
                     message=f"new model in use and every game re-signed ({why})",
                     finished=datetime.now().isoformat(timespec="seconds"))
    except BaseException as e:
        _save_status(store, state="failed", dollars=round(spent, 2), message=f"{type(e).__name__}: {e}",
                     finished=datetime.now().isoformat(timespec="seconds"))
        raise


def sweep(store: Store, log=print) -> list[str]:
    """Delete our pods that are past their deadline (written by Runner.arm_dead_mans_switch) by SWEEP_GRACE_S: a
    second guard for when a pod's own switch fails. Returns the pods deleted."""
    gone = []
    for pod in gpu.list_pods():
        run_dir = _gpu_dir(store) / pod["name"].removeprefix(gpu.POD_PREFIX)
        f = run_dir / "deadline"
        if f.exists() and time.time() > float(f.read_text()) + SWEEP_GRACE_S:
            gpu.delete_pod(pod["id"])
            log(f"deleted pod {pod['id']} ({pod['name']}): past its deadline")
            gone.append(pod["id"])
    return gone
