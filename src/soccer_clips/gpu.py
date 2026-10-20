"""
Cloud GPU jobs on RunPod: start a pod, upload, run, download, delete the pod.

`soccer-clips gpu run <job>` does it all from this machine and deletes the pod
whatever happens (success, error, timeout, Ctrl-C). As a second line of
defence the pod stops its own container after the time limit, so GPU billing
ends even if this machine dies mid-run; `soccer-clips gpu pods` / `cleanup`
list and delete any leftovers.

The API key (RUNPOD_API_KEY in .env) is only used here; it is never sent to
the pod. Code goes up with `git archive HEAD`, so only committed files (never
.env) leave the machine. Each run gets a folder <data root>/gpu/<run id>/ with
its log, the pod's details and the results.
"""

import json
import os
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from soccer_clips.config import REPO_ROOT, ConfigError
from soccer_clips.store import Store

API = "https://rest.runpod.io/v1"
POD_PREFIX = "soccer-clips-"
IMAGE = "runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404"  # PyTorch 2.8, CUDA 12.8, Ubuntu 24.04, sshd
GPU_TYPES = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090", "NVIDIA L40S"]  # in order of preference
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise ConfigError("RUNPOD_API_KEY is not set in .env")
    return key


def api(method: str, path: str, body: dict | None = None) -> dict | list | None:
    req = urllib.request.Request(
        API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
        # Cloudflare in front of the API rejects urllib's default User-Agent (error 1010).
        headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json",
                 "User-Agent": "soccer-clips/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode()
    except urllib.error.HTTPError as e:
        hint = (" (the API key may be read-only: it needs read/write access to pods)"
                if e.code in (401, 403) and method != "GET" else "")
        raise RuntimeError(f"RunPod API {method} {path}: {e.code} {e.read().decode()[:500]}{hint}") from None
    return json.loads(text) if text.strip() else None


def list_pods() -> list[dict]:
    return [p for p in api("GET", "/pods") or [] if p.get("name", "").startswith(POD_PREFIX)]


def delete_pod(pod_id: str) -> None:
    api("DELETE", f"/pods/{pod_id}")


@dataclass
class Job:
    """What to send, what to run, what to bring back. Paths are relative to the data root."""

    name: str
    uploads: list[str]  # files or folders under the data root; nothing else is sent
    script: str  # bash, run in /workspace/code with the venv active; data root is $W
    downloads: list[str]  # files or folders under the remote data root, copied into the run folder
    exclude: list[str] = field(default_factory=list)  # tar --exclude patterns for the upload


def data_root(store: Store) -> Path:
    return store.settings.games_dir.parent


SETUP = r"""
set -euo pipefail
W=/workspace/data
cd /workspace/code
python -m venv --system-site-packages /workspace/venv
. /workspace/venv/bin/activate
pip install -q -e . ultralytics lap 2>&1 | grep -v "^\s*$" | tail -2 || true
yolo settings sync=False > /dev/null
mkdir -p $W/full $W/proxy $W/clips $W/reels $W/gallery
printf "SOCCER_FULL_DIR=$W/full\nSOCCER_PROXY_DIR=$W/proxy\nSOCCER_CLIPS_DIR=$W/clips\nSOCCER_GAMES_DIR=$W/games\nSOCCER_REELS_DIR=$W/reels\nSOCCER_GALLERY_DIR=$W/gallery\n" > .env
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__)"
"""


class Runner:
    def __init__(self, store: Store, job: Job, gpu_types: list[str], max_hours: float, log: Callable[[str], None]):
        self.store, self.job, self.gpu_types, self.max_hours = store, job, gpu_types, max_hours
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + job.name
        self.dir = data_root(store) / "gpu" / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._log = log
        self.pod: dict | None = None
        self.ssh_args: list[str] = []

    def log(self, msg: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {msg}"
        self._log(line)
        with open(self.dir / "log.txt", "a") as fh:
            fh.write(line + "\n")

    def ssh(self, command: str, stdin=None, stdout=None, check=True, timeout=None) -> subprocess.CompletedProcess:
        return subprocess.run(["ssh", *self.ssh_args, command], stdin=stdin, stdout=stdout, check=check,
                              timeout=timeout)

    # ---- steps ----

    def start_pod(self, wait_s: int = 1200) -> None:
        """Create the pod, retrying while RunPod has no machine free (up to wait_s)."""
        if not SSH_KEY.with_suffix(".pub").exists():
            raise ConfigError(f"no SSH key at {SSH_KEY}.pub")
        # No allowedCudaVersions filter: together with an exposed TCP port it left no
        # machines at all (2026-10-19). The job's setup checks CUDA works instead.
        body = {
            "name": POD_PREFIX + self.run_id, "cloudType": "SECURE", "computeType": "GPU",
            "gpuTypeIds": self.gpu_types, "gpuTypePriority": "custom", "gpuCount": 1,
            "imageName": IMAGE, "ports": ["22/tcp"], "containerDiskInGb": 40, "volumeInGb": 0,
            "interruptible": False,
            "env": {"SSH_PUBLIC_KEY": SSH_KEY.with_suffix(".pub").read_text().strip()},
        }
        deadline = time.time() + wait_s
        while True:
            try:
                self.pod = api("POST", "/pods", body)
                break
            except RuntimeError as e:
                if "no instances currently available" not in str(e) or time.time() > deadline:
                    raise
                self.log("no machine free for those GPU types; retrying in 60 s")
                time.sleep(60)
        (self.dir / "pod.json").write_text(json.dumps(self.pod, indent=2))
        self.log(f"pod {self.pod['id']} created ({self.pod.get('gpu', {}).get('displayName', '?')}, "
                 f"${self.pod.get('costPerHr', '?')}/h)")

    def wait_for_ssh(self, timeout_s: int = 900) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            pod = api("GET", f"/pods/{self.pod['id']}")
            ip, port = pod.get("publicIp"), (pod.get("portMappings") or {}).get("22")
            if ip and port:
                self.pod = pod
                self.ssh_args = ["-p", str(port), "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=accept-new",
                                 "-o", f"UserKnownHostsFile={self.dir / 'known_hosts'}", "-o", "ConnectTimeout=15",
                                 "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
                                 "-o", "BatchMode=yes", f"root@{ip}"]
                if self.ssh("true", check=False, stdout=subprocess.DEVNULL, timeout=60).returncode == 0:
                    self.log(f"ssh ready at {ip}:{port}")
                    return
            time.sleep(10)
        raise TimeoutError("pod never became reachable over ssh")

    def arm_dead_mans_switch(self) -> None:
        """Stop the container (and so GPU billing) after max_hours even if we lose contact."""
        secs = int(self.max_hours * 3600)
        self.ssh(f"nohup bash -c 'sleep {secs}; kill -TERM 1' < /dev/null > /dev/null 2>&1 &")
        self.log(f"pod will stop itself after {self.max_hours:g} h")

    def upload(self) -> None:
        root = data_root(self.store)
        code = subprocess.run(["git", "-C", str(REPO_ROOT), "archive", "--format=tar", "HEAD"],
                              check=True, capture_output=True).stdout
        subprocess.run(["ssh", *self.ssh_args, "mkdir -p /workspace/code && tar -xf - -C /workspace/code"],
                       input=code, check=True)
        excludes = sum((["--exclude", e] for e in self.job.exclude), [])
        listing = self.dir / "uploads.txt"  # exactly what is sent, kept for the record
        listing.write_text("\n".join(self.job.uploads) + "\n")
        tar = subprocess.Popen(["tar", "-czf", "-", "-C", str(root), *excludes, "-T", str(listing)],
                               stdout=subprocess.PIPE)
        self.ssh("mkdir -p /workspace/data && tar -xzf - -C /workspace/data", stdin=tar.stdout)
        if tar.wait():
            raise RuntimeError("tar of the upload failed")
        size = self.ssh("du -sh /workspace/data | cut -f1", stdout=subprocess.PIPE).stdout.decode().strip()
        self.log(f"uploaded code ({len(code) // 1024} KB) and data ({size})")

    def run_script(self) -> None:
        script = SETUP + "\n" + self.job.script
        remote_log = self.dir / "remote.txt"
        with open(remote_log, "ab") as out:
            proc = subprocess.run(["ssh", *self.ssh_args, "bash -s"], input=script.encode(), stdout=out,
                                  stderr=subprocess.STDOUT, timeout=self.max_hours * 3600)
        if proc.returncode:
            raise RuntimeError(f"job script failed (exit {proc.returncode}); see {remote_log}")
        self.log("job script finished")

    def download(self) -> None:
        paths = " ".join(self.job.downloads)  # our own patterns, unquoted so the remote shell expands them
        cmd = f"cd /workspace/data && tar -czf - $(ls -d {paths} 2>/dev/null)"
        out = self.dir / "results"
        out.mkdir(exist_ok=True)
        with subprocess.Popen(["ssh", *self.ssh_args, cmd], stdout=subprocess.PIPE) as proc:
            with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tf:
                tf.extractall(out, filter="data")
        self.log(f"downloaded results to {out}")

    def teardown(self) -> None:
        if not self.pod:
            return
        if self.ssh_args:
            try:  # don't leave footage on someone else's disk, even briefly
                self.ssh("rm -rf /workspace/data /workspace/code /workspace/venv ~/.config/Ultralytics",
                         check=False, timeout=120)
            except Exception as e:
                self.log(f"couldn't wipe the pod ({e}); deleting it anyway")
        for _ in range(5):
            try:
                delete_pod(self.pod["id"])
                self.log(f"pod {self.pod['id']} deleted")
                return
            except Exception as e:
                self.log(f"delete failed ({e}), retrying")
                time.sleep(15)
        self.log(f"COULD NOT DELETE pod {self.pod['id']}: run `soccer-clips gpu cleanup`")

    def run(self) -> Path:
        t0 = time.time()
        try:
            leftovers = list_pods()
            if leftovers:
                self.log(f"warning: {len(leftovers)} soccer-clips pod(s) already exist: "
                         f"{[p['id'] for p in leftovers]} (`soccer-clips gpu cleanup` deletes them)")
            self.start_pod()
            self.wait_for_ssh()
            self.arm_dead_mans_switch()
            self.upload()
            self.run_script()
            self.download()
        except BaseException as e:  # KeyboardInterrupt too: always tear down
            self.log(f"FAILED: {type(e).__name__}: {e}")
            raise
        finally:
            self.teardown()
            hours = (time.time() - t0) / 3600
            rate = float((self.pod or {}).get("costPerHr") or 0)
            self.log(f"total {hours * 60:.0f} min, about ${hours * rate:.2f}")
        return self.dir / "results"


# ---- jobs ----

def ball_train_job(store: Store, stock: list[str], train_bases: list[str], epochs: int) -> Job:
    """
    Optionally score `stock` bases on the bench frames, fine-tune `train_bases` (plus the best
    other stock base, when stock bases are given), and score the fine-tuned models on the bench.
    """
    from soccer_clips.track import ball_labels

    root = data_root(store)
    for need, how in [("labels/dataset/data.yaml", "soccer-clips ball export"), ("labels/ball.jsonl", "label some frames"),
                      ("labels/dataset/val_frames.json", "soccer-clips ball export")]:
        if not (root / need).exists():
            raise ConfigError(f"{root / need} is missing; run `{how}` first")
    # Full frame images: the bench frames (scoring) and the validation frames (choosing checkpoints).
    wanted = [(t.game_id, t.frame) for t in ball_labels.load_tasks(store.settings).sets.get("bench", [])]
    wanted += [tuple(v) for v in json.loads((root / "labels/dataset/val_frames.json").read_text())]
    frames = sorted({str(ball_labels.frame_image(store, store.load_game(g), f).relative_to(root)) for g, f in wanted})
    games = [str(p.relative_to(root)) for p in sorted((root / "games").glob("*/*/game.json"))]
    quiet = '2>&1 | (grep -vE "━|^\\s*$" || true)'  # progress bars out; never fails the job on its own
    fixed = " ".join(train_bases)
    script = "mkdir -p $W/out\nBEST=\n"
    if stock:
        script += f"""
soccer-clips ball eval {" ".join(stock)} --json $W/out/stock.json {quiet}
BEST=$(python - <<'PY'
import json
s = json.load(open("/workspace/data/out/stock.json"))
fixed = "{fixed}".split()
others = {{k: v for k, v in s.items() if k.removesuffix("-centre") not in fixed}}
key = lambda k: (others[k]["0.1"]["recall"] or 0, others[k]["0.1"]["precision"] or 0)
print(max(others, key=key).removesuffix("-centre") if others else "")
PY
)
echo "best other stock base: $BEST"
"""
    script += f"""
soccer-clips ball train {fixed} $BEST --epochs {epochs} {quiet}
soccer-clips ball eval $(for b in {fixed} $BEST; do echo ball_$b; done) --json $W/out/finetuned.json {quiet}
"""
    uploads = games + ["labels/ball_tasks.json", "labels/ball.jsonl", "labels/dataset"] + frames
    if (root / "labels/ball_others.json").exists():
        uploads.append("labels/ball_others.json")
    return Job(
        name="ball-train",
        uploads=uploads,
        exclude=["*.part.jpg"],
        script=script,
        downloads=["out", "models/ball_*.pt", "labels/runs/*/results.csv", "labels/runs/*/args.yaml",
                   "labels/runs/*/selection.json"],
    )
