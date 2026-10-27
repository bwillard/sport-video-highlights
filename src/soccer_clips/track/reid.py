"""
Player appearance (re-ID): a signature per player crop, so the same girl matches across gaps
and games. The model is OSNet-AIN (Zhou et al., "Omni-Scale Feature Learning for Person
Re-Identification"; the torchreid MSMT17 weights, MIT licence), written out here.

Crops: every CROP_EVERY frames, each tracked box tall enough, kept as JPEG per game
(tracks/reid_crops.npz). Training needs no naming of other players: two tracklets on screen
at the same moment are two different people, and a tracklet over a few seconds is one person.
Each labelled player's stretches (the picks) are one person across all games.

Yardstick (evaluate): at each labelled stretch of #38 in a held-out game, is her tracklet the
teammate that looks most like her (her look learnt from the other games)? And for everyone on
our team: does a tracklet match itself 10-20 s later better than the teammates on screen then?
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from soccer_clips import ffmpeg
from soccer_clips.models import Game
from soccer_clips.store import Store

CROP_EVERY = 30  # frames between crops (about 1 s)
MIN_H = 40  # px: smaller boxes are a blur
SIZE = (128, 256)  # w, h fed to the model (the usual person re-ID shape)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
STOCK = "osnet_ain_x1_0_msmt17.pt"  # torchreid model zoo
TUNED = "reid_osnet_ain.pt"
NEAR_S = 5  # a tracklet within this many seconds of a moment is one person (few swaps)
PLAYER = 38
PLAYER_ROLES = (0, 1)  # ours and the opponents (people.ROLE_CODES): spectators and refs aren't trained on


# ---- the model ----

def _model():
    import torch
    from torch import nn

    class ConvLayer(nn.Module):
        def __init__(self, cin, cout, k, stride=1, pad=0, IN=False):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, k, stride, pad, bias=False)
            self.bn = nn.InstanceNorm2d(cout, affine=True) if IN else nn.BatchNorm2d(cout)

        def forward(self, x):
            return torch.relu(self.bn(self.conv(x)))

    class Conv1x1(ConvLayer):
        def __init__(self, cin, cout):
            super().__init__(cin, cout, 1)

    class Conv1x1Linear(nn.Module):
        def __init__(self, cin, cout, bn=True):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, 1, bias=False)
            self.bn = nn.BatchNorm2d(cout) if bn else None

        def forward(self, x):
            x = self.conv(x)
            return self.bn(x) if self.bn is not None else x

    class LightConv3x3(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 1, bias=False)
            self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False, groups=cout)
            self.bn = nn.BatchNorm2d(cout)

        def forward(self, x):
            return torch.relu(self.bn(self.conv2(self.conv1(x))))

    class Stream(nn.Module):
        def __init__(self, c, depth):
            super().__init__()
            self.layers = nn.Sequential(*[LightConv3x3(c, c) for _ in range(depth)])

        def forward(self, x):
            return self.layers(x)

    class Gate(nn.Module):
        def __init__(self, c, reduction=16):
            super().__init__()
            self.fc1 = nn.Conv2d(c, c // reduction, 1)
            self.fc2 = nn.Conv2d(c // reduction, c, 1)

        def forward(self, x):
            g = x.mean((2, 3), keepdim=True)
            return x * torch.sigmoid(self.fc2(torch.relu(self.fc1(g))))

    class Block(nn.Module):
        """Omni-scale block: four streams of 1-4 light 3x3 convs, mixed by a shared gate. in_IN puts an
        instance norm on the residual branch (style-free features); otherwise a batch norm."""

        def __init__(self, cin, cout, in_IN):
            super().__init__()
            mid = cout // 4
            self.conv1 = Conv1x1(cin, mid)
            self.conv2 = nn.ModuleList([Stream(mid, t) for t in range(1, 5)])
            self.gate = Gate(mid)
            self.conv3 = Conv1x1Linear(mid, cout, bn=not in_IN)
            self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None
            self.IN = nn.InstanceNorm2d(cout, affine=True) if in_IN else None

        def forward(self, x):
            x1 = self.conv1(x)
            x3 = self.conv3(sum(self.gate(s(x1)) for s in self.conv2))
            if self.IN is not None:
                x3 = self.IN(x3)
            return torch.relu(x3 + (self.downsample(x) if self.downsample is not None else x))

    class OSNetAIN(nn.Module):
        def __init__(self):
            super().__init__()
            c = (64, 256, 384, 512)
            self.conv1 = ConvLayer(3, c[0], 7, 2, 3, IN=True)
            self.maxpool = nn.MaxPool2d(3, 2, 1)
            self.conv2 = nn.Sequential(Block(c[0], c[1], True), Block(c[1], c[1], True))
            self.pool2 = nn.Sequential(Conv1x1(c[1], c[1]), nn.AvgPool2d(2, 2))
            self.conv3 = nn.Sequential(Block(c[1], c[2], False), Block(c[2], c[2], True))
            self.pool3 = nn.Sequential(Conv1x1(c[2], c[2]), nn.AvgPool2d(2, 2))
            self.conv4 = nn.Sequential(Block(c[2], c[3], True), Block(c[3], c[3], False))
            self.conv5 = Conv1x1(c[3], c[3])
            self.fc = nn.Sequential(nn.Linear(c[3], 512), nn.BatchNorm1d(512), nn.ReLU())

        def forward(self, x):
            x = self.maxpool(self.conv1(x))
            x = self.pool3(self.conv3(self.pool2(self.conv2(x))))
            x = self.conv5(self.conv4(x))
            return self.fc(x.mean((2, 3)))

    return OSNetAIN()


def load_model(path: Path, device: str = "cpu"):
    """OSNet-AIN with the weights at path (stock torchreid file or ours), in eval mode."""
    import torch

    m = _model()
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k.removeprefix("module."): v for k, v in sd.items() if not k.removeprefix("module.").startswith("classifier")}
    m.load_state_dict(sd)
    return m.to(device).eval()


def to_tensor(imgs: list[np.ndarray] | np.ndarray):
    """BGR crops (any size) -> normalised (n, 3, 256, 128) float tensor."""
    import torch

    x = np.stack([cv2.resize(cv2.cvtColor(i, cv2.COLOR_BGR2RGB), SIZE, interpolation=cv2.INTER_CUBIC)
                  for i in imgs]).astype(np.float32) / 255
    return torch.from_numpy(((x - MEAN) / STD).transpose(0, 3, 1, 2).copy())


def embed(model, imgs, device: str = "cpu", batch: int = 128) -> np.ndarray:
    """Unit-length signatures (n, 512) of BGR crops."""
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            out.append(model(to_tensor(imgs[i:i + batch]).to(device)).float().cpu().numpy())
    f = np.vstack(out) if out else np.zeros((0, 512), np.float32)
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)


# ---- crops ----

def crops_path(store: Store, game: Game) -> Path:
    return store.game_dir(game.year, game.id) / "tracks" / "reid_crops.npz"


def _read_every(store: Store, game: Game, start: int, end: int):
    """Frames start, start + CROP_EVERY, ... below end (start a multiple of CROP_EVERY), full size BGR."""
    fps = game.fps_num / game.fps_den
    seek_s, _ = ffmpeg.cut_window(game, start / fps, end / fps)
    w, h = game.width, game.height
    proc = subprocess.Popen([store.settings.ffmpeg, "-v", "error", "-threads", "1", "-ss", f"{seek_s:.6f}",
                             "-i", str(store.original_path(game)), "-frames:v", str(end - start), "-map", "0:v:0",
                             "-vf", f"select=not(mod(n\\,{CROP_EVERY}))", "-fps_mode", "passthrough",
                             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    try:
        for f in range(start, end, CROP_EVERY):
            buf = proc.stdout.read(w * h * 3)
            if len(buf) < w * h * 3:
                break
            yield f, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def _crop_chunk(args):
    settings, game_id, start, end, rows = args
    cv2.setNumThreads(1)
    from soccer_clips.track import people

    store = Store(settings)
    game = store.load_game(game_id)
    t = people.tracks(store, game)
    by_frame: dict[int, list[int]] = {}
    for i in rows:
        by_frame.setdefault(int(t.frame[i]), []).append(int(i))
    out_rows, jpgs = [], []
    W, H = game.width, game.height
    for f, img in _read_every(store, game, start, end):
        for i in by_frame.get(f, []):
            x1, y1, x2, y2 = (t.box[i] * [W, H, W, H]).round().astype(int)
            c = img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            if c.size:
                out_rows.append(i)
                jpgs.append(cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())
    return out_rows, jpgs


def extract_crops(store: Store, game: Game, workers: int = 4, log=print) -> int:
    """Crops of every tracked box at frames on the CROP_EVERY grid, at least MIN_H tall. Returns crops saved."""
    import multiprocessing as mp

    from soccer_clips.track import ball_game, people

    t = people.tracks(store, game)
    if t is None:
        raise ValueError(f"{game.id}: no player tracks yet")
    _, roles = people.roles_for(store, game)
    h = (t.box[:, 3] - t.box[:, 1]) * game.height
    sel = np.flatnonzero((t.frame % CROP_EVERY == 0) & (h >= MIN_H))
    span = CROP_EVERY * 300  # about 5 minutes per chunk
    todo = []
    for s in range(0, int(t.frame.max()) + 1, span):
        rows = sel[(t.frame[sel] >= s) & (t.frame[sel] < s + span)]
        if len(rows):
            todo.append((store.settings, game.id, s, s + span, rows))
    got_rows, got = [], []
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, (r, j) in enumerate(pool.imap_unordered(_crop_chunk, todo), 1):
            got_rows += r
            got += j
            log(f"{game.id}: crops {n}/{len(todo)} chunks")
    order = np.argsort(got_rows, kind="stable")
    rows = np.array(got_rows, np.int64)[order]
    blobs = [got[i] for i in order]
    lengths = np.array([len(b) for b in blobs], np.int64)
    ball_game.save_npz(crops_path(store, game), data=np.frombuffer(b"".join(blobs), np.uint8),
                       offsets=np.concatenate([[0], np.cumsum(lengths)]), frame=t.frame[rows], track=t.track[rows],
                       role=roles[rows].astype(np.int8), height=h[rows].astype(np.float32))
    return len(rows)


class Crops:
    """One game's crops (JPEG) with frame, track, role and the player labelled there (-1: nobody)."""

    def __init__(self, path: Path, named: list[tuple[int, int, int, int]], fps: float):
        with np.load(path) as z:
            self.data, self.offsets = z["data"], z["offsets"]
            self.frame, self.track, self.role = z["frame"], z["track"], z["role"]
        self.fps = fps
        self.player = np.full(len(self.frame), -1, np.int64)
        for k, a, b, n in named:
            self.player[(self.track == k) & (self.frame >= a) & (self.frame < b)] = n
        self.her = self.player == PLAYER
        self.named = {int(n): np.flatnonzero(self.player == n) for n in np.unique(self.player) if n >= 0}
        self.grid = {}  # frame -> crop indices at that frame
        for i, f in enumerate(self.frame.tolist()):
            self.grid.setdefault(f, []).append(i)
        self.by_track = {}  # track -> crop indices, in frame order
        for i in np.argsort(self.track, kind="stable").tolist():
            self.by_track.setdefault(int(self.track[i]), []).append(i)

    def __len__(self):
        return len(self.frame)

    def image(self, i: int) -> np.ndarray:
        return cv2.imdecode(self.data[self.offsets[i]:self.offsets[i + 1]], cv2.IMREAD_COLOR)

    def near(self, i: int, s: float = NEAR_S, same_label: bool = True) -> list[int]:
        """Crops of i's tracklet within s seconds of it (and labelled as the same player, or nobody, as i)."""
        idx = np.array(self.by_track[int(self.track[i])])
        keep = np.abs(self.frame[idx] - self.frame[i]) <= s * self.fps
        if same_label:
            keep &= self.player[idx] == self.player[i]
        return idx[keep].tolist()


def labels(store: Store, game: Game) -> list[tuple[int, int, int, int]]:
    """Every player's labelled stretches as (track, start, end, player)."""
    from soccer_clips.track import people

    t = people.tracks(store, game)
    return [(s["track"], s["start"], s["end"], s["player"]) for s in people.segments(t, people.load_picks(store, game))]


# ---- training data (on the GPU pod: no store, just files) ----

def write_bundle(store: Store, game_ids: list[str], out: Path) -> list[Path]:
    """Copy each game's crops into out/<game>.npz and the labels and frame rates into out/index.json."""
    import shutil

    out.mkdir(parents=True, exist_ok=True)
    index, files = {}, []
    for gid in game_ids:
        game = store.load_game(gid)
        src = crops_path(store, game)
        if not src.exists():
            raise ValueError(f"{gid}: no crops yet (soccer-clips reid crops)")
        shutil.copy(src, out / f"{gid}.npz")
        files.append(out / f"{gid}.npz")
        index[gid] = {"fps": game.fps_num / game.fps_den, "named": labels(store, game)}
    (out / "index.json").write_text(json.dumps(index, indent=1))
    files.append(out / "index.json")
    return files


def load_bundle(folder: Path) -> dict[str, Crops]:
    index = json.loads((folder / "index.json").read_text())
    return {gid: Crops(folder / f"{gid}.npz", [tuple(x) for x in v["named"]], v["fps"]) for gid, v in index.items()}


def _augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flip, a little shift and zoom, brightness and contrast (never hue: the kit colour matters)."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
    h, w = img.shape[:2]
    dx, dy = rng.uniform(-0.08, 0.08, 2) * (w, h)
    z = rng.uniform(0.9, 1.1)
    m = np.array([[z, 0, dx + (1 - z) * w / 2], [0, z, dy + (1 - z) * h / 2]], np.float32)
    img = cv2.warpAffine(np.ascontiguousarray(img), m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    a, b = rng.uniform(0.75, 1.25), rng.uniform(-25, 25)
    return np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


def _named_crop(games: dict[str, Crops], player: int, rng: np.random.Generator) -> tuple[Crops, int]:
    """A random labelled crop of the player from any game (games weighted by how many they have)."""
    have = [(g, g.named[player]) for g in games.values() if player in g.named]
    sizes = np.array([len(idx) for _, idx in have], float)
    g, idx = have[rng.choice(len(have), p=sizes / sizes.sum())]
    return g, int(idx[rng.integers(len(idx))])


def sample_batch(games: dict[str, Crops], rng: np.random.Generator, people_n: int = 16, per: int = 4,
                 her_share: float = 0.4, named_share: float = 0.3) -> tuple[list[np.ndarray], np.ndarray]:
    """
    One batch: up to people_n tracklets on screen at one moment (so all different people), per crops
    each from within NEAR_S of it. Some batches are at a moment when PLAYER is labelled (her_share), some
    when another player is (named_share); a labelled player's crops then come from all their labelled
    stretches in all the games, so each named player is one person across games.
    Returns crops and an identity per crop.
    """
    gids = list(games)
    players = sorted({n for g in games.values() for n in g.named})
    others = [n for n in players if n != PLAYER]
    r = rng.random()
    anchor = PLAYER if PLAYER in players and r < her_share else \
        (others[rng.integers(len(others))] if others and r < her_share + named_share else None)
    while True:
        if anchor is not None:
            g, i0 = _named_crop(games, anchor, rng)
            f = int(g.frame[i0])
        else:
            g = games[rng.choice(gids)]
            frames = list(g.grid)
            f = frames[rng.integers(len(frames))]
        here = [i for i in rng.permutation(g.grid[f]).tolist() if g.role[i] in PLAYER_ROLES]
        named, seen = [], set()
        for i in here:  # labelled first, one tracklet per player (two at once would be a labelling slip)
            if g.player[i] >= 0 and g.player[i] not in seen:
                named.append(i)
                seen.add(int(g.player[i]))
        here = (named + [i for i in here if g.player[i] < 0])[:people_n]
        if len(here) >= 2:
            break
    imgs, ids = [], []
    for n, i in enumerate(here):
        if g.player[i] >= 0:
            pool = [_named_crop(games, int(g.player[i]), rng) for _ in range(per)]
        else:
            near = g.near(i)
            pool = [(g, j) for j in rng.choice(near, per, replace=len(near) < per)]
        for gg, j in pool:
            imgs.append(_augment(gg.image(j), rng))
            ids.append(n)
    return imgs, np.array(ids)


def supcon_loss(f, ids, temperature: float = 0.1):
    """Supervised contrastive loss on unit-length features: pull same-identity crops together."""
    import torch

    sim = f @ f.T / temperature
    n = len(f)
    eye = torch.eye(n, dtype=torch.bool, device=f.device)
    same = (ids[:, None] == ids[None]) & ~eye
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    has = same.any(1)
    return -((logp * same).sum(1)[has] / same.sum(1)[has]).mean()


def train(games: dict[str, Crops], stock: Path, steps: int = 3000, lr: float = 2e-4, device: str = "cuda",
          seed: int = 0, workers: int = 8, log=print):
    """Fine-tune the stock OSNet-AIN on the games' crops. Returns the model (eval mode)."""
    import torch

    torch.manual_seed(seed)
    model = load_model(stock, device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    class Batches(torch.utils.data.IterableDataset):
        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            rng = np.random.default_rng(seed * 1000 + (info.id if info else 0))
            cv2.setNumThreads(1)
            while True:
                imgs, ids = sample_batch(games, rng)
                yield to_tensor(imgs), torch.from_numpy(ids)

    loader = torch.utils.data.DataLoader(Batches(), batch_size=None, num_workers=workers,
                                         persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
    it = iter(loader)
    total = 0.0
    for step in range(1, steps + 1):
        x, ids = next(it)
        x, ids = x.to(device, non_blocking=True), ids.to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            f = model(x)
        f = torch.nn.functional.normalize(f.float(), dim=1)
        loss = supcon_loss(f, ids)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        total += loss.item()
        if step % 100 == 0:
            log(f"step {step}/{steps}: loss {total / 100:.3f}")
            total = 0.0
    return model.eval()


# ---- the yardstick ----

def _auc(p, n) -> float:
    p, n = np.sort(np.asarray(p)), np.asarray(n)
    if not len(p) or not len(n):
        return float("nan")
    right, left = np.searchsorted(p, n, side="right"), np.searchsorted(p, n, side="left")
    return float(((len(p) - right) + 0.5 * (right - left)).mean() / len(p))  # P(her > teammate), ties half


def _mean(f: np.ndarray, idx: list[int]) -> np.ndarray:
    v = f[idx].mean(0)
    return v / max(np.linalg.norm(v), 1e-9)


def _moments(g: Crops, player: int) -> list[int]:
    """One crop per labelled stretch of the player (the middle of each run of their crops along a tracklet)."""
    by_track: dict[int, list[int]] = {}
    for i in g.named.get(player, []).tolist():
        by_track.setdefault(int(g.track[i]), []).append(i)
    out = []
    for idx in by_track.values():
        idx = sorted(idx, key=lambda i: g.frame[i])
        for run in np.split(np.array(idx), np.flatnonzero(np.diff(g.frame[idx]) > CROP_EVERY) + 1):
            out.append(int(run[len(run) // 2]))
    return out


def evaluate(games: dict[str, Crops], feats: dict[str, np.ndarray], test: list[str]) -> dict:
    """
    For each test game and each labelled player, with the player's look = mean of their crops in the
    other games:
      her_top1: at each of PLAYER's labelled stretches (middle moment), her tracklet looks most like her
                among the teammates on screen then (each tracklet: mean of its crops within NEAR_S);
      her_auc:  her tracklets against those teammates, by likeness to her look;
      chance:   1 / (teammates on screen + 1);
      mates_top1: the same as her_top1 for every other labelled player (pooled), with mates_chance;
      same_top1: an our-team tracklet's crops now (within NEAR_S) match its own crops 10-20 s later better
                than the other teammates on screen then (labels come from the tracker, so a few are wrong).
    """
    out = {}
    for gid in test:
        g, f = games[gid], feats[gid]
        res = {}
        for player in g.named:
            theirs = [feats[x][games[x].named[player]] for x in games if x != gid and player in games[x].named]
            if not theirs:
                continue
            theirs = np.vstack(theirs)
            look = _mean(theirs, list(range(len(theirs))))
            top1, chance, pos, neg = [], [], [], []
            for i in _moments(g, player):
                mates = [j for j in g.grid[int(g.frame[i])] if g.role[j] == 0 and g.track[j] != g.track[i]]
                if not mates:
                    continue
                s_p = float(_mean(f, g.near(i)) @ look)
                s_m = [float(_mean(f, g.near(j, same_label=False)) @ look) for j in mates]
                top1.append(s_p > max(s_m))
                chance.append(1 / (len(mates) + 1))
                pos.append(s_p)
                neg += s_m
            res[player] = (top1, chance, pos, neg)
        her = res.pop(PLAYER, ([], [], [], []))
        mates = [x for v in res.values() for x in v[0]]
        mates_chance = [x for v in res.values() for x in v[1]]
        rng = np.random.default_rng(0)
        same = []
        ours = np.flatnonzero(g.role == 0)
        for i in rng.choice(ours, min(2000, len(ours)), replace=False).tolist():
            later = [j for j in g.by_track[int(g.track[i])] if 10 * g.fps <= g.frame[j] - g.frame[i] <= 20 * g.fps]
            if not later:
                continue
            j = later[len(later) // 2]
            others = [m for m in g.grid[int(g.frame[j])] if g.role[m] == 0 and g.track[m] != g.track[i]]
            if not others:
                continue
            q = _mean(f, g.near(i, same_label=False))
            s_self = float(_mean(f, g.near(j, same_label=False)) @ q)
            same.append(s_self > max(float(_mean(f, g.near(m, same_label=False)) @ q) for m in others))
        mean = lambda xs: float(np.mean(xs)) if xs else None  # noqa: E731
        out[gid] = {"stretches": len(her[0]), "her_top1": mean(her[0]), "chance": mean(her[1]),
                    "her_auc": _auc(her[2], her[3]), "mates_top1": mean(mates), "mates_chance": mean(mates_chance),
                    "mates_n": len(mates), "mates_players": len(res),
                    "same_top1": mean(same), "same_n": len(same)}
    return out


def features(model, games: dict[str, Crops], device: str = "cpu") -> dict[str, np.ndarray]:
    return {gid: embed(model, [g.image(i) for i in range(len(g))], device) for gid, g in games.items()}


def folds(game_ids: list[str], n: int) -> list[list[str]]:
    return [game_ids[i::n] for i in range(n)]


def summary(results: dict) -> dict:
    """Means over games of each score, per model."""
    keys = ("her_top1", "chance", "her_auc", "mates_top1", "mates_chance", "same_top1")
    return {name: {k: float(np.nanmean([r[k] for r in per.values() if r.get(k) is not None] or [np.nan]))
                   for k in keys} for name, per in results.items()}


def run_experiment(folder: Path, stock: Path, out: Path, n_folds: int = 3, steps: int = 3000,
                   device: str = "cuda", log=print) -> dict:
    """
    Cross-validated: for each fold, fine-tune on the other games and score stock and tuned on the
    fold's games. Then fine-tune on all games and save the model to out/TUNED.
    """
    import torch

    games = load_bundle(folder)
    ids = sorted(games)
    stock_model = load_model(stock, device)
    stock_feats = features(stock_model, games, device)
    results = {"stock": {}, "tuned": {}}
    for k, test in enumerate(folds(ids, n_folds)):
        log(f"fold {k + 1}/{n_folds}: testing on {', '.join(test)}")
        train_games = {g: games[g] for g in ids if g not in test}
        model = train(train_games, stock, steps=steps, device=device, log=log)
        results["stock"].update(evaluate(games, stock_feats, test))
        tuned = features(model, {g: games[g] for g in test}, device)
        # her look comes from the training games, as the tuned model sees them
        tuned.update(features(model, train_games, device))
        results["tuned"].update(evaluate(games, tuned, test))
        log(json.dumps(summary(results)))
    model = train(games, stock, steps=steps, device=device, log=log)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / TUNED)
    results["summary"] = summary(results)
    (out / "reid_results.json").write_text(json.dumps(results, indent=1))
    return results


# ---- signatures for use (our team's crops, tuned model) ----

SIG_ROLES = (0,)  # ours: the players we name


def sigs_path(store: Store, game: Game) -> Path:
    return store.game_dir(game.year, game.id) / "tracks" / "reid_sigs.npz"


def model_path(store: Store) -> Path:
    """The tuned model if there is one, else the stock weights."""
    from soccer_clips.track.benchmark import models_dir

    tuned = models_dir(store) / TUNED
    return tuned if tuned.exists() else models_dir(store) / STOCK


def compute_signatures(store: Store, game: Game, device: str = "cpu", log=print) -> int:
    """Signatures of the game's our-team crops (from extract_crops) with model_path(). Returns crops done.
    About 17 crops/s on this machine's CPU (OSNet's many small convolutions), thousands on a GPU."""
    import torch

    from soccer_clips.track import ball_game

    torch.set_num_threads(4)
    c = Crops(crops_path(store, game), [], game.fps_num / game.fps_den)
    idx = np.flatnonzero(np.isin(c.role, SIG_ROLES))
    weights = model_path(store)
    model = load_model(weights, device)
    sigs = np.zeros((len(idx), 512), np.float16)
    step = 2048
    for k in range(0, len(idx), step):
        part = idx[k:k + step]
        sigs[k:k + len(part)] = embed(model, [c.image(i) for i in part], device)
        log(f"{game.id}: signatures {k + len(part)}/{len(idx)}")
    ball_game.save_npz(sigs_path(store, game), frame=c.frame[idx], track=c.track[idx], sig=sigs,
                       model=np.array(weights.name))
    return len(idx)


# ---- suggestions: "is this #N?" for any player with labels ----

PIECE = 10  # crops (~10 s) per piece of a tracklet: a swap mid-tracklet only spoils some pieces


def load_sigs(store: Store, game: Game) -> dict | None:
    p = sigs_path(store, game)
    if not p.exists():
        return None
    with np.load(p) as z:
        return {"frame": z["frame"], "track": z["track"], "sig": z["sig"].astype(np.float32)}


def _looks_path(store: Store, game: Game) -> Path:
    return store.game_dir(game.year, game.id) / "tracks" / "reid_looks.npz"


def player_sums(store: Store, game: Game) -> dict[int, tuple[np.ndarray, int]]:
    """Sum and count of signatures in each player's labelled stretches in this game (cached on disk
    until the signatures, tracklets or picks change)."""
    from soccer_clips.track import ball_game, people

    files = [sigs_path(store, game), people.path(store, game, people.DEFAULT_MODEL), people._picks_path(store, game)]
    if not files[0].exists() or not files[1].exists():
        return {}
    key = " ".join(f"{f.stat().st_mtime_ns}" if f.exists() else "-" for f in files)
    cache = _looks_path(store, game)
    if cache.exists():
        with np.load(cache) as z:
            if str(z["key"]) == key:
                return {int(p): (s, int(n)) for p, s, n in zip(z["players"], z["sums"], z["counts"])}
    s = load_sigs(store, game)
    t = people.Tracks.load(people.path(store, game, people.DEFAULT_MODEL))  # not the shared cache: may be another game than the one open
    out: dict[int, list] = {}
    for seg in people.segments(t, people.load_picks(store, game)):
        m = (s["track"] == seg["track"]) & (s["frame"] >= seg["start"]) & (s["frame"] < seg["end"])
        if m.any():
            acc = out.setdefault(seg["player"], [np.zeros(512, np.float32), 0])
            acc[0] += s["sig"][m].sum(0)
            acc[1] += int(m.sum())
    players = sorted(out)
    ball_game.save_npz(cache, key=np.array(key), players=np.array(players, np.int64),
                       sums=np.array([out[p][0] for p in players], np.float32).reshape(-1, 512),
                       counts=np.array([out[p][1] for p in players], np.int64))
    return {p: (out[p][0], out[p][1]) for p in players}


def player_look(store: Store, player: int, skip: str | None = None) -> tuple[np.ndarray | None, int]:
    """(unit-length mean signature of the player's labelled crops over all games but `skip`, crops used)."""
    total, n = np.zeros(512, np.float32), 0
    for g in store.list_games():
        if g.id == skip:
            continue
        got = player_sums(store, g).get(player)
        if got:
            total += got[0]
            n += got[1]
    return (total / max(np.linalg.norm(total), 1e-9), n) if n else (None, 0)


_pieces_cache: dict = {}
_theirs_cache: dict = {}


def _pieces(store: Store, game: Game) -> dict | None:
    """The game's signatures as pieces of ~PIECE crops along each tracklet (>= 3 crops): track, first and last
    frame, middle frame and unit mean signature of each. Kept in memory until the signatures change."""
    path = sigs_path(store, game)
    if not path.exists():
        return None
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _pieces_cache:
        s = load_sigs(store, game)
        order = np.lexsort((s["frame"], s["track"]))
        tr, fr, sig = s["track"][order], s["frame"][order], s["sig"][order]
        out = {"track": [], "lo": [], "hi": [], "mid": [], "vec": []}
        for idx in np.split(np.arange(len(tr)), np.flatnonzero(np.diff(tr)) + 1):
            for piece in np.array_split(idx, max(1, round(len(idx) / PIECE))):
                if len(piece) < 3:
                    continue
                out["track"].append(int(tr[piece[0]]))
                out["lo"].append(int(fr[piece[0]]))
                out["hi"].append(int(fr[piece[-1]]))
                out["mid"].append(int(fr[piece[len(piece) // 2]]))
                out["vec"].append(sig[piece].mean(0))
        vec = np.array(out["vec"], np.float32).reshape(-1, 512)
        _pieces_cache.clear()  # one game at a time, like the tracklets
        _pieces_cache[key] = {"track": np.array(out["track"], np.int64), "lo": np.array(out["lo"], np.int64),
                              "hi": np.array(out["hi"], np.int64), "mid": np.array(out["mid"], np.int64),
                              "vec": vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)}
    return _pieces_cache[key]


def _not_ours(t, roles: np.ndarray) -> set[int]:
    """Tracklets mostly called opponents or others by the team split (unknown counts as ours); cached per split."""
    key = id(roles)
    if key not in _theirs_cache:
        ids, inv = np.unique(t.track, return_inverse=True)
        theirs = np.bincount(inv, weights=roles >= 1, minlength=len(ids)) / np.bincount(inv, minlength=len(ids))
        _theirs_cache.clear()
        _theirs_cache[key] = (roles, set(ids[theirs > 0.5].tolist()))  # holds roles so its id isn't reused
    return _theirs_cache[key][1]


def suggest(store: Store, game: Game, player: int, limit: int = 20) -> dict:
    """
    Our-team tracklets (by the current team split) in the game that look most like `player` (their labelled
    crops in every game), best first, one per tracklet (its best ~PIECE-crop piece):
    {track, frame, x, y, box, score, start, end}. Left out: tracklets with any label in this game or a pick
    for this player, and anything on screen while the player is labelled elsewhere (one place at a time).
    """
    from soccer_clips.track import people

    pc = _pieces(store, game)
    if pc is None:
        return {"suggestions": [], "note": "no appearance signatures for this game yet (soccer-clips reid sigs)"}
    look, n = player_look(store, player)
    if look is None:
        return {"suggestions": [], "note": f"no labelled stretches of #{player} yet"}
    t = people.tracks(store, game)
    picks = people.load_picks(store, game)
    segs = people.segments(t, picks)
    skip = {x["track"] for x in segs} | {p["track"] for p in picks if p["player"] == player}
    # our team as the team split has it now (the crops' roles date from when they were cut: in a game with
    # no labels yet both teams were "team", and they split once she's labelled)
    skip |= _not_ours(t, people.roles_for(store, game)[1])
    ok = ~np.isin(pc["track"], list(skip))
    busy = sorted((x["start"], x["end"]) for x in segs if x["player"] == player)
    if busy:  # a piece overlapping a stretch where the player is labelled elsewhere
        starts, ends = np.array([a for a, _ in busy]), np.array([b for _, b in busy])
        k = np.searchsorted(starts, pc["hi"], side="right") - 1
        ok &= ~((k >= 0) & (pc["lo"] < ends[np.maximum(k, 0)]))
    idx = np.flatnonzero(ok)
    score = pc["vec"][idx] @ look
    idx, score = idx[np.argsort(-score, kind="stable")], np.sort(score)[::-1]
    _, first = np.unique(pc["track"][idx], return_index=True)  # each tracklet's best piece
    first = np.sort(first)
    out = []
    for j in first[:limit]:
        k, f = int(pc["track"][idx[j]]), int(pc["mid"][idx[j]])
        a, b = t.window(f, f + 1)
        i = a + int(np.flatnonzero(t.track[a:b] == k)[0])
        x1, y1, x2, y2 = t.box[i].tolist()
        out.append({"track": k, "frame": f, "x": round((x1 + x2) / 2, 4), "y": round((y1 + y2) / 2, 4),
                    "box": [round(v, 4) for v in (x1, y1, x2, y2)], "score": round(float(score[j]), 3),
                    "start": t.first[k], "end": t.last[k] + 1})
    return {"suggestions": out, "look_crops": n, "candidates": len(first)}


def guess(store: Store, game: Game, track: int, frame: int) -> dict:
    """
    Every player with labelled crops, ranked by how much `track` around `frame` (its nearest ~PIECE-crop piece,
    else all its crops) looks like them: {players: [{player, score, busy, rejected}]}. busy = labelled on another
    tracklet at `frame` (one place at a time); rejected = turned down for this tracklet; both ranked last.
    """
    from soccer_clips.track import people

    pc = _pieces(store, game)
    if pc is None:
        return {"players": [], "note": "no appearance signatures for this game yet"}
    on = np.flatnonzero(pc["track"] == track)
    if len(on):
        vec = pc["vec"][on[np.argmin(np.abs(pc["mid"][on] - frame))]]
    else:  # too short for a piece: whatever crops it has
        sg = load_sigs(store, game)
        mine = sg["sig"][sg["track"] == track]
        if not len(mine):
            return {"players": [], "note": "no crops of this tracklet (too small or too short)"}
        vec = mine.mean(0)
        vec = vec / max(np.linalg.norm(vec), 1e-9)
    totals: dict[int, np.ndarray] = {}
    for g in store.list_games():
        for p, (total, _) in player_sums(store, g).items():
            totals[p] = totals.get(p, 0) + total
    t = people.tracks(store, game)
    picks = people.load_picks(store, game)
    busy = {x["player"] for x in people.segments(t, picks) if x["start"] <= frame < x["end"] and x["track"] != track}
    rejected = {p["player"] for p in picks if p["track"] == track and p["kind"] != "is"}
    out = [{"player": p, "score": round(float(vec @ (v / max(np.linalg.norm(v), 1e-9))), 3), "busy": p in busy,
            "rejected": p in rejected} for p, v in totals.items()]
    out.sort(key=lambda x: (x["busy"] or x["rejected"], -x["score"]))
    return {"players": out}


# ---- automatic labels: where a player is, from appearance alone ----

AUTO_MARGIN = 0.02  # a piece must beat every other unlabelled teammate on screen by this much (cosine)
AUTO_MIN = 0.9  # and look at least this much like the player (2026-10-27: her own pieces ~0.95 against her look
# from the other games, pieces labelled as teammates ~0.45-0.69)


def _look_without(store: Store, game: Game, player: int, hidden: list[dict]) -> np.ndarray | None:
    """The player's look from every game, leaving out this game's `hidden` stretches (for the yardstick)."""
    total, n = np.zeros(512, np.float32), 0
    for g in store.list_games():
        got = player_sums(store, g).get(player)
        if got:
            total += got[0]
            n += got[1]
    if hidden:
        s = load_sigs(store, game)
        for x in hidden:
            m = (s["track"] == x["track"]) & (s["frame"] >= x["start"]) & (s["frame"] < x["end"])
            total -= s["sig"][m].sum(0)
            n -= int(m.sum())
    return total / max(np.linalg.norm(total), 1e-9) if n > 0 else None


def auto_label(store: Store, game: Game, player: int, margin: float = AUTO_MARGIN, min_score: float = AUTO_MIN,
               hidden: list[dict] | None = None, others_compete: bool = False) -> list[dict]:
    """
    Stretches where `player` is probably on screen and unlabelled: {track, start, end, frame, margin} (frame: a
    crop frame of the tracklet in the middle of the stretch, for showing it). A piece (~PIECE
    crops of a tracklet) is claimed when it looks at least `min_score` like the player and more like them than
    every other unlabelled our-team piece on screen at the same time, by at least `margin`. Nothing is claimed
    while the player is labelled elsewhere, nor on stretches labelled as anyone, nor on tracklets turned
    down for the player. For the yardstick: `hidden` =
    labelled stretches of the player to treat as unlabelled; `others_compete` = stretches labelled as other
    players stay in (a claim there is a known mistake).
    """
    from soccer_clips.track import people

    pc = _pieces(store, game)
    hidden = hidden or []
    look = _look_without(store, game, player, hidden)
    if pc is None or look is None:
        return []
    t = people.tracks(store, game)
    hid = {(x["track"], x["start"], x["end"]) for x in hidden}
    picks = people.load_picks(store, game)
    segs = [x for x in people.segments(t, picks) if (x["track"], x["start"], x["end"]) not in hid]
    # tracklets turned down for this player (X, or the marker said to be wrong) aren't claimed again
    rejected = {p["track"] for p in picks if p["player"] == player and p["kind"] != "is"}
    ok = ~np.isin(pc["track"], list(_not_ours(t, people.roles_for(store, game)[1]) | rejected))
    for x in segs:  # labelled pieces aren't candidates; the player labelled elsewhere rules out everyone else
        over = (pc["lo"] < x["end"]) & (pc["hi"] >= x["start"])
        if x["player"] == player or not others_compete:
            ok &= ~(over & (pc["track"] == x["track"]))
        if x["player"] == player:
            ok &= ~over
    idx = np.flatnonzero(ok)
    score = pc["vec"][idx] @ look
    # each piece against the best other candidate on screen at any of its crop frames
    lo, hi = pc["lo"][idx] // CROP_EVERY, pc["hi"][idx] // CROP_EVERY
    counts = hi - lo + 1
    owner = np.repeat(np.arange(len(idx)), counts)
    cell = np.repeat(lo, counts) + (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts))
    order = np.lexsort((-score[owner], cell))
    cell_s, owner_s = cell[order], owner[order]
    starts = np.flatnonzero(np.r_[True, np.diff(cell_s) > 0])
    best = np.full(cell_s.max() + 1 if len(cell_s) else 1, -np.inf)
    second = best.copy()
    best[cell_s[starts]] = score[owner_s[starts]]
    has2 = np.r_[starts[1:], len(cell_s)] - starts > 1
    second[cell_s[starts[has2]]] = score[owner_s[starts[has2] + 1]]
    rival = np.where(best[cell] == score[owner], second[cell], best[cell])
    worst = np.full(len(idx), -np.inf)
    np.maximum.at(worst, owner, rival)
    lead = score - worst
    out = []
    for j in np.flatnonzero((lead >= margin) & (score >= min_score)):
        p = idx[j]
        out.append({"track": int(pc["track"][p]), "start": int(pc["lo"][p]), "end": int(pc["hi"][p]) + CROP_EVERY,
                    "frames": [int(pc["mid"][p])], "margin": round(float(lead[j]), 4)})
    out.sort(key=lambda x: x["start"])
    merged = []  # consecutive pieces of one tracklet become one stretch
    for x in out:
        if merged and merged[-1]["track"] == x["track"] and x["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], x["end"])
            merged[-1]["margin"] = min(merged[-1]["margin"], x["margin"])
            merged[-1]["frames"] += x["frames"]
        else:
            merged.append(dict(x))
    for x in merged:
        frames = x.pop("frames")
        x["frame"] = frames[len(frames) // 2]
    return merged


def evaluate_auto(store: Store, game_ids: list[str], players: list[int] | None = None,
                  settings: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.8, 0.0), (0.85, 0.0), (0.85, 0.02),
                                                               (0.9, 0.0), (0.9, 0.02)), log=print) -> dict:
    """
    Hide every other labelled stretch of a player in a game, auto-label, and score (two rounds, so each
    stretch is hidden once). At each crop frame of a hidden stretch: right = claimed on the hidden tracklet,
    wrong = claimed on another, missed = no claim. false_s = seconds claimed on stretches labelled as another
    player (they compete here, so each is a known mistake). Per group (PLAYER as "her", everyone else as
    "others") and (min_score, margin) setting, summed over the games.
    """
    from soccer_clips.track import people

    res = {grp: {st: {"right": 0, "wrong": 0, "missed": 0, "false_s": 0.0, "claimed_s": 0.0} for st in settings}
           for grp in ("her", "others")}
    for gid in game_ids:
        game = store.load_game(gid)
        if _pieces(store, game) is None:
            continue
        t = people.tracks(store, game)
        segs = people.segments(t, people.load_picks(store, game))
        s = load_sigs(store, game)
        fps = game.fps_num / game.fps_den
        for player in sorted({x["player"] for x in segs} if players is None else set(players)):
            mine = [x for x in segs if x["player"] == player]
            theirs = [x for x in segs if x["player"] != player]
            r = res["her" if player == PLAYER else "others"]
            for half in (0, 1):
                hidden = mine[half::2]
                if not hidden:
                    continue
                for st in settings:
                    claims = auto_label(store, game, player, margin=st[1], min_score=st[0], hidden=hidden,
                                        others_compete=True)
                    r[st]["claimed_s"] += sum(c["end"] - c["start"] for c in claims) / fps
                    for c in claims:
                        for x in theirs:
                            if x["track"] == c["track"]:
                                r[st]["false_s"] += max(0, min(c["end"], x["end"]) - max(c["start"], x["start"])) / fps
                    for x in hidden:
                        frames = s["frame"][(s["track"] == x["track"]) & (s["frame"] >= x["start"]) & (s["frame"] < x["end"])]
                        for f in frames.tolist():
                            at = [c["track"] for c in claims if c["start"] <= f < c["end"]]
                            if x["track"] in at:
                                r[st]["right"] += 1
                            elif at:
                                r[st]["wrong"] += 1
                            else:
                                r[st]["missed"] += 1
        log(f"{gid}: done")
    for per in res.values():
        for r in per.values():
            claimed = r["right"] + r["wrong"]
            r["precision"] = round(r["right"] / claimed, 3) if claimed else None
            r["coverage"] = round(r["right"] / max(1, claimed + r["missed"]), 3)
    return res


_auto_cache: dict = {}


def auto_all(store: Store, game: Game) -> dict[int, list[dict]]:
    """auto_label for every player labelled in any game; a stretch two players both claim is dropped.
    Cached until the game's signatures, tracklets or picks change, or labels change in another game."""
    from soccer_clips.track import people

    files = [sigs_path(store, game), people.path(store, game, people.DEFAULT_MODEL)]
    if not all(f.exists() for f in files):
        return {}
    looks = [_looks_path(store, g) for g in store.list_games()]
    pp = people._picks_path(store, game)
    key = (game.id, *(f.stat().st_mtime_ns for f in files), pp.stat().st_mtime_ns if pp.exists() else 0,
           *(f.stat().st_mtime_ns if f.exists() else 0 for f in looks))
    if key not in _auto_cache:
        players = sorted({p for g in store.list_games() for p in player_sums(store, g)})
        claims = {p: auto_label(store, game, p) for p in players}
        flat = sorted((c["track"], c["start"], c["end"], p) for p, cs in claims.items() for c in cs)
        clash = set()
        for (k1, a1, b1, p1), (k2, a2, b2, p2) in zip(flat, flat[1:]):
            if k1 == k2 and a2 < b1 and p1 != p2:
                clash |= {(k1, a1, p1), (k2, a2, p2)}
        out = {p: [c for c in cs if (c["track"], c["start"], p) not in clash] for p, cs in claims.items()}
        _auto_cache.clear()
        _auto_cache[key] = {p: cs for p, cs in out.items() if cs}
    return _auto_cache[key]
