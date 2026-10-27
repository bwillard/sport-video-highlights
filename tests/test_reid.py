import json

import cv2
import numpy as np
import pytest

from conftest import make_video
from soccer_clips.scan import scan
from soccer_clips.store import Store
from soccer_clips.track import ball_game, reid

COLOURS = [(40, 40, 200), (200, 40, 40), (40, 200, 40), (200, 200, 40), (40, 200, 200), (200, 40, 200)]


def fake_bundle(folder, games=("a", "b"), steps=40):
    """Games where 6 tracklets are always on screen, each a solid colour; tracklet 0 is her (labelled)."""
    rng = np.random.default_rng(0)
    folder.mkdir()
    index = {}
    for gid in games:
        blobs, frame, track = [], [], []
        for f in range(0, steps * reid.CROP_EVERY, reid.CROP_EVERY):
            for k, c in enumerate(COLOURS):
                img = np.clip(np.array(c) + rng.normal(0, 20, (60, 30, 3)), 0, 255).astype(np.uint8)
                blobs.append(cv2.imencode(".jpg", img)[1].tobytes())
                frame.append(f)
                track.append(k)
        lengths = [len(b) for b in blobs]
        np.savez(folder / f"{gid}.npz", data=np.frombuffer(b"".join(blobs), np.uint8),
                 offsets=np.concatenate([[0], np.cumsum(lengths)]), frame=np.array(frame), track=np.array(track),
                 role=np.zeros(len(frame), np.int8), height=np.full(len(frame), 60.0, np.float32))
        index[gid] = {"fps": 30.0, "her": [[0, 0, steps * reid.CROP_EVERY]]}
    (folder / "index.json").write_text(json.dumps(index))
    return reid.load_bundle(folder)


def test_batches_are_different_people_on_screen_together(tmp_path):
    games = fake_bundle(tmp_path / "b")
    rng = np.random.default_rng(1)
    for _ in range(10):
        imgs, ids = reid.sample_batch(games, rng, people_n=4, per=3)
        assert len(imgs) == 12 and np.bincount(ids).tolist() == [3, 3, 3, 3]
        means = np.array([im.reshape(-1, 3).mean(0) for im in imgs])
        colour = [tuple(m > m.mean()) for m in means]  # which channels are high (survives brightness changes)
        # each identity is one colour (one person); different identities are different people
        assert all(len({colour[i] for i in np.flatnonzero(ids == n)}) == 1 for n in range(4))
        assert len({colour[i] for i in range(len(imgs))}) == 4


def test_supcon_prefers_grouped_features():
    import torch

    ids = torch.tensor([0, 0, 1, 1, 2, 2])
    grouped = torch.nn.functional.normalize(torch.eye(3)[ids] + 0.01 * torch.randn(6, 3), dim=1)
    mixed = torch.nn.functional.normalize(torch.eye(3)[torch.tensor([0, 1, 2, 0, 1, 2])], dim=1)
    assert reid.supcon_loss(grouped, ids) < reid.supcon_loss(mixed, ids)


def test_auc():
    assert reid._auc([3, 4], [1, 2]) == 1.0 and reid._auc([1], [2]) == 0.0 and reid._auc([1], [1]) == 0.5


def test_train_and_evaluate(tmp_path):
    import torch

    games = fake_bundle(tmp_path / "b")
    stock = tmp_path / "stock.pt"
    torch.save(reid._model().state_dict(), stock)
    model = reid.train(games, stock, steps=2, device="cpu", workers=0, log=lambda m: None)
    feats = reid.features(model, games)
    r = reid.evaluate(games, feats, ["a"])["a"]
    assert r["stretches"] == 1 and r["chance"] == pytest.approx(1 / 6)
    assert 0 <= r["her_auc"] <= 1 and r["same_n"] > 0


def test_every_nth_frame_is_exact(settings):
    make_video(settings, "2026/gu11-test-2026-10-10.mp4", duration_s=6, audio=False)
    store = Store(settings)
    scan(settings, store)
    game = store.load_game("gu11-test-2026-10-10")
    got = list(reid._read_every(store, game, 30, 150))
    assert [f for f, _ in got] == [30, 60, 90, 120]
    for f, img in got:
        ref = next(ball_game.read_frames(store, game, f, f + 1))
        assert np.abs(ref.astype(int) - img).mean() < 0.5
