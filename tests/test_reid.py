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
    """Games where 6 tracklets are always on screen, each a solid colour; tracklet 0 is her and 1 is #5 (labelled)."""
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
        index[gid] = {"fps": 30.0, "named": [[0, 0, steps * reid.CROP_EVERY, 38], [1, 0, steps * reid.CROP_EVERY, 5]]}
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
    assert r["mates_players"] == 1 and r["mates_n"] == 1


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


def test_suggestions_rank_lookalikes_and_respect_one_place_at_a_time(settings, monkeypatch):
    from soccer_clips.track import people

    make_video(settings, "2026/gu11-test-2026-10-10.mp4", duration_s=6, audio=False)
    store = Store(settings)
    scan(settings, store)
    game = store.load_game("gu11-test-2026-10-10")
    # four tracklets: 1 = her (labelled, frames 0-89), 2 = a teammate who looks different, 3 = a look-alike
    # later (90-179), 4 = a look-alike on screen while she's labelled (0-89): can't be her
    spans = {1: (0, 90), 2: (0, 180), 3: (90, 180), 4: (0, 90)}
    xs = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.7}
    rows = [(f, k) for k, (a, b) in spans.items() for f in range(a, b)]
    frame = np.array([f for f, _ in rows], np.int32)
    track = np.array([k for _, k in rows], np.int32)
    x1 = np.array([xs[k] for k in track], np.float32)
    n = len(rows)
    ball_game.save_npz(people.path(store, game, people.DEFAULT_MODEL), frame=frame, track=track, x1=x1, y1=np.full(n, 0.2, np.float32),
                       x2=x1 + 0.1, y2=np.full(n, 0.6, np.float32), conf=np.full(n, 0.9, np.float32),
                       det=np.arange(n, dtype=np.int32), ranges=np.array([[0, 180]], np.int32))
    people.add_pick(store, game, 10, 38, "is", 0.15, 0.4)
    rng = np.random.default_rng(0)
    her, other = rng.normal(size=512), rng.normal(size=512)
    looks = {1: her, 2: other, 3: her, 4: her}
    on_grid = frame % reid.CROP_EVERY == 0
    sig = np.array([looks[k] + 0.3 * rng.normal(size=512) for k in track[on_grid]])
    sig /= np.linalg.norm(sig, axis=1, keepdims=True)
    ball_game.save_npz(reid.sigs_path(store, game), frame=frame[on_grid], track=track[on_grid],
                       sig=sig.astype(np.float16), model=np.array("test"))
    monkeypatch.setattr(reid, "PIECE", 3)
    got = reid.suggest(store, game, 38)
    assert [s["track"] for s in got["suggestions"]] == [3, 2]
    assert got["suggestions"][0]["score"] > 0.5 > got["suggestions"][1]["score"]
    assert reid.suggest(store, game, 7)["note"] == "no labelled stretches of #7 yet"
    # who a tracklet looks like: the look-alike is #38 first; one on screen while she's labelled elsewhere is
    # "busy" for her
    assert reid.guess(store, game, 3, 120)["players"][0] == {"player": 38, "score": pytest.approx(1, abs=0.2), "busy": False,
                                                                 "rejected": False}
    assert reid.guess(store, game, 4, 30)["players"][0]["busy"] is True
    # automatic labels: the look-alike while she isn't labelled; never the different-looking teammate, nor
    # the look-alike on screen while she's labelled elsewhere
    claims = reid.auto_label(store, game, 38, min_score=0.5)
    assert [c["track"] for c in claims] == [3] and claims[0]["start"] <= claims[0]["frame"] < claims[0]["end"]
    assert reid.auto_label(store, game, 38, min_score=0.99) == []
    assert {p: [c["track"] for c in cs] for p, cs in reid.auto_all(store, game).items()} == {38: [3]}
    # "no" takes a tracklet out of the suggestions, without counting as a swap
    people.add_pick(store, game, 100, 38, "no", 0.55, 0.4, track=3)
    assert [s["track"] for s in reid.suggest(store, game, 38)["suggestions"]] == [2]
    assert reid.auto_label(store, game, 38, min_score=0.5) == []  # nor claimed automatically
    assert people.label_points(store, game, 38)[1] == []
    assert reid.guess(store, game, 3, 120)["players"][-1]["rejected"] is True  # #38 is now last for it
    # the players review: a page of answers, a labelled stretch turned down, and each page taken back
    assert [x["track"] for x in reid.labelled_scores(store, game, 38)] == [1]
    segs = lambda: {(x["track"], x["player"]) for x in people.segments(people.tracks(store, game), people.load_picks(store, game))}
    assert people.answer_tracklets(store, game, 38, [{"track": 2, "frame": 30, "start": 0, "yes": True},
                                                     {"track": 4, "frame": 30, "start": 0, "yes": False}], "b1") == 2
    assert segs() == {(1, 38), (2, 38)}
    assert reid.guess(store, game, 4, 30)["players"][-1]["rejected"] is True
    assert people.answer_tracklets(store, game, 38, [{"track": 1, "frame": 30, "start": 0, "yes": False, "labelled": True},
                                                     {"track": 2, "frame": 30, "start": 0, "yes": True, "labelled": True}], "b2") == 1
    assert segs() == {(2, 38)}
    assert people.undo_answers(store, game, "b2") == 1 and segs() == {(1, 38), (2, 38)}
    assert people.undo_answers(store, game, "b1") == 2 and segs() == {(1, 38)}
    assert [p["kind"] for p in people.load_picks(store, game)] == ["is", "no"]


def test_strip_shows_crops_across_the_stretch(settings):
    make_video(settings, "2026/gu11-test-2026-10-10.mp4", duration_s=2, audio=False)
    store = Store(settings)
    scan(settings, store)
    game = store.load_game("gu11-test-2026-10-10")
    blobs, frame, track = [], [], []
    for f in range(0, 300, 30):
        for k, c in ((1, (0, 0, 200)), (2, (200, 0, 0))):
            blobs.append(cv2.imencode(".png", np.full((60, 30, 3), c, np.uint8))[1].tobytes())
            frame.append(f)
            track.append(k)
    reid.crops_path(store, game).parent.mkdir(parents=True, exist_ok=True)
    np.savez(reid.crops_path(store, game), data=np.frombuffer(b"".join(blobs), np.uint8),
             offsets=np.concatenate([[0], np.cumsum([len(b) for b in blobs])]), frame=np.array(frame), track=np.array(track))
    img = cv2.imdecode(np.frombuffer(reid.strip(store, game, 1, 0, 300, height=120), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[0] == 120 and img.shape[1] == 3 * 60 + 2 * 3  # three crops (30x60 -> 60x120), two gaps
    assert img[60, 30, 2] > 150 and img[60, 30, 0] < 60  # track 1's red, not track 2's blue
    one = cv2.imdecode(np.frombuffer(reid.strip(store, game, 2, 100, 110), np.uint8), cv2.IMREAD_COLOR)
    assert one.shape[1] == 60  # no crop inside: the nearest one
    assert reid.strip(store, game, 9, 0, 300) is None
