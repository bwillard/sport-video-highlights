import json
import subprocess

import pytest
from fastapi.testclient import TestClient

from conftest import make_video
from soccer_clips.server.app import create_app
from soccer_clips.store import Store

GAME_A = "a-2026-01-01"
GAME_B = "b-2026-01-08"
PRESET = "preview_540p"


@pytest.fixture
def client(settings):
    make_video(settings, f"2026/{GAME_A}.mp4", duration_s=6)
    make_video(settings, f"2026/{GAME_B}.mp4", duration_s=6, audio=False)
    app = create_app(Store(settings))
    with TestClient(app) as c:
        c.app_state = app.state
        yield c


def add_clip(client, game, start, end):
    return client.post(f"/api/games/{game}/clips", json={"start_s": start, "end_s": end}).json()["id"]


def render(client, reel_id):
    job = client.post(f"/api/reels/{reel_id}/render", json={"preset": PRESET}).json()
    client.app_state.jobs.wait()
    job = client.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "done", job["error"]
    return client.get(f"/api/reels/{reel_id}").json()


def probe(settings, path) -> dict:
    out = subprocess.run(
        [settings.ffprobe, "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_type,duration,nb_read_frames,width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


@pytest.fixture
def reel(client):
    """Two clips from game A (out of order) and one from game B (no audio)."""
    a1 = add_clip(client, GAME_A, 3.0, 4.0)
    a2 = add_clip(client, GAME_A, 0.5, 2.0)
    b1 = add_clip(client, GAME_B, 1.0, 2.0)
    items = [{"game_id": GAME_A, "clip_id": a1}, {"game_id": GAME_A, "clip_id": a2},
             {"game_id": GAME_B, "clip_id": b1}]
    r = client.post("/api/reels", json={"title": "Fall 2026: vs Rovers!", "items": items})
    assert r.status_code == 201
    return r.json()["reel"]


def test_render_segments_and_output(client, settings, reel):
    view = render(client, reel["id"])
    [r] = view["renders"]
    assert r["preset"] == PRESET and r["stale"] == [] and r["skipped"] == []

    # segments follow reel order, are contiguous, and have the clips' lengths
    segs = r["segments"]
    assert [(s["game_id"], round(s["source_start_s"], 2)) for s in segs] == [(GAME_A, 3.0), (GAME_A, 0.5), (GAME_B, 1.0)]
    assert segs[0]["reel_start_s"] == 0
    for prev, nxt in zip(segs, segs[1:]):
        assert nxt["reel_start_s"] == pytest.approx(prev["reel_end_s"])
    for s, length in zip(segs, [1.0, 1.5, 1.0]):
        assert s["reel_end_s"] - s["reel_start_s"] == pytest.approx(length, abs=0.03)

    # output: 540p, every frame present (30 + 45 + 30), audio in sync with video
    out = settings.reels_dir / reel["id"] / f"{PRESET}.mp4"
    streams = probe(settings, out)
    assert (streams["video"]["width"], streams["video"]["height"]) == (960, 540)
    assert int(streams["video"]["nb_read_frames"]) == 105
    assert float(streams["audio"]["duration"]) == pytest.approx(float(streams["video"]["duration"]), abs=0.05)
    assert r["duration_s"] == pytest.approx(segs[-1]["reel_end_s"], abs=0.05)
    assert (settings.reels_dir / reel["id"] / f"{PRESET}.render.json").exists()

    # served inline (so <video> plays it) with a readable filename, and seekable
    resp = client.get(f"/reels/{reel['id']}/{PRESET}.mp4", headers={"Range": "bytes=0-99"})
    assert resp.status_code == 206
    assert 'filename="fall-2026-vs-rovers-preview_540p.mp4"' in resp.headers["content-disposition"]
    assert resp.headers["content-disposition"].startswith("inline")


def test_segment_maps_back_to_source_time(client, reel):
    from soccer_clips.models import Segment

    segs = [Segment(**s) for s in render(client, reel["id"])["renders"][0]["segments"]]
    # 0.25s into the second segment is 0.25s after that clip's start in game A
    assert segs[1].source_time(segs[1].reel_start_s + 0.25) == pytest.approx(0.75, abs=0.02)


def test_rerender_reuses_cached_intermediates(client, settings, reel):
    render(client, reel["id"])
    cache = settings.reels_dir / reel["id"] / "cache"
    before = {p.name: p.stat().st_mtime_ns for p in cache.glob("*.mp4")}
    assert len(before) == 3
    render(client, reel["id"])
    assert {p.name: p.stat().st_mtime_ns for p in cache.glob("*.mp4")} == before


def test_stale_after_edits_and_missing_clips_skipped(client, reel):
    rid = reel["id"]
    render(client, rid)
    items = reel["items"]

    # edit a clip's range
    client.patch(f"/api/games/{GAME_A}/clips/{items[0]['clip_id']}", json={"end_s": 4.5})
    assert client.get(f"/api/reels/{rid}").json()["renders"][0]["stale"] == ["1 clip(s) had their start/end changed"]

    # reorder
    client.patch(f"/api/reels/{rid}", json={"items": list(reversed(items))})
    assert "clips were added, removed or reordered" in client.get(f"/api/reels/{rid}").json()["renders"][0]["stale"]

    # delete a clip: shown as missing, flagged stale, skipped on the next render
    client.delete(f"/api/games/{GAME_B}/clips/{items[2]['clip_id']}")
    view = client.get(f"/api/reels/{rid}").json()
    assert [i["missing"] for i in view["items"]] == [True, False, False]
    assert "1 clip(s) were deleted" in view["renders"][0]["stale"]
    r = render(client, rid)["renders"][0]
    assert r["stale"] == []
    assert r["skipped"] == [items[2]]
    assert len(r["segments"]) == 2


def test_reel_crud(client, settings, reel):
    rid = reel["id"]
    assert [v["reel"]["id"] for v in client.get("/api/reels").json()] == [rid]
    assert client.patch(f"/api/reels/{rid}", json={"title": "Season"}).json()["reel"]["title"] == "Season"
    assert client.patch(f"/api/reels/{rid}", json={"items": [{"game_id": GAME_A, "clip_id": "nope"}]}).status_code == 404
    assert client.post("/api/reels", json={"title": "x", "items": [{"game_id": "nope", "clip_id": "x"}]}).status_code == 404

    empty = client.post("/api/reels", json={"title": "empty"}).json()["reel"]
    assert client.post(f"/api/reels/{empty['id']}/render", json={"preset": PRESET}).status_code == 422

    assert client.delete(f"/api/reels/{rid}").status_code == 204
    assert not (settings.reels_dir / rid).exists()
    assert client.get(f"/api/reels/{rid}").status_code == 404
