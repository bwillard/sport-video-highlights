import json
import subprocess

import pytest
from fastapi.testclient import TestClient

from conftest import make_video
from soccer_clips.models import DEFAULT_TAGS
from soccer_clips.server.app import create_app
from soccer_clips.store import Store

GAME = "g-2026-01-01"


@pytest.fixture
def client(settings):
    make_video(settings, f"2026/{GAME}.mp4")
    app = create_app(Store(settings))
    with TestClient(app) as c:  # startup runs scan
        c.app_state = app.state
        yield c


def count_frames(settings, path) -> int:
    out = subprocess.run(
        [settings.ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(json.loads(out)["streams"][0]["nb_read_frames"])


def test_tags_default_written_once(client, settings):
    tags = client.get("/api/tags").json()
    assert [t["name"] for t in tags] == [t.name for t in DEFAULT_TAGS]
    assert tags[0] == {"name": "goal", "hotkey": "1"}
    assert (settings.games_dir / "tags.json").exists()


def test_clip_crud(client):
    r = client.post(f"/api/games/{GAME}/clips", json={"start_s": 0.5, "end_s": 1.5, "tags": ["shot"]})
    assert r.status_code == 201
    clip = r.json()
    assert len(clip["id"]) == 8

    r = client.patch(f"/api/games/{GAME}/clips/{clip['id']}",
                     json={"rating": 4, "involves_her": True, "note": "nice"})
    assert r.json()["rating"] == 4 and r.json()["involves_her"] and r.json()["note"] == "nice"
    assert r.json()["tags"] == ["shot"]  # untouched fields kept

    # explicit null clears a field
    assert client.patch(f"/api/games/{GAME}/clips/{clip['id']}", json={"rating": None}).json()["rating"] is None

    client.post(f"/api/games/{GAME}/clips", json={"start_s": 0.1, "end_s": 0.4})
    starts = [c["start_s"] for c in client.get(f"/api/games/{GAME}/clips").json()]
    assert starts == [0.1, 0.5]  # sorted by start

    assert client.delete(f"/api/games/{GAME}/clips/{clip['id']}").status_code == 204
    assert len(client.get(f"/api/games/{GAME}/clips").json()) == 1
    assert client.delete(f"/api/games/{GAME}/clips/{clip['id']}").status_code == 404


@pytest.mark.parametrize("body", [
    {"start_s": 2, "end_s": 1},     # end before start
    {"start_s": -1, "end_s": 1},    # negative
    {"start_s": 1, "end_s": 99},    # past the end of the 3s video
    {"start_s": 0, "end_s": 1, "rating": 6},
])
def test_invalid_clips_rejected(client, body):
    assert client.post(f"/api/games/{GAME}/clips", json=body).status_code == 422
    assert client.get(f"/api/games/{GAME}/clips").json() == []


def test_kickoff_set_and_clear(client):
    assert client.patch(f"/api/games/{GAME}", json={"kickoff_1h_s": 1.25}).json()["kickoff_1h_s"] == 1.25
    game = client.patch(f"/api/games/{GAME}", json={"kickoff_2h_s": 2.0}).json()
    assert (game["kickoff_1h_s"], game["kickoff_2h_s"]) == (1.25, 2.0)
    assert client.patch(f"/api/games/{GAME}", json={"kickoff_1h_s": None}).json()["kickoff_1h_s"] is None


@pytest.mark.parametrize("fast", [False, True])
def test_export_clip(client, settings, fast):
    clip = client.post(f"/api/games/{GAME}/clips", json={"start_s": 1.0, "end_s": 2.0}).json()
    [job] = client.post(f"/api/games/{GAME}/export", json={"clip_ids": [clip["id"]], "fast": fast}).json()
    client.app_state.jobs.wait()

    job = client.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "done", job["error"]
    out = settings.clips_dir / f"2026/{GAME}/{clip['id']}.mp4"
    assert job["output"] == f"2026/{GAME}/{clip['id']}.mp4"
    assert out.exists()
    assert not list(out.parent.glob("*.part.mp4"))
    if not fast:
        # 1.0s -> 2.0s at 29.97 fps is frames 30..60: exactly 30 frames.
        assert count_frames(settings, out) == 30

    [saved] = client.get(f"/api/games/{GAME}/clips").json()
    assert saved["export"] == job["output"]
    assert client.get(f"/clips/{GAME}/{clip['id']}.mp4").status_code == 200

    # changing the range invalidates the export; deleting removes the file
    moved = client.patch(f"/api/games/{GAME}/clips/{clip['id']}", json={"end_s": 2.5}).json()
    assert moved["export"] is None
    client.delete(f"/api/games/{GAME}/clips/{clip['id']}")
    assert not out.exists()


def test_export_unknown_clip_is_404(client):
    assert client.post(f"/api/games/{GAME}/export", json={"clip_ids": ["nope"]}).status_code == 404


def test_callout_set_validate_and_clear(client):
    clip = client.post(f"/api/games/{GAME}/clips", json={"start_s": 1.0, "end_s": 2.0}).json()
    url = f"/api/games/{GAME}/clips/{clip['id']}"
    callout = {"t_s": 1.5, "x": 0.4, "y": 0.6, "r": 0.05}
    assert client.patch(url, json={"callout": callout}).json()["callout"] == callout
    assert client.patch(url, json={"callout": {**callout, "t_s": 2.5}}).status_code == 422  # outside the clip
    assert client.patch(url, json={"callout": {**callout, "x": 1.5}}).status_code == 422  # off the frame
    # moving the end keeps a callout that's still inside; moving the start past it drops it
    assert client.patch(url, json={"end_s": 2.2}).json()["callout"] == callout
    assert client.patch(url, json={"start_s": 1.6}).json()["callout"] is None
    client.patch(url, json={"start_s": 1.0, "callout": callout})
    assert client.patch(url, json={"callout": None}).json()["callout"] is None
