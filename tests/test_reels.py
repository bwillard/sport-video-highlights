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


def make_reel(client, card_s):
    """Two clips from game A (out of order) and one from game B (no audio)."""
    a1 = add_clip(client, GAME_A, 3.0, 4.0)
    a2 = add_clip(client, GAME_A, 0.5, 2.0)
    b1 = add_clip(client, GAME_B, 1.0, 2.0)
    items = [{"game_id": GAME_A, "clip_id": a1}, {"game_id": GAME_A, "clip_id": a2},
             {"game_id": GAME_B, "clip_id": b1}]
    r = client.post("/api/reels", json={"title": "Fall 2026: vs Rovers!", "items": items, "card_s": card_s})
    assert r.status_code == 201
    return r.json()["reel"]


@pytest.fixture
def reel(client):
    """Without title cards, so segment times are just the clips."""
    return make_reel(client, card_s=0)


@pytest.fixture
def carded_reel(client):
    """With 1s title cards."""
    return make_reel(client, card_s=1)


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


def test_title_cards(client, settings, carded_reel):
    rid = carded_reel["id"]
    client.patch(f"/api/games/{GAME_A}", json={"kickoff_1h_s": 0.0})
    a1 = carded_reel["items"][0]["clip_id"]
    client.patch(f"/api/games/{GAME_A}/clips/{a1}", json={"note": "Shot from outside the box"})
    [r] = render(client, rid)["renders"]
    card = 30 / (30000 / 1001)  # 1s rounds to 30 frames = 1.001s

    assert r["start_card_lines"] == ["Fall 2026: vs Rovers!", "2 games · January 1 – January 8, 2026"]
    segs = r["segments"]
    # game A has a kickoff, so its cards show the match minute; game B has none, so video time
    assert [s["card_lines"] for s in segs] == [["1'", "Shot from outside the box"], ["1'"], ["0:01"]]
    # layout: start card, then card + clip for each clip
    assert segs[0]["card_start_s"] == pytest.approx(card, abs=0.03)
    assert segs[0]["reel_start_s"] == pytest.approx(2 * card, abs=0.03)
    for prev, nxt in zip(segs, segs[1:]):
        assert nxt["card_start_s"] == pytest.approx(prev["reel_end_s"])
        assert nxt["reel_start_s"] - nxt["card_start_s"] == pytest.approx(card, abs=0.03)

    # every frame present: 4 cards x 30 + clips 30 + 45 + 30
    out = settings.reels_dir / rid / f"{PRESET}.mp4"
    streams = probe(settings, out)
    assert int(streams["video"]["nb_read_frames"]) == 4 * 30 + 105
    assert float(streams["audio"]["duration"]) == pytest.approx(float(streams["video"]["duration"]), abs=0.05)

    # card text changes make the render out of date
    client.patch(f"/api/games/{GAME_A}/clips/{a1}", json={"note": "Goal!"})
    assert client.get(f"/api/reels/{rid}").json()["renders"][0]["stale"] == ["1 clip card(s) changed (note or kickoff time)"]
    client.patch(f"/api/reels/{rid}", json={"title": "Season"})
    assert "title card changed" in client.get(f"/api/reels/{rid}").json()["renders"][0]["stale"]
    client.patch(f"/api/reels/{rid}", json={"card_s": 2})
    assert "card length changed" in client.get(f"/api/reels/{rid}").json()["renders"][0]["stale"]
    assert client.patch(f"/api/reels/{rid}", json={"card_s": 99}).status_code == 422


def test_draw_card_wraps_long_notes():
    from soccer_clips.reel.cards import draw_card

    img = draw_card(["2H 7'", "word " * 200], 960, 540)
    assert img.size == (960, 540)
    # text stays inside the frame: the outer 5% border is all background
    px = img.load()
    border = [px[x, y] for x in range(960) for y in list(range(0, 27)) + list(range(513, 540))]
    assert set(border) == {(17, 17, 17)}


def frame_at(settings, path, t):
    from io import BytesIO

    from PIL import Image

    png = subprocess.run([settings.ffmpeg, "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    return Image.open(BytesIO(png)).convert("RGB")


def test_callout_freeze(client, settings, reel):
    from soccer_clips.models import Segment

    rid = reel["id"]
    a2 = reel["items"][1]["clip_id"]  # game A 0.5s -> 2.0s
    callout = {"t_s": 1.0, "x": 0.5, "y": 0.5, "r": 0.1}
    client.patch(f"/api/games/{GAME_A}/clips/{a2}", json={"callout": callout})
    [r] = render(client, rid)["renders"]
    fps = 30000 / 1001
    freeze = 45 / fps  # default 1.5s = 45 frames

    segs = [Segment(**s) for s in r["segments"]]
    s = segs[1]
    assert s.callout.model_dump() == callout and s.freeze_s == pytest.approx(freeze)
    # the freeze starts on frame 15 of the clip (1.0s - 0.5s)
    assert s.freeze_at_s - s.reel_start_s == pytest.approx(15 / fps, abs=0.001)
    assert s.reel_end_s - s.reel_start_s == pytest.approx(1.5 + freeze, abs=0.03)
    assert segs[2].reel_start_s == pytest.approx(s.reel_end_s)
    assert segs[0].freeze_at_s is None

    # jump-back: before, during and after the freeze
    assert s.source_time(s.reel_start_s + 0.2) == pytest.approx(0.7, abs=0.02)
    assert s.source_time(s.freeze_at_s + 1.0) == pytest.approx(1.0, abs=0.02)
    assert s.source_time(s.freeze_at_s + freeze + 0.3) == pytest.approx(1.3, abs=0.02)

    out = settings.reels_dir / rid / f"{PRESET}.mp4"
    streams = probe(settings, out)
    assert int(streams["video"]["nb_read_frames"]) == 105 + 45
    assert float(streams["audio"]["duration"]) == pytest.approx(float(streams["video"]["duration"]), abs=0.05)

    # the held frame has a white ring around the callout and is dimmed outside it
    held = frame_at(settings, out, s.freeze_at_s + 0.7)
    w, h = held.size
    ring_x = round(w * 0.5 + 0.1 * h)  # right edge of the ring
    assert min(held.getpixel((ring_x, h // 2))) > 200
    before = frame_at(settings, out, s.freeze_at_s - 0.05)
    corner = lambda img: sum(img.getpixel((w - 20, 20)))
    assert corner(held) < corner(before) * 0.7

    # moving the callout makes the render out of date; turning freezes off changes the length
    client.patch(f"/api/games/{GAME_A}/clips/{a2}", json={"callout": {**callout, "x": 0.3}})
    assert client.get(f"/api/reels/{rid}").json()["renders"][0]["stale"] == ["1 callout(s) added, moved or removed"]
    client.patch(f"/api/reels/{rid}", json={"callout_s": 0})
    [r] = render(client, rid)["renders"]
    assert all(seg["freeze_at_s"] is None for seg in r["segments"])
    assert int(probe(settings, out)["video"]["nb_read_frames"]) == 105


def test_start_card_lines():
    from datetime import date, datetime

    from soccer_clips.models import Game, Reel
    from soccer_clips.reel.cards import start_card_lines

    def game(gid, played_on, description=None):
        return Game(id=gid, year=2026, played_on=played_on, description=description, original="x.mp4",
                    duration_s=60, fps_num=30000, fps_den=1001, width=1920, height=1080, created_at=datetime.now())

    saa = game("saa", date(2026, 10, 10), "Home vs SAA, 3-1 win")
    reel = Reel(title="Fall highlights")
    assert start_card_lines(reel, [saa, saa]) == ["Fall highlights", "Home vs SAA, 3-1 win", "Saturday, October 10, 2026"]
    # a title that repeats the description doesn't show it twice
    assert start_card_lines(Reel(title="home vs saa, 3-1 WIN"), [saa]) == ["home vs saa, 3-1 WIN", "Saturday, October 10, 2026"]
    assert start_card_lines(reel, [game("x", None)]) == ["Fall highlights"]
    # several games: count + date range
    games = [saa, game("a", date(2026, 9, 19)), game("b", None)]
    assert start_card_lines(reel, games) == ["Fall highlights", "3 games · September 19 – October 10, 2026"]
    assert start_card_lines(reel, [saa, game("old", date(2025, 11, 2))])[1] == "2 games · November 2, 2025 – October 10, 2026"


def test_game_info_edit_updates_title_card(client):
    a1 = add_clip(client, GAME_A, 1.0, 2.0)
    rid = client.post("/api/reels", json={"title": "Rovers", "card_s": 1,
                                          "items": [{"game_id": GAME_A, "clip_id": a1}]}).json()["reel"]["id"]
    g = client.patch(f"/api/games/{GAME_A}", json={"played_on": "2026-01-03", "description": "Away vs Rovers"}).json()
    assert (g["played_on"], g["description"]) == ("2026-01-03", "Away vs Rovers")
    [r] = render(client, rid)["renders"]
    assert r["start_card_lines"] == ["Rovers", "Away vs Rovers", "Saturday, January 3, 2026"]
    client.patch(f"/api/games/{GAME_A}", json={"description": "Away vs Rovers, 2-2"})
    assert client.get(f"/api/reels/{rid}").json()["renders"][0]["stale"] == ["title card changed"]
    assert client.patch(f"/api/games/{GAME_A}", json={"description": None}).json()["description"] is None


def test_draw_card_keeps_body_lines_separate():
    from soccer_clips.reel.cards import draw_card

    one = draw_card(["Title", "Home vs SAA Saturday, October 10, 2026"], 960, 540)
    two = draw_card(["Title", "Home vs SAA", "Saturday, October 10, 2026"], 960, 540)
    rows = lambda img: sum(1 for y in range(540) if any(img.getpixel((x, y)) != (17, 17, 17) for x in range(0, 960, 2)))
    assert rows(two) > rows(one)  # an extra text row
