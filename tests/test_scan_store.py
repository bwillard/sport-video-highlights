from datetime import date

import pytest
from fastapi.testclient import TestClient

from conftest import make_video
from soccer_clips.scan import parse_played_on, scan
from soccer_clips.server.app import create_app
from soccer_clips.store import Store


@pytest.mark.parametrize(
    "stem, expected",
    [
        ("gu11-vs-saa-2026-10-10", date(2026, 10, 10)),
        ("gu11black-vs-ebu-2026-10-10", date(2026, 10, 10)),
        ("no-date-here", None),
        ("bad-date-2026-13-45", None),
    ],
)
def test_parse_played_on(stem, expected):
    assert parse_played_on(stem) == expected


def test_scan_registers_new_videos_once(settings):
    make_video(settings, "2026/gu11-vs-saa-2026-10-10.mp4")
    make_video(settings, "notayear/ignored-2026-01-01.mp4")
    store = Store(settings)

    added = scan(settings, store)
    assert [g.id for g in added] == ["gu11-vs-saa-2026-10-10"]
    assert scan(settings, store) == []

    game = store.load_game("gu11-vs-saa-2026-10-10")
    assert game.year == 2026
    assert game.played_on == date(2026, 10, 10)
    assert game.original == "2026/gu11-vs-saa-2026-10-10.mp4"
    assert (game.fps_num, game.fps_den) == (30000, 1001)
    assert (game.width, game.height) == (320, 180)
    assert game.has_audio
    assert game.duration_s == pytest.approx(3, abs=0.1)
    assert (store.game_dir(2026, game.id) / "probe.json").exists()


def test_frame_at_uses_exact_fps(settings):
    make_video(settings, "2026/g-2026-01-01.mp4")
    store = Store(settings)
    scan(settings, store)
    game = store.load_game("g-2026-01-01")
    # 70 min in: 29.97 fps gives 125874 frames; rounding to 30 fps would give 126000.
    assert game.frame_at(70 * 60) == 125874


def test_edit_game_round_trips(settings):
    make_video(settings, "2026/g-2026-01-01.mp4")
    store = Store(settings)
    scan(settings, store)
    with store.edit_game("g-2026-01-01") as game:
        game.opponent = "Rovers"
        game.kickoff_1h_s = 312.5
    reloaded = store.load_game("g-2026-01-01")
    assert reloaded.opponent == "Rovers"
    assert reloaded.kickoff_1h_s == 312.5
    assert not list(store.game_dir(2026, "g-2026-01-01").glob("*.tmp"))


def test_server_lists_games_and_serves_byte_ranges(settings):
    video = make_video(settings, "2026/g-2026-01-01.mp4")
    with TestClient(create_app(Store(settings))) as client:  # startup runs scan
        assert [g["id"] for g in client.get("/api/games").json()] == ["g-2026-01-01"]
        r = client.get("/video/g-2026-01-01", headers={"Range": "bytes=100-199"})
        assert r.status_code == 206
        assert r.content == video.read_bytes()[100:200]
        assert client.get("/video/nope").status_code == 404
