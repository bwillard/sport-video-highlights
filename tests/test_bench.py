import subprocess

from conftest import make_video
from soccer_clips.models import Callout, Clip
from soccer_clips.scan import scan
from soccer_clips.store import Store
from soccer_clips.track import bench

GAME = "gu11-test-2026-10-10"


def frames(settings, path) -> int:
    out = subprocess.run([settings.ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return int(out)


def test_add_and_cut_only_redoes_changed(settings):
    make_video(settings, f"2026/{GAME}.mp4", duration_s=4)
    store = Store(settings)
    scan(settings, store)
    with store.edit_game(GAME) as g:
        g.clips.append(Clip(id="c1", start_s=1.0, end_s=2.0, players=[38], note="pass",
                            callout=Callout(t_s=1.5, x=0.5, y=0.5)))
        g.clips.append(Clip(id="c2", start_s=2.0, end_s=3.0, players=[38, 7], callout=Callout(t_s=2.5, x=0.1, y=0.1)))

    a = bench.add_clip(store, GAME, "c1", "first")
    assert (a.why, a.callout_player, a.callout.t_s) == ("pass", 38, 1.5)
    b = bench.add_clip(store, GAME, "c2", "second", why="two players")
    assert b.callout is not None and b.callout_player is None  # whose callout is unknown
    bench.add_clip(store, GAME, "c1", "first", why="replaced")
    assert [(c.name, c.why) for c in bench.load_set(settings).clips] == [("second", "two players"), ("first", "replaced")]

    out = settings.games_dir.parent / "bench" / "clips"
    assert sorted(bench.cut(store)) == ["first", "second"]
    assert frames(settings, out / "first.mp4") == 30
    assert bench.cut(store) == []

    s = bench.load_set(settings)
    s.clips[0].end_s = 3.5
    bench.save_set(settings, s)
    assert bench.cut(store) == ["second"]
    assert frames(settings, out / "second.mp4") == 45
