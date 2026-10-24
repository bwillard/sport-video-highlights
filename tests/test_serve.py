import io
import json

from soccer_clips import cli


class FakeChild:
    started = []

    def __init__(self, cmd):
        FakeChild.started.append(cmd)
        self.stopped = False

    def terminate(self):
        self.stopped = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.stopped = True


def test_restarts_on_code_change_but_not_during_a_render(monkeypatch):
    FakeChild.started = []
    statuses = iter(["running", "running", "done"])  # what /api/jobs says each time it's asked
    ticks = [{(1, "/x/ball_track.py")}, set(), set()]  # a change, then two quiet 5 s ticks

    def fake_urlopen(url, timeout=None):
        return io.BytesIO(json.dumps([{"status": next(statuses)}]).encode())

    monkeypatch.setattr("subprocess.Popen", FakeChild)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("watchfiles.watch", lambda *a, **k: iter(ticks))
    cli._serve_reloading("0.0.0.0", 8799)
    # started once, held off twice while the render ran, restarted once it was done
    assert len(FakeChild.started) == 2
    assert FakeChild.started[0][-4:] == ["--host", "0.0.0.0", "--port", "8799"]
    assert "--no-reload" in FakeChild.started[0]
