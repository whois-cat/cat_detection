import sqlite3

import pytest

from decider.config import FeedConfig, FeederConfig, parse
from decider.display import DisplayThrottle, status_text
from decider.feeder import CLOCK_RATE, Feeder, observations
from decider.journal import FeedJournal
from decider.zone_state import ZoneSummary


class FakeClient:
    def __init__(self):
        self.state = "closed"
        self.calls = []
        self.fail_close = False

    def force_closed(self):
        self.calls.append(("force_closed",))

    def set_door(self, desired, reason):
        self.calls.append(("door", desired, reason))
        if desired == "close" and self.fail_close:
            return False
        self.state = desired
        return True

    def set_display_text(self, text, interval=1):
        self.calls.append(("display", text, interval))
        return True

    def feed(self, grain_num=1):
        self.calls.append(("feed", grain_num))
        return True


class Clock:
    def __init__(self):
        self.t = 1_790_000_000.0

    def __call__(self):
        return self.t


def cat(name, p=0.97, box=(0.4, 0.4, 0.2, 0.2)):
    others = (1 - p) / 3
    cats = {n: others for n in ("alisa", "chuzh", "ellie", "felisis")}
    cats[name] = p
    return {"box": list(box), "score": 0.9, "cats": cats}


def make(tmp_path, **kw):
    clock = Clock()
    cfg = FeederConfig(id="feeder1", camera="grey", api_base_url="http://x", serial_number="S",
                       allowed_cats=["chuzh", "ellie"], door_close_timeout_sec=5, **kw)
    client = FakeClient()
    decisions = []
    f = Feeder(cfg, client, FeedJournal(tmp_path / "j.db"), decisions.append, monotonic=clock, wall=clock)
    f.start()
    return f, client, clock, decisions

def feed_frames(f, clock, seconds, dets, fps=5):
    for _ in range(int(seconds * fps)):
        clock.t += 1 / fps
        f.handle_result({"camera": "grey", "pts": int(clock.t * CLOCK_RATE), "dets": dets})


def doors(client):
    return [c[1:] for c in client.calls if c[0] == "door"]


def test_observations():
    r = {"dets": [cat("chuzh", 0.97), cat("alisa", 0.4), {"box": [0.9, 0.9, 0.05, 0.05], "score": 0.5}]}
    obs = observations(r, unknown_conf=0.5, polygon=[[0, 0], [0.8, 0], [0.8, 0.8], [0, 0.8]])
    assert obs[0] == ("chuzh", 0.97, True)
    assert obs[1][0] == "unknown" and obs[1][1] == pytest.approx(0.4)
    assert obs[2] == ("cat", None, False)  # no classifier; centre outside the polygon


def test_allowed_cat_opens_then_leaves(tmp_path):
    f, client, clock, decisions = make(tmp_path)
    feed_frames(f, clock, 2, [cat("chuzh")])
    assert doors(client) == []  # still debouncing (3 s)
    feed_frames(f, clock, 2, [cat("chuzh")])
    assert doors(client) == [("open", "chuzh")]
    assert ("display", "chuzh", 5) in client.calls
    assert any(d.get("event") == "opened" for d in decisions)

    feed_frames(f, clock, 7, [])  # gone longer than door_close_timeout_sec
    assert doors(client)[-1] == ("close", "no_cat")
    closed = [d for d in decisions if d.get("event") == "closed"]
    assert closed and closed[0]["state"] == "closed" and closed[0]["door"] == "closed"
    row = sqlite3.connect(tmp_path / "j.db").execute("SELECT cat, close_reason FROM door_sessions").fetchone()
    assert row == ("chuzh", "cat_left")


def test_not_allowed_cat_stays_closed(tmp_path):
    f, client, clock, decisions = make(tmp_path)
    feed_frames(f, clock, 10, [cat("alisa")])
    assert doors(client) == []
    assert decisions[-1]["reason"] == "not_allowed:alisa"
    # Reports are sent on change only, not per frame…
    n = len(decisions)
    assert n < 5
    feed_frames(f, clock, 1, [cat("alisa")])
    assert len(decisions) == n
    # …unless asked for (e.g. after reconnecting to the hub).
    f.request_report()
    feed_frames(f, clock, 1, [cat("alisa")])
    assert len(decisions) == n + 1


def test_silence_closes_open_door(tmp_path):
    f, client, clock, _ = make(tmp_path, stream_blip_grace_sec=10)
    feed_frames(f, clock, 5, [cat("chuzh")])
    assert client.state == "open"
    clock.t += 5
    f.tick()
    assert client.state == "open"  # a blip: held
    clock.t += 6
    f.tick()
    assert doors(client)[-1] == ("close", "stream_lost")
    assert f.fsm.state == "closed"


def test_status_display(tmp_path):
    f, client, clock, decisions = make(tmp_path, status_display=True)
    feed_frames(f, clock, 5, [cat("chuzh", 0.93)])
    texts = [c[1] for c in client.calls if c[0] == "display"]
    assert texts[0].startswith("AC9") or texts[0].startswith("C")
    assert texts[-1] == "OC9"
    assert "chuzh" not in texts  # the status replaces the plain name
    assert len(texts) <= 6  # at most ~1/s over 5 s
    assert decisions[-1]["display"] == "OC9"


def test_status_text():
    s = ZoneSummary(n_cats=1, identity="alisa", present=True, meal_sec=0, identity_score=0.84)
    assert status_text("closed", s, "close", "not_allowed:alisa") == "CA8N"
    assert status_text("open", s, "open", "alisa") == "OA8"
    assert status_text("closed", ZoneSummary(0, None, False, 0), "close", "no_cat") == "C-"
    assert status_text("closed", ZoneSummary(2, "alisa", True, 0, 0.9), "close", "multi_cat") == "C29M"


def test_display_throttle():
    sent = []
    th = DisplayThrottle(lambda t: sent.append(t) or True, min_gap_sec=1, refresh_sec=10)
    th.set("A"); th.flush(0.0)
    th.set("B"); th.flush(0.5)   # too soon
    th.flush(1.0)
    th.flush(2.0)                # unchanged: nothing
    th.flush(11.0)               # refresh
    assert sent == ["A", "B", "B"]


CONFIG = """
# comments may mention ${UNSET_IN_A_COMMENT}
data_dir: /data
cameras: [{id: grey, rtsp: "rtsp://x"}, {id: beige, rtsp: "rtsp://y"}]
streamhub: {hub_listen: ":9001"}
decider:
  feeders:
    - id: feeder1
      camera: grey
      api_base_url: https://feeder.example
      serial_number: "${SERIAL}"
      allowed_cats: [chuzh, ellie]
      unknown_conf: 0.8
      feed: {mode: scheduled, times: ["07:00"], tz: America/Chicago}
"""


def test_dry_run_client_never_calls_api():
    from decider.feeder_client import DryRunClient
    c = DryRunClient("f")
    assert c.set_door("open", "x") and c.state == "open"
    assert c.feed(2) and c.set_display_text("t")


def test_config(monkeypatch):
    monkeypatch.setenv("SERIAL", "AF0")
    c = parse(CONFIG)
    assert not c.dry_run
    dry = parse(CONFIG.replace("decider:\n", "decider:\n  dry_run: true\n"))
    assert dry.dry_run and dry.journal_db == "/data/feed_journal/journal.dry-run.db"
    monkeypatch.setenv("DECIDER_DRY_RUN", "1")
    assert parse(CONFIG).dry_run
    monkeypatch.setenv("DECIDER_DRY_RUN", "0")
    assert not parse(CONFIG).dry_run
    f = c.feeders[0]
    assert (c.hub, c.journal_db) == ("127.0.0.1:9001", "/data/feed_journal/journal.db")
    assert f.serial_number == "AF0" and f.unknown_conf == 0.8 and f.door_close_timeout_sec == 30
    assert f.feed == FeedConfig(mode="scheduled", times=["07:00"], tz="America/Chicago")


@pytest.mark.parametrize("bad, err", [
    ("allowed_cats: [chuzh, ellie]", "unset"),                        # SERIAL not set
    ("camera: grey", "unknown camera"),
    ("unknown_conf: 0.8", "unknown keys"),
])
def test_config_errors(monkeypatch, bad, err):
    if err != "unset":
        monkeypatch.setenv("SERIAL", "AF0")
    text = CONFIG
    if err == "unknown camera":
        text = text.replace("camera: grey", "camera: pink")
    if err == "unknown keys":
        text = text.replace("unknown_conf: 0.8", "unknown_conf: 0.8\n      bogus: 1")
    with pytest.raises(ValueError, match=err):
        parse(text)


def test_example_config_is_valid(monkeypatch):
    from pathlib import Path
    for var in ("CAM_GREY_PASSWORD", "CAM_BEIGE_PASSWORD", "CAM_BLACK_PASSWORD"):
        monkeypatch.setenv(var, "x")
    c = parse((Path(__file__).parents[2] / "config.example.yaml").read_text())
    assert c.dry_run and [f.id for f in c.feeders] == ["feeder1", "feeder2", "feeder3"]
    assert c.feeders[2].allowed_cats == ["felisis", "unknown"] and c.feeders[1].feed.mode == "scheduled"


def test_confident_other_cat_does_not_open_unknown_feeder(tmp_path):
    """feeder3-like: allowed felisis + unknown. alisa at 87% (above unknown_conf
    0.8, below classifier_min_conf 0.9) must keep the door shut."""
    clock = Clock()
    cfg = FeederConfig(id="feeder3", camera="grey", api_base_url="http://x", serial_number="S",
                       allowed_cats=["felisis", "unknown"], unknown_conf=0.8)
    client, decisions = FakeClient(), []
    f = Feeder(cfg, client, FeedJournal(tmp_path / "j.db"), decisions.append, monotonic=clock, wall=clock)
    f.start()
    feed_frames(f, clock, 6, [cat("alisa", 0.87)])
    assert doors(client) == []
    assert decisions[-1]["reason"] == "not_allowed:alisa"
    # A genuinely unsure cat is still admitted as unknown.
    feed_frames(f, clock, 40, [])
    feed_frames(f, clock, 6, [{"box": [0.4, 0.4, 0.2, 0.2], "score": 0.9,
                               "cats": {"felisis": 0.5, "alisa": 0.3, "chuzh": 0.1, "ellie": 0.1}}])
    assert doors(client) == [("open", "unknown")]
