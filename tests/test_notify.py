"""Telegram notifications are opt-in and never break a push."""
import json

from training import notify


def test_not_configured_sends_nothing(monkeypatch, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert notify.telegram("hi") is False
    assert "not configured" in capsys.readouterr().err


def test_sends_text_to_the_chat(monkeypatch):
    sent = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"ok": true}'

    def urlopen(request, timeout):
        sent["url"], sent["body"] = request.full_url, json.loads(request.data)
        return _Resp()

    monkeypatch.setattr(notify.urllib.request, "urlopen", urlopen)
    assert notify.telegram("12 frames", token="T0K", chat_id="42") is True
    assert sent["url"] == "https://api.telegram.org/botT0K/sendMessage"
    assert sent["body"]["chat_id"] == "42" and sent["body"]["text"] == "12 frames"


def test_network_failure_is_reported_not_raised(monkeypatch):
    def urlopen(request, timeout):
        raise OSError("offline")
    monkeypatch.setattr(notify.urllib.request, "urlopen", urlopen)
    assert notify.telegram("x", token="t", chat_id="1") is False
