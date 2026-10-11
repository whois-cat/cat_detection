"""Optional Telegram notifications (stdlib only).

Set TELEGRAM_BOT_TOKEN (from @BotFather) and TELEGRAM_CHAT_ID (your chat with
the bot) in .env; without them nothing is sent.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request


def telegram(text: str, *, token: str | None = None, chat_id: str | None = None) -> bool:
    """Send ``text``; False (with a note on stderr) when not configured or failed."""
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("telegram: not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)", file=sys.stderr)
        return False
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=json.dumps({"chat_id": chat_id, "text": text,
                         "disable_web_page_preview": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return bool(json.load(response).get("ok"))
    except OSError as exc:
        print(f"telegram: send failed: {exc}", file=sys.stderr)
        return False
