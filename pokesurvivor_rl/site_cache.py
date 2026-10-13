"""Local caching mirror of pokesurvivor.com for training browsers.

Every file is downloaded from the real site at most once and then served from
`site_cache/` on disk, so 12 parallel browsers (and every restart) cost the site
nothing after the first load. Requests are throttled and back off on HTTP 429.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ORIGIN = "https://pokesurvivor.com"
CACHE_DIR = Path(__file__).resolve().parents[1] / "site_cache"
UA = "Mozilla/5.0 (pokesurvivor-rl local cache; one download per file)"
NEVER_CACHE = ("/version.json",)          # the game's self-update check
BLOCK = ("/sw.js",)                       # no service worker in the mirror

_fetch_lock = threading.Lock()            # one upstream request at a time per process
_last_fetch = [0.0]
MIN_GAP = 0.25                            # seconds between upstream requests
COOLDOWN = 600                            # after a 429, leave the site alone this long
_blocked_until = [0.0]
# PS_MIRROR_OFFLINE=1: never contact the site, missing files are simply 404
OFFLINE = os.environ.get("PS_MIRROR_OFFLINE") == "1"


def _paths(key: str):
    h = hashlib.sha1(key.encode()).hexdigest()
    return CACHE_DIR / h[:2] / h, CACHE_DIR / h[:2] / (h + ".json")


def _fetch_upstream(path_q: str):
    """Download ORIGIN+path_q politely. Returns (status, headers, body)."""
    if OFFLINE or time.time() < _blocked_until[0]:
        return 404, {"Content-Type": "text/plain"}, b""
    with _fetch_lock:
        delay = 2.0
        for _ in range(3):
            if time.time() < _blocked_until[0]:
                break
            wait = MIN_GAP - (time.time() - _last_fetch[0])
            if wait > 0:
                time.sleep(wait)
            _last_fetch[0] = time.time()
            req = urllib.request.Request(ORIGIN + path_q, headers={"User-Agent": UA})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    return r.status, {"Content-Type": r.headers.get("Content-Type", "application/octet-stream")}, r.read()
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    _blocked_until[0] = time.time() + COOLDOWN
                    break
                if e.code >= 500:
                    time.sleep(delay)
                    delay *= 2
                    continue
                return e.code, {"Content-Type": "text/plain"}, b""
            except Exception:
                time.sleep(delay)
                delay *= 2
        return 503, {"Content-Type": "text/plain"}, b"upstream unavailable"


def get(path_q: str):
    if path_q.startswith(BLOCK):
        return 404, {"Content-Type": "text/plain"}, b""
    if path_q.startswith(NEVER_CACHE):
        return 404, {"Content-Type": "application/json"}, b"{}"
    body_p, meta_p = _paths(path_q)
    if body_p.exists() and meta_p.exists():
        return 200, json.loads(meta_p.read_text()), body_p.read_bytes()
    status, headers, body = _fetch_upstream(path_q)
    if status == 200:
        store(path_q, headers, body)
    return status, headers, body


def store(path_q: str, headers: dict, body: bytes):
    body_p, meta_p = _paths(path_q)
    body_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = body_p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(body)
    os.replace(tmp, body_p)  # atomic: other processes never see half a file
    meta_p.write_text(json.dumps(headers))


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        status, headers, body = get(self.path)
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


_server = None


def start() -> str:
    """Start (once per process) the mirror on a free localhost port; returns the game URL."""
    global _server
    if _server is None:
        _server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        _server.daemon_threads = True
        threading.Thread(target=_server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{_server.server_address[1]}/en/"
