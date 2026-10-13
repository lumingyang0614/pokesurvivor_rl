"""Gymnasium environment that plays https://pokesurvivor.com through Playwright.

The agent controls movement (8 directions + stand still). Attacks are automatic in
the game; level-up cards, move bonds and items are chosen by a simple heuristic in
bridge.js so the policy can focus on dodging, kiting and collecting XP.
"""
from __future__ import annotations

import json
import socket
import time
import urllib.request
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import websocket
from playwright.sync_api import sync_playwright

GAME_URL = "https://pokesurvivor.com/en/"
BRIDGE_JS = (Path(__file__).parent / "bridge.js").read_text(encoding="utf-8")

N_SECT, N_CH, N_FEATS, N_NEAR = 16, 8, 14, 8 * 4
OBS_DIM = N_SECT * N_CH + N_FEATS + N_NEAR

# Flags for the persistent (logged-in) profile: Google refuses sign-in from browsers
# that announce they are automated.
PROFILE_ARGS = ["--disable-blink-features=AutomationControlled"]
PROFILE_IGNORE = ["--enable-automation"]

STARTERS = {"bulbasaur": 1, "charmander": 4, "squirtle": 7, "pikachu": 25}


class PokeSurvivorEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        headless: bool = True,
        frame_skip: int = 3,
        partner: int = 4,
        max_seconds: float = 20 * 60,
        realtime: bool = False,
        render: bool | None = None,
        auto_upgrade: bool = True,
        save_path: str | None = None,
        profile_dir: str | None = None,
        record_dir: str | None = None,
        viewport=(960, 540),
    ):
        super().__init__()
        self.headless = headless
        self.frame_skip = frame_skip
        self.partner = partner
        self.max_seconds = max_seconds
        self.realtime = realtime
        self.record_dir = record_dir
        self.viewport = viewport
        self.auto_upgrade = auto_upgrade
        # JSON file that keeps coins + permanent upgrades across browser restarts
        self.save_path = Path(save_path) if save_path else None
        # persistent browser profile with a logged-in game account (see login.py);
        # progress then lives in the account's cloud save instead of save_path
        self.profile_dir = profile_dir
        if profile_dir:
            self.save_path = None
        self.account_user = None
        self.last_bought: list = []
        self.progress: dict = {}
        # skip canvas drawing when nobody watches (headless training without video)
        self.render_frames = (not headless or bool(record_dir)) if render is None else render
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_DIM,), np.float32)
        self._pw = self._browser = self._ctx = self.page = None
        self._ws = None
        self._msg_id = 0
        self.last_stats: dict = {}

    # ------------------------------------------------------------------ browser
    def _launch(self):
        self._pw = sync_playwright().start()
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0))
            self._port = sk.getsockname()[1]
        args = ["--mute-audio", "--autoplay-policy=no-user-gesture-required",
                f"--remote-debugging-port={self._port}"]
        kw = dict(viewport={"width": self.viewport[0], "height": self.viewport[1]})
        if self.record_dir:
            kw["record_video_dir"] = self.record_dir
            kw["record_video_size"] = kw["viewport"]
        if self.profile_dir:
            self._ctx = self._pw.chromium.launch_persistent_context(
                self.profile_dir, headless=self.headless, args=args + PROFILE_ARGS,
                ignore_default_args=PROFILE_IGNORE, **kw)
        else:
            self._browser = self._pw.chromium.launch(headless=self.headless, args=args)
            self._ctx = self._browser.new_context(**kw)
        if not self.render_frames:
            self._ctx.add_init_script("window.__RL_NORENDER = true;")
        self._ctx.add_init_script(BRIDGE_JS)
        self.page = self._ctx.new_page()
        self.page.goto(GAME_URL, wait_until="load", timeout=120_000)
        # boot needs animation frames for fonts/UI; pump until the game objects exist
        for _ in range(600):
            if self.page.evaluate("() => window.RL && RL.ready() && !!G.DB && G.DB.ready && true"):
                break
            time.sleep(0.1)
        # wait for the DB to finish its async ready() (tap-to-start screen shows up)
        for _ in range(600):
            ok = self.page.evaluate(
                "() => { RL.pump(1); const b=document.getElementById('boot'); return !!b && b.classList.contains('hidden'); }"
            )
            if ok:
                break
            time.sleep(0.05)
        if self.profile_dir:
            for _ in range(200):  # cloud save loads asynchronously
                self.account_user = self.page.evaluate(
                    "() => G.DB.loggedIn && !G.DB.guest && G.DB.save ? (G.DB.user || 'account') : null")
                if self.account_user:
                    break
                time.sleep(0.1)
            if not self.account_user:
                self.close()
                raise RuntimeError("Not logged in - run `python login.py` first")
        self.partner = self.page.evaluate(f"() => RL.boot({int(self.partner)}, {'true' if self.profile_dir else 'false'})")
        if self.save_path and self.save_path.exists():
            self.page.evaluate("p => RL.setProgress(p)", json.loads(self.save_path.read_text()))
        # Hot loop goes over a raw CDP websocket: ~0.2 ms per call instead of ~1.7 ms
        # through Playwright's driver.
        targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self._port}/json/list"))
        url = next(t["webSocketDebuggerUrl"] for t in targets
                   if t["type"] == "page" and "pokesurvivor" in t["url"])
        self._ws = websocket.create_connection(url, suppress_origin=True, timeout=60)

    def _eval(self, expr: str):
        """Evaluate a JS expression returning a JSON string, via raw CDP."""
        self._msg_id += 1
        mid = self._msg_id
        self._ws.send(json.dumps({"id": mid, "method": "Runtime.evaluate",
                                  "params": {"expression": f"JSON.stringify({expr})", "returnByValue": True}}))
        while True:
            m = json.loads(self._ws.recv())
            if m.get("id") == mid:
                break
        res = m.get("result", {})
        if "exceptionDetails" in res:
            raise RuntimeError(res["exceptionDetails"].get("exception", {}).get("description", res))
        return json.loads(res["result"]["value"])

    # ------------------------------------------------------------------ gym api
    @staticmethod
    def _obs(o) -> np.ndarray:
        v = np.concatenate([o["grid"], o["feats"], o["near"]]).astype(np.float32)
        return np.clip(v, -1.0, 1.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.page is None:
            self._launch()
        try:
            self._eval("RL.ready()")
        except Exception:
            self.close()
            self._launch()
        r = self._eval(f"RL.reset({int(self.partner)}, {'true' if self.auto_upgrade else 'false'})")
        self.last_stats = r["stats"]
        self.last_bought = r["bought"]
        self.progress = r["progress"]
        if self.save_path:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_path.write_text(json.dumps(self.progress))
        self._t_last = time.perf_counter()
        return self._obs(r["obs"]), {}

    def step(self, action):
        try:
            r = self._eval(f"RL.step({int(action)}, {self.frame_skip})")
        except Exception as e:  # page reloaded / browser died: end episode, relaunch on reset
            print(f"[env] browser lost ({str(e).splitlines()[0][:120]}); relaunching on reset")
            st = dict(self.last_stats or {})
            self.close()
            return (np.zeros(self.observation_space.shape, np.float32), 0.0, True, False,
                    {"stats": st, "crashed": True})
        st = r["stats"]
        self.last_stats = st
        reward = (
            0.02 * self.frame_skip            # alive bonus (~0.6 / second)
            - 6.0 * r["hpLost"]               # losing a full health bar costs 6
            + 0.03 * r["dKills"]
            + 1.0 * r["dProgress"]            # XP collected, 1.0 per full level
            + 3.0 * r["dBoss"]
        )
        terminated = bool(r["done"])
        if terminated:
            reward -= 5.0
        truncated = (not terminated) and st["time"] >= self.max_seconds
        if self.realtime:
            target = self.frame_skip / 30.0
            dt = time.perf_counter() - self._t_last
            if dt < target:
                time.sleep(target - dt)
            self._t_last = time.perf_counter()
        info = {"stats": st}
        if terminated or truncated:
            info["episode_stats"] = st
        return self._obs(r["obs"]), float(reward), terminated, truncated, info

    def upgrade_levels(self) -> dict:
        return self.progress.get("pupg", {}).get(str(self.partner), {})

    def close(self):
        if self.page is not None and self.save_path:
            try:  # bank the coins of an unfinished run, then save
                self._eval(f"(() => {{ if (G.Game.player && G.Game.state !== 'ui' && G.Game.state !== 'over') G.Game.quit(); return RL.getProgress(); }})()")
                self.save_path.write_text(json.dumps(self._eval("RL.getProgress()")))
            except Exception:
                pass
        try:
            if self._ws:
                self._ws.close()
            if self._ctx:
                self._ctx.close()
            if self._browser:
                self._browser.close()
            if self._pw:
                self._pw.stop()
        finally:
            self._pw = self._browser = self._ctx = self.page = self._ws = None
