"""Sign in to PokeSurvivor once, by hand, in a dedicated browser profile.

    python login.py

A browser window opens on the game. Sign in yourself (Google, or the game's own
name/password account). The window closes by itself once the game reports you are
logged in. Afterwards `python play.py --account` reuses that login and your progress
(coins, upgrades, stats) is saved to your account. Your password never touches this
project: it is typed only into the browser window.
"""
import argparse
import time

from playwright.sync_api import sync_playwright

from pokesurvivor_rl.env import GAME_URL, PROFILE_ARGS, PROFILE_IGNORE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="browser_profile")
    p.add_argument("--timeout", type=int, default=600, help="seconds to wait for the login")
    args = p.parse_args()

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            args.profile, headless=False, args=PROFILE_ARGS, ignore_default_args=PROFILE_IGNORE,
            viewport={"width": 960, "height": 540})
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(GAME_URL, wait_until="load")
        print("Sign in inside the browser window (Google or a game account)...")
        deadline = time.time() + args.timeout
        user = None
        while time.time() < deadline:
            try:
                user = page.evaluate("() => window.G && G.DB && G.DB.loggedIn && !G.DB.guest ? (G.DB.user || 'ok') : null")
            except Exception:
                user = None  # page navigating / popup flow
            if user:
                break
            time.sleep(1)
        if user:
            time.sleep(3)  # let the cloud save finish loading
            print(f"Logged in as: {user}")
        else:
            print("No login detected (timed out)")
        ctx.close()


if __name__ == "__main__":
    main()
