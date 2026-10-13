"""Watch a trained agent play PokeSurvivor in a visible browser window.

    python play.py                                   # latest checkpoint (or models/best.zip)
    python play.py --model models/best.zip           # a specific model
    python play.py --account                         # use the account logged in via login.py
                                                     # (results also go to the leaderboard)
    python play.py --record videos                   # also save a .webm video
    python play.py --random                          # baseline: random (valid) actions
"""
import argparse
from pathlib import Path

import numpy as np

from pokesurvivor_rl.env import PokeSurvivorEnv


def latest_checkpoint():
    files = sorted(Path("checkpoints").glob("maskppo*/*.zip"), key=lambda f: f.stat().st_mtime)
    if files:
        return files[-1]
    best = Path("models/best.zip")  # shipped with the repo
    return best if best.exists() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--partner", type=int, default=4)
    p.add_argument("--speed", type=float, default=1.0, help="1.0 = real time, 0 = as fast as possible")
    p.add_argument("--record", type=str, default=None, help="directory to save videos into")
    p.add_argument("--random", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--save", type=str, default="progress/play.json",
                   help="guest save file (coins, tickets, Pokemon, upgrades) kept between sessions")
    p.add_argument("--account", action="store_true",
                   help="play with the account logged in via login.py (progress saved to the account)")
    p.add_argument("--no-ranking", action="store_true",
                   help="with --account: do not submit results to the public leaderboard")
    p.add_argument("--no-meta", action="store_true",
                   help="do not claim missions / pull gacha / switch partner / buy upgrades")
    p.add_argument("--auto-partner", action="store_true",
                   help="switch to the strongest owned partner between runs (default: keep the partner)")
    p.add_argument("--invest", type=str, default=None,
                   help="'auto' or a Pokedex number: max the current partner first, then put coins into "
                        "this Pokemon and switch to it once its upgrades are maxed")
    args = p.parse_args()

    model = None
    if not args.random:
        from sb3_contrib import MaskablePPO
        path = args.model or latest_checkpoint()
        if path is None:
            raise SystemExit("No checkpoint found - train first, or pass --random")
        print("loading", path)
        model = MaskablePPO.load(path, device="cpu")

    env = PokeSurvivorEnv(headless=args.headless, partner=args.partner, record_dir=args.record,
                          realtime=args.speed > 0, save_path=args.save, meta=not args.no_meta,
                          choose_partner=args.auto_partner,
                          profile_dir="browser_profile" if args.account else None,
                          ranking=args.account and not args.no_ranking,
                          invest=args.invest)
    rng = np.random.default_rng()
    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            meta = info["meta"]
            if ep == 0 and env.account_user:
                print("account:", env.account_user, "| leaderboard:", "on" if env.ranking else "off")
            if meta.get("claimed"):
                print("   missions claimed:", ", ".join(meta["claimed"]))
            if meta.get("pulls"):
                print("   gacha:", ", ".join(meta["pulls"]))
            if meta.get("investTarget"):
                print("   investing in:", meta["investTarget"], "(switched!)" if meta.get("switched") else "")
            if meta.get("bought"):
                print("   bought upgrades:", ", ".join(meta["bought"]))
            print(f"   partner: {meta.get('partner')}  coins: {meta.get('coins')}  "
                  f"tickets: {meta.get('tickets')}  upgrades: {meta.get('upgrades') or '-'}")
            done, ret = False, 0.0
            while not done:
                mask = env.action_masks()
                if model is None:
                    a = rng.choice(np.flatnonzero(mask))
                else:
                    a, _ = model.predict(obs, action_masks=mask, deterministic=True)
                obs, r, term, trunc, info = env.step(int(a))
                ret += r
                done = term or trunc
            st = info["stats"]
            m, s = divmod(int(st["time"]), 60)
            print(f"episode {ep + 1}: survived {m}:{s:02d}, level {st['level']}, kills {st['kills']}, "
                  f"bosses {st['bosses']}, rifts won {st['riftsWon']}/{st['riftsEntered']}, return {ret:.1f}")
            print("   moves:", ", ".join(st["moves"]))
    finally:
        env.close()
        if args.record:
            print("videos saved in", args.record)


if __name__ == "__main__":
    main()
