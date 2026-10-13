"""Watch a trained agent play PokeSurvivor in a visible browser window.

    python play.py                                   # latest checkpoint (or models/best.zip)
    python play.py --model models/best.zip           # the best evaluated model
    python play.py --model checkpoints/ppo_pokesurvivor/final.zip
    python play.py --record videos                   # also save a .webm video
    python play.py --random                          # baseline: random moves
"""
import argparse
from collections import deque
from pathlib import Path

import numpy as np

from pokesurvivor_rl.env import PokeSurvivorEnv


def latest_checkpoint():
    files = sorted(Path("checkpoints").glob("**/*.zip"), key=lambda f: f.stat().st_mtime)
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
                   help="file that keeps coins + permanent upgrades between sessions")
    p.add_argument("--account", action="store_true",
                   help="play with the account logged in via login.py (progress saved to the account)")
    p.add_argument("--no-upgrade", action="store_true", help="do not spend coins on upgrades")
    args = p.parse_args()

    model = None
    if not args.random:
        from stable_baselines3 import PPO
        path = args.model or latest_checkpoint()
        if path is None:
            raise SystemExit("No checkpoint found - train first, or pass --random")
        print("loading", path)
        model = PPO.load(path, device="cpu")

    env = PokeSurvivorEnv(headless=args.headless, partner=args.partner, record_dir=args.record,
                          realtime=args.speed > 0, save_path=args.save,
                          auto_upgrade=not args.no_upgrade,
                          profile_dir="browser_profile" if args.account else None)
    if args.speed > 0:
        env.frame_skip = 3
    n_stack = 4
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            if ep == 0 and env.account_user:
                print("account:", env.account_user, "| partner dex:", env.partner)
            if env.last_bought:
                print("   bought upgrades:", ", ".join(env.last_bought))
            print(f"   coins: {env.progress.get('coins', 0)}  upgrades: {env.upgrade_levels() or '-'}")
            # same as VecFrameStack: zeros, then the first observation last
            frames = deque([np.zeros_like(obs)] * (n_stack - 1) + [obs], maxlen=n_stack)
            done, ret = False, 0.0
            while not done:
                if model is None:
                    a = env.action_space.sample()
                else:
                    a, _ = model.predict(np.concatenate(frames), deterministic=True)
                obs, r, term, trunc, info = env.step(int(a))
                frames.append(obs)
                ret += r
                done = term or trunc
            st = info["stats"]
            m, s = divmod(int(st["time"]), 60)
            print(f"episode {ep + 1}: survived {m}:{s:02d}, level {st['level']}, kills {st['kills']}, "
                  f"bosses {st['bosses']}, return {ret:.1f}")
            print("   moves:", ", ".join(st["moves"]))
    finally:
        env.close()
        if args.record:
            print("videos saved in", args.record)


if __name__ == "__main__":
    main()
