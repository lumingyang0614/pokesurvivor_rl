"""Compare checkpoints under the same conditions and copy the best to models/best.zip.

Each checkpoint plays N normal runs as a fresh guest Charmander with no meta
(no upgrades, gacha or partner switch), so results only reflect the policy.

    python evaluate.py --last 8 --episodes 10
"""
import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def run_checkpoint(path: str, episodes: int, partner: int) -> dict:
    from sb3_contrib import MaskablePPO
    from pokesurvivor_rl.env import PokeSurvivorEnv

    model = MaskablePPO.load(path, device="cpu")
    env = PokeSurvivorEnv(headless=True, partner=partner, meta=False)
    times, levels, kills = [], [], []
    try:
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
                obs, _, term, trunc, info = env.step(int(a))
                done = term or trunc
            st = info["stats"]
            times.append(st["time"]); levels.append(st["level"]); kills.append(st["kills"])
    finally:
        env.close()
    return {"model": path, "survive_mean": float(np.mean(times)), "survive_median": float(np.median(times)),
            "survive_max": float(np.max(times)), "level_mean": float(np.mean(levels)),
            "kills_mean": float(np.mean(kills))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="checkpoints/maskppo_v2")
    p.add_argument("--last", type=int, default=8, help="evaluate the N most recent checkpoints")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--partner", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    ckpts = sorted(Path(args.dir).glob("*.zip"), key=lambda f: f.stat().st_mtime)[-args.last:]
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(run_checkpoint, [str(c) for c in ckpts],
                              [args.episodes] * len(ckpts), [args.partner] * len(ckpts)))
    results.sort(key=lambda r: r["survive_mean"], reverse=True)
    for r in results:
        print(f"{Path(r['model']).name:32s} survive {r['survive_mean']:6.1f}s (median {r['survive_median']:.0f}, "
              f"max {r['survive_max']:.0f})  level {r['level_mean']:.1f}  kills {r['kills_mean']:.0f}")
    best = results[0]
    out = Path("models"); out.mkdir(exist_ok=True)
    shutil.copy(best["model"], out / "best.zip")
    (out / "best.json").write_text(json.dumps({**best, "episodes": args.episodes,
                                               "partner": args.partner}, indent=2))
    print("best ->", out / "best.zip")


if __name__ == "__main__":
    main()
