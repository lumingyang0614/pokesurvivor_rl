"""Run episodes with random (masked) actions and report speed / stats.

    python scripts/smoke_test.py [--show] [--late]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from pokesurvivor_rl.env import OBS_DIM, PokeSurvivorEnv

env = PokeSurvivorEnv(headless="--show" not in sys.argv, curriculum=1.0 if "--late" in sys.argv else 0.0,
                      save_path="progress/smoke_test.json")
rng = np.random.default_rng(0)
for ep in range(2):
    obs, info = env.reset()
    assert obs.shape == (OBS_DIM,), obs.shape
    print("reset:", info)
    t0, n, R, choices, done = time.time(), 0, 0.0, 0, False
    while not done and n < 4000:
        m = env.action_masks()
        a = rng.choice(np.flatnonzero(m))
        choices += a >= 9
        obs, r, term, trunc, info = env.step(a)
        R += r
        n += 1
        done = term or trunc
    print(f"steps {n} (menu/switch actions {choices}) return {R:.1f} {n / (time.time() - t0):.0f} steps/s")
    print("   ", info["stats"])
    print("    picks:", env._eval("RL.picks"))
env.close()
