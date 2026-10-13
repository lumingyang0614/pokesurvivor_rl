"""Run one episode with random actions and report speed / stats."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pokesurvivor_rl.env import PokeSurvivorEnv

env = PokeSurvivorEnv(headless="--show" not in sys.argv)
obs, _ = env.reset()
print("obs shape", obs.shape, "nonzero", int((obs != 0).sum()))
t0, n, R = time.time(), 0, 0.0
done = False
while not done and n < 3000:
    obs, r, term, trunc, info = env.step(np.random.randint(9))
    R += r; n += 1; done = term or trunc
    if n % 500 == 0:
        print(n, info["stats"], f"{n/(time.time()-t0):.1f} steps/s")
print("steps", n, "return", round(R, 2), info["stats"], f"{n/(time.time()-t0):.1f} steps/s")
print("picks", env.page.evaluate("() => RL.picks"))
env.page.screenshot(path=sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "smoke.png")
env.close()
