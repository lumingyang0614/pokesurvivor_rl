import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from pokesurvivor_rl.env import PokeSurvivorEnv
def mk(): return lambda: PokeSurvivorEnv(headless=True)
if __name__ == "__main__":
    n = int(sys.argv[1])
    v = SubprocVecEnv([mk() for _ in range(n)])
    v.reset()
    for _ in range(50): v.step(np.random.randint(9, size=n))
    t = time.perf_counter(); k = 300
    for _ in range(k): v.step(np.random.randint(9, size=n))
    print(n, "envs:", round(n * k / (time.perf_counter() - t)), "steps/s")
    v.close()
