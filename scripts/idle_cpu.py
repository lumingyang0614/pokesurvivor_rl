import sys, time, psutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pokesurvivor_rl.env import PokeSurvivorEnv
env = PokeSurvivorEnv(headless=True); env.reset()
me = psutil.Process()
def snap():
    d = {}
    for p in psutil.process_iter(["name", "exe"]):
        try:
            if (p.info["exe"] or "").find("ms-playwright") >= 0 or p.info["name"].startswith("node") or p.pid == me.pid:
                d[p.pid] = (p.info["name"], p.cpu_times().user + p.cpu_times().system)
        except Exception: pass
    return d
a = snap(); time.sleep(5); b = snap()
print("IDLE 5s cpu-seconds:", {b[k][0]: round(b[k][1]-a[k][1],2) for k in b if k in a and b[k][1]-a[k][1] > 0.05})
a = snap(); t = time.time(); n = 0
while time.time() - t < 5: env.step(1); n += 1
b = snap()
print("STEPPING", n, "steps; cpu-seconds:", {f"{b[k][0]}{k}": round(b[k][1]-a[k][1],2) for k in b if k in a and b[k][1]-a[k][1] > 0.05})
env.close()
