"""Diagnostic: scripted bot that walks to the nearest XP gem."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pokesurvivor_rl.env import PokeSurvivorEnv
env = PokeSurvivorEnv(headless=True)
env.reset()
for ep in range(2):
    env.reset(); done = False; n = 0
    while not done:
        d = env._eval("""(() => { const h=G.Game.player; let b=null,bd=1e9;
          for (const p of G.Pickups.all()) { const d=Math.hypot(p.x-h.x,p.y-h.y); if(d<bd){bd=d;b=p;} }
          return b ? [b.x-h.x, b.y-h.y] : null })()""")
        a = 0 if d is None else 1 + int(round(math.atan2(d[1], d[0]) / (2*math.pi/8))) % 8
        _, r, t, tr, info = env.step(a); done = t or tr; n += 1
        if n % 300 == 0: print(" ", {k: info["stats"][k] for k in ("time","level","kills","hp")})
    print("episode", info["stats"])
print(env._eval("G.Pickups.all().slice(0,5).map(p=>[p.kind,p.value])"))
env.close()
