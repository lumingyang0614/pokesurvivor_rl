import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pokesurvivor_rl.env import PokeSurvivorEnv
env = PokeSurvivorEnv(headless=True)
env.reset()
for _ in range(100): env.step(0)
print(env.page.evaluate("""() => {
  const t0=performance.now(); for(let i=0;i<300;i++) RL.pump(1); const t1=performance.now();
  for(let i=0;i<300;i++) RL.observe(); const t2=performance.now();
  return {pumpMs:(t1-t0)/300, obsMs:(t2-t1)/300, enemies:G.EnemyMgr.all().length, proj:G.Projectiles.all().length, pick:G.Pickups.all().length}
}"""))
t=time.perf_counter()
for _ in range(200): env.page.evaluate("() => 1")
print("roundtrip ms", (time.perf_counter()-t)/200*1000)
env.close()
