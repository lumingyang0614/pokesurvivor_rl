import asyncio, json, sys
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page(viewport={"width": 960, "height": 540})
    pg.on("console", lambda m: print("CONSOLE", m.type, m.text[:200]))
    pg.goto("https://pokesurvivor.com/en/", wait_until="load")
    pg.wait_for_function("window.G && G.DB && G.Game && G.Game.state", timeout=60000)
    pg.wait_for_timeout(3000)
    print(pg.evaluate("""() => ({state: G.Game.state, mode: G.DB.mode, save: !!G.DB.save,
        starters: (G.STARTERS||[]).slice(0,30).map(s=>s.dex||s), partner: G.DB.save && G.DB.save.partner})"""))
    print(pg.evaluate("""() => { G.DB.playAsGuest(); return {save: !!G.DB.save, partner: G.DB.save && G.DB.save.partner, keys: G.DB.save && Object.keys(G.DB.save)} }"""))
    print(pg.evaluate("() => JSON.stringify({owned: G.DB.save.owned, starter: G.DB.save.starter})"))
    print(pg.evaluate("""() => { const s=G.DB.save; G.DB.grant && G.DB.grant(4,'starter'); s.partner=4; s.starter=4;
        document.getElementById('tap-start') && document.getElementById('tap-start').classList.add('hidden');
        G.Ranking.submitSolo=()=>{}; G.Ranking.runStarted=()=>{}; G.Ranking.submitGroup=()=>{};
        G.Flow.play(); return G.Game.state }"""))
    pg.wait_for_timeout(4000)
    print(pg.evaluate("""() => { const h=G.Game.player; const E=G.EnemyMgr.all();
        return {state:G.Game.state, hp:h.hp, maxHp:h.maxHp, x:h.x, y:h.y, spd:h.spd, lvl:h.level, moves:h.moves.map(m=>m.id), nE:E.length,
          e0: E[0] && Object.keys(E[0]).filter(k=>typeof E[0][k]!=='object').map(k=>k+'='+E[0][k]).join(' '),
          proj: G.Projectiles.all ? G.Projectiles.all().length : Object.keys(G.Projectiles),
          haz: Object.keys(G.Hazards), pick: G.Pickups.all().length, cam:[G.Camera.w,G.Camera.h,G.Camera.scale]} }"""))
    pg.screenshot(path=sys.argv[1] + "/s1.png")
    b.close()
