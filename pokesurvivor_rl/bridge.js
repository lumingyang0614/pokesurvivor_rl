// Injected into pokesurvivor.com before any game script runs.
// - Takes over requestAnimationFrame so Python can step the game deterministically
//   (fixed 1/30 s frames, as fast as the CPU allows).
// - Exposes window.RL with reset / step / observe helpers.
(() => {
  const FRAME_MS = 1000 / 30;
  const realRAF = window.requestAnimationFrame.bind(window);
  let queue = [];
  let fakeNow = 0;
  let manual = true;
  window.requestAnimationFrame = (cb) => {
    if (!manual) return realRAF(cb);
    queue.push(cb);
    return queue.length;
  };
  window.cancelAnimationFrame = () => {};

  // The game reloads itself between runs when a new version is deployed
  // (G.Updater polls version.json). Hide new versions so a run never vanishes;
  // a fresh browser launch still loads the latest game.
  const realFetch = window.fetch.bind(window);
  window.fetch = (input, init) => {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    if (url.includes("version.json")) return Promise.resolve(new Response("{}", { status: 404 }));
    return realFetch(input, init);
  };

  // Training mode: give the main game canvas a do-nothing 2D context so no CPU is
  // spent drawing frames nobody looks at. Game logic is unaffected.
  if (window.__RL_NORENDER) {
    const sink = new Proxy(function () {}, {
      get: (t, k) => (k === Symbol.toPrimitive ? () => 0 : k === "width" ? 0 : sink),
      set: () => true,
      apply: () => sink,
    });
    const realGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, ...rest) {
      if (this.id === "game" && type === "2d") return sink;
      return realGetContext.call(this, type, ...rest);
    };
  }

  function pump(n) {
    for (let i = 0; i < n; i++) {
      fakeNow += FRAME_MS;
      const q = queue;
      queue = [];
      for (const cb of q) {
        try { cb(fakeNow); } catch (e) { console.error(e); }
      }
    }
  }

  const N_SECT = 16;
  const SENSE = 360;        // world units the agent "sees"
  const WALL_RANGE = 160;
  const DIRS = [];
  for (let i = 0; i < N_SECT; i++) {
    const a = (i / N_SECT) * Math.PI * 2;
    DIRS.push([Math.cos(a), Math.sin(a)]);
  }
  // 8-way movement + idle
  const ACTIONS = [[0, 0]];
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2;
    ACTIONS.push([Math.cos(a), Math.sin(a)]);
  }

  let axis = [0, 0];
  let hooked = false;
  let prev = null;
  let lastSlotSwitch = 0;

  function sectorOf(dx, dy) {
    let a = Math.atan2(dy, dx);
    if (a < 0) a += Math.PI * 2;
    return Math.floor((a / (Math.PI * 2)) * N_SECT + 0.5) % N_SECT;
  }

  // ---- heuristics for the menus (the RL agent only controls movement) ----
  // Only the active move slot fires (several moves share a slot only after a rare
  // "bond"), so pour upgrades into the active move and global stats.
  const STAT_SCORE = { atk: 42, cd: 38, hp: 30, def: 30, mag: 28, rgn: 25, spd: 22, xp: 20, heal: 5 };
  function scoreOption(o, h) {
    const hpf = h.hp / h.maxHp;
    if (o.type === "evo") return 100;
    if (o.type === "new") return h.moves.length < 2 ? 32 : 12;
    if (o.type === "up") {
      const id = o.key.slice(3);
      return h.activeGroup().some((m) => m.id === id) ? 65 : 18;
    }
    const id = o.key.slice(5);
    if (id === "heal") return hpf < 0.5 ? 90 : 5;
    if (id === "hp" && hpf < 0.5) return 50;
    if (o.color === "#ffd23f") return 70; // stat that unlocks an evolution
    return STAT_SCORE[id] || 10;
  }

  function hookUI() {
    if (hooked) return;
    hooked = true;
    G.RunUI.showLevelUp = (lvl, options, cb) => {
      const h = G.Game.player;
      let best = 0, bs = -1;
      options.forEach((o, i) => {
        const s = scoreOption(o, h) + Math.random();
        if (s > bs) { bs = s; best = i; }
      });
      RL.picks.push(options[best].key);
      cb(options[best]);
    };
    G.RunUI.showBond = (opt, cb) => {
      // always fuse: bonded moves fire together
      if (opt.mode === "add") cb([opt.loose[0]]);
      else cb([opt.loose[0], opt.loose[1]]);
    };
    G.RunUI.showItemChoice = (cur, neu, cb) => cb(neu);
    // never touch the public leaderboard / profile from training runs
    G.Ranking.submitSolo = () => {};
    G.Ranking.submitGroup = () => {};
    G.Ranking.runStarted = () => {};
    if (G.Social) G.Social.publishProfile = () => {};
    G.Input.axis = () => axis;
    try { G.Audio.music = () => {}; G.Audio.sfx = () => {}; } catch (e) {}
  }

  // ---- permanent upgrades ("Mejoras" menu), bought with coins between runs ----
  // weight = how much we want it; the cheapest cost/weight option is bought first.
  const UPGRADE_WANT = { atk: 1.4, hp: 1.2, cd: 1.2, def: 1.0, magnet: 0.9, regen: 0.8,
                         crit: 0.9, spd: 0.6, xp: 0.7, coins: 0.5 };
  function autoUpgrade(dex) {
    const save = G.DB.save, ups = G.DB.upgradesOf(dex), bought = [];
    for (;;) {
      let best = null, bestScore = Infinity, bestCost = 0;
      for (const id in UPGRADE_WANT) {
        const def = G.Upgrades.BY[id], lvl = ups[id] || 0;
        if (!def || lvl >= def.max) continue;
        const c = G.Upgrades.cost(id, lvl), sc = c / UPGRADE_WANT[id];
        if (sc < bestScore) { bestScore = sc; best = id; bestCost = c; }
      }
      if (!best || save.coins < bestCost) break;
      save.coins -= bestCost;
      ups[best] = (ups[best] || 0) + 1;
      bought.push(best + ">" + ups[best]);
    }
    if (bought.length && G.DB.commit) G.DB.commit(true);
    return bought;
  }

  function pickSlot(h) {
    // fire the slot with the most total move levels
    let best = h.active, bs = -1;
    h.slots.forEach((g, i) => {
      let s = 0;
      for (const m of g) s += (m.kind === "buff" ? 0.5 : 1) * (m.lvl || 1) * (m.evolved ? 2 : 1);
      if (s > bs + 0.01) { bs = s; best = i; }
    });
    if (best !== h.active && G.Game.st.time - lastSlotSwitch > 3) {
      lastSlotSwitch = G.Game.st.time;
      h.setActive(best);
    }
  }

  function hazardAt(list, x, y, r) {
    for (const z of list) {
      if (z.kind === "zone") {
        if ((x - z.x) ** 2 + (y - z.y) ** 2 < (z.r + 0.6 * r) ** 2) return true;
      } else if (z.len) {
        const c = Math.cos(z.angle), s = Math.sin(z.angle);
        const ox = x - z.x, oy = y - z.y, t = ox * c + oy * s;
        if (t >= -r && t <= z.len + r && Math.abs(-ox * s + oy * c) <= z.w / 2 + 0.6 * r) return true;
      }
    }
    return false;
  }

  function observe() {
    const h = G.Game.player;
    const C = 8;
    const grid = new Float32Array(N_SECT * C);
    const near = new Float32Array(8 * 4);
    if (!h) return { grid: Array.from(grid), feats: new Array(14).fill(0), near: Array.from(near) };
    const px = h.x, py = h.y;
    const enemies = [];
    let bossAlive = 0;
    for (const e of G.EnemyMgr.all()) {
      if (e.dead) continue;
      const dx = e.x - px, dy = e.y - py, d = Math.hypot(dx, dy);
      if (e.boss) bossAlive = 1;
      if (d > SENSE) continue;
      const k = sectorOf(dx, dy) * C, p = 1 - d / SENSE;
      grid[k + 0] = Math.max(grid[k + 0], p);
      grid[k + 1] += 0.15 * p;
      if (e.boss) grid[k + 7] = Math.max(grid[k + 7], p);
      enemies.push([d, dx, dy, e]);
    }
    for (const pr of G.Projectiles.all()) {
      if (pr.friendly) continue;
      const dx = pr.x - px, dy = pr.y - py, d = Math.hypot(dx, dy);
      if (d > SENSE) continue;
      // weight by whether it is heading toward us
      const sp = Math.hypot(pr.vx, pr.vy) || 1;
      const toward = Math.max(0, -(dx * pr.vx + dy * pr.vy) / (d * sp + 1e-6));
      const k = sectorOf(dx, dy) * C;
      grid[k + 2] = Math.max(grid[k + 2], (1 - d / SENSE) * (0.4 + 0.6 * toward));
    }
    for (const it of G.Pickups.all()) {
      const dx = it.x - px, dy = it.y - py, d = Math.hypot(dx, dy);
      if (d > SENSE) continue;
      const k = sectorOf(dx, dy) * C, p = 1 - d / SENSE;
      if (it.kind === "xp" || it.kind === "xpBig" || it.kind === "coin") grid[k + 3] = Math.max(grid[k + 3], p);
      else grid[k + 4] = Math.max(grid[k + 4], p);
    }
    const haz = G.Hazards.all || [];
    for (let i = 0; i < N_SECT; i++) {
      const [cx, cy] = DIRS[i];
      let wall = 0;
      for (let r = 20; r <= WALL_RANGE; r += 20) {
        if (!G.World.isFree(px + cx * r, py + cy * r, h.r)) { wall = 1 - (r - 20) / WALL_RANGE; break; }
      }
      grid[i * C + 5] = wall;
      if (haz.length) {
        for (const r of [0, 40, 80, 130]) {
          if (hazardAt(haz, px + cx * r, py + cy * r, h.r)) { grid[i * C + 6] = Math.max(grid[i * C + 6], 1 - r / 160); break; }
        }
      }
    }
    for (let i = 0; i < grid.length; i++) if (grid[i] > 1) grid[i] = 1;
    enemies.sort((a, b) => a[0] - b[0]);
    for (let i = 0; i < Math.min(8, enemies.length); i++) {
      const [d, dx, dy, e] = enemies[i];
      near[i * 4 + 0] = dx / SENSE;
      near[i * 4 + 1] = dy / SENSE;
      near[i * 4 + 2] = Math.min(1, e.dmg / Math.max(1, h.hp));
      near[i * 4 + 3] = e.maxHp ? e.hp / e.maxHp : 0;
    }
    const feats = [
      h.hp / h.maxHp,
      Math.min(1, h.maxHp / 600),
      Math.min(1, h.level / 60),
      Math.min(1, h.spd / 250),
      h.xpNext ? h.xp / h.xpNext : 0,
      Math.min(1, G.Game.st.time / 1200),
      Math.min(1, G.EnemyMgr.all().length / 150),
      h.invuln > 0 ? 1 : 0,
      h.reduction || 0,
      h.inLiquid ? 1 : 0,
      h.slowT > 0 ? 1 : 0,
      bossAlive,
      G.Rift.inArena ? 1 : 0,
      Math.min(1, h.slots.length / 4),
    ];
    return { grid: Array.from(grid), feats, near: Array.from(near) };
  }

  function stats() {
    const h = G.Game.player;
    if (!h) return null;
    return {
      hp: h.hp, maxHp: h.maxHp, level: h.level, xp: h.xp, kills: h.kills,
      bosses: h.bosses, time: G.Game.st.time, dead: !!h.dead,
      progress: h.level - 1 + (h.xpNext ? Math.min(1, h.xp / h.xpNext) : 0),
      moves: h.moves.map((m) => m.id + ":" + m.lvl),
    };
  }

  window.RL = {
    picks: [],
    ready() {
      return !!(window.G && G.DB && G.Game && G.Flow && G.RunUI && G.Ranking);
    },
    setManual(m) {
      manual = m;
      if (!m) { const q = queue; queue = []; for (const cb of q) realRAF(cb); }
    },
    pump,
    boot(partner, account) {
      if (!G.DB.save) G.DB.playAsGuest();
      const s = G.DB.save;
      if (account) {
        // real account: never hand out Pokemon; fall back to the account's partner
        if (G.DB.owns(partner)) s.partner = partner;
        else if (!s.partner) { G.DB.grant(partner, "starter"); s.partner = partner; }
      } else {
        if (G.DB.grant && !G.DB.owns(partner)) G.DB.grant(partner, "starter");
        s.partner = partner;
      }
      s.starter = s.starter || s.partner;
      for (const id of ["tap-start", "boot"]) {
        const el = document.getElementById(id);
        if (el) el.classList.add("hidden");
      }
      hookUI();
      return s.partner;
    },
    reset(partner, upgrade) {
      hookUI();
      // finish the previous run properly so the game pays out its coins
      if (G.Game.player && G.Game.state !== "ui" && G.Game.state !== "over") G.Game.quit();
      if (partner) G.DB.save.partner = partner;
      G.UI.hideAll();
      if (G.Game.state !== "ui") G.Game.toUI();
      const bought = upgrade ? autoUpgrade(G.DB.save.partner) : [];
      G.Game.startRun();
      axis = [0, 0];
      lastSlotSwitch = -99;
      RL.picks = [];
      pump(2);
      prev = stats();
      return { obs: observe(), stats: prev, bought, progress: RL.getProgress() };
    },
    getProgress() {
      const s = G.DB.save;
      return { coins: s.coins, pupg: s.pupg || {} };
    },
    setProgress(p) {
      const s = G.DB.save;
      if (!p) return false;
      s.coins = p.coins || 0;
      s.pupg = Object.assign(s.pupg || {}, p.pupg || {});
      return true;
    },
    step(action, frames) {
      const a = ACTIONS[action] || ACTIONS[0];
      axis = [a[0], a[1]];
      const h0 = G.Game.player;
      let hpLost = 0, xpGain = 0;
      const before = stats();
      for (let i = 0; i < frames; i++) {
        const hp = G.Game.player ? G.Game.player.hp : 0;
        pump(1);
        const h = G.Game.player;
        if (!h || h !== h0) break;
        if (h.hp < hp) hpLost += hp - h.hp;
        if (G.Game.state === "over" || h.dead) break;
      }
      const h = G.Game.player;
      if (h && h === h0 && !h.dead) pickSlot(h);
      const after = stats() || before;
      const done = !h || h !== h0 || h.dead || G.Game.state === "over";
      const r = {
        obs: observe(),
        stats: after,
        hpLost: hpLost / (before ? before.maxHp : 100),
        dKills: after.kills - before.kills,
        dLevel: after.level - before.level,
        dProgress: after.progress - before.progress,
        dBoss: after.bosses - before.bosses,
        dt: after.time - before.time,
        done,
        state: G.Game.state,
      };
      prev = after;
      return r;
    },
    observe,
    stats,
  };
})();
