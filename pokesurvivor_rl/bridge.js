// Injected into pokesurvivor.com before any game script runs.
// - Takes over requestAnimationFrame so Python can step the game deterministically
//   (fixed 1/30 s frames, as fast as the CPU allows).
// - Exposes window.RL: reset / step / observe / action mask, plus the between-run
//   "meta" (missions, gacha, partner choice, permanent upgrades).
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

  // ------------------------------------------------------------------ constants
  const N_SECT = 16;
  const SENSE = 360;          // world units the agent "sees"
  const TERRAIN_RANGE = 400;  // how far walls / liquid are probed (about the whole screen)
  const C = 9;                // grid channels per sector
  // 0 enemy proximity, 1 enemy density, 2 hostile projectiles, 3 xp/coins,
  // 4 special pickups, 5 walls & solid props, 6 liquid (water slows, lava hurts),
  // 7 telegraphed hazards, 8 boss
  const DIRS = [];
  for (let i = 0; i < N_SECT; i++) {
    const a = (i / N_SECT) * Math.PI * 2;
    DIRS.push([Math.cos(a), Math.sin(a)]);
  }
  // actions: 0 idle, 1-8 move, 9-11 pick an option of the open menu (level-up card,
  // move bond, held-item swap), 12-15 switch active move slot
  const N_ACTIONS = 16;
  const MOVES = [[0, 0]];
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2;
    MOVES.push([Math.cos(a), Math.sin(a)]);
  }
  const KINDS = ["melee", "dash", "buff", "projectile", "cone", "orbit", "nova", "beam",
                 "boomerang", "aura", "chain", "trail", "mine", "meteor", "turret"];
  const STATS = ["hp", "atk", "spd", "cd", "def", "rgn", "mag", "xp", "heal"];
  const CARD_TYPES = ["new", "up", "evo", "stat"];
  const CARD_DIM = CARD_TYPES.length + STATS.length + 1 + KINDS.length + 5;
  const SLOT_DIM = 4 + KINDS.length + 2;
  const FEAT_DIM = 36;
  const ITEMS = ["leftovers", "choiceband", "quickclaw", "focussash", "shellbell", "amuletcoin",
                 "luckyegg", "lifeorb", "choicescarf", "rockyhelmet", "assaultvest"];
  const CHOICE_KINDS = ["card", "bond", "item"];
  const BOND_DIM = KINDS.length + 4;
  const OPT_DIM = BOND_DIM + ITEMS.length;
  const CHOICE_DIM = CHOICE_KINDS.length + ITEMS.length + 3 * OPT_DIM;
  // map objects per sector: 0 big chest (stand next to it 3.5 s: coins, gacha tickets),
  // 1 small chest (break it), 2 grass (breaking it sometimes drops healing),
  // 3 spring / altar (heal / power-up), 4 trap; then nearest big chest, small chest
  // and spring/altar further away
  const MAP_CH = 5;
  const MAP_DIM = N_SECT * MAP_CH + 10;
  const CHEST_FAR = 2400;    // big chests are searched this far (every 15 steps: ~1 ms)
  const FAR = 900;            // range for rift / boss direction features

  let axis = [0, 0];
  let hooked = false;
  // menus waiting for the agent, oldest first: {kind: card|bond|item, options, cb}
  let pendingQ = [];
  let learnCards = true;     // false: heuristic picks cards / bonds / items (old models)
  let ranking = false;       // submit results to the public leaderboard (account play only)
  let realRanking = null;
  let lastPos = null;
  let riftsEntered = 0, riftsWon = 0, riftsLost = 0, wasInArena = false;
  let chests = 0, bigChests = 0;
  let farChest = null, farAge = 0, farOpened = -1;   // cached far big-chest search

  function sectorOf(dx, dy) {
    let a = Math.atan2(dy, dx);
    if (a < 0) a += Math.PI * 2;
    return Math.floor((a / (Math.PI * 2)) * N_SECT + 0.5) % N_SECT;
  }
  function oneHot(list, v) { return list.map((x) => (x === v ? 1 : 0)); }

  // ------------------------------------------------- heuristic fallbacks / menus
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
    if (o.color === "#ffd23f") return 70;
    return STAT_SCORE[id] || 10;
  }
  function heuristicPick(options, h) {
    let best = 0, bs = -1;
    options.forEach((o, i) => {
      const s = scoreOption(o, h) + Math.random();
      if (s > bs) { bs = s; best = i; }
    });
    return best;
  }

  // up to 3 bond choices: which loose move joins the existing group ("add"), or which
  // two loose moves form a new group ("pair"); moves of the active slot come first
  function bondCandidates(opt, h) {
    const act = h.activeGroup();
    const dps = (m) => ((m.dmg || 0) * (m.count || 1)) / Math.max(0.15, m.cd || 1);
    const score = (ms) => ms.reduce((a, m) => a + dps(m) + (act.includes(m) ? 1e4 : 0), 0);
    let c;
    if (opt.mode === "add") {
      c = opt.loose.map((m) => ({ moves: [m], group: opt.group.concat([m]) }));
    } else {
      c = [];
      for (let i = 0; i < opt.loose.length; i++)
        for (let j = i + 1; j < opt.loose.length; j++)
          c.push({ moves: [opt.loose[i], opt.loose[j]], group: [opt.loose[i], opt.loose[j]] });
    }
    c.sort((a, b) => score(b.group) - score(a.group));
    return c.slice(0, 3);
  }

  function hookUI() {
    if (hooked) return;
    hooked = true;
    G.RunUI.showLevelUp = (lvl, options, cb) => {
      const h = G.Game.player;
      if (learnCards && h) { pendingQ.push({ kind: "card", options, cb }); return; }
      const i = heuristicPick(options, h);
      RL.picks.push(options[i].key);
      cb(options[i]);
    };
    G.RunUI.showBond = (opt, cb) => {
      // bonded moves fire together; the agent picks which moves to fuse
      if (!learnCards || !G.Game.player) {
        if (opt.mode === "add") cb([opt.loose[0]]);
        else cb([opt.loose[0], opt.loose[1]]);
        return;
      }
      pendingQ.push({ kind: "bond", options: bondCandidates(opt, G.Game.player), cb: (c) => cb(c.moves) });
    };
    G.RunUI.showItemChoice = (cur, neu, cb) => {
      if (!learnCards || !G.Game.player) { cb(neu); return; }
      pendingQ.push({ kind: "item", options: [{ id: cur }, { id: neu }], cb: (o) => cb(o.id) });
    };
    // leaderboard / profile only when explicitly enabled (logged-in account play),
    // never from training
    realRanking = { submitSolo: G.Ranking.submitSolo, runStarted: G.Ranking.runStarted,
                    publishProfile: G.Social && G.Social.publishProfile };
    G.Ranking.submitSolo = function () { if (ranking) return realRanking.submitSolo.apply(this, arguments); };
    G.Ranking.runStarted = function () { if (ranking) return realRanking.runStarted.apply(this, arguments); };
    G.Ranking.submitGroup = () => {};
    if (G.Social) G.Social.publishProfile = function () {
      if (ranking && realRanking.publishProfile) return realRanking.publishProfile.apply(this, arguments);
    };
    G.Input.axis = () => axis;
    const realEvent = G.Progress.event;
    G.Progress.event = function (name, data) {
      if (name === "rift") riftsWon++;
      if (name === "chest") { bigChests++; chests--; } // big chest (its sound was counted too)
      return realEvent.apply(this, arguments);
    };
    // silent, but chests break inside the game's world module: count them by their sound
    try { G.Audio.music = () => {}; G.Audio.sfx = (name) => { if (name === "chest") chests++; }; } catch (e) {}
  }

  // --------------------------------------------------------------- meta game
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
    return bought;
  }

  // rough "how strong is this partner with its permanent upgrades"
  function partnerPower(dex) {
    const d = G.DEX_BY[dex];
    if (!d) return 0;
    const save = G.DB.save;
    const p = G.Upgrades.perks((save.pupg && save.pupg[dex]) || {});
    return (d.hp + p.hp) * d.atk * p.atkMul / p.cdMul * Math.sqrt(d.spd * p.spdMul) *
           (1 + 0.5 * p.crit) / (1 - p.dmgReduce);
  }

  function upgradesMaxed(dex) {
    const ups = (G.DB.save.pupg && G.DB.save.pupg[dex]) || {};
    return Object.keys(UPGRADE_WANT).every((id) => (ups[id] || 0) >= G.Upgrades.BY[id].max);
  }

  // base-stat potential (as if fully upgraded): used to pick an investment target
  function basePotential(dex) {
    const d = G.DEX_BY[dex];
    return d ? d.hp * d.atk * Math.sqrt(d.spd) : 0;
  }

  // "Farm with the current partner, invest in a target, switch once it is maxed":
  // coins go to the current partner until its upgrades are maxed, then into the
  // target; when the target is maxed too, it becomes the partner.
  function investPlan(target) {
    const save = G.DB.save, cur = save.partner, out = { target: null, bought: [], switched: false };
    if (target === "auto") {
      let best = null, bp = basePotential(cur) * 1.15;
      for (const k of Object.keys(save.owned || {})) {
        const dex = +k;
        if (dex === cur || !G.DEX_BY[dex] || !G.SPRITE_META[dex]) continue;
        const p = basePotential(dex);
        if (p > bp) { bp = p; best = dex; }
      }
      target = best;
    }
    target = target ? +target : null;
    out.bought = autoUpgrade(cur);
    if (!target || target === cur || !G.DB.owns(target) || !upgradesMaxed(cur)) return out;
    out.target = (G.DEX_BY[target] || {}).name;
    out.bought.push(...autoUpgrade(target).map((b) => out.target + ":" + b));
    if (upgradesMaxed(target)) {
      save.partner = target;
      save.partnerShiny = !!(save.shiny && save.shiny[target]);
      out.switched = true;
    }
    return out;
  }

  function claimMissions() {
    const save = G.DB.save, got = [];
    if (!save.missions || !G.Progress.claim) return got;
    try { G.Progress.ensure && G.Progress.ensure(save); } catch (e) {}
    for (const [list, weekly] of [[save.missions.daily || [], false], [save.missions.weekly || [], true]]) {
      for (const m of list) {
        try { if (G.Progress.claim(m, weekly)) got.push(m.id); } catch (e) {}
      }
    }
    return got;
  }

  function gacha() {
    const save = G.DB.save, got = [];
    save.tickets = save.tickets || { t1: 0, t10: 0 };
    const gen = 1 + Math.floor(Math.random() * G.Gacha.BANNERS.length);
    const take = (res) => { for (const r of res || []) got.push((G.DEX_BY[r.dex] || {}).name + (r.shiny ? "*" : "") + ":" + r.rarity); };
    while (save.tickets.t10 >= 1) take(G.Gacha.pull(save, 10, gen, "t10"));
    while (save.tickets.t1 >= 10) take(G.Gacha.pull(save, 10, gen, "t1"));
    while (save.tickets.t1 >= 1) take(G.Gacha.pull(save, 1, gen, "t1"));
    return got;
  }

  function choosePartner() {
    const save = G.DB.save;
    let best = save.partner, bp = partnerPower(save.partner) * 1.1; // switching has to be clearly better
    for (const k of Object.keys(save.owned || {})) {
      const dex = +k;
      if (!G.DEX_BY[dex] || !G.SPRITE_META[dex]) continue;
      const p = partnerPower(dex);
      if (p > bp) { bp = p; best = dex; }
    }
    if (best !== save.partner) {
      save.partner = best;
      save.partnerShiny = !!(save.shiny && save.shiny[best]);
    }
    return best;
  }

  // ------------------------------------------------------------- observation
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

  function moveFeat(def, inst, h) {
    // def: static move data, inst: player's instance (or null for a new move)
    const m = inst || def;
    const dps = ((m.dmg || 0) * (m.count || 1)) / Math.max(0.15, m.cd || 1);
    return [
      ...oneHot(KINDS, m.kind),
      Math.min(1, dps / 120),
      inst ? (inst.lvl || 1) / (inst.maxLvl || 5) : 0,
      inst && h.activeGroup().includes(inst) ? 1 : 0,
      inst && inst.evolved ? 1 : 0,
      m.kind === "buff" ? 1 : 0,
    ];
  }

  function cardFeat(o, h) {
    const f = [...oneHot(CARD_TYPES, o.type)];
    const statId = o.type === "stat" ? o.key.slice(5) : null;
    f.push(...oneHot(STATS, statId));
    f.push(o.type === "stat" && o.color === "#ffd23f" ? 1 : 0); // unlocks an evolution
    if (o.type === "stat") {
      f.push(...new Array(KINDS.length + 5).fill(0));
    } else {
      const id = o.key.slice(o.key.indexOf(":") + 1);
      const inst = h.moves.find((m) => m.id === id) || null;
      f.push(...moveFeat(G.Moves.BY_ID[id] || {}, inst, h));
    }
    return f;
  }

  function mapFeat(h) {
    const f = new Array(MAP_DIM).fill(0);
    if (!h) return f;
    const near = [null, null, null], nd = [Infinity, Infinity, Infinity];
    const add = (o, ch, far) => {
      const dx = o.x - h.x, dy = o.y - h.y, d = Math.hypot(dx, dy);
      if (far >= 0 && d < nd[far]) { nd[far] = d; near[far] = o; }
      if (d > SENSE) return;
      const k = sectorOf(dx, dy) * MAP_CH + ch;
      f[k] = Math.max(f[k], 1 - d / SENSE);
    };
    for (const o of G.World.propsInCircle(h.x, h.y, FAR)) {
      if (o.type === "chest") add(o, 1, 1);
      else if (o.type === "grass") add(o, 2, -1);
    }
    for (const o of G.World.specialsNear(h.x, h.y, FAR)) {
      if (o.type === "bigchest") add(o, 0, 0);
      else if (o.type === "trap") add(o, 4, -1);
      else if ((o.type === "altar" && !o.used) || (o.type === "spring" && o.charge > 0.15)) add(o, 3, 2);
    }
    if (--farAge <= 0 || farOpened !== bigChests) {
      farAge = 15;
      farOpened = bigChests;
      farChest = null;
      let best = Infinity;
      for (const o of G.World.specialsNear(h.x, h.y, CHEST_FAR)) {
        const d = o.type === "bigchest" ? Math.hypot(o.x - h.x, o.y - h.y) : Infinity;
        if (d < best) { best = d; farChest = o; }
      }
    }
    if (!near[0] && farChest) near[0] = farChest;
    let at = N_SECT * MAP_CH;
    near.forEach((o, i) => {
      if (o) {
        const scale = i === 0 ? CHEST_FAR : FAR;
        f[at] = 1;
        f[at + 1] = Math.max(-1, Math.min(1, (o.x - h.x) / scale));
        f[at + 2] = Math.max(-1, Math.min(1, (o.y - h.y) / scale));
      }
      at += 3;
    });
    f[at] = near[0] ? Math.min(1, (near[0].open || 0) / 3.5) : 0;
    return f;
  }

  function choiceFeat(h) {
    const f = new Array(CHOICE_DIM).fill(0);
    if (!h) return f;
    ITEMS.forEach((id, i) => { f[CHOICE_KINDS.length + i] = h.item === id ? 1 : 0; });
    const p = pendingQ[0];
    if (!p) return f;
    f[CHOICE_KINDS.indexOf(p.kind)] = 1;
    const base = CHOICE_KINDS.length + ITEMS.length;
    p.options.slice(0, 3).forEach((o, i) => {
      const at = base + i * OPT_DIM;
      if (p.kind === "bond") {
        const act = h.activeGroup();
        let dps = 0, lvl = 0;
        for (const m of o.group) {
          const k = KINDS.indexOf(m.kind);
          if (k >= 0) f[at + k] = 1;
          dps += ((m.dmg || 0) * (m.count || 1)) / Math.max(0.15, m.cd || 1);
          lvl += m.lvl || 1;
        }
        f[at + KINDS.length] = Math.min(1, dps / 200);
        f[at + KINDS.length + 1] = Math.min(1, lvl / 20);
        f[at + KINDS.length + 2] = o.group.length / 4;
        f[at + KINDS.length + 3] = o.group.some((m) => act.includes(m)) ? 1 : 0;
      } else if (p.kind === "item") {
        const k = ITEMS.indexOf(o.id);
        if (k >= 0) f[at + BOND_DIM + k] = 1;
      }
    });
    return f;
  }

  function observe() {
    const h = G.Game.player;
    const grid = new Float32Array(N_SECT * C);
    const near = new Float32Array(8 * 4);
    const slots = new Array(4 * SLOT_DIM).fill(0);
    const cards = new Array(3 * CARD_DIM).fill(0);
    if (!h) return { grid: Array.from(grid), feats: new Array(FEAT_DIM).fill(0), near: Array.from(near), slots, cards,
                     choice: choiceFeat(h), map: mapFeat(h) };
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
      if (e.boss) grid[k + 8] = Math.max(grid[k + 8], p);
      enemies.push([d, dx, dy, e]);
    }
    for (const pr of G.Projectiles.all()) {
      if (pr.friendly) continue;
      const dx = pr.x - px, dy = pr.y - py, d = Math.hypot(dx, dy);
      if (d > SENSE) continue;
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
    // terrain: walls / solid props and liquid, probed along 16 rays
    const liq = G.World.liquidEffect(h.mon.types) || {};
    const liqVal = liq.dmg ? 1 : liq.slow ? 0.5 : 0.15;
    const haz = G.Hazards.all || [];
    for (let i = 0; i < N_SECT; i++) {
      const [cx, cy] = DIRS[i];
      let wall = 0, water = 0;
      for (let r = 16; r <= TERRAIN_RANGE; r += 24) {
        const x = px + cx * r, y = py + cy * r, near = 1 - (r - 16) / TERRAIN_RANGE;
        if (!water && G.World.isLiquid(x, y)) water = near * liqVal;
        if (!G.World.isFree(x, y, h.r, true)) { wall = near; break; }
      }
      grid[i * C + 5] = wall;
      grid[i * C + 6] = water;
      if (haz.length) {
        for (const r of [0, 45, 90, 140, 190, 240]) {
          if (hazardAt(haz, px + cx * r, py + cy * r, h.r)) { grid[i * C + 7] = Math.max(grid[i * C + 7], 1 - r / 280); break; }
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
    h.slots.slice(0, 4).forEach((g, i) => {
      const lead = g[0];
      const lvl = g.reduce((a, m) => a + (m.lvl || 1), 0);
      const dps = g.reduce((a, m) => a + ((m.dmg || 0) * (m.count || 1)) / Math.max(0.15, m.cd || 1), 0);
      slots.splice(i * SLOT_DIM, SLOT_DIM, 1, i === h.active ? 1 : 0, g.length / 4, Math.min(1, lvl / 20),
        ...oneHot(KINDS, lead && lead.kind), g.some((m) => m.evolved) ? 1 : 0, Math.min(1, dps / 200));
    });
    const p = pendingQ[0];
    if (p && p.kind === "card") p.options.slice(0, 3).forEach((o, i) => cards.splice(i * CARD_DIM, CARD_DIM, ...cardFeat(o, h)));
    // how much of the commanded movement actually happened (blocked by a wall -> low)
    let moved = 1;
    if (lastPos && (axis[0] || axis[1])) {
      const want = h.spd * 3 / 30;
      moved = Math.min(1, Math.hypot(px - lastPos[0], py - lastPos[1]) / Math.max(1, want));
    }
    const d = h.mon;
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
      pendingQ.length ? 1 : 0,
      moved,
      Math.min(1, d.hp / 200),
      Math.min(1, d.atk / 2),
      Math.min(1, d.spd / 200),
      Math.min(1, (h.atkMul || 1) / 4),
      Math.min(1, (h.cdMul || 1)),
      Math.min(1, h.critChance || 0),
      Math.min(1, (h.magnet || 78) / 300),
      Math.min(1, (h.regen || 0) / 10),
      h.item ? 1 : 0,
      Math.min(1, (G.Game.st.time % 240) / 240),
      ...riftFeats(h),
    ];
    return { grid: Array.from(grid), feats, near: Array.from(near), slots, cards, choice: choiceFeat(h), map: mapFeat(h) };
  }

  // rift (portal to a legendary fight) and the nearest map boss, often beyond SENSE
  function riftFeats(h) {
    const f = new Array(10).fill(0);
    const rift = G.Rift.rift, arena = G.Rift.arena;
    if (rift) {
      f[0] = 1;
      f[1] = Math.max(-1, Math.min(1, (rift.x - h.x) / FAR));
      f[2] = Math.max(-1, Math.min(1, (rift.y - 10 - h.y) / FAR));
      f[3] = Math.min(1, rift.ttl / 45);
    }
    if (arena) {
      f[4] = arena.phase === "fight" ? 1 : 0.5;
      const b = G.EnemyMgr.get(arena.bossId);
      f[5] = b && !b.dead ? b.hp / b.maxHp : 0;
      f[6] = Math.max(0, 1 - arena.t / 120);
    }
    let best = null, bd = Infinity;
    for (const e of G.EnemyMgr.all()) {
      if (e.dead || !e.boss) continue;
      const d = Math.hypot(e.x - h.x, e.y - h.y);
      if (d < bd) { bd = d; best = e; }
    }
    if (best) {
      f[7] = Math.max(-1, Math.min(1, (best.x - h.x) / FAR));
      f[8] = Math.max(-1, Math.min(1, (best.y - h.y) / FAR));
      f[9] = best.hp / best.maxHp;
    }
    return f;
  }

  function actionMask() {
    const m = new Array(N_ACTIONS).fill(false);
    const h = G.Game.player;
    if (pendingQ.length) {
      for (let i = 0; i < Math.min(3, pendingQ[0].options.length); i++) m[9 + i] = true;
      return m;
    }
    for (let i = 0; i < 9; i++) m[i] = true;
    if (h && !h.hasItem("choiceband"))
      for (let i = 0; i < Math.min(4, h.slots.length); i++) if (i !== h.active) m[12 + i] = true;
    return m;
  }

  function stats() {
    const h = G.Game.player;
    if (!h) return null;
    return {
      hp: h.hp, maxHp: h.maxHp, level: h.level, xp: h.xp, kills: h.kills,
      bosses: h.bosses, time: G.Game.st.time, dead: !!h.dead,
      progress: h.level - 1 + (h.xpNext ? Math.min(1, h.xp / h.xpNext) : 0),
      moves: h.moves.map((m) => m.id + ":" + m.lvl + (m.evolved ? "*" : "")),
      partner: (G.DEX_BY[h.dex] || {}).name || String(h.dex),
      riftsEntered, riftsWon, riftsLost, inArena: G.Rift.inArena, chests, bigChests,
    };
  }

  function result(before, hpLost, h0) {
    const h = G.Game.player;
    const after = stats() || before;
    const obs = observe();
    if (after.time !== before.time) lastPos = h ? [h.x, h.y] : null;
    return {
      obs,
      mask: actionMask(),
      stats: after,
      hpLost: hpLost / (before ? before.maxHp : 100),
      dKills: after.kills - before.kills,
      dProgress: after.progress - before.progress,
      dBoss: after.bosses - before.bosses,
      dRiftIn: after.riftsEntered - before.riftsEntered,
      dRiftWon: after.riftsWon - before.riftsWon,
      dRiftLost: after.riftsLost - before.riftsLost,
      dChest: after.chests - before.chests,
      dBigChest: after.bigChests - before.bigChests,
      dt: after.time - before.time,
      done: !h || h !== h0 || (h.dead && !G.Rift.inArena) || G.Game.state === "over",
      choice: pendingQ.length > 0,
    };
  }

  window.RL = {
    picks: [],
    ready() {
      return !!(window.G && G.DB && G.Game && G.Flow && G.RunUI && G.Ranking);
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
        s.partner = s.partner && G.DB.owns(s.partner) ? s.partner : partner;
      }
      s.starter = s.starter || s.partner;
      for (const id of ["tap-start", "boot"]) {
        const el = document.getElementById(id);
        if (el) el.classList.add("hidden");
      }
      hookUI();
      return s.partner;
    },
    // opts: {meta, learnCards, startTime, startLevels, partner}
    reset(opts) {
      hookUI();
      learnCards = opts.learnCards !== false;
      // only a real logged-in account may ever reach the leaderboard
      ranking = !!opts.ranking && G.DB.loggedIn && !G.DB.guest && !(opts.startTime > 0);
      pendingQ = [];
      // finish the previous run properly so the game pays out its coins / tickets
      if (G.Game.player && G.Game.state !== "ui" && G.Game.state !== "over") G.Game.quit();
      G.UI.hideAll();
      if (G.Game.state !== "ui") G.Game.toUI();
      const meta = { claimed: [], pulls: [], bought: [] };
      if (opts.partner && G.DB.owns(opts.partner)) G.DB.save.partner = opts.partner;
      if (opts.meta) {
        meta.claimed = claimMissions();
        meta.pulls = gacha();
        // favourites the user wants levelled first; they get coins before the partner,
        // but the partner itself stays unchanged
        for (const fav of opts.favorites || []) {
          if (+fav === G.DB.save.partner || !G.DB.owns(+fav)) continue;
          const name = (G.DEX_BY[+fav] || {}).name;
          meta.bought.push(...autoUpgrade(+fav).map((b) => name + ":" + b));
        }
        if (opts.invest) {
          const plan = investPlan(opts.invest);
          meta.bought.push(...plan.bought);
          meta.investTarget = plan.target;
          meta.switched = plan.switched;
        } else {
          if (opts.choosePartner !== false) choosePartner();
          meta.bought.push(...autoUpgrade(G.DB.save.partner));
        }
        if (G.DB.commit) G.DB.commit(true);
      }
      meta.partner = (G.DEX_BY[G.DB.save.partner] || {}).name;
      meta.coins = G.DB.save.coins;
      meta.tickets = G.DB.save.tickets;
      meta.owned = Object.keys(G.DB.save.owned || {}).length;
      meta.legendaries = Object.keys(G.DB.save.owned || {})
        .map((k) => G.DEX_BY[+k]).filter((d) => d && d.leg)
        .map((d) => d.name + (G.DB.save.shiny && G.DB.save.shiny[d.dex] ? "*" : ""));
      meta.upgrades = (G.DB.save.pupg || {})[G.DB.save.partner] || {};
      G.Game.startRun();
      axis = [0, 0];
      lastPos = null;
      riftsEntered = riftsWon = riftsLost = 0;
      chests = bigChests = 0;
      farChest = null; farAge = 0;
      wasInArena = false;
      RL.picks = [];
      // curriculum: jump ahead in time with a matching number of level-ups
      const h = G.Game.player;
      if (opts.startTime > 0) G.Game.debugTime(opts.startTime);
      for (let i = 0; i < (opts.startLevels || 0); i++) G.Game.debugLevelUp();
      if (opts.startLevels) h.xpNext = G.Player.xpFor(h.level);
      pump(2);
      const before = stats();
      return Object.assign(result(before, 0, h), { meta, partnerDex: G.DB.save.partner });
    },
    step(action, frames) {
      const h0 = G.Game.player;
      const before = stats();
      let hpLost = 0;
      if (pendingQ.length) {
        // menu choice: level-up card, bond or item swap (time is frozen while it is open)
        const p = pendingQ.shift();
        const fallback = p.kind === "card" ? heuristicPick(p.options, h0) : p.kind === "item" ? 1 : 0;
        const i = action >= 9 && action <= 11 && action - 9 < p.options.length ? action - 9 : fallback;
        const o = p.options[i];
        RL.picks.push(p.kind === "card" ? o.key : p.kind === "item" ? "item:" + o.id
                      : "bond:" + o.group.map((m) => m.id).join("+"));
        p.cb(o);
        if (!pendingQ.length) pump(1); // lets the game raise the next queued menu, if any
        return result(before, 0, h0);
      }
      if (action >= 12 && action <= 15) {
        if (h0 && action - 12 < h0.slots.length) h0.setActive(action - 12);
      } else if (action >= 0 && action <= 8) {
        axis = MOVES[action].slice();
      }
      for (let i = 0; i < frames; i++) {
        const hp = G.Game.player ? G.Game.player.hp : 0;
        pump(1);
        const h = G.Game.player;
        if (!h || h !== h0) break;
        if (h.hp < hp) hpLost += hp - h.hp;
        const inArena = G.Rift.inArena;
        if (inArena && !wasInArena) riftsEntered++;
        if (!inArena && wasInArena && riftsWon + riftsLost < riftsEntered) riftsLost++;
        wasInArena = inArena;
        if (G.Game.state === "over" || (h.dead && !inArena) || pendingQ.length) break;
      }
      return result(before, hpLost, h0);
    },
    getSave() {
      return JSON.stringify(G.DB.save);
    },
    setSave(json) {
      const s = JSON.parse(json);
      if (!s || !G.DB.save) return false;
      Object.assign(G.DB.save, s);
      return true;
    },
    observe,
    actionMask,
    stats,
    dims: { N_SECT, C, FEAT_DIM, SLOT_DIM, CARD_DIM, CHOICE_DIM, MAP_DIM, N_ACTIONS },
  };
})();
