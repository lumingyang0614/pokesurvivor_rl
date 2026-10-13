"""Live terminal monitor of training results (best runs so far).

    python monitor.py            # refreshes every 5 s, Ctrl+C to quit
    python monitor.py --once     # print once
"""
import argparse
import json
import os
import re
import statistics as stats
import time
from pathlib import Path

EPISODES = Path("logs/episodes.jsonl")
TRAIN_LOG = Path("logs/train_v2.log")
ACCOUNT = Path("logs/account_episodes.jsonl")


def mmss(sec):
    sec = int(sec or 0)
    return f"{sec // 60}:{sec % 60:02d}"


def load(path=EPISODES):
    eps = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                eps.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return eps


def train_status():
    if not TRAIN_LOG.exists():
        return "-", "-"
    tail = TRAIN_LOG.read_text(encoding="utf-8", errors="ignore")[-6000:]
    steps = re.findall(r"total_timesteps\s*\|\s*(\d+)", tail)
    fps = re.findall(r"fps\s*\|\s*(\d+)", tail)
    return (f"{int(steps[-1]):,}" if steps else "-"), (fps[-1] if fps else "-")


def bar(value, best, width=24):
    n = int(width * value / best) if best else 0
    return "█" * n + "·" * (width - n)


def render():
    eps = load()
    normal = [e for e in eps if not e.get("start_time")]
    late = [e for e in eps if e.get("start_time")]
    steps, fps = train_status()
    out = []
    out.append("PokeSurvivor RL monitor  " + time.strftime("%H:%M:%S"))
    out.append(f"timesteps {steps}   speed {fps} steps/s   episodes {len(eps)} "
               f"(normal {len(normal)}, late-game practice {len(late)})")
    out.append("")
    if normal:
        best = max(normal, key=lambda e: e["time"])
        out.append(f"BEST RUN  survived {mmss(best['time'])}  level {best['level']}  kills {best['kills']}  "
                   f"bosses {best['bosses']}  rifts won {best.get('riftsWon', 0)}/{best.get('riftsEntered', 0)}  "
                   f"(at step {best['step']:,})")
        out.append(f"          moves: {', '.join(best.get('moves') or [])}")
        out.append("")
        recent = normal[-100:]
        t = [e["time"] for e in recent]
        out.append(f"last {len(recent)} runs: mean {mmss(stats.mean(t))}  median {mmss(stats.median(t))}  "
                   f"best {mmss(max(t))}  level {stats.mean(e['level'] for e in recent):.1f}  "
                   f"bosses {sum(e['bosses'] for e in recent)}  rifts won {sum(e.get('riftsWon', 0) for e in recent)}"
                   f"/{sum(e.get('riftsEntered', 0) for e in recent)}")
        out.append("")
        out.append("progress (best / mean survival per 200k steps):")
        buckets = {}
        for e in normal:
            buckets.setdefault(e["step"] // 200_000, []).append(e["time"])
        top = max(max(v) for v in buckets.values())
        for k in sorted(buckets)[-12:]:
            v = buckets[k]
            out.append(f"  {k * 0.2:5.1f}M  best {mmss(max(v)):>6} {bar(max(v), top)}  mean {mmss(stats.mean(v))}")
        out.append("")
        out.append("top 5 runs:")
        for e in sorted(normal, key=lambda e: e["time"], reverse=True)[:5]:
            out.append(f"  {mmss(e['time']):>6}  lv {e['level']:>3}  kills {e['kills']:>4}  bosses {e['bosses']}  "
                       f"rifts {e.get('riftsWon', 0)}/{e.get('riftsEntered', 0)}  step {e['step']:,}")
    else:
        out.append("no finished normal runs logged yet...")
    if late:
        out.append("")
        lasted = [e["time"] - e["start_time"] for e in late[-100:]]
        bl = max(late, key=lambda e: e["time"] - e["start_time"])
        out.append(f"late-game practice: latest start now up to {mmss(late[-1].get('max_start'))}, "
                   f"free levels {late[-1].get('level_rate', 0):.1f}/min, survives {mmss(stats.mean(lasted))} on average")
        out.append(f"   best: started at {mmss(bl['start_time'])}, lasted {mmss(bl['time'] - bl['start_time'])} "
                   f"(reached {mmss(bl['time'])}, level {bl['level']})")
    acc = load(ACCOUNT)
    if acc:
        last = acc[-1]
        best = max(acc, key=lambda e: e["time"])
        ago = int(time.time() - last.get("wall", time.time()))
        out.append("")
        out.append(f"ACCOUNT {last.get('account')}: {len(acc)} runs, coins now {last.get('coins')}, "
                   f"best {mmss(best['time'])} (lv {best['level']}, bosses {best['bosses']}, "
                   f"rifts {best.get('riftsWon', 0)}/{best.get('riftsEntered', 0)}), "
                   f"last run {mmss(last['time'])} {ago // 60} min ago")
        out.append(f"   upgrades: {last.get('upgrades')}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--every", type=float, default=5.0)
    args = p.parse_args()
    if args.once:
        print(render())
        return
    os.system("")  # enable ANSI escape codes on Windows terminals
    try:
        while True:
            print("\033[2J\033[H" + render(), flush=True)
            time.sleep(args.every)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
