"""Train a MaskablePPO agent to survive in PokeSurvivor.

    python train.py --envs 12 --steps 20000000
    tensorboard --logdir runs
"""
import argparse
import json
import os
import time
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from pokesurvivor_rl.env import PokeSurvivorEnv


def make_env(rank, partner, frame_skip, curriculum, switch_partner):
    def _thunk():
        import time
        time.sleep(2.0 * rank)  # stagger page loads so the site is not hit all at once
        return PokeSurvivorEnv(headless=True, partner=partner, frame_skip=frame_skip,
                               curriculum=curriculum, choose_partner=switch_partner,
                               save_path=f"progress/train_env{rank}.json")
    return _thunk


class GameStatsCallback(BaseCallback):
    """Logs in-game results of finished episodes (normal and late-game starts separately)
    to TensorBoard and appends every episode to logs/episodes.jsonl (read by monitor.py)."""

    def __init__(self, log_path="logs/episodes.jsonl"):
        super().__init__()
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(exist_ok=True)

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals["infos"]):
            st = info.get("episode_stats")
            if not st:
                continue
            rec = {k: st.get(k) for k in ("time", "start_time", "level", "kills", "bosses", "riftsEntered",
                                          "riftsWon", "partner", "max_start", "level_rate")}
            rec.update(env=i, step=int(self.num_timesteps), wall=time.time(), moves=st.get("moves"))
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if st.get("start_time", 0) > 0:
                self.logger.record_mean("late/survive_sec", st["time"] - st["start_time"])
                self.logger.record_mean("late/start_time", st["start_time"])
                self.logger.record_mean("late/max_start", st["max_start"])
                self.logger.record_mean("late/level_rate", st["level_rate"])
            else:
                self.logger.record_mean("game/survive_sec", st["time"])
                self.logger.record_mean("game/level", st["level"])
                self.logger.record_mean("game/kills", st["kills"])
                self.logger.record_mean("game/bosses", st["bosses"])
                self.logger.record_mean("game/rifts_entered", st["riftsEntered"])
                self.logger.record_mean("game/rifts_won", st["riftsWon"])
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=12)
    p.add_argument("--steps", type=int, default=20_000_000)
    p.add_argument("--partner", type=int, default=4, help="starting partner (Pokedex number, 4 = Charmander)")
    p.add_argument("--frame-skip", type=int, default=3)
    p.add_argument("--curriculum", type=float, default=0.4, help="share of runs that start in the late game")
    p.add_argument("--switch-partner", action="store_true",
                   help="let the meta step switch to stronger gacha partners (default: train one partner)")
    p.add_argument("--online", action="store_true",
                   help="let the local mirror download missing files from the site (default: offline, "
                        "only cached files - training needs no images)")
    p.add_argument("--resume", type=str, default=None, help="path to a .zip checkpoint")
    p.add_argument("--name", type=str, default="maskppo_v2")
    args = p.parse_args()
    if not args.online:
        os.environ["PS_MIRROR_OFFLINE"] = "1"  # inherited by the env subprocesses

    venv = SubprocVecEnv([make_env(i, args.partner, args.frame_skip, args.curriculum, args.switch_partner) for i in range(args.envs)])
    venv = VecMonitor(venv)

    ckpt_dir = Path("checkpoints") / args.name
    if args.resume:
        model = MaskablePPO.load(args.resume, env=venv, device="cpu", tensorboard_log="runs")
    else:
        model = MaskablePPO(
            "MlpPolicy",
            venv,
            n_steps=1024,
            batch_size=2048,
            n_epochs=5,
            gamma=0.998,          # ~50 s horizon: card choices pay off slowly
            gae_lambda=0.95,
            learning_rate=2.5e-4,
            ent_coef=0.01,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            tensorboard_log="runs",
            device="cpu",
            verbose=1,
        )
    callbacks = [
        CheckpointCallback(save_freq=max(1, 100_000 // args.envs), save_path=str(ckpt_dir), name_prefix="model"),
        GameStatsCallback(),
    ]
    try:
        model.learn(total_timesteps=args.steps, callback=callbacks, tb_log_name=args.name,
                    reset_num_timesteps=not args.resume)
    finally:
        model.save(ckpt_dir / "final")
        venv.close()
        print("saved", ckpt_dir / "final.zip")


if __name__ == "__main__":
    main()
