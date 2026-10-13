"""Train a PPO agent to survive in PokeSurvivor.

    python train.py --envs 8 --steps 3000000
    tensorboard --logdir runs
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor

from pokesurvivor_rl.env import PokeSurvivorEnv


def make_env(rank, partner, frame_skip):
    def _thunk():
        return PokeSurvivorEnv(headless=True, partner=partner, frame_skip=frame_skip,
                               save_path=f"progress/train_env{rank}.json")
    return _thunk


class GameStatsCallback(BaseCallback):
    """Logs in-game survival time / level / kills of finished episodes."""

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            st = info.get("episode_stats")
            if st:
                self.logger.record_mean("game/survive_sec", st["time"])
                self.logger.record_mean("game/level", st["level"])
                self.logger.record_mean("game/kills", st["kills"])
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--partner", type=int, default=4, help="Pokedex number of the partner (4 = Charmander)")
    p.add_argument("--frame-skip", type=int, default=3)
    p.add_argument("--resume", type=str, default=None, help="path to a .zip checkpoint")
    p.add_argument("--name", type=str, default="ppo_pokesurvivor")
    args = p.parse_args()

    venv = SubprocVecEnv([make_env(i, args.partner, args.frame_skip) for i in range(args.envs)])
    venv = VecMonitor(venv)
    venv = VecFrameStack(venv, n_stack=4)

    ckpt_dir = Path("checkpoints") / args.name
    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cuda", tensorboard_log="runs")
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            n_steps=512,
            batch_size=1024,
            n_epochs=6,
            gamma=0.995,
            gae_lambda=0.95,
            learning_rate=2.5e-4,
            ent_coef=0.01,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            tensorboard_log="runs",
            device="cuda",
            verbose=1,
        )
    callbacks = [
        CheckpointCallback(save_freq=max(1, 50_000 // args.envs), save_path=str(ckpt_dir), name_prefix="model"),
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
