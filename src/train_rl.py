"""Train a Stable-Baselines3 PPO agent in the simulated pendulum environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from src.envs import RotaryPendulumEnv


def make_env(seed: int, domain_randomization: bool, start: str) -> Monitor:
    env = RotaryPendulumEnv(domain_randomization=domain_randomization, seed=seed, start=start)
    return Monitor(env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model-path", type=Path, default=Path("models/ppo_rotary_pendulum"))
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--start", choices=["upright", "down", "random"], default="upright")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    args = parser.parse_args()

    env = make_env(args.seed, args.domain_randomization, args.start)
    if args.check_env:
        check_env(RotaryPendulumEnv(seed=args.seed, start=args.start), warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log="results/tensorboard" if args.tensorboard else None,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=args.progress_bar)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    print(f"Saved model to {args.model_path}.zip")


if __name__ == "__main__":
    main()
