"""Train a Stable-Baselines3 agent in the simulated pendulum environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from src.envs import RotaryPendulumEnv


def make_env(
    seed: int,
    domain_randomization: bool,
    arm_limit_deg: float,
    initial_perturbation: float,
    voltage_limit: float,
    soft_arm_limit: bool,
    sensor_noise: bool,
    max_episode_steps: int,
    reset_mode: str,
    reward_mode: str,
) -> Monitor:
    env = RotaryPendulumEnv(
        max_episode_steps=max_episode_steps,
        domain_randomization=domain_randomization,
        seed=seed,
        arm_limit_rad=np.deg2rad(arm_limit_deg),
        initial_perturbation=initial_perturbation,
        voltage_limit=voltage_limit,
        soft_arm_limit=soft_arm_limit,
        terminate_on_arm_limit=not soft_arm_limit,
        sensor_noise=sensor_noise,
        reset_mode=reset_mode,
        reward_mode=reward_mode,
    )
    return Monitor(env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model-path", type=Path, default=Path("models/sac_qube_swingup"))
    parser.add_argument("--continue-from", type=Path, default=None)
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--arm-limit-deg", type=float, default=90.0)
    parser.add_argument("--initial-perturbation", type=float, default=0.25)
    parser.add_argument("--voltage-limit", type=float, default=5.0)
    parser.add_argument("--soft-arm-limit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sensor-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-episode-steps", type=int, default=3000)
    parser.add_argument("--reset-mode", choices=["down", "upright", "mixed"], default="down")
    parser.add_argument("--reward-mode", choices=["report_balance", "recovery"], default="report_balance")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    args = parser.parse_args()

    env = make_env(
        args.seed,
        args.domain_randomization,
        args.arm_limit_deg,
        args.initial_perturbation,
        args.voltage_limit,
        args.soft_arm_limit,
        args.sensor_noise,
        args.max_episode_steps,
        args.reset_mode,
        args.reward_mode,
    )
    if args.check_env:
        check_env(
            RotaryPendulumEnv(
                max_episode_steps=args.max_episode_steps,
                seed=args.seed,
                arm_limit_rad=np.deg2rad(args.arm_limit_deg),
                initial_perturbation=args.initial_perturbation,
                voltage_limit=args.voltage_limit,
                soft_arm_limit=args.soft_arm_limit,
                terminate_on_arm_limit=not args.soft_arm_limit,
                sensor_noise=args.sensor_noise,
                reset_mode=args.reset_mode,
                reward_mode=args.reward_mode,
            ),
            warn=True,
        )

    tensorboard_log = "results/tensorboard" if args.tensorboard else None
    if args.continue_from is not None:
        model_cls = PPO if args.algo == "ppo" else SAC
        model = model_cls.load(args.continue_from, env=env)
    elif args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            tensorboard_log=tensorboard_log,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            ent_coef=0.01,
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            tensorboard_log=tensorboard_log,
            learning_rate=3e-4,
            buffer_size=300_000,
            learning_starts=1_000,
            batch_size=256,
            gamma=0.995,
            tau=0.01,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
        )
    model.learn(total_timesteps=args.timesteps, progress_bar=args.progress_bar)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    print(f"Saved model to {args.model_path}.zip")


if __name__ == "__main__":
    main()
