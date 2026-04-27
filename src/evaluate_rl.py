"""Evaluate a trained Stable-Baselines3 policy and optionally save a video."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

from src.envs import RotaryPendulumEnv
from src.video import save_video_or_gif


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/ppo_rotary_pendulum.zip"))
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--video", type=str, default="videos/rl_policy.mp4")
    parser.add_argument("--arm-limit-deg", type=float, default=60.0)
    parser.add_argument("--initial-perturbation", type=float, default=0.25)
    parser.add_argument("--voltage-limit", type=float, default=2.0)
    parser.add_argument("--render-style", choices=["qube", "cartpole"], default="qube")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    model_cls = PPO if args.algo == "ppo" else SAC
    model = model_cls.load(args.model_path)
    env = RotaryPendulumEnv(
        max_episode_steps=args.steps,
        seed=args.seed,
        arm_limit_rad=np.deg2rad(args.arm_limit_deg),
        initial_perturbation=args.initial_perturbation,
        voltage_limit=args.voltage_limit,
        render_style=args.render_style,
    )
    frames = []
    returns = []
    upright_ratios = []
    recovery_times = []
    episode_lengths = []
    min_abs_alpha_deg = []
    max_abs_theta_deg = []
    termination_reasons = []

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        episode_return = 0.0
        upright_count = 0
        first_recovery_time = None
        min_abs_alpha = np.inf
        max_abs_theta = 0.0
        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            min_abs_alpha = min(min_abs_alpha, abs(info["alpha"]))
            max_abs_theta = max(max_abs_theta, abs(info["theta"]))
            if abs(info["alpha"]) < np.deg2rad(10):
                upright_count += 1
                if first_recovery_time is None:
                    first_recovery_time = step * env.params.dt
            if episode == 0 and step % 2 == 0:
                frames.append(env.render())
            if terminated or truncated:
                break
        returns.append(episode_return)
        episode_steps = step + 1
        upright_ratios.append(upright_count / episode_steps)
        recovery_times.append(first_recovery_time)
        episode_lengths.append(episode_steps)
        min_abs_alpha_deg.append(float(np.rad2deg(min_abs_alpha)))
        max_abs_theta_deg.append(float(np.rad2deg(max_abs_theta)))
        termination_reasons.append(info.get("termination_reason", "unknown"))

    video_path = None if args.video.lower() == "none" else Path(args.video)
    if video_path is not None and frames:
        save_video_or_gif(video_path, frames, fps=25)

    print(f"Mean return over {args.episodes} episodes: {np.mean(returns):.3f}")
    print(f"Episode returns: {[round(value, 3) for value in returns]}")
    print(f"Mean upright ratio: {np.mean(upright_ratios):.3f}")
    print(f"Upright ratios: {[round(value, 3) for value in upright_ratios]}")
    print(f"Recovery times [s]: {[None if value is None else round(value, 3) for value in recovery_times]}")
    print(f"Episode lengths: {episode_lengths}")
    print(f"Closest alpha to upright [deg]: {[round(value, 2) for value in min_abs_alpha_deg]}")
    print(f"Max abs theta [deg]: {[round(value, 2) for value in max_abs_theta_deg]}")
    print(f"Termination reasons: {termination_reasons}")


if __name__ == "__main__":
    main()
