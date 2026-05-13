"""Evaluate a trained Stable-Baselines3 policy and optionally save a video."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

from src.envs import RotaryPendulumEnv
from src.video import save_video_or_gif

REPORT_BALANCE_PRESET = {
    "algo": "sac",
    "episodes": 10,
    "steps": 1500,
    "video": "none",
    "arm_limit_deg": 60.0,
    "initial_perturbation": 0.08,
    "voltage_limit": 5.0,
    "soft_arm_limit": True,
    "sensor_noise": False,
    "reset_mode": "upright",
    "reward_mode": "report_balance",
    "obs_mode": "base6",
    "action_filter_alpha": 1.0,
    "voltage_slew_rate": 0.0,
    "action_rate_penalty": 0.0,
    "motor_dead_voltage": 0.0,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=["custom", "report-balance"],
        default="custom",
        help="Use a named simulation evaluation setup matching the report balance policy.",
    )
    parser.add_argument("--model-path", type=Path, default=Path("models/sac_qube_classical_env_speed_obs.zip"))
    parser.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--video", type=str, default="videos/rl_policy.mp4")
    parser.add_argument("--arm-limit-deg", type=float, default=90.0)
    parser.add_argument("--initial-perturbation", type=float, default=0.25)
    parser.add_argument("--voltage-limit", type=float, default=10.0)
    parser.add_argument("--soft-arm-limit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sensor-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reset-mode", choices=["down", "upright", "mixed"], default="down")
    parser.add_argument(
        "--reward-mode",
        choices=["report_balance", "recovery", "recovery_speed", "hardware_recovery"],
        default="recovery_speed",
    )
    parser.add_argument("--obs-mode", choices=["base6", "speed7", "speed8"], default="speed7")
    parser.add_argument("--action-filter-alpha", type=float, default=1.0)
    parser.add_argument("--voltage-slew-rate", type=float, default=0.0)
    parser.add_argument("--action-rate-penalty", type=float, default=0.0)
    parser.add_argument("--motor-dead-voltage", type=float, default=0.0)
    parser.add_argument("--render-style", choices=["qube", "cartpole"], default="qube")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    if args.preset == "report-balance":
        for name, value in REPORT_BALANCE_PRESET.items():
            setattr(args, name, value)

    model_cls = PPO if args.algo == "ppo" else SAC
    model = model_cls.load(args.model_path)
    env = RotaryPendulumEnv(
        max_episode_steps=args.steps,
        seed=args.seed,
        arm_limit_rad=np.deg2rad(args.arm_limit_deg),
        initial_perturbation=args.initial_perturbation,
        voltage_limit=args.voltage_limit,
        soft_arm_limit=args.soft_arm_limit,
        terminate_on_arm_limit=not args.soft_arm_limit,
        sensor_noise=args.sensor_noise,
        reset_mode=args.reset_mode,
        reward_mode=args.reward_mode,
        obs_mode=args.obs_mode,
        render_style=args.render_style,
        action_filter_alpha=args.action_filter_alpha,
        voltage_slew_rate=args.voltage_slew_rate,
        action_rate_penalty=args.action_rate_penalty,
        motor_dead_voltage=args.motor_dead_voltage,
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
