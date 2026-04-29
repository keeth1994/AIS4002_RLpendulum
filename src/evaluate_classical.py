"""Evaluate the energy swing-up plus PD baseline controller."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.controllers import EnergySwingUpPDController
from src.envs import RotaryPendulumEnv
from src.video import save_video_or_gif


def voltage_to_action(voltage_command: np.ndarray, voltage_limit: float) -> np.ndarray:
    """Convert a voltage command to the normalized environment action."""
    voltage = float(np.asarray(voltage_command, dtype=np.float64).reshape(-1)[0])
    action = np.clip(voltage / voltage_limit, -1.0, 1.0)
    return np.array([action], dtype=np.float32)


def run_baseline(
    steps: int,
    seed: int,
    video_path: Path | None,
    plot_path: Path | None,
    csv_path: Path | None = None,
    arm_limit_deg: float = 60.0,
    open_loop: bool = False,
    initial_perturbation: float = 0.25,
    voltage_limit: float = 5.0,
    render_style: str = "qube",
    controller_kwargs: dict | None = None,
    reset_mode: str = "down",
    reward_mode: str = "recovery",
    terminate_on_arm_limit: bool = True,
    soft_arm_limit: bool = False,
) -> dict:
    env = RotaryPendulumEnv(
        max_episode_steps=steps,
        seed=seed,
        arm_limit_rad=np.deg2rad(arm_limit_deg),
        initial_perturbation=initial_perturbation,
        voltage_limit=voltage_limit,
        render_style=render_style,
        reset_mode=reset_mode,
        reward_mode=reward_mode,
        soft_arm_limit=soft_arm_limit,
        terminate_on_arm_limit=terminate_on_arm_limit,
    )
    obs, info = env.reset(seed=seed)
    controller = EnergySwingUpPDController(env.params, **(controller_kwargs or {}))

    rows = []
    frames = []
    total_reward = 0.0
    min_abs_alpha = np.inf
    max_abs_theta = 0.0
    termination_reason = "running"
    for step in range(steps):
        time_s = step * env.params.dt
        if open_loop:
            voltage_command = controller.open_loop_swingup(time_s, env.state.copy())
        else:
            voltage_command = controller.command(env.state.copy(), time_s)
        action = voltage_to_action(voltage_command, env.params.voltage_limit)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        min_abs_alpha = min(min_abs_alpha, abs(info["alpha"]))
        max_abs_theta = max(max_abs_theta, abs(info["theta"]))
        termination_reason = info.get("termination_reason", "unknown")
        rows.append([step * env.params.dt, info["theta"], info["alpha"], info["theta_dot"], info["alpha_dot"], info["voltage"], reward])
        if video_path is not None and step % 2 == 0:
            frames.append(env.render())
        if terminated or truncated:
            break

    data = np.asarray(rows)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time", "theta", "alpha", "theta_dot", "alpha_dot", "voltage", "reward"])
            writer.writerows(rows)

    if plot_path is not None:
        import matplotlib.pyplot as plt

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        theta_deg = np.rad2deg(np.unwrap(data[:, 1]))
        alpha_deg = np.rad2deg(np.unwrap(data[:, 2]))
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        axes[0].plot(data[:, 0], alpha_deg)
        axes[0].axhline(0.0, color="black", linewidth=0.8)
        axes[0].axhline(180.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        axes[0].set_ylabel("alpha unwrap [deg]")
        axes[1].plot(data[:, 0], theta_deg)
        axes[1].set_ylabel("theta [deg]")
        axes[2].plot(data[:, 0], data[:, 5])
        axes[2].set_ylabel("voltage [V]")
        axes[2].set_xlabel("time [s]")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)

    if video_path is not None and frames:
        save_video_or_gif(video_path, frames, fps=25)

    upright_ratio = float(np.mean(np.abs(data[:, 2]) < np.deg2rad(12))) if len(data) else 0.0
    return {
        "steps": int(len(data)),
        "total_reward": float(total_reward),
        "upright_ratio": upright_ratio,
        "final_alpha_deg": float(np.rad2deg(data[-1, 2])) if len(data) else 0.0,
        "closest_alpha_deg": float(np.rad2deg(min_abs_alpha)) if len(data) else 0.0,
        "max_abs_theta_deg": float(np.rad2deg(max_abs_theta)) if len(data) else 0.0,
        "termination_reason": termination_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--video", type=str, default="videos/classical_baseline.mp4")
    parser.add_argument("--plot", type=str, default="results/classical_baseline.png")
    parser.add_argument("--csv", type=str, default="results/classical_baseline.csv")
    parser.add_argument("--arm-limit-deg", type=float, default=90.0)
    parser.add_argument("--open-loop", action="store_true")
    parser.add_argument("--initial-perturbation", type=float, default=0.25)
    parser.add_argument("--voltage-limit", type=float, default=10.0)
    parser.add_argument("--render-style", choices=["qube", "cartpole"], default="qube")
    parser.add_argument("--reset-mode", choices=["down", "upright", "mixed"], default="down")
    parser.add_argument("--reward-mode", choices=["report_balance", "recovery"], default="recovery")
    parser.add_argument("--terminate-on-arm-limit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--soft-arm-limit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--energy-gain", type=float, default=50.0)
    parser.add_argument("--balance-voltage-limit", type=float, default=5.0)
    parser.add_argument("--swingup-voltage-limit", type=float, default=10.0)
    parser.add_argument("--swingup-frequency-hz", type=float, default=1.5)
    parser.add_argument("--swingup-amplitude", type=float, default=10.0)
    parser.add_argument("--swingup-mode", choices=["oscillatory", "energy"], default="oscillatory")
    parser.add_argument("--swingup-accel-limit", type=float, default=6.0)
    parser.add_argument("--arm-centering-gain", type=float, default=1.3235294117647058)
    parser.add_argument("--arm-centering-rate-gain", type=float, default=1.134453781512605)
    args = parser.parse_args()

    video_path = None if args.video.lower() == "none" else Path(args.video)
    plot_path = None if args.plot.lower() == "none" else Path(args.plot)
    csv_path = None if args.csv.lower() == "none" else Path(args.csv)
    metrics = run_baseline(
        args.steps,
        args.seed,
        video_path,
        plot_path,
        csv_path,
        args.arm_limit_deg,
        args.open_loop,
        args.initial_perturbation,
        args.voltage_limit,
        args.render_style,
        {
            "energy_gain": args.energy_gain,
            "balance_voltage_limit": args.balance_voltage_limit,
            "swingup_voltage_limit": args.swingup_voltage_limit,
            "swingup_frequency_hz": args.swingup_frequency_hz,
            "swingup_amplitude": args.swingup_amplitude,
            "swingup_mode": args.swingup_mode,
            "swingup_accel_limit": args.swingup_accel_limit,
            "arm_centering_gain": args.arm_centering_gain,
            "arm_centering_rate_gain": args.arm_centering_rate_gain,
        },
        args.reset_mode,
        args.reward_mode,
        args.terminate_on_arm_limit,
        args.soft_arm_limit,
    )
    print("Classical baseline metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
