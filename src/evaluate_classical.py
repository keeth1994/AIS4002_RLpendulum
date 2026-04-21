"""Evaluate the energy swing-up plus PD baseline controller."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.controllers import EnergySwingUpPDController
from src.envs import RotaryPendulumEnv


def run_baseline(
    steps: int,
    seed: int,
    video_path: Path | None,
    plot_path: Path | None,
    csv_path: Path | None = None,
) -> dict:
    env = RotaryPendulumEnv(max_episode_steps=steps, seed=seed)
    obs, info = env.reset(seed=seed, options={"start": "down"})
    controller = EnergySwingUpPDController(env.params)

    rows = []
    frames = []
    total_reward = 0.0
    for step in range(steps):
        action = controller(env.state.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rows.append([step * env.params.dt, info["theta"], info["alpha"], info["theta_dot"], info["alpha_dot"], info["torque"], reward])
        if video_path is not None and step % 2 == 0:
            frames.append(env.render())
        if terminated or truncated:
            break

    data = np.asarray(rows)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time", "theta", "alpha", "theta_dot", "alpha_dot", "torque", "reward"])
            writer.writerows(rows)

    if plot_path is not None:
        import matplotlib.pyplot as plt

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        axes[0].plot(data[:, 0], np.rad2deg(data[:, 2]))
        axes[0].axhline(0.0, color="black", linewidth=0.8)
        axes[0].set_ylabel("alpha [deg]")
        axes[1].plot(data[:, 0], np.rad2deg(data[:, 1]))
        axes[1].set_ylabel("theta [deg]")
        axes[2].plot(data[:, 0], data[:, 5])
        axes[2].set_ylabel("torque [Nm]")
        axes[2].set_xlabel("time [s]")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)

    if video_path is not None and frames:
        import imageio.v2 as imageio

        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=25)

    upright_ratio = float(np.mean(np.abs(data[:, 2]) < np.deg2rad(12))) if len(data) else 0.0
    return {
        "steps": int(len(data)),
        "total_reward": float(total_reward),
        "upright_ratio": upright_ratio,
        "final_alpha_deg": float(np.rad2deg(data[-1, 2])) if len(data) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--video", type=str, default="videos/classical_baseline.mp4")
    parser.add_argument("--plot", type=str, default="results/classical_baseline.png")
    parser.add_argument("--csv", type=str, default="results/classical_baseline.csv")
    args = parser.parse_args()

    video_path = None if args.video.lower() == "none" else Path(args.video)
    plot_path = None if args.plot.lower() == "none" else Path(args.plot)
    csv_path = None if args.csv.lower() == "none" else Path(args.csv)
    metrics = run_baseline(args.steps, args.seed, video_path, plot_path, csv_path)
    print("Classical baseline metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
