"""Compare classical-controller telemetry from simulation and hardware."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def wrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    return (angle_deg + 180.0) % 360.0 - 180.0


def load_sim_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"time", "theta", "alpha", "theta_dot", "alpha_dot", "voltage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Simulation CSV missing columns: {sorted(missing)}")

    out = pd.DataFrame()
    out["time_s"] = df["time"].astype(float)
    out["theta_deg"] = np.rad2deg(df["theta"].astype(float).to_numpy())
    out["alpha_deg"] = wrap_deg(np.rad2deg(df["alpha"].astype(float).to_numpy()))
    out["theta_dot_rad"] = df["theta_dot"].astype(float)
    out["alpha_dot_rad"] = df["alpha_dot"].astype(float)
    out["voltage_applied"] = df["voltage"].astype(float)
    out["source"] = "simulation"
    return out


def load_hw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"time_s", "theta_deg", "alpha_deg", "theta_dot_rad", "alpha_dot_rad", "voltage_applied"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hardware CSV missing columns: {sorted(missing)}")

    out = pd.DataFrame()
    out["time_s"] = df["time_s"].astype(float)
    out["theta_deg"] = df["theta_deg"].astype(float)
    out["alpha_deg"] = wrap_deg(df["alpha_deg"].astype(float).to_numpy())
    out["theta_dot_rad"] = df["theta_dot_rad"].astype(float)
    out["alpha_dot_rad"] = df["alpha_dot_rad"].astype(float)
    out["voltage_applied"] = df["voltage_applied"].astype(float)
    out["source"] = "hardware"
    return out


def trim_time(df: pd.DataFrame, duration: float | None) -> pd.DataFrame:
    if duration is None:
        return df
    return df[df["time_s"] <= duration].copy()


def summarize(df: pd.DataFrame) -> dict[str, float]:
    alpha_deg = df["alpha_deg"].to_numpy()
    theta_deg = df["theta_deg"].to_numpy()
    voltage = df["voltage_applied"].to_numpy()
    upright_ratio = float(np.mean(np.abs(alpha_deg) < 12.0))
    closest_alpha_deg = float(np.min(np.abs(alpha_deg)))
    max_abs_theta_deg = float(np.max(np.abs(theta_deg)))
    rms_voltage = float(np.sqrt(np.mean(voltage**2)))
    return {
        "duration_s": float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]) if len(df) > 1 else 0.0,
        "upright_ratio": upright_ratio,
        "closest_alpha_deg": closest_alpha_deg,
        "max_abs_theta_deg": max_abs_theta_deg,
        "rms_voltage": rms_voltage,
    }


def add_summary_box(ax: plt.Axes, title: str, summary: dict[str, float], x: float, color: str) -> None:
    text = (
        f"{title}\n"
        f"upright<12deg: {summary['upright_ratio']:.3f}\n"
        f"closest |alpha|: {summary['closest_alpha_deg']:.1f} deg\n"
        f"max |theta|: {summary['max_abs_theta_deg']:.1f} deg\n"
        f"RMS voltage: {summary['rms_voltage']:.2f} V"
    )
    ax.text(
        x,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": color},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-csv", type=Path, required=True)
    parser.add_argument("--hw-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/classical_sim_vs_hw.png"))
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--title", type=str, default="Classical Controller: Simulation vs Hardware")
    args = parser.parse_args()

    sim = trim_time(load_sim_csv(args.sim_csv), args.duration)
    hw = trim_time(load_hw_csv(args.hw_csv), args.duration)

    sim_summary = summarize(sim)
    hw_summary = summarize(hw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(sim["time_s"], sim["alpha_deg"], label="Simulation", linewidth=1.8)
    axes[0].plot(hw["time_s"], hw["alpha_deg"], label="Hardware", linewidth=1.4, alpha=0.9)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].axhline(180.0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
    axes[0].axhline(-180.0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
    axes[0].set_ylabel("alpha [deg]")
    axes[0].legend(loc="upper right")

    axes[1].plot(sim["time_s"], sim["theta_deg"], linewidth=1.8)
    axes[1].plot(hw["time_s"], hw["theta_deg"], linewidth=1.4, alpha=0.9)
    axes[1].set_ylabel("theta [deg]")

    axes[2].plot(sim["time_s"], sim["voltage_applied"], linewidth=1.8)
    axes[2].plot(hw["time_s"], hw["voltage_applied"], linewidth=1.4, alpha=0.9)
    axes[2].set_ylabel("voltage [V]")

    axes[3].plot(sim["time_s"], sim["alpha_dot_rad"], linewidth=1.8)
    axes[3].plot(hw["time_s"], hw["alpha_dot_rad"], linewidth=1.4, alpha=0.9)
    axes[3].set_ylabel("alpha_dot [rad/s]")
    axes[3].set_xlabel("time [s]")

    add_summary_box(axes[0], "Simulation", sim_summary, 0.01, "C0")
    add_summary_box(axes[0], "Hardware", hw_summary, 0.28, "C1")

    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print("Simulation summary:")
    for key, value in sim_summary.items():
        print(f"  {key}: {value:.3f}")
    print("Hardware summary:")
    for key, value in hw_summary.items():
        print(f"  {key}: {value:.3f}")
    print(f"Wrote comparison plot to {args.output}")


if __name__ == "__main__":
    main()
