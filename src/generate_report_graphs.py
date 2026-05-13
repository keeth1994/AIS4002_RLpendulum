"""Generate report figures from the retained experiment logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rad_to_deg(angle: np.ndarray) -> np.ndarray:
    return np.rad2deg(angle)


def load_sim_classical(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame(
        {
            "time_s": df["time"],
            "theta_deg": rad_to_deg(df["theta"].to_numpy()),
            "alpha_deg": rad_to_deg(wrap_angle(df["alpha"].to_numpy())),
            "voltage_applied": df["voltage"],
            "mode": "sim_classical",
        }
    )


def load_hw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def style_axis(axis, ylabel: str) -> None:
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)


def shade_mode(axis, df: pd.DataFrame, mode: str, color: str, label: str) -> None:
    in_segment = False
    start = None
    label_used = False
    for _, row in df.iterrows():
        active = row["mode"] == mode
        if active and not in_segment:
            start = float(row["time_s"])
            in_segment = True
        if in_segment and not active:
            end = float(row["time_s"])
            axis.axvspan(start, end, color=color, alpha=0.12, label=None if label_used else label)
            label_used = True
            in_segment = False
    if in_segment:
        axis.axvspan(start, float(df["time_s"].iloc[-1]), color=color, alpha=0.12, label=None if label_used else label)


def plot_classical_validation(sim_path: Path, hw_path: Path, output: Path) -> None:
    sim = load_sim_classical(sim_path)
    hw = load_hw(hw_path)
    hw = hw[hw["time_s"] <= min(6.0, hw["time_s"].max())]
    sim = sim[sim["time_s"] <= hw["time_s"].max()]

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(sim["time_s"], sim["theta_deg"], label="simulation", linewidth=1.4)
    axes[0].plot(hw["time_s"], hw["theta_deg"], label="hardware", linewidth=1.0, alpha=0.85)
    style_axis(axes[0], "arm theta [deg]")
    axes[0].legend(loc="upper right")

    axes[1].plot(sim["time_s"], sim["alpha_deg"], label="simulation", linewidth=1.4)
    axes[1].plot(hw["time_s"], hw["alpha_deg"], label="hardware", linewidth=1.0, alpha=0.85)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    style_axis(axes[1], "pendulum alpha [deg]")

    axes[2].plot(sim["time_s"], sim["voltage_applied"], label="simulation", linewidth=1.4)
    axes[2].plot(hw["time_s"], hw["voltage_applied"], label="hardware", linewidth=1.0, alpha=0.85)
    style_axis(axes[2], "motor voltage [V]")
    axes[2].set_xlabel("time [s]")

    fig.suptitle("Classical controller validation: simulation vs hardware")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_hybrid_hardware(path: Path, output: Path) -> None:
    df = load_hw(path)
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    for axis in axes:
        shade_mode(axis, df, "rl_residual", "#2ca02c", "RL residual active")

    axes[0].plot(df["time_s"], df["theta_deg"], color="#1f77b4", linewidth=1.2)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    style_axis(axes[0], "theta [deg]")

    axes[1].plot(df["time_s"], df["alpha_deg"], color="#ff7f0e", linewidth=1.2)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].axhline(10.0, color="black", linestyle="--", linewidth=0.7, alpha=0.4)
    axes[1].axhline(-10.0, color="black", linestyle="--", linewidth=0.7, alpha=0.4)
    style_axis(axes[1], "alpha [deg]")

    axes[2].plot(df["time_s"], df["voltage_applied"], color="#d62728", linewidth=1.0)
    style_axis(axes[2], "voltage [V]")

    axes[3].plot(df["time_s"], df["rl_command"], color="#9467bd", linewidth=1.0)
    style_axis(axes[3], "RL action")
    axes[3].set_xlabel("time [s]")
    axes[0].legend(loc="upper right")

    fig.suptitle("Hardware run: classical swing-up with bounded RL residual")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def longest_upright_seconds(df: pd.DataFrame, threshold_deg: float = 10.0) -> float:
    times = df["time_s"].to_numpy()
    if len(times) < 2:
        return 0.0
    mask = np.abs(df["alpha_deg"].to_numpy()) < threshold_deg
    best = 0.0
    start = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            best = max(best, times[index - 1] - times[start])
            start = None
    if start is not None:
        best = max(best, times[-1] - times[start])
    return float(best)


def summarize_log(path: Path) -> dict[str, float | str]:
    df = load_hw(path)
    return {
        "log": path.name,
        "duration_s": float(df["time_s"].iloc[-1]),
        "upright_ratio": float((df["alpha_deg"].abs() < 10.0).mean()),
        "longest_upright_s": longest_upright_seconds(df),
        "max_abs_theta_deg": float(df["theta_deg"].abs().max()),
        "mean_abs_voltage_v": float(df["voltage_applied"].abs().mean()),
        "final_theta_deg": float(df["theta_deg"].iloc[-1]),
        "final_alpha_deg": float(df["alpha_deg"].iloc[-1]),
    }


def plot_repeat_summary(logs: list[Path], output: Path, metrics_output: Path) -> None:
    metrics = pd.DataFrame([summarize_log(path) for path in logs])
    metrics.to_csv(metrics_output, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(metrics["log"], metrics["upright_ratio"], color="#2ca02c")
    axes[0].set_ylabel("upright ratio")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(metrics["log"], metrics["max_abs_theta_deg"], color="#1f77b4")
    axes[1].set_ylabel("max |theta| [deg]")
    axes[1].grid(True, axis="y", alpha=0.25)

    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    fig.suptitle("Hardware repeatability summary")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/report_figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_classical_validation(
        args.results_dir / "classical_sim_example_calibrated_defaults.csv",
        args.results_dir / "classical_hw_example_calm_kick_com13.csv",
        args.output_dir / "classical_validation_sim_vs_hw.png",
    )
    plot_hybrid_hardware(
        args.results_dir / "hw_example_rl_residual_06_repeat.csv",
        args.output_dir / "hardware_rl_residual_timeseries.png",
    )
    plot_repeat_summary(
        [
            args.results_dir / "hw_example_rl_residual_06.csv",
            args.results_dir / "hw_example_rl_residual_06_repeat.csv",
        ],
        args.output_dir / "hardware_repeatability_summary.png",
        args.output_dir / "hardware_metrics.csv",
    )
    print(f"Wrote report figures to {args.output_dir}")


if __name__ == "__main__":
    main()
