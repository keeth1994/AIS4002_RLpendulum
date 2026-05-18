"""Compare the retained RL balance policy in simulation against hardware logs.

The hardware report run uses classical swing-up first and then switches to a
bounded RL residual near upright. A fair sim-to-real comparison should therefore
align both traces at the RL handoff and compare the upright balance phase.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC

from src.controllers import ExampleEnergyPDController
from src.envs import RotaryPendulumEnv
from src.evaluate_rl import build_policy_observation, voltage_from_action


def load_model(path: Path, algo: str):
    if algo == "ppo":
        return PPO.load(path, device="cpu")
    if algo == "sac":
        return SAC.load(path, device="cpu")
    try:
        return SAC.load(path, device="cpu")
    except Exception:
        return PPO.load(path, device="cpu")


def load_hardware_balance_phase(
    path: Path,
    start_alpha_deg: float,
    start_theta_dot: float,
    start_alpha_dot: float,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    balance_modes = {"rl_residual", "rl_balance"}
    mask = df["mode"].isin(balance_modes)
    if not mask.any():
        raise ValueError(f"{path} does not contain an RL balance/residual phase")
    ready_mask = (
        mask
        & (df["alpha_deg"].astype(float).abs() <= start_alpha_deg)
        & (df["theta_dot_rad"].astype(float).abs() <= start_theta_dot)
        & (df["alpha_dot_rad"].astype(float).abs() <= start_alpha_dot)
    )
    first_index = int(np.flatnonzero((ready_mask if ready_mask.any() else mask).to_numpy())[0])
    out = df.iloc[first_index:].copy()
    out["phase_time_s"] = out["time_s"] - float(out["time_s"].iloc[0])
    return out


def simulate_balance_phase(
    model,
    hw_phase: pd.DataFrame,
    voltage_limit: float,
    arm_limit_deg: float,
    obs_mode: str,
    action_filter_alpha: float,
    voltage_slew_rate: float,
    motor_dead_voltage: float,
    residual_max: float,
    blend_seconds: float,
    sim_controller: str,
) -> pd.DataFrame:
    env = RotaryPendulumEnv(
        max_episode_steps=max(1, len(hw_phase)),
        seed=0,
        arm_limit_rad=np.deg2rad(arm_limit_deg),
    )
    env.base_params = replace(env.base_params, voltage_limit=voltage_limit, dt=1.0 / 300.0)
    env.params = env.base_params
    env.reset(seed=0)

    first = hw_phase.iloc[0]
    env.state = np.array(
        [
            float(first["theta_rad"]),
            float(first["alpha_rad"]),
            float(first["theta_dot_rad"]),
            float(first["alpha_dot_rad"]),
        ],
        dtype=np.float64,
    )
    controller = ExampleEnergyPDController(
        dt=env.params.dt,
        voltage_limit=voltage_limit,
        swingup_u_max=1.6,
        balance_range_deg=30.0,
        startup_kick_voltage=1.5,
        startup_kick_seconds=0.15,
    )

    rows = []
    previous_voltage = 0.0
    for step in range(len(hw_phase)):
        time_s = step * env.params.dt
        classical_voltage = float(controller.command(env.state.copy(), time_s)[0])
        obs = build_policy_observation(env.state, obs_mode, previous_voltage, voltage_limit)
        obs = np.clip(obs, model.observation_space.low, model.observation_space.high).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        rl_voltage = voltage_from_action(
            action,
            previous_voltage,
            env.params.dt,
            argparse.Namespace(
                voltage_limit=voltage_limit,
                action_filter_alpha=action_filter_alpha,
                voltage_slew_rate=voltage_slew_rate,
                motor_dead_voltage=motor_dead_voltage,
            ),
        )
        if sim_controller == "pure-rl":
            voltage = rl_voltage
        else:
            residual = float(np.clip(rl_voltage - classical_voltage, -residual_max, residual_max))
            blend = float(np.clip(time_s / max(blend_seconds, 1e-6), 0.0, 1.0))
            voltage = float(np.clip(classical_voltage + blend * residual, -voltage_limit, voltage_limit))
        _, _, terminated, truncated, info = env.step([voltage])
        previous_voltage = voltage
        rows.append(
            {
                "phase_time_s": step * env.params.dt,
                "theta_deg": np.rad2deg(info["theta"]),
                "alpha_deg": np.rad2deg(info["alpha"]),
                "theta_dot_rad": info["theta_dot"],
                "alpha_dot_rad": info["alpha_dot"],
                "rl_voltage": rl_voltage,
                "classical_voltage": classical_voltage,
                "voltage_applied": voltage,
            }
        )
        if terminated or truncated:
            break
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, voltage_col: str) -> dict[str, float]:
    return {
        "duration_s": float(df["phase_time_s"].iloc[-1]) if len(df) else 0.0,
        "upright_ratio": float((df["alpha_deg"].abs() < 10.0).mean()) if len(df) else 0.0,
        "max_abs_theta_deg": float(df["theta_deg"].abs().max()) if len(df) else 0.0,
        "rms_voltage": float(np.sqrt(np.mean(df[voltage_col].to_numpy(dtype=float) ** 2))) if len(df) else 0.0,
    }


def add_summary(axis: plt.Axes, title: str, summary: dict[str, float], x: float, color: str) -> None:
    text = (
        f"{title}\n"
        f"upright<10deg: {summary['upright_ratio']:.3f}\n"
        f"max |theta|: {summary['max_abs_theta_deg']:.1f} deg\n"
        f"RMS voltage: {summary['rms_voltage']:.2f} V"
    )
    axis.text(
        x,
        0.97,
        text,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.86},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw-csv", type=Path, default=Path("results/hw_rl_report_run.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("models/sac_report_balance_centered_5v_100k.zip"))
    parser.add_argument("--algo", choices=["auto", "sac", "ppo"], default="sac")
    parser.add_argument("--output", type=Path, default=Path("results/report_figures/rl_balance_sim_vs_hw.png"))
    parser.add_argument("--voltage-limit", type=float, default=5.0)
    parser.add_argument("--arm-limit-deg", type=float, default=90.0)
    parser.add_argument("--obs-mode", choices=["base6", "speed7", "speed8"], default="base6")
    parser.add_argument("--action-filter-alpha", type=float, default=1.0)
    parser.add_argument("--voltage-slew-rate", type=float, default=0.0)
    parser.add_argument("--motor-dead-voltage", type=float, default=0.0)
    parser.add_argument("--residual-max", type=float, default=0.6)
    parser.add_argument("--blend-seconds", type=float, default=0.8)
    parser.add_argument(
        "--sim-controller",
        choices=["pure-rl", "residual"],
        default="pure-rl",
        help="Use pure RL in simulation, or mirror the hardware residual structure.",
    )
    parser.add_argument(
        "--start-alpha-deg",
        type=float,
        default=2.0,
        help="Start comparison at the first RL sample inside this upright window.",
    )
    parser.add_argument("--start-theta-dot", type=float, default=1.0)
    parser.add_argument("--start-alpha-dot", type=float, default=1.0)
    args = parser.parse_args()

    model = load_model(args.model_path, args.algo)
    hw = load_hardware_balance_phase(
        args.hw_csv,
        args.start_alpha_deg,
        args.start_theta_dot,
        args.start_alpha_dot,
    )
    sim = simulate_balance_phase(
        model,
        hw,
        args.voltage_limit,
        args.arm_limit_deg,
        args.obs_mode,
        args.action_filter_alpha,
        args.voltage_slew_rate,
        args.motor_dead_voltage,
        args.residual_max,
        args.blend_seconds,
        args.sim_controller,
    )

    hw_summary = summarize(hw, "voltage_applied")
    sim_summary = summarize(sim, "voltage_applied")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(sim["phase_time_s"], sim["alpha_deg"], label="simulation", linewidth=1.8)
    axes[0].plot(hw["phase_time_s"], hw["alpha_deg"], label="hardware", linewidth=1.2, alpha=0.9)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].axhline(10.0, color="black", linestyle="--", linewidth=0.7, alpha=0.35)
    axes[0].axhline(-10.0, color="black", linestyle="--", linewidth=0.7, alpha=0.35)
    axes[0].set_ylabel("alpha [deg]")
    axes[0].legend(loc="upper right")
    add_summary(axes[0], "Simulation", sim_summary, 0.01, "C0")
    add_summary(axes[0], "Hardware", hw_summary, 0.28, "C1")

    axes[1].plot(sim["phase_time_s"], sim["theta_deg"], linewidth=1.8)
    axes[1].plot(hw["phase_time_s"], hw["theta_deg"], linewidth=1.2, alpha=0.9)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_ylabel("theta [deg]")

    axes[2].plot(sim["phase_time_s"], sim["voltage_applied"], linewidth=1.8)
    axes[2].plot(hw["phase_time_s"], hw["voltage_applied"], linewidth=1.2, alpha=0.9)
    axes[2].set_ylabel("voltage [V]")

    axes[3].plot(sim["phase_time_s"], sim["alpha_dot_rad"], linewidth=1.8)
    axes[3].plot(hw["phase_time_s"], hw["alpha_dot_rad"], linewidth=1.2, alpha=0.9)
    axes[3].set_ylabel("alpha_dot [rad/s]")
    axes[3].set_xlabel("time since RL handoff [s]")

    for axis in axes:
        axis.grid(True, alpha=0.25)

    fig.suptitle("RL balance phase: simulation vs hardware")
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print("Simulation RL-phase summary:")
    for key, value in sim_summary.items():
        print(f"  {key}: {value:.3f}")
    print("Hardware RL-phase summary:")
    for key, value in hw_summary.items():
        print(f"  {key}: {value:.3f}")
    print(f"Wrote comparison plot to {args.output}")


if __name__ == "__main__":
    main()
