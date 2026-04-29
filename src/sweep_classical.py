"""Sweep classical energy swing-up controller parameters and save aggregate metrics."""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np

from src.evaluate_classical import run_baseline


def parse_float_list(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--arm-limit-deg", type=float, default=90.0)
    parser.add_argument("--initial-perturbation", type=float, default=0.25)
    parser.add_argument("--voltage-limit", type=float, default=10.0)
    parser.add_argument("--energy-gains", type=str, default="36,60,80,100")
    parser.add_argument("--arm-centering-gains", type=str, default="0.1,0.2,0.35")
    parser.add_argument("--arm-centering-rate-gains", type=str, default="0.06")
    parser.add_argument("--balance-voltage-limits", type=str, default="5.0")
    parser.add_argument("--swingup-voltage-limits", type=str, default="10.0")
    parser.add_argument("--swingup-accel-limits", type=str, default="6.0")
    parser.add_argument("--swingup-frequency-hz", type=float, default=1.5)
    parser.add_argument("--swingup-amplitude", type=float, default=10.0)
    parser.add_argument("--render-style", choices=["qube", "cartpole"], default="qube")
    parser.add_argument("--csv", type=Path, default=Path("results/classical_sweep.csv"))
    args = parser.parse_args()

    seeds = [int(value) for value in parse_float_list(args.seeds)]
    energy_gains = parse_float_list(args.energy_gains)
    arm_centering_gains = parse_float_list(args.arm_centering_gains)
    arm_centering_rate_gains = parse_float_list(args.arm_centering_rate_gains)
    balance_voltage_limits = parse_float_list(args.balance_voltage_limits)
    swingup_voltage_limits = parse_float_list(args.swingup_voltage_limits)
    swingup_accel_limits = parse_float_list(args.swingup_accel_limits)

    rows: list[dict] = []
    parameter_grid = product(
        energy_gains,
        arm_centering_gains,
        arm_centering_rate_gains,
        balance_voltage_limits,
        swingup_voltage_limits,
        swingup_accel_limits,
    )

    for (
        energy_gain,
        arm_centering_gain,
        arm_centering_rate_gain,
        balance_voltage_limit,
        swingup_voltage_limit,
        swingup_accel_limit,
    ) in parameter_grid:
        metrics = []
        for seed in seeds:
            metric = run_baseline(
                steps=args.steps,
                seed=seed,
                video_path=None,
                plot_path=None,
                csv_path=None,
                arm_limit_deg=args.arm_limit_deg,
                open_loop=False,
                initial_perturbation=args.initial_perturbation,
                voltage_limit=args.voltage_limit,
                render_style=args.render_style,
                controller_kwargs={
                    "energy_gain": energy_gain,
                    "arm_centering_gain": arm_centering_gain,
                    "arm_centering_rate_gain": arm_centering_rate_gain,
                    "balance_voltage_limit": balance_voltage_limit,
                    "swingup_voltage_limit": swingup_voltage_limit,
                    "swingup_frequency_hz": args.swingup_frequency_hz,
                    "swingup_amplitude": args.swingup_amplitude,
                    "swingup_mode": "oscillatory",
                    "swingup_accel_limit": swingup_accel_limit,
                },
                reset_mode="down",
                reward_mode="recovery",
                terminate_on_arm_limit=False,
                soft_arm_limit=True,
            )
            metrics.append(metric)

        row = {
            "energy_gain": energy_gain,
            "arm_centering_gain": arm_centering_gain,
            "arm_centering_rate_gain": arm_centering_rate_gain,
            "balance_voltage_limit": balance_voltage_limit,
            "swingup_voltage_limit": swingup_voltage_limit,
            "swingup_accel_limit": swingup_accel_limit,
            "mean_upright_ratio": float(np.mean([m["upright_ratio"] for m in metrics])),
            "mean_closest_alpha_deg": float(np.mean([m["closest_alpha_deg"] for m in metrics])),
            "best_closest_alpha_deg": float(np.min([m["closest_alpha_deg"] for m in metrics])),
            "mean_max_abs_theta_deg": float(np.mean([m["max_abs_theta_deg"] for m in metrics])),
            "arm_limit_count": int(sum(m["termination_reason"] == "arm_limit" for m in metrics)),
            "time_limit_count": int(sum(m["termination_reason"] == "time_limit" for m in metrics)),
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["mean_closest_alpha_deg"], r["arm_limit_count"], -r["mean_upright_ratio"]))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote sweep results to {args.csv}")
    print("Top 10 parameter sets:")
    for row in rows[:10]:
        print(
            "  "
            f"energy_gain={row['energy_gain']:.3f}, "
            f"arm_centering_gain={row['arm_centering_gain']:.3f}, "
            f"arm_centering_rate_gain={row['arm_centering_rate_gain']:.3f}, "
            f"balance_voltage_limit={row['balance_voltage_limit']:.3f}, "
            f"swingup_voltage_limit={row['swingup_voltage_limit']:.3f}, "
            f"swingup_accel_limit={row['swingup_accel_limit']:.3f}, "
            f"mean_closest_alpha_deg={row['mean_closest_alpha_deg']:.3f}, "
            f"best_closest_alpha_deg={row['best_closest_alpha_deg']:.3f}, "
            f"mean_upright_ratio={row['mean_upright_ratio']:.3f}, "
            f"arm_limit_count={row['arm_limit_count']}"
        )


if __name__ == "__main__":
    main()
