"""Summarize QUBE hardware CSV logs for reports."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def longest_segment(rows: list[dict[str, str]], predicate) -> tuple[int, str | None, str | None]:
    best_len = 0
    best_start = None
    best_end = None
    current_start = None
    for index, row in enumerate(rows):
        if predicate(row):
            if current_start is None:
                current_start = index
        elif current_start is not None:
            length = index - current_start
            if length > best_len:
                best_len = length
                best_start = rows[current_start]["time_s"]
                best_end = rows[index - 1]["time_s"]
            current_start = None
    if current_start is not None:
        length = len(rows) - current_start
        if length > best_len:
            best_len = length
            best_start = rows[current_start]["time_s"]
            best_end = rows[-1]["time_s"]
    return best_len, best_start, best_end


def mean_abs(values: list[float]) -> float:
    return sum(abs(value) for value in values) / max(len(values), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--upright-deg", type=float, default=10.0)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv_path.open()))
    if not rows:
        raise SystemExit(f"No rows found in {args.csv_path}")

    theta = [float(row["theta_deg"]) for row in rows]
    alpha = [float(row["alpha_deg"]) for row in rows]
    voltage = [float(row["voltage_applied"]) for row in rows]
    modes = Counter(row.get("mode", "") for row in rows)
    upright_count = sum(abs(value) < args.upright_deg for value in alpha)
    longest_upright = longest_segment(rows, lambda row: abs(float(row["alpha_deg"])) < args.upright_deg)
    balance_modes = {"example_balance", "rl_balance", "rl_residual"}
    longest_balance = longest_segment(rows, lambda row: row.get("mode", "") in balance_modes)

    print(f"log: {args.csv_path}")
    print(f"rows: {len(rows)}")
    print(f"duration_s: {float(rows[-1]['time_s']):.3f}")
    print(f"modes: {dict(modes)}")
    print(f"upright_ratio_abs_alpha_lt_{args.upright_deg:g}_deg: {upright_count / len(rows):.3f}")
    print(f"longest_upright_samples: {longest_upright[0]}")
    print(f"longest_upright_time_s: {longest_upright[1]} to {longest_upright[2]}")
    print(f"longest_balance_mode_samples: {longest_balance[0]}")
    print(f"longest_balance_mode_time_s: {longest_balance[1]} to {longest_balance[2]}")
    print(f"closest_alpha_to_upright_deg: {min(abs(value) for value in alpha):.3f}")
    print(f"max_abs_theta_deg: {max(abs(value) for value in theta):.3f}")
    print(f"mean_abs_theta_deg: {mean_abs(theta):.3f}")
    print(f"mean_abs_alpha_deg: {mean_abs(alpha):.3f}")
    print(f"mean_abs_voltage_v: {mean_abs(voltage):.3f}")
    print(
        "final_state: "
        f"mode={rows[-1].get('mode', '')}, "
        f"theta_deg={theta[-1]:+.3f}, "
        f"alpha_deg={alpha[-1]:+.3f}, "
        f"voltage_v={voltage[-1]:+.3f}"
    )


if __name__ == "__main__":
    main()
