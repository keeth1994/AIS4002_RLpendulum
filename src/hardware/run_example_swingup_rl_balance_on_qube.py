"""Run course-example swing-up with an RL balance handoff on the QUBE.

This follows the practical report-style workflow:
- use the known energy swing-up controller to reach the capture region
- use an SB3 policy only for upright balance
- fall back to swing-up if the policy loses the pendulum
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

from src.hardware.run_classical_on_qube import ExampleEnergyController, clip
from src.hardware.run_policy_on_qube import (
    build_policy_observation,
    clip_state_velocities,
    load_model,
    read_qube_angles,
    transform_qube_angles,
    wrap_angle,
)

QUBE_DEG_TO_RAD = math.pi / 180.0

REPORT_RESIDUAL_PRESET = {
    "rate_hz": 300.0,
    "startup_settle_seconds": 0.5,
    "voltage_limit": 5.0,
    "rl_voltage_gain": 0.8,
    "rl_min_voltage": 0.0,
    "rl_blend_seconds": 0.8,
    "rl_residual_max": 0.6,
    "motor_sign": 1.0,
    "velocity_filter": 0.08,
    "handoff_deg": 18.0,
    "handoff_theta_deg": 70.0,
    "handoff_theta_dot": 6.0,
    "handoff_alpha_dot": 12.0,
    "handoff_stable_seconds": 0.08,
    "exit_deg": 35.0,
    "exit_theta_deg": 90.0,
    "exit_theta_dot": 10.0,
    "exit_alpha_dot": 18.0,
    "example_u_max": 1.6,
    "example_balance_range_deg": 30.0,
    "example_startup_kick_voltage": 1.5,
    "example_startup_kick_seconds": 0.15,
    "arm_soft_limit_deg": 90.0,
    "arm_hard_stop_deg": 130.0,
}


def load_qube_class(example_dir: Path):
    sys.path.insert(0, str(example_dir.resolve()))
    from QUBE import QUBE  # type: ignore

    return QUBE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=["custom", "report-residual"],
        default="custom",
        help="Use a named hardware configuration. 'report-residual' is the validated classical swing-up + bounded RL residual setup.",
    )
    parser.add_argument("--port", type=str, default="COM13")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--example-dir", type=Path, default=Path("EXAMPLE_CODE"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--algo", choices=["auto", "sac", "ppo"], default="sac")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--rate-hz", type=float, default=300.0)
    parser.add_argument("--startup-delay", type=float, default=3.0)
    parser.add_argument("--startup-settle-seconds", type=float, default=0.35)
    parser.add_argument("--voltage-limit", type=float, default=5.0)
    parser.add_argument("--rl-voltage-gain", type=float, default=1.0)
    parser.add_argument("--rl-min-voltage", type=float, default=0.0)
    parser.add_argument("--rl-blend-seconds", type=float, default=1.0)
    parser.add_argument(
        "--rl-residual-max",
        type=float,
        default=0.0,
        help=(
            "If >0, keep the example balance controller as the baseline and "
            "let RL add only this many volts of bounded residual correction."
        ),
    )
    parser.add_argument("--motor-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--theta-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--alpha-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--center-trim-deg", type=float, default=0.0)
    parser.add_argument("--velocity-filter", type=float, default=0.15)
    parser.add_argument("--handoff-deg", type=float, default=12.0)
    parser.add_argument("--exit-deg", type=float, default=22.0)
    parser.add_argument("--handoff-theta-deg", type=float, default=35.0)
    parser.add_argument("--handoff-theta-dot", type=float, default=3.5)
    parser.add_argument("--handoff-alpha-dot", type=float, default=5.0)
    parser.add_argument("--handoff-stable-seconds", type=float, default=0.25)
    parser.add_argument("--exit-theta-deg", type=float, default=55.0)
    parser.add_argument("--exit-theta-dot", type=float, default=5.0)
    parser.add_argument("--exit-alpha-dot", type=float, default=8.0)
    parser.add_argument("--example-u-max", type=float, default=1.6)
    parser.add_argument("--example-balance-range-deg", type=float, default=30.0)
    parser.add_argument("--example-startup-kick-voltage", type=float, default=1.5)
    parser.add_argument("--example-startup-kick-seconds", type=float, default=0.15)
    parser.add_argument("--arm-soft-limit-deg", type=float, default=90.0)
    parser.add_argument("--arm-hard-stop-deg", type=float, default=130.0)
    parser.add_argument("--arm-limit-brake-gain", type=float, default=10.0)
    parser.add_argument("--arm-limit-rate-gain", type=float, default=0.5)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.preset == "report-residual":
        for name, value in REPORT_RESIDUAL_PRESET.items():
            setattr(args, name, value)

    QUBE = load_qube_class(args.example_dir)
    model = load_model(args.model_path, args.algo)
    obs_dim = int(np.prod(model.observation_space.shape))
    controller = ExampleEnergyController(
        dt=1.0 / args.rate_hz,
        swingup_u_max=args.example_u_max,
        balance_range_deg=args.example_balance_range_deg,
        startup_kick_voltage=args.example_startup_kick_voltage,
        startup_kick_seconds=args.example_startup_kick_seconds,
    )

    csv_path = args.csv
    if csv_path is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = Path(f"results/qube_example_rl_balance_{stamp}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    qube = QUBE(args.port, args.baudrate)
    qube.setMotorVoltage(0.0)
    qube.setRGB(0, 0, 999)
    qube.update()

    print("Center the arm and let the pendulum hang down. Resetting encoders in 3 seconds.")
    time.sleep(args.startup_delay)
    if hasattr(qube, "reset_buffers"):
        qube.reset_buffers()
    qube.resetMotorEncoder()
    qube.resetPendulumEncoder()
    qube.update()
    if hasattr(qube, "reset_buffers"):
        qube.reset_buffers()
    time.sleep(0.2)

    center_trim_rad = args.center_trim_deg * QUBE_DEG_TO_RAD
    settle_samples = max(3, int(max(args.startup_settle_seconds, 0.0) * max(args.rate_hz, 1.0)))
    sleep_s = 1.0 / max(args.rate_hz, 1.0)
    theta = 0.0
    alpha = np.pi
    for _ in range(settle_samples):
        qube.setMotorVoltage(0.0)
        qube.update()
        theta_raw, alpha_raw = read_qube_angles(qube, "down")
        theta, alpha = transform_qube_angles(
            theta_raw,
            alpha_raw,
            args.theta_sign,
            args.alpha_sign,
            center_trim_rad,
        )
        time.sleep(sleep_s)

    previous_state = np.array([theta, alpha, 0.0, 0.0], dtype=np.float32)
    filtered_state = previous_state.copy()
    previous_voltage = 0.0
    rl_mode = False
    handoff_ready_s = 0.0
    handoff_start_s: float | None = None
    dt_target = 1.0 / args.rate_hz
    next_tick = time.monotonic()
    start = next_tick
    last_time = start
    end_time = start + args.duration
    print_every = max(1, int(args.rate_hz / 10.0))
    step = 0

    fieldnames = [
        "time_s",
        "mode",
        "theta_deg",
        "alpha_deg",
        "theta_rad",
        "alpha_rad",
        "theta_dot_rad",
        "alpha_dot_rad",
        "rl_command",
        "voltage_cmd",
        "voltage_applied",
        "safety_limited",
        "motor_current",
        "motor_rpm",
        "dry_run",
    ]

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        try:
            while time.monotonic() < end_time:
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(next_tick - now)
                    continue
                next_tick += dt_target
                dt = max(now - last_time, 1e-6)
                last_time = now

                qube.update()
                theta_deg = args.theta_sign * qube.getMotorAngle() - args.center_trim_deg
                raw_alpha_deg = args.alpha_sign * qube.getPendulumAngle()
                alpha_deg = raw_alpha_deg - 180.0 if raw_alpha_deg > 0.0 else raw_alpha_deg + 180.0
                while alpha_deg > 180.0:
                    alpha_deg -= 360.0
                while alpha_deg < -180.0:
                    alpha_deg += 360.0

                theta = theta_deg * QUBE_DEG_TO_RAD
                alpha = wrap_angle(alpha_deg * QUBE_DEG_TO_RAD)
                theta_dot = (theta - float(previous_state[0])) / dt
                alpha_dot = wrap_angle(alpha - float(previous_state[1])) / dt
                state = np.array([theta, alpha, theta_dot, alpha_dot], dtype=np.float32)
                state = clip_state_velocities(state)
                previous_state = state
                filtered_state[:2] = state[:2]
                filtered_state[2:] = (
                    (1.0 - args.velocity_filter) * filtered_state[2:]
                    + args.velocity_filter * state[2:]
                )
                filtered_state = clip_state_velocities(filtered_state)

                abs_alpha_deg = abs(alpha_deg)
                theta_dot_abs = abs(float(filtered_state[2]))
                alpha_dot_abs = abs(float(filtered_state[3]))
                if rl_mode and (
                    abs_alpha_deg >= args.exit_deg
                    or abs(theta_deg) >= args.exit_theta_deg
                    or theta_dot_abs >= args.exit_theta_dot
                    or alpha_dot_abs >= args.exit_alpha_dot
                ):
                    rl_mode = False
                    previous_voltage = 0.0
                    handoff_start_s = None

                rl_command = 0.0
                output = controller.step(theta_deg, raw_alpha_deg, dt)
                classical_voltage_cmd = float(output["voltage"])
                mode = str(output["mode"])
                if not rl_mode:
                    voltage_cmd = classical_voltage_cmd
                    handoff_ready = (
                        mode == "example_balance"
                        and abs_alpha_deg <= args.handoff_deg
                        and abs(theta_deg) <= args.handoff_theta_deg
                        and theta_dot_abs <= args.handoff_theta_dot
                        and alpha_dot_abs <= args.handoff_alpha_dot
                    )
                    handoff_ready_s = handoff_ready_s + dt if handoff_ready else 0.0
                    if handoff_ready_s >= args.handoff_stable_seconds:
                        rl_mode = True
                        previous_voltage = 0.0
                        handoff_start_s = now - start

                if rl_mode:
                    obs = build_policy_observation(
                        filtered_state,
                        obs_dim,
                        previous_voltage,
                        args.voltage_limit,
                    )
                    obs = np.clip(
                        obs,
                        model.observation_space.low,
                        model.observation_space.high,
                    ).astype(np.float32)
                    action, _ = model.predict(obs, deterministic=True)
                    rl_command = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
                    rl_voltage_cmd = (
                        args.motor_sign
                        * clip(rl_command, -1.0, 1.0)
                        * args.voltage_limit
                        * args.rl_voltage_gain
                    )
                    rl_voltage_cmd = clip(rl_voltage_cmd, -args.voltage_limit, args.voltage_limit)
                    if abs(rl_voltage_cmd) < args.rl_min_voltage and abs(rl_voltage_cmd) > 1e-6:
                        rl_voltage_cmd = math.copysign(args.rl_min_voltage, rl_voltage_cmd)

                    blend_elapsed = (now - start) - (handoff_start_s or (now - start))
                    blend = clip(
                        blend_elapsed / max(args.rl_blend_seconds, 1e-6),
                        0.0,
                        1.0,
                    )
                    if args.rl_residual_max > 0.0:
                        residual = clip(
                            rl_voltage_cmd - classical_voltage_cmd,
                            -args.rl_residual_max,
                            args.rl_residual_max,
                        )
                        voltage_cmd = classical_voltage_cmd + blend * residual
                        mode = "rl_residual"
                    else:
                        voltage_cmd = (1.0 - blend) * classical_voltage_cmd + blend * rl_voltage_cmd
                        mode = "rl_balance"
                    handoff_ready_s = args.handoff_stable_seconds

                voltage = clip(voltage_cmd, -args.voltage_limit, args.voltage_limit)
                safety_limited = 0
                theta_dot_rad = float(filtered_state[2])
                if args.arm_soft_limit_deg > 0.0:
                    if theta_deg > args.arm_soft_limit_deg:
                        over_rad = (theta_deg - args.arm_soft_limit_deg) * QUBE_DEG_TO_RAD
                        brake_voltage = -(
                            args.arm_limit_brake_gain * over_rad
                            + args.arm_limit_rate_gain * theta_dot_rad
                        )
                        voltage = min(voltage, brake_voltage)
                        safety_limited = 1
                    elif theta_deg < -args.arm_soft_limit_deg:
                        over_rad = (theta_deg + args.arm_soft_limit_deg) * QUBE_DEG_TO_RAD
                        brake_voltage = -(
                            args.arm_limit_brake_gain * over_rad
                            + args.arm_limit_rate_gain * theta_dot_rad
                        )
                        voltage = max(voltage, brake_voltage)
                        safety_limited = 1
                    voltage = clip(voltage, -args.voltage_limit, args.voltage_limit)

                if args.arm_hard_stop_deg > 0.0 and abs(theta_deg) > args.arm_hard_stop_deg:
                    print(
                        f"Arm hard stop: theta={theta_deg:+.2f} deg exceeds "
                        f"{args.arm_hard_stop_deg:.2f} deg. Stopping motor."
                    )
                    qube.setMotorVoltage(0.0)
                    break

                qube.setMotorVoltage(0.0 if args.dry_run else voltage)
                previous_voltage = 0.0 if args.dry_run else voltage

                row = {
                    "time_s": round(now - start, 6),
                    "mode": mode,
                    "theta_deg": theta_deg,
                    "alpha_deg": alpha_deg,
                    "theta_rad": theta,
                    "alpha_rad": alpha,
                    "theta_dot_rad": float(filtered_state[2]),
                    "alpha_dot_rad": float(filtered_state[3]),
                    "rl_command": rl_command,
                    "voltage_cmd": voltage_cmd,
                    "voltage_applied": 0.0 if args.dry_run else voltage,
                    "safety_limited": safety_limited,
                    "motor_current": qube.getMotorCurrent(),
                    "motor_rpm": qube.getMotorRPM(),
                    "dry_run": int(args.dry_run),
                }
                writer.writerow(row)
                file.flush()

                if step % print_every == 0:
                    print(
                        f"{row['time_s']:7.3f}s mode={mode} "
                        f"theta={theta_deg:+7.2f}deg alpha={alpha_deg:+7.2f}deg "
                        f"voltage={row['voltage_applied']:+6.3f}V"
                    )
                step += 1
        finally:
            qube.setMotorVoltage(0.0)
            qube.setRGB(999, 0, 0)
            qube.update()

    print(f"Wrote hybrid RL-balance hardware log to {csv_path}")


if __name__ == "__main__":
    main()
