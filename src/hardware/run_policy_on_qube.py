"""Run a trained SB3 policy on the QUBE using the course serial API.

Start with ``--dry-run`` and a low ``--voltage-limit``. Reset the motor encoder
with the arm centered and reset the pendulum encoder while it hangs down.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

QUBE_DEG_TO_RAD = np.pi / 180.0
QUBE_ARM_LENGTH_M = 0.085
QUBE_PENDULUM_LENGTH_M = 0.1161
THETA_DOT_OBS_LIMIT = 30.0
ALPHA_DOT_OBS_LIMIT = 40.0


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def load_qube_class(example_dir: Path):
    sys.path.insert(0, str(example_dir.resolve()))
    from QUBE import QUBE  # type: ignore

    return QUBE


def load_model(model_path: Path, algo: str):
    if algo == "sac":
        return SAC.load(model_path, device="cpu")
    if algo == "ppo":
        return PPO.load(model_path, device="cpu")
    try:
        return SAC.load(model_path, device="cpu")
    except Exception:
        return PPO.load(model_path, device="cpu")


def read_qube_angles(
    qube,
    pendulum_reset_position: str = "down",
) -> tuple[float, float]:
    theta = qube.getMotorAngle() * QUBE_DEG_TO_RAD

    # Repo convention is alpha=0 upright and alpha=pi hanging down.
    raw_from_down = qube.getPendulumAngle() * QUBE_DEG_TO_RAD
    if pendulum_reset_position == "upright":
        alpha = wrap_angle(raw_from_down)
    else:
        alpha = wrap_angle(raw_from_down + np.pi)
    return float(theta), float(alpha)


def transform_qube_angles(
    theta: float,
    alpha: float,
    theta_sign: float,
    alpha_sign: float,
    center_trim_rad: float,
) -> tuple[float, float]:
    theta = theta_sign * theta - center_trim_rad
    alpha = wrap_angle(alpha_sign * alpha)
    return float(theta), float(alpha)


def read_qube_state(
    qube,
    previous_state: np.ndarray,
    dt: float,
    pendulum_reset_position: str = "down",
    theta_sign: float = 1.0,
    alpha_sign: float = 1.0,
    center_trim_rad: float = 0.0,
) -> np.ndarray:
    theta, alpha = read_qube_angles(qube, pendulum_reset_position)
    theta, alpha = transform_qube_angles(theta, alpha, theta_sign, alpha_sign, center_trim_rad)

    previous_theta, previous_alpha, _, _ = previous_state
    theta_dot = (theta - previous_theta) / dt
    alpha_dot = wrap_angle(alpha - previous_alpha) / dt
    return np.array([theta, alpha, theta_dot, alpha_dot], dtype=np.float32)


def clip_state_velocities(
    state: np.ndarray,
    theta_dot_limit: float = THETA_DOT_OBS_LIMIT,
    alpha_dot_limit: float = ALPHA_DOT_OBS_LIMIT,
) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).copy()
    state[2] = np.clip(state[2], -theta_dot_limit, theta_dot_limit)
    state[3] = np.clip(state[3], -alpha_dot_limit, alpha_dot_limit)
    return state


def initialize_state_from_hardware(
    qube,
    pendulum_reset_position: str,
    theta_sign: float,
    alpha_sign: float,
    center_trim_rad: float,
    settle_seconds: float,
    sample_hz: float,
) -> np.ndarray:
    """Read several zero-voltage samples and start with measured angles and zero velocity."""
    qube.setMotorVoltage(0.0)
    samples = max(3, int(max(0.0, settle_seconds) * max(sample_hz, 1.0)))
    sleep_s = 1.0 / max(sample_hz, 1.0)
    theta = 0.0
    alpha = 0.0 if pendulum_reset_position == "upright" else np.pi
    for _ in range(samples):
        qube.update()
        theta_raw, alpha_raw = read_qube_angles(qube, pendulum_reset_position)
        theta, alpha = transform_qube_angles(
            theta_raw,
            alpha_raw,
            theta_sign,
            alpha_sign,
            center_trim_rad,
        )
        time.sleep(sleep_s)
    return np.array([theta, alpha, 0.0, 0.0], dtype=np.float32)


def build_policy_observation(
    state: np.ndarray,
    obs_dim: int,
    last_voltage: float = 0.0,
    voltage_limit: float = 1.0,
) -> np.ndarray:
    theta, alpha, theta_dot, alpha_dot = np.asarray(state, dtype=np.float32).reshape(4)
    if obs_dim == 4:
        return np.array([theta, alpha, theta_dot, alpha_dot], dtype=np.float32)
    if obs_dim == 6:
        return np.array(
            [
                np.sin(theta),
                np.cos(theta),
                np.sin(alpha),
                np.cos(alpha),
                theta_dot,
                alpha_dot,
            ],
            dtype=np.float32,
        )
    if obs_dim == 7:
        pendulum_speed = np.sqrt(
            (
                -QUBE_ARM_LENGTH_M * np.sin(theta) * theta_dot
                + QUBE_PENDULUM_LENGTH_M * np.cos(alpha) * alpha_dot
            )
            ** 2
            + (
                QUBE_ARM_LENGTH_M * np.cos(theta) * theta_dot
                + QUBE_PENDULUM_LENGTH_M * np.sin(alpha) * alpha_dot
            )
            ** 2
        )
        return np.array(
            [
                np.sin(theta),
                np.cos(theta),
                np.sin(alpha),
                np.cos(alpha),
                theta_dot,
                alpha_dot,
                pendulum_speed,
            ],
            dtype=np.float32,
        )
    if obs_dim == 8:
        pendulum_speed = np.sqrt(
            (
                -QUBE_ARM_LENGTH_M * np.sin(theta) * theta_dot
                + QUBE_PENDULUM_LENGTH_M * np.cos(alpha) * alpha_dot
            )
            ** 2
            + (
                QUBE_ARM_LENGTH_M * np.cos(theta) * theta_dot
                + QUBE_PENDULUM_LENGTH_M * np.sin(alpha) * alpha_dot
            )
            ** 2
        )
        return np.array(
            [
                np.sin(theta),
                np.cos(theta),
                np.sin(alpha),
                np.cos(alpha),
                theta_dot,
                alpha_dot,
                pendulum_speed,
                last_voltage / max(voltage_limit, 1e-6),
            ],
            dtype=np.float32,
        )
    raise ValueError(f"Unsupported policy observation dimension: {obs_dim}")


def apply_voltage_shaping(command: float, voltage_limit: float, voltage_gain: float, min_voltage: float) -> float:
    voltage = float(np.clip(command, -1.0, 1.0) * voltage_limit * voltage_gain)
    voltage = float(np.clip(voltage, -voltage_limit, voltage_limit))
    if abs(voltage) < 1e-6:
        return 0.0
    if abs(voltage) < min_voltage:
        return float(np.sign(voltage) * min_voltage)
    return voltage


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def apply_voltage_dynamics(
    target_voltage: float,
    previous_voltage: float,
    dt: float,
    filter_alpha: float,
    slew_rate: float,
    voltage_limit: float,
) -> float:
    voltage = previous_voltage + clip(filter_alpha, 0.0, 1.0) * (target_voltage - previous_voltage)
    if slew_rate > 0.0:
        max_delta = slew_rate * dt
        voltage = clip(voltage, previous_voltage - max_delta, previous_voltage + max_delta)
    return clip(voltage, -voltage_limit, voltage_limit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/sac_qube_swingup_tuned_reward_100k.zip"))
    parser.add_argument("--algo", choices=["auto", "sac", "ppo"], default="auto")
    parser.add_argument("--port", type=str, default="COM3")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--example-dir", type=Path, default=Path("EXAMPLE_CODE"))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--voltage-limit", type=float, default=8.0)
    parser.add_argument("--voltage-gain", type=float, default=4.0)
    parser.add_argument("--min-voltage", type=float, default=1.2)
    parser.add_argument("--velocity-filter", type=float, default=0.15)
    parser.add_argument("--startup-settle-seconds", type=float, default=0.25)
    parser.add_argument("--zero-velocity-seconds", type=float, default=0.15)
    parser.add_argument("--voltage-filter-alpha", type=float, default=1.0)
    parser.add_argument("--voltage-slew-rate", type=float, default=0.0)
    parser.add_argument("--motor-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--theta-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--alpha-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--center-trim-deg", type=float, default=0.0)
    parser.add_argument("--arm-soft-limit-deg", type=float, default=90.0)
    parser.add_argument("--arm-hard-stop-deg", type=float, default=130.0)
    parser.add_argument("--arm-limit-brake-gain", type=float, default=10.0)
    parser.add_argument("--arm-limit-rate-gain", type=float, default=0.5)
    parser.add_argument("--startup-kick-voltage", type=float, default=0.0)
    parser.add_argument("--startup-kick-seconds", type=float, default=0.0)
    parser.add_argument("--startup-kick-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--pendulum-reset-position", choices=["down", "upright"], default="down")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    QUBE = load_qube_class(args.example_dir)
    model = load_model(args.model_path, args.algo)
    obs_dim = int(np.prod(model.observation_space.shape))

    qube = QUBE(args.port, args.baudrate)
    qube.setMotorVoltage(0.0)
    qube.setRGB(0, 0, 999)
    qube.update()

    if args.pendulum_reset_position == "upright":
        print("Center the arm and hold the pendulum upright. Resetting encoders in 3 seconds.")
    else:
        print("Center the arm and let the pendulum hang down. Resetting encoders in 3 seconds.")
    time.sleep(3.0)
    if hasattr(qube, "reset_buffers"):
        qube.reset_buffers()
    qube.resetMotorEncoder()
    qube.resetPendulumEncoder()
    qube.update()
    if hasattr(qube, "reset_buffers"):
        qube.reset_buffers()
    time.sleep(0.2)
    if args.startup_kick_voltage > 0.0 and args.startup_kick_seconds > 0.0:
        kick_voltage = args.startup_kick_sign * min(args.startup_kick_voltage, args.voltage_limit)
        print(
            f"Applying startup kick: {kick_voltage:+.2f} V for "
            f"{args.startup_kick_seconds:.2f} s"
        )
        if args.dry_run:
            qube.setMotorVoltage(0.0)
        else:
            qube.setMotorVoltage(kick_voltage)
        time.sleep(args.startup_kick_seconds)
        qube.setMotorVoltage(0.0)
        qube.update()
        time.sleep(0.05)

    dt_target = 1.0 / args.rate_hz
    center_trim_rad = args.center_trim_deg * QUBE_DEG_TO_RAD
    previous_state = initialize_state_from_hardware(
        qube,
        args.pendulum_reset_position,
        args.theta_sign,
        args.alpha_sign,
        center_trim_rad,
        args.startup_settle_seconds,
        args.rate_hz,
    )
    filtered_state = previous_state.copy()
    previous_voltage = 0.0
    previous_time = time.monotonic()
    control_start_time = previous_time
    end_time = previous_time + args.duration
    print_every = max(1, int(args.rate_hz / 10.0))
    step = 0
    csv_file = None
    writer = None
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args.csv.open("w", newline="")
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "time_s",
                "theta_rad",
                "alpha_rad",
                "theta_dot_rad",
                "alpha_dot_rad",
                "command",
                "voltage_cmd",
                "voltage_applied",
                "safety_limited",
                "motor_current",
                "motor_rpm",
                "dry_run",
            ],
        )
        writer.writeheader()

    try:
        while time.monotonic() < end_time:
            now = time.monotonic()
            dt = now - previous_time
            if dt < dt_target:
                time.sleep(max(0.0, dt_target - dt))
                continue

            previous_time = now
            qube.update()
            state = read_qube_state(
                qube,
                previous_state,
                max(dt, 1e-6),
                args.pendulum_reset_position,
                args.theta_sign,
                args.alpha_sign,
                center_trim_rad,
            )
            state = clip_state_velocities(state)
            previous_state = state
            filtered_state[:2] = state[:2]
            filtered_state[2:] = (
                (1.0 - args.velocity_filter) * filtered_state[2:]
                + args.velocity_filter * state[2:]
            )
            filtered_state = clip_state_velocities(filtered_state)
            if now - control_start_time < args.zero_velocity_seconds:
                filtered_state[2:] = 0.0

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
            command = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
            voltage_cmd = args.motor_sign * apply_voltage_shaping(
                command,
                args.voltage_limit,
                args.voltage_gain,
                args.min_voltage,
            )
            voltage = apply_voltage_dynamics(
                voltage_cmd,
                previous_voltage,
                max(dt, 1e-6),
                args.voltage_filter_alpha,
                args.voltage_slew_rate,
                args.voltage_limit,
            )
            safety_limited = 0
            theta, alpha, theta_dot, alpha_dot = filtered_state
            theta_deg = float(theta / QUBE_DEG_TO_RAD)
            if args.arm_soft_limit_deg > 0.0:
                if theta_deg > args.arm_soft_limit_deg:
                    over_rad = (theta_deg - args.arm_soft_limit_deg) * QUBE_DEG_TO_RAD
                    brake_voltage = -(
                        args.arm_limit_brake_gain * over_rad
                        + args.arm_limit_rate_gain * float(theta_dot)
                    )
                    voltage = min(voltage, brake_voltage)
                    safety_limited = 1
                elif theta_deg < -args.arm_soft_limit_deg:
                    over_rad = (theta_deg + args.arm_soft_limit_deg) * QUBE_DEG_TO_RAD
                    brake_voltage = -(
                        args.arm_limit_brake_gain * over_rad
                        + args.arm_limit_rate_gain * float(theta_dot)
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

            if args.dry_run:
                qube.setMotorVoltage(0.0)
            else:
                qube.setMotorVoltage(voltage)
            previous_voltage = 0.0 if args.dry_run else voltage

            if step % print_every == 0:
                print(
                    f"theta={theta:+.3f} alpha={alpha:+.3f} "
                    f"theta_dot={theta_dot:+.3f} alpha_dot={alpha_dot:+.3f} "
                    f"command={command:+.3f} voltage={voltage:+.3f} "
                    f"safety={safety_limited} dry_run={args.dry_run}"
                )
            if writer is not None:
                writer.writerow(
                    {
                        "time_s": round(now - (end_time - args.duration), 6),
                        "theta_rad": float(theta),
                        "alpha_rad": float(alpha),
                        "theta_dot_rad": float(theta_dot),
                        "alpha_dot_rad": float(alpha_dot),
                        "command": command,
                        "voltage_cmd": voltage_cmd,
                        "voltage_applied": 0.0 if args.dry_run else voltage,
                        "safety_limited": safety_limited,
                        "motor_current": qube.getMotorCurrent(),
                        "motor_rpm": qube.getMotorRPM(),
                        "dry_run": int(args.dry_run),
                    }
                )
                csv_file.flush()
            step += 1
    finally:
        qube.setMotorVoltage(0.0)
        qube.setRGB(999, 0, 0)
        qube.update()
        if csv_file is not None:
            csv_file.close()
            print(f"Wrote policy hardware log to {args.csv}")


if __name__ == "__main__":
    main()
