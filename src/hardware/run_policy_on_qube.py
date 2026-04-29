"""Run a trained SB3 policy on the QUBE using the course serial API.

Start with ``--dry-run`` and a low ``--voltage-limit``. Reset the motor encoder
with the arm centered and reset the pendulum encoder while it hangs down.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

QUBE_DEG_TO_RAD = np.pi / 180.0


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


def read_qube_state(qube, previous_state: np.ndarray, dt: float) -> np.ndarray:
    theta = qube.getMotorAngle() * QUBE_DEG_TO_RAD

    # Encoder reset is done while hanging down. Repo convention is alpha=0
    # upright and alpha=pi hanging down.
    raw_from_down = qube.getPendulumAngle() * QUBE_DEG_TO_RAD
    alpha = wrap_angle(raw_from_down + np.pi)

    previous_theta, previous_alpha, _, _ = previous_state
    theta_dot = wrap_angle(theta - previous_theta) / dt
    alpha_dot = wrap_angle(alpha - previous_alpha) / dt
    return np.array([theta, alpha, theta_dot, alpha_dot], dtype=np.float32)


def build_policy_observation(state: np.ndarray, obs_dim: int) -> np.ndarray:
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
    raise ValueError(f"Unsupported policy observation dimension: {obs_dim}")


def apply_voltage_shaping(command: float, voltage_limit: float, voltage_gain: float, min_voltage: float) -> float:
    voltage = float(np.clip(command, -1.0, 1.0) * voltage_limit * voltage_gain)
    voltage = float(np.clip(voltage, -voltage_limit, voltage_limit))
    if abs(voltage) < 1e-6:
        return 0.0
    if abs(voltage) < min_voltage:
        return float(np.sign(voltage) * min_voltage)
    return voltage


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
    parser.add_argument("--motor-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    QUBE = load_qube_class(args.example_dir)
    model = load_model(args.model_path, args.algo)
    obs_dim = int(np.prod(model.observation_space.shape))

    qube = QUBE(args.port, args.baudrate)
    qube.setMotorVoltage(0.0)
    qube.setRGB(0, 0, 999)
    qube.update()

    print("Center the arm and let the pendulum hang down. Resetting encoders in 3 seconds.")
    time.sleep(3.0)
    qube.resetMotorEncoder()
    qube.resetPendulumEncoder()
    qube.update()
    time.sleep(0.2)

    dt_target = 1.0 / args.rate_hz
    previous_state = np.array([0.0, np.pi, 0.0, 0.0], dtype=np.float32)
    filtered_state = previous_state.copy()
    previous_time = time.monotonic()
    end_time = previous_time + args.duration
    print_every = max(1, int(args.rate_hz / 10.0))
    step = 0

    try:
        while time.monotonic() < end_time:
            now = time.monotonic()
            dt = now - previous_time
            if dt < dt_target:
                time.sleep(max(0.0, dt_target - dt))
                continue

            previous_time = now
            qube.update()
            state = read_qube_state(qube, previous_state, max(dt, 1e-6))
            previous_state = state
            filtered_state[:2] = state[:2]
            filtered_state[2:] = (
                (1.0 - args.velocity_filter) * filtered_state[2:]
                + args.velocity_filter * state[2:]
            )

            obs = build_policy_observation(filtered_state, obs_dim)
            action, _ = model.predict(obs, deterministic=True)
            command = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
            voltage = args.motor_sign * apply_voltage_shaping(
                command,
                args.voltage_limit,
                args.voltage_gain,
                args.min_voltage,
            )

            if args.dry_run:
                qube.setMotorVoltage(0.0)
            else:
                qube.setMotorVoltage(voltage)

            if step % print_every == 0:
                theta, alpha, theta_dot, alpha_dot = filtered_state
                print(
                    f"theta={theta:+.3f} alpha={alpha:+.3f} "
                    f"theta_dot={theta_dot:+.3f} alpha_dot={alpha_dot:+.3f} "
                    f"command={command:+.3f} voltage={voltage:+.3f} dry_run={args.dry_run}"
                )
            step += 1
    finally:
        qube.setMotorVoltage(0.0)
        qube.setRGB(999, 0, 0)
        qube.update()


if __name__ == "__main__":
    main()
