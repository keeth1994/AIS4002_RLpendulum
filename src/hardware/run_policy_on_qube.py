"""Skeleton for running a trained policy on the physical QUBE-Servo 2.

This file deliberately does not guess the course/lab Quanser API. Fill in the
read/write functions using your local installation instructions, then test with
the motor disabled or voltage-limited before attempting closed-loop control.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


def read_qube_state() -> np.ndarray:
    """Return [theta, alpha, theta_dot, alpha_dot] from the QUBE encoders."""
    raise NotImplementedError("Connect this to the course Quanser/QUBE API.")


def write_motor_voltage(voltage: float) -> None:
    """Send clipped motor voltage to the QUBE amplifier."""
    raise NotImplementedError("Connect this to the course Quanser/QUBE API.")


def stop_motor() -> None:
    """Put the motor in a safe zero-voltage state."""
    try:
        write_motor_voltage(0.0)
    except NotImplementedError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/ppo_rotary_pendulum.zip"))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--voltage-limit", type=float, default=2.0)
    args = parser.parse_args()

    model = PPO.load(args.model_path)
    dt = 1.0 / args.rate_hz
    end_time = time.monotonic() + args.duration

    try:
        while time.monotonic() < end_time:
            obs = read_qube_state().astype(np.float32)
            action, _ = model.predict(obs, deterministic=True)
            voltage = float(np.clip(action[0], -args.voltage_limit, args.voltage_limit))
            write_motor_voltage(voltage)
            time.sleep(dt)
    finally:
        stop_motor()


if __name__ == "__main__":
    main()
