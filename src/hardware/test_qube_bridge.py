"""Minimal smoke test for the Python-controlled QUBE serial bridge.

Use this after flashing QUBE/examples/Python_Serial/Python_Serial.ino.
It verifies that:
1. the PC can talk to the Teensy bridge,
2. encoder readings change when you move the mechanism by hand,
3. the motor responds to a small commanded voltage pulse.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def load_qube_class(example_dir: Path):
    sys.path.insert(0, str(example_dir.resolve()))
    from QUBE import QUBE  # type: ignore

    return QUBE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="COM3")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--example-dir", type=Path, default=Path("EXAMPLE_CODE"))
    parser.add_argument("--startup-delay", type=float, default=3.0)
    parser.add_argument("--sense-seconds", type=float, default=5.0)
    parser.add_argument("--pulse-voltage", type=float, default=1.0)
    parser.add_argument("--pulse-seconds", type=float, default=1.0)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--no-pulse", action="store_true")
    args = parser.parse_args()

    QUBE = load_qube_class(args.example_dir)
    qube = QUBE(args.port, args.baudrate)
    dt = 1.0 / args.rate_hz

    qube.setMotorVoltage(0.0)
    qube.setRGB(0, 0, 999)
    qube.update()

    print("Waiting for Teensy reboot and serial bridge startup...")
    time.sleep(args.startup_delay)
    qube.reset_buffers()

    print("Resetting encoders. Put arm near center and pendulum hanging down.")
    qube.resetMotorEncoder()
    qube.resetPendulumEncoder()
    qube.update()
    time.sleep(0.2)

    print(f"Sensing for {args.sense_seconds:.1f}s. Move the arm and pendulum gently by hand.")
    sense_end = time.monotonic() + args.sense_seconds
    while time.monotonic() < sense_end:
        try:
            qube.update()
        except TimeoutError as exc:
            print(f"Bridge timeout while sensing: {exc}")
            break
        print(
            f"theta={qube.getMotorAngle():+8.2f} deg  "
            f"alpha_raw={qube.getPendulumAngle():+8.2f} deg  "
            f"rpm={qube.getMotorRPM():+7.1f}  "
            f"current={qube.getMotorCurrent():+7.1f} mA"
        )
        time.sleep(dt)

    if args.no_pulse:
        qube.setMotorVoltage(0.0)
        qube.update()
        print("Skipping motor pulse. Sense path only.")
        return

    print(
        f"Applying +{args.pulse_voltage:.2f} V for {args.pulse_seconds:.1f}s. "
        "Keep clear of the arm."
    )
    pulse_end = time.monotonic() + args.pulse_seconds
    while time.monotonic() < pulse_end:
        try:
            qube.update()
        except TimeoutError as exc:
            print(f"Bridge timeout during motor pulse: {exc}")
            break
        qube.setMotorVoltage(args.pulse_voltage)
        print(
            f"theta={qube.getMotorAngle():+8.2f} deg  "
            f"alpha_raw={qube.getPendulumAngle():+8.2f} deg  "
            f"rpm={qube.getMotorRPM():+7.1f}  "
            f"current={qube.getMotorCurrent():+7.1f} mA  "
            f"voltage={args.pulse_voltage:+5.2f} V"
        )
        time.sleep(dt)

    qube.setMotorVoltage(0.0)
    qube.update()
    print("Motor voltage set back to 0 V.")


if __name__ == "__main__":
    main()
