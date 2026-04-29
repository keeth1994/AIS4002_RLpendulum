"""Run the classical controller on the QUBE and log data directly to CSV.

This uses the course serial API in EXAMPLE_CODE/QUBE.py rather than scraping
Arduino Serial output, which is more reliable for report-quality plots.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

QUBE_DEG_TO_RAD = math.pi / 180.0


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def load_qube_class(example_dir: Path):
    sys.path.insert(0, str(example_dir.resolve()))
    from QUBE import QUBE  # type: ignore

    return QUBE


class ExampleClassicalController:
    """Classical controller copied from EXAMPLE_CODE/inverted_pendulum.py."""

    def __init__(self, dt: float) -> None:
        self.dt = dt

        self.m_p = 0.1
        self.l = 0.095
        self.l_com = self.l / 2.0
        self.J = (1.0 / 3.0) * self.m_p * self.l * self.l
        self.g = 9.81

        self.Er = 0.015
        self.ke = 50.0
        self.u_max = 2.5
        self.balance_range_deg = 20.0

        s = 0.33
        self.kp_theta = 1.0 * s
        self.kd_theta = 0.125 * s
        self.kp_pos = 0.07 * s
        self.kd_pos = 0.06 * s

        self.wc = 500.0 / (2.0 * math.pi)
        self.y_k_last = 0.0
        self.y2_k_last = 0.0
        self.prev_angle_deg = 0.0
        self.prev_pos_deg = 0.0
        self.balance_mode = False
        self.t_balance = 0.0
        self.reset_mode = False
        self.t_reset = 0.0

    def swingup_voltage(self, angle_deg: float) -> tuple[float, float]:
        angular_v_deg = (angle_deg - self.prev_angle_deg) / self.dt
        angular_v_rad = angular_v_deg * QUBE_DEG_TO_RAD
        angle_rad = angle_deg * QUBE_DEG_TO_RAD
        self.prev_angle_deg = angle_deg

        energy = 0.5 * self.J * angular_v_rad**2 + self.m_p * self.g * self.l_com * (
            1.0 - math.cos(angle_rad)
        )
        u = self.ke * (energy - self.Er) * (-angular_v_rad * math.cos(angle_rad))
        u_sat = clip(u, -self.u_max, self.u_max)
        voltage = u_sat * (8.4 * 0.095 * 0.085) / 0.042
        return voltage, angular_v_rad

    def settle_voltage(self, angle_deg: float) -> tuple[float, float]:
        settle_angle_deg = angle_deg + 360.0 - 2.0 * angle_deg if angle_deg < 0.0 else -360.0 + 2.0 * angle_deg
        return self.swingup_voltage(settle_angle_deg)

    def balance_voltage(self, position_deg: float, angle_deg: float) -> tuple[float, float, float]:
        u_dot_deg = (angle_deg - self.prev_angle_deg) / self.dt
        y_k = self.y_k_last + self.wc * self.dt * (u_dot_deg - self.y_k_last)
        self.y_k_last = y_k

        v_deg = (position_deg - self.prev_pos_deg) / self.dt
        y2_k = self.y2_k_last + self.wc * self.dt * (v_deg - self.y2_k_last)
        self.y2_k_last = y2_k

        u_pos = self.kp_pos * position_deg + self.kd_pos * y2_k
        u_ang = self.kp_theta * angle_deg + self.kd_theta * y_k
        voltage = u_pos + u_ang

        self.prev_angle_deg = angle_deg
        self.prev_pos_deg = position_deg
        return voltage, y_k * QUBE_DEG_TO_RAD, y2_k * QUBE_DEG_TO_RAD

    def step(self, position_deg: float, angle_deg: float, now_s: float) -> dict[str, float | str | bool]:
        if (not self.balance_mode) and (-self.balance_range_deg < angle_deg < self.balance_range_deg):
            self.balance_mode = True
            self.t_balance = now_s
        elif self.balance_mode and not (-self.balance_range_deg < angle_deg < self.balance_range_deg):
            self.balance_mode = False
            if (now_s - self.t_balance) > 1.0:
                self.reset_mode = True
                self.t_reset = now_s

        if self.reset_mode:
            voltage, alpha_dot = self.settle_voltage(angle_deg)
            if (now_s - self.t_reset) >= 2.0:
                self.reset_mode = False
            return {
                "mode": "reset",
                "voltage": voltage,
                "alpha_dot": alpha_dot,
                "theta_dot": 0.0,
            }

        if self.balance_mode:
            voltage, alpha_dot, theta_dot = self.balance_voltage(position_deg, angle_deg)
            return {
                "mode": "balance",
                "voltage": voltage,
                "alpha_dot": alpha_dot,
                "theta_dot": theta_dot,
            }

        voltage, alpha_dot = self.swingup_voltage(angle_deg)
        return {
            "mode": "swingup",
            "voltage": voltage,
            "alpha_dot": alpha_dot,
            "theta_dot": 0.0,
        }


def normalize_example_angle(raw_pendulum_deg: float) -> float:
    wrapped = math.fmod(raw_pendulum_deg, 360.0)
    angle_deg = -180.0 + wrapped if wrapped > 0.0 else 180.0 + wrapped
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="COM3")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--example-dir", type=Path, default=Path("EXAMPLE_CODE"))
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--rate-hz", type=float, default=300.0)
    parser.add_argument("--startup-delay", type=float, default=3.0)
    parser.add_argument("--hard-voltage-limit", type=float, default=5.0)
    parser.add_argument("--motor-voltage-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--theta-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--alpha-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--center-trim-deg", type=float, default=0.0)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = Path(f"results/qube_classical_direct_{stamp}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    QUBE = load_qube_class(args.example_dir)
    qube = QUBE(args.port, args.baudrate)

    controller = ExampleClassicalController(dt=1.0 / args.rate_hz)
    fieldnames = [
        "time_s",
        "mode",
        "theta_deg",
        "alpha_deg",
        "theta_rad",
        "alpha_rad",
        "theta_dot_rad",
        "alpha_dot_rad",
        "voltage_cmd",
        "voltage_applied",
        "motor_current",
        "motor_rpm",
        "dry_run",
    ]

    qube.setMotorVoltage(0.0)
    qube.setRGB(0, 0, 999)
    qube.update()

    print("Center the arm and let the pendulum hang down. Resetting encoders in 3 seconds.")
    time.sleep(args.startup_delay)
    qube.resetMotorEncoder()
    qube.resetPendulumEncoder()
    qube.update()
    time.sleep(0.2)

    dt_target = 1.0 / args.rate_hz
    next_tick = time.monotonic()
    start = next_tick
    end_time = start + args.duration
    print_every = max(1, int(args.rate_hz / 10.0))
    step = 0

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

                qube.update()
                theta_deg = args.theta_sign * qube.getMotorAngle() - args.center_trim_deg
                raw_alpha_deg = args.alpha_sign * qube.getPendulumAngle()
                alpha_deg = normalize_example_angle(raw_alpha_deg)

                output = controller.step(theta_deg, alpha_deg, now_s=now - start)
                voltage_cmd = float(output["voltage"])
                voltage_applied = clip(
                    args.motor_voltage_sign * voltage_cmd,
                    -args.hard_voltage_limit,
                    args.hard_voltage_limit,
                )

                if args.dry_run:
                    qube.setMotorVoltage(0.0)
                else:
                    qube.setMotorVoltage(voltage_applied)

                row = {
                    "time_s": round(now - start, 6),
                    "mode": output["mode"],
                    "theta_deg": theta_deg,
                    "alpha_deg": alpha_deg,
                    "theta_rad": theta_deg * QUBE_DEG_TO_RAD,
                    "alpha_rad": alpha_deg * QUBE_DEG_TO_RAD,
                    "theta_dot_rad": float(output["theta_dot"]),
                    "alpha_dot_rad": float(output["alpha_dot"]),
                    "voltage_cmd": voltage_cmd,
                    "voltage_applied": voltage_applied,
                    "motor_current": qube.getMotorCurrent(),
                    "motor_rpm": qube.getMotorRPM(),
                    "dry_run": int(args.dry_run),
                }
                writer.writerow(row)
                file.flush()

                if step % print_every == 0:
                    print(
                        f"{row['time_s']:7.3f}s mode={row['mode']} "
                        f"theta={row['theta_deg']:+7.2f}deg alpha={row['alpha_deg']:+7.2f}deg "
                        f"voltage={row['voltage_applied']:+6.3f}V"
                    )
                step += 1
        finally:
            qube.setMotorVoltage(0.0)
            qube.setRGB(999, 0, 0)
            qube.update()

    print(f"Wrote classical hardware log to {csv_path}")


if __name__ == "__main__":
    main()
