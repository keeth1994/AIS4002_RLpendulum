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

from src.envs.rotary_pendulum import RotaryPendulumParams, wrap_angle

QUBE_DEG_TO_RAD = math.pi / 180.0


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def wrap_deg180(angle_deg: float) -> float:
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


def load_qube_class(example_dir: Path):
    sys.path.insert(0, str(example_dir.resolve()))
    from QUBE import QUBE  # type: ignore

    return QUBE


class HybridClassicalController:
    """Energy swing-up plus PD balance controller aligned with the simulator."""

    def __init__(
        self,
        dt: float,
        swingup_voltage_limit: float = 5.0,
        balance_voltage_limit: float = 5.0,
        energy_gain: float = 50.0,
        energy_pump_sign: float = 1.0,
        energy_deadband: float = 0.002,
        swingup_accel_limit: float = 8.0,
        arm_centering_gain: float = 2.0,
        arm_centering_rate_gain: float = 0.3,
        velocity_filter_alpha: float = 0.10,
        startup_kick_voltage: float = 2.5,
        startup_alpha_threshold_deg: float = 8.0,
        startup_alpha_dot_threshold: float = 0.25,
        startup_theta_target_deg: float = 20.0,
        startup_min_seconds: float = 0.0,
        startup_max_seconds: float = 0.0,
        startup_mode: str = "rock",
        startup_frequency_hz: float = 1.4,
        startup_centering_scale: float = 0.25,
    ) -> None:
        if startup_mode not in ("rock", "sine", "none"):
            raise ValueError("startup_mode must be 'rock', 'sine', or 'none'")
        self.dt = dt
        self.params = RotaryPendulumParams(voltage_limit=swingup_voltage_limit, dt=dt)
        self.swingup_voltage_limit = swingup_voltage_limit
        self.upright_threshold_deg = 20.0
        self.balance_exit_deg = 35.0
        self.balance_mode = False
        self.balance_voltage_limit = balance_voltage_limit
        self.energy_gain = energy_gain
        self.energy_pump_sign = 1.0 if energy_pump_sign >= 0.0 else -1.0
        self.energy_deadband = energy_deadband
        self.swingup_accel_limit = swingup_accel_limit
        self.startup_kick_voltage = min(abs(startup_kick_voltage), swingup_voltage_limit)
        self.startup_alpha_threshold_deg = startup_alpha_threshold_deg
        self.startup_alpha_dot_threshold = startup_alpha_dot_threshold
        self.startup_theta_target_deg = startup_theta_target_deg
        self.startup_min_seconds = startup_min_seconds
        self.startup_max_seconds = max(0.0, startup_max_seconds)
        self.startup_mode = startup_mode
        self.startup_frequency_hz = max(0.0, startup_frequency_hz)
        self.startup_centering_scale = max(0.0, startup_centering_scale)

        self.theta_gain = -1.3235294117647058
        self.alpha_gain = 18.90760723931717
        self.theta_dot_gain = -1.134453781512605
        self.alpha_dot_gain = 2.3634509049146462
        self.arm_centering_gain = arm_centering_gain
        self.arm_centering_rate_gain = arm_centering_rate_gain

        self.reference_energy = (
            2.0 * self.params.pendulum_mass * self.params.gravity * self.params.pendulum_com
        )
        self.accel_to_voltage = (
            self.params.motor_resistance * self.params.arm_length * self.params.arm_mass
        ) / max(self.params.motor_torque_constant, 1e-9)

        self.filtered_theta_dot_rad = 0.0
        self.filtered_alpha_dot_rad = 0.0
        self.velocity_filter_alpha = clip(velocity_filter_alpha, 0.0, 1.0)
        self.startup_direction = 1.0
        self.startup_active = startup_mode != "none"
        self.startup_elapsed_s = 0.0
        self.previous_theta_rad: float | None = None
        self.previous_alpha_from_down_rad: float | None = None

    def _arm_centering_voltage(self, theta_rad: float, theta_dot_rad: float) -> float:
        return self.arm_centering_gain * theta_rad + self.arm_centering_rate_gain * theta_dot_rad

    def _pendulum_energy(self, alpha_from_down_rad: float, alpha_dot_rad: float) -> float:
        kinetic = 0.5 * self.params.pendulum_inertia * alpha_dot_rad**2
        potential = (
            self.params.pendulum_mass
            * self.params.gravity
            * self.params.pendulum_com
            * (1.0 - math.cos(alpha_from_down_rad))
        )
        return kinetic + potential

    def step(
        self,
        position_deg: float,
        raw_pendulum_deg: float,
        dt_s: float | None = None,
    ) -> dict[str, float | str]:
        theta_rad = position_deg * QUBE_DEG_TO_RAD
        alpha_from_down_rad = raw_pendulum_deg * QUBE_DEG_TO_RAD
        alpha_balance_rad = float(wrap_angle(alpha_from_down_rad - math.pi))

        effective_dt = self.dt if dt_s is None or dt_s <= 0.0 else dt_s
        effective_dt = clip(effective_dt, self.dt * 0.25, self.dt * 5.0)
        if self.previous_theta_rad is None or self.previous_alpha_from_down_rad is None:
            raw_theta_dot_rad = 0.0
            raw_alpha_dot_rad = 0.0
        else:
            raw_theta_dot_rad = (theta_rad - self.previous_theta_rad) / effective_dt
            raw_alpha_dot_rad = (
                alpha_from_down_rad - self.previous_alpha_from_down_rad
            ) / effective_dt
        self.previous_theta_rad = theta_rad
        self.previous_alpha_from_down_rad = alpha_from_down_rad

        self.filtered_theta_dot_rad += self.velocity_filter_alpha * (
            raw_theta_dot_rad - self.filtered_theta_dot_rad
        )
        self.filtered_alpha_dot_rad += self.velocity_filter_alpha * (
            raw_alpha_dot_rad - self.filtered_alpha_dot_rad
        )

        abs_alpha_deg = abs(math.degrees(alpha_balance_rad))
        if (not self.balance_mode) and abs_alpha_deg < self.upright_threshold_deg:
            self.balance_mode = True
        elif self.balance_mode and abs_alpha_deg > self.balance_exit_deg:
            self.balance_mode = False

        energy = self._pendulum_energy(alpha_from_down_rad, self.filtered_alpha_dot_rad)
        energy_error = energy - self.reference_energy
        pump_signal = (
            self.energy_pump_sign
            * energy_error
            * self.filtered_alpha_dot_rad
            * math.cos(alpha_from_down_rad)
        )
        arm_centering_voltage = self._arm_centering_voltage(
            theta_rad, self.filtered_theta_dot_rad
        )

        if self.balance_mode:
            voltage = -(
                self.theta_gain * theta_rad
                + self.alpha_gain * alpha_balance_rad
                + self.theta_dot_gain * self.filtered_theta_dot_rad
                + self.alpha_dot_gain * self.filtered_alpha_dot_rad
            )
            voltage = clip(voltage, -self.balance_voltage_limit, self.balance_voltage_limit)
            return {
                "mode": "balance",
                "voltage": voltage,
                "alpha_deg": math.degrees(alpha_balance_rad),
                "raw_alpha_deg": raw_pendulum_deg,
                "alpha_dot": self.filtered_alpha_dot_rad,
                "theta_dot": self.filtered_theta_dot_rad,
                "alpha_from_down_deg": math.degrees(alpha_from_down_rad),
                "energy_j": energy,
                "energy_error_j": energy_error,
                "pump_signal": pump_signal,
                "accel_cmd": 0.0,
                "arm_centering_voltage": arm_centering_voltage,
                "dt_s": effective_dt,
            }

        # A pure energy law produces zero command at the exact hanging-down
        # equilibrium because both energy error injection terms are zero there.
        # Use a small arm kick to bootstrap motion from rest, then let the
        # energy controller take over once pendulum speed is measurable.
        startup_amplitude_small = abs(raw_pendulum_deg) < self.startup_alpha_threshold_deg
        startup_speed_small = abs(self.filtered_alpha_dot_rad) < self.startup_alpha_dot_threshold
        startup_conditions = startup_amplitude_small
        if self.startup_mode == "rock":
            startup_conditions = startup_amplitude_small and startup_speed_small
        if self.startup_active:
            self.startup_elapsed_s += effective_dt
        startup_min_elapsed = self.startup_elapsed_s >= self.startup_min_seconds
        startup_max_elapsed = (
            self.startup_max_seconds > 0.0
            and self.startup_elapsed_s >= self.startup_max_seconds
        )
        if self.startup_active and startup_max_elapsed:
            self.startup_active = False
        elif self.startup_active and startup_min_elapsed and not startup_conditions:
            self.startup_active = False

        if self.startup_active and ((not startup_min_elapsed) or startup_conditions):
            mode = self.startup_mode
            if self.startup_mode == "sine":
                ramp = min(1.0, self.startup_elapsed_s / 0.5)
                voltage = (
                    ramp
                    * self.startup_kick_voltage
                    * math.sin(2.0 * math.pi * self.startup_frequency_hz * self.startup_elapsed_s)
                )
                voltage -= self.startup_centering_scale * arm_centering_voltage
                voltage = clip(voltage, -self.swingup_voltage_limit, self.swingup_voltage_limit)
            else:
                if position_deg >= self.startup_theta_target_deg:
                    self.startup_direction = -1.0
                elif position_deg <= -self.startup_theta_target_deg:
                    self.startup_direction = 1.0
                elif abs(position_deg) < 1e-6:
                    self.startup_direction = 1.0
                voltage = self.startup_direction * self.startup_kick_voltage
            return {
                "mode": mode,
                "voltage": voltage,
                "alpha_deg": math.degrees(alpha_balance_rad),
                "raw_alpha_deg": raw_pendulum_deg,
                "alpha_dot": self.filtered_alpha_dot_rad,
                "theta_dot": self.filtered_theta_dot_rad,
                "alpha_from_down_deg": math.degrees(alpha_from_down_rad),
                "energy_j": energy,
                "energy_error_j": energy_error,
                "pump_signal": pump_signal,
                "accel_cmd": 0.0,
                "arm_centering_voltage": arm_centering_voltage,
                "dt_s": effective_dt,
            }

        if abs(pump_signal) < self.energy_deadband:
            accel_cmd = 0.0
        else:
            accel_cmd = clip(
                self.energy_gain * pump_signal,
                -self.swingup_accel_limit,
                self.swingup_accel_limit,
            )
        voltage = (
            self.accel_to_voltage * accel_cmd
            - arm_centering_voltage
        )
        voltage = clip(voltage, -self.swingup_voltage_limit, self.swingup_voltage_limit)
        return {
            "mode": "swingup",
            "voltage": voltage,
            "alpha_deg": math.degrees(alpha_balance_rad),
            "raw_alpha_deg": raw_pendulum_deg,
            "alpha_dot": self.filtered_alpha_dot_rad,
            "theta_dot": self.filtered_theta_dot_rad,
            "alpha_from_down_deg": math.degrees(alpha_from_down_rad),
            "energy_j": energy,
            "energy_error_j": energy_error,
            "pump_signal": pump_signal,
            "accel_cmd": accel_cmd,
            "arm_centering_voltage": arm_centering_voltage,
            "dt_s": effective_dt,
        }


class ExampleEnergyController:
    """Reference controller from EXAMPLE_CODE/inverted_pendulum.py.

    This preserves the example's state machine, gains, and angle convention.
    Two hardware-bridge fixes are applied: angle differences are wrapped across
    +/-180 deg, and the energy calculation uses radians instead of raw degrees.
    """

    def __init__(
        self,
        dt: float,
        energy_reference: float = 0.015,
        energy_gain: float = 50.0,
        swingup_u_max: float = 2.5,
        balance_range_deg: float = 20.0,
        startup_kick_voltage: float = 1.5,
        startup_kick_seconds: float = 0.15,
    ) -> None:
        self.dt = dt

        self.m_p = 0.1
        self.l = 0.095
        self.l_com = self.l / 2.0
        self.J = (1.0 / 3.0) * self.m_p * self.l * self.l
        self.g = 9.81

        self.Er = energy_reference
        self.ke = energy_gain
        self.u_max = swingup_u_max
        self.balance_range_deg = balance_range_deg
        self.startup_kick_voltage = startup_kick_voltage
        self.startup_kick_seconds = max(0.0, startup_kick_seconds)

        s = 0.33
        self.kp_theta = 1.0 * s
        self.kd_theta = 0.125 * s
        self.kp_pos = 0.07 * s
        self.kd_pos = 0.06 * s

        self.wc = 500.0 / (2.0 * math.pi)
        self.wc3 = 500.0 / (2.0 * math.pi)
        self.y_k_last = 0.0
        self.y2_k_last = 0.0
        self.y3_k_last = 0.0
        self.prev_angle_deg: float | None = None
        self.prev_pos_deg: float | None = None
        self.balance_mode = False
        self.elapsed_s = 0.0
        self.t_balance_s = 0.0
        self.reset_mode = False
        self.t_reset_s = 0.0
        self.last_energy_j = 0.0
        self.last_u = 0.0
        self.last_u_sat = 0.0

    def _angle_delta_deg(self, current_deg: float, previous_deg: float) -> float:
        return wrap_deg180(current_deg - previous_deg)

    def _filter_voltage(self, voltage: float, dt_s: float) -> float:
        y3_k = self.y3_k_last + self.wc3 * dt_s * (voltage - self.y3_k_last)
        self.y3_k_last = y3_k
        return y3_k

    def _swingup_voltage(self, angle_deg: float, dt_s: float) -> tuple[float, float]:
        if self.prev_angle_deg is None:
            angular_v_rad = 0.0
        else:
            angular_v_rad = (
                self._angle_delta_deg(angle_deg, self.prev_angle_deg)
                * QUBE_DEG_TO_RAD
                / dt_s
            )
        self.prev_angle_deg = angle_deg

        angle_rad = angle_deg * QUBE_DEG_TO_RAD
        energy = (
            0.5 * self.J * angular_v_rad**2
            + self.m_p * self.g * self.l_com * (1.0 - math.cos(angle_rad))
        )
        u = self.ke * (energy - self.Er) * (-angular_v_rad * math.cos(angle_rad))
        u_sat = clip(u, -self.u_max, self.u_max)

        voltage = u_sat * (8.4 * 0.095 * 0.085) / 0.042
        self.last_energy_j = energy
        self.last_u = u
        self.last_u_sat = u_sat
        return self._filter_voltage(voltage, dt_s), angular_v_rad

    def _settle_voltage(self, angle_deg: float, dt_s: float) -> tuple[float, float]:
        settle_angle_deg = (
            angle_deg + 360.0 - 2.0 * angle_deg
            if angle_deg < 0.0
            else -360.0 + 2.0 * angle_deg
        )
        return self._swingup_voltage(settle_angle_deg, dt_s)

    def _balance_voltage(self, position_deg: float, angle_deg: float, dt_s: float) -> tuple[float, float, float]:
        if self.prev_angle_deg is None:
            u_dot_deg = 0.0
        else:
            u_dot_deg = self._angle_delta_deg(angle_deg, self.prev_angle_deg) / dt_s
        y_k = self.y_k_last + self.wc * dt_s * (u_dot_deg - self.y_k_last)
        self.y_k_last = y_k

        if self.prev_pos_deg is None:
            v_deg = 0.0
        else:
            v_deg = (position_deg - self.prev_pos_deg) / dt_s
        y2_k = self.y2_k_last + self.wc * dt_s * (v_deg - self.y2_k_last)
        self.y2_k_last = y2_k

        u_pos = self.kp_pos * position_deg + self.kd_pos * y2_k
        u_ang = self.kp_theta * angle_deg + self.kd_theta * y_k
        voltage = u_pos + u_ang

        self.prev_angle_deg = angle_deg
        self.prev_pos_deg = position_deg
        return voltage, y_k, y2_k

    def _output(
        self,
        mode: str,
        voltage: float,
        position_deg: float,
        raw_pendulum_deg: float,
        angle_deg: float,
        alpha_dot_rad: float,
        theta_dot_rad: float,
        dt_s: float,
    ) -> dict[str, float | str]:
        return {
            "mode": mode,
            "voltage": voltage,
            "alpha_deg": angle_deg,
            "raw_alpha_deg": raw_pendulum_deg,
            "alpha_dot": alpha_dot_rad,
            "theta_dot": theta_dot_rad,
            "alpha_from_down_deg": raw_pendulum_deg,
            "energy_j": self.last_energy_j,
            "energy_error_j": self.last_energy_j - self.Er,
            "pump_signal": self.last_u,
            "accel_cmd": self.last_u_sat,
            "arm_centering_voltage": 0.0,
            "dt_s": dt_s,
        }

    def step(
        self,
        position_deg: float,
        raw_pendulum_deg: float,
        dt_s: float | None = None,
    ) -> dict[str, float | str]:
        effective_dt = self.dt if dt_s is None or dt_s <= 0.0 else dt_s
        effective_dt = clip(effective_dt, self.dt * 0.25, self.dt * 5.0)
        self.elapsed_s += effective_dt
        angle_deg = normalize_example_angle(raw_pendulum_deg)

        if (not self.balance_mode) and (-self.balance_range_deg < angle_deg < self.balance_range_deg):
            self.balance_mode = True
            self.t_balance_s = self.elapsed_s
        elif self.balance_mode and not (-self.balance_range_deg < angle_deg < self.balance_range_deg):
            self.balance_mode = False
            if self.elapsed_s - self.t_balance_s > 1.0:
                self.reset_mode = True
                self.t_reset_s = self.elapsed_s

        if self.reset_mode:
            voltage, alpha_dot_rad = self._settle_voltage(angle_deg, effective_dt)
            if self.elapsed_s - self.t_reset_s >= 2.0:
                self.reset_mode = False
            return self._output(
                "example_reset",
                voltage,
                position_deg,
                raw_pendulum_deg,
                angle_deg,
                alpha_dot_rad,
                0.0,
                effective_dt,
            )

        # The reference script gets an initial kick from prevAngle=0 while the
        # pendulum is actually at 180 deg. Make that bootstrap explicit.
        if (
            not self.balance_mode
            and self.elapsed_s <= self.startup_kick_seconds
            and abs(abs(angle_deg) - 180.0) < 5.0
        ):
            return self._output(
                "example_kick",
                self.startup_kick_voltage,
                position_deg,
                raw_pendulum_deg,
                angle_deg,
                0.0,
                0.0,
                effective_dt,
            )

        if self.balance_mode:
            voltage, alpha_dot_deg, theta_dot_deg = self._balance_voltage(position_deg, angle_deg, effective_dt)
            return self._output(
                "example_balance",
                voltage,
                position_deg,
                raw_pendulum_deg,
                angle_deg,
                alpha_dot_deg * QUBE_DEG_TO_RAD,
                theta_dot_deg * QUBE_DEG_TO_RAD,
                effective_dt,
            )

        voltage, alpha_dot_rad = self._swingup_voltage(angle_deg, effective_dt)
        return self._output(
            "example_swingup",
            voltage,
            position_deg,
            raw_pendulum_deg,
            angle_deg,
            alpha_dot_rad,
            0.0,
            effective_dt,
        )


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
    parser.add_argument(
        "--controller",
        choices=["hybrid", "example"],
        default="hybrid",
        help="hybrid uses the simulator-aligned controller; example uses EXAMPLE_CODE/inverted_pendulum.py.",
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--rate-hz", type=float, default=300.0)
    parser.add_argument("--startup-delay", type=float, default=3.0)
    parser.add_argument("--hard-voltage-limit", type=float, default=5.0)
    parser.add_argument("--balance-voltage-limit", type=float, default=5.0)
    parser.add_argument("--energy-gain", type=float, default=50.0)
    parser.add_argument("--energy-pump-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--energy-deadband", type=float, default=0.002)
    parser.add_argument("--swingup-accel-limit", type=float, default=8.0)
    parser.add_argument("--arm-centering-gain", type=float, default=2.0)
    parser.add_argument("--arm-centering-rate-gain", type=float, default=0.3)
    parser.add_argument("--velocity-filter-alpha", type=float, default=0.10)
    parser.add_argument("--startup-kick-voltage", type=float, default=2.5)
    parser.add_argument("--startup-alpha-threshold-deg", type=float, default=8.0)
    parser.add_argument("--startup-alpha-dot-threshold", type=float, default=0.25)
    parser.add_argument("--startup-theta-target-deg", type=float, default=20.0)
    parser.add_argument("--startup-min-seconds", type=float, default=0.0)
    parser.add_argument("--startup-max-seconds", type=float, default=0.0)
    parser.add_argument("--startup-mode", choices=["rock", "sine", "none"], default="rock")
    parser.add_argument("--startup-frequency-hz", type=float, default=1.4)
    parser.add_argument("--startup-centering-scale", type=float, default=0.25)
    parser.add_argument("--example-energy-ref", type=float, default=0.015)
    parser.add_argument("--example-energy-gain", type=float, default=50.0)
    parser.add_argument("--example-u-max", type=float, default=2.5)
    parser.add_argument("--example-balance-range-deg", type=float, default=20.0)
    parser.add_argument("--example-startup-kick-voltage", type=float, default=1.5)
    parser.add_argument("--example-startup-kick-seconds", type=float, default=0.15)
    parser.add_argument("--motor-voltage-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--theta-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--alpha-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--center-trim-deg", type=float, default=0.0)
    parser.add_argument(
        "--arm-soft-limit-deg",
        type=float,
        default=0.0,
        help="Start overriding the command with a centering brake beyond this arm angle. 0 disables.",
    )
    parser.add_argument(
        "--arm-hard-stop-deg",
        type=float,
        default=0.0,
        help="Immediately stop the run if abs(theta) exceeds this arm angle. 0 disables.",
    )
    parser.add_argument("--arm-limit-brake-gain", type=float, default=10.0)
    parser.add_argument("--arm-limit-rate-gain", type=float, default=0.5)
    parser.add_argument(
        "--motor-health-check-seconds",
        type=float,
        default=1.0,
        help="Stop if commanded voltage produces no current, RPM, or arm motion for this long. 0 disables.",
    )
    parser.add_argument("--motor-health-min-voltage", type=float, default=1.0)
    parser.add_argument("--motor-health-min-theta-change-deg", type=float, default=1.0)
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

    if args.controller == "example":
        controller = ExampleEnergyController(
            dt=1.0 / args.rate_hz,
            energy_reference=args.example_energy_ref,
            energy_gain=args.example_energy_gain,
            swingup_u_max=args.example_u_max,
            balance_range_deg=args.example_balance_range_deg,
            startup_kick_voltage=args.example_startup_kick_voltage,
            startup_kick_seconds=args.example_startup_kick_seconds,
        )
    else:
        controller = HybridClassicalController(
            dt=1.0 / args.rate_hz,
            swingup_voltage_limit=args.hard_voltage_limit,
            balance_voltage_limit=min(args.balance_voltage_limit, args.hard_voltage_limit),
            energy_gain=args.energy_gain,
            energy_pump_sign=args.energy_pump_sign,
            energy_deadband=args.energy_deadband,
            swingup_accel_limit=args.swingup_accel_limit,
            arm_centering_gain=args.arm_centering_gain,
            arm_centering_rate_gain=args.arm_centering_rate_gain,
            velocity_filter_alpha=args.velocity_filter_alpha,
            startup_kick_voltage=args.startup_kick_voltage,
            startup_alpha_threshold_deg=args.startup_alpha_threshold_deg,
            startup_alpha_dot_threshold=args.startup_alpha_dot_threshold,
            startup_theta_target_deg=args.startup_theta_target_deg,
            startup_min_seconds=args.startup_min_seconds,
            startup_max_seconds=args.startup_max_seconds,
            startup_mode=args.startup_mode,
            startup_frequency_hz=args.startup_frequency_hz,
            startup_centering_scale=args.startup_centering_scale,
        )
    fieldnames = [
        "time_s",
        "mode",
        "theta_deg",
        "alpha_deg",
        "raw_alpha_deg",
        "theta_rad",
        "alpha_rad",
        "theta_dot_rad",
        "alpha_dot_rad",
        "alpha_from_down_deg",
        "energy_j",
        "energy_error_j",
        "pump_signal",
        "accel_cmd",
        "arm_centering_voltage",
        "dt_s",
        "voltage_cmd",
        "voltage_applied",
        "safety_limited",
        "motor_current",
        "motor_rpm",
        "dry_run",
    ]

    qube.setMotorVoltage(0.0)
    qube.setRGB(0, 0, 999)
    qube.update()

    print("Center the arm and let the pendulum hang down. Resetting encoders in 3 seconds.")
    time.sleep(args.startup_delay)
    qube.reset_buffers()
    qube.resetMotorEncoder()
    qube.resetPendulumEncoder()
    qube.update()
    time.sleep(0.2)

    dt_target = 1.0 / args.rate_hz
    next_tick = time.monotonic()
    start = next_tick
    end_time = start + args.duration
    last_control_time = start
    health_check_start = start
    health_check_theta_deg: float | None = None
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

                try:
                    qube.update()
                except TimeoutError as exc:
                    print(f"Bridge timeout while reading QUBE state: {exc}")
                    break
                control_dt = now - last_control_time
                last_control_time = now
                theta_deg = args.theta_sign * qube.getMotorAngle() - args.center_trim_deg
                raw_alpha_deg = args.alpha_sign * qube.getPendulumAngle()
                output = controller.step(theta_deg, raw_alpha_deg, control_dt)
                alpha_deg = float(output["alpha_deg"])
                voltage_cmd = float(output["voltage"])
                voltage_applied = clip(
                    args.motor_voltage_sign * voltage_cmd,
                    -args.hard_voltage_limit,
                    args.hard_voltage_limit,
                )
                safety_limited = 0
                theta_dot_rad = float(output["theta_dot"])
                if args.arm_soft_limit_deg > 0.0:
                    if theta_deg > args.arm_soft_limit_deg:
                        over_rad = (theta_deg - args.arm_soft_limit_deg) * QUBE_DEG_TO_RAD
                        brake_voltage = -(
                            args.arm_limit_brake_gain * over_rad
                            + args.arm_limit_rate_gain * theta_dot_rad
                        )
                        voltage_applied = min(voltage_applied, brake_voltage)
                        safety_limited = 1
                    elif theta_deg < -args.arm_soft_limit_deg:
                        over_rad = (theta_deg + args.arm_soft_limit_deg) * QUBE_DEG_TO_RAD
                        brake_voltage = -(
                            args.arm_limit_brake_gain * over_rad
                            + args.arm_limit_rate_gain * theta_dot_rad
                        )
                        voltage_applied = max(voltage_applied, brake_voltage)
                        safety_limited = 1
                    voltage_applied = clip(
                        voltage_applied,
                        -args.hard_voltage_limit,
                        args.hard_voltage_limit,
                    )

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
                    qube.setMotorVoltage(voltage_applied)

                if not args.dry_run and args.motor_health_check_seconds > 0.0:
                    if health_check_theta_deg is None:
                        health_check_theta_deg = theta_deg
                        health_check_start = now
                    commanded = abs(voltage_applied) >= args.motor_health_min_voltage
                    moving = abs(theta_deg - health_check_theta_deg) >= args.motor_health_min_theta_change_deg
                    motor_feedback = abs(qube.getMotorRPM()) > 1.0 or abs(qube.getMotorCurrent()) > 5.0
                    if moving or motor_feedback or not commanded:
                        health_check_theta_deg = theta_deg
                        health_check_start = now
                    elif now - health_check_start >= args.motor_health_check_seconds:
                        print(
                            "Motor drive inactive: commanded "
                            f"{voltage_applied:+.2f} V for {now - health_check_start:.2f} s, "
                            f"but theta changed only {theta_deg - health_check_theta_deg:+.2f} deg, "
                            f"rpm={qube.getMotorRPM():+.1f}, current={qube.getMotorCurrent():+.1f} mA. "
                            "Power-cycle the QUBE/amp and verify with test_qube_bridge."
                        )
                        qube.setMotorVoltage(0.0)
                        break

                row = {
                    "time_s": round(now - start, 6),
                    "mode": output["mode"],
                    "theta_deg": theta_deg,
                    "alpha_deg": alpha_deg,
                    "raw_alpha_deg": float(output["raw_alpha_deg"]),
                    "theta_rad": theta_deg * QUBE_DEG_TO_RAD,
                    "alpha_rad": alpha_deg * QUBE_DEG_TO_RAD,
                    "theta_dot_rad": float(output["theta_dot"]),
                    "alpha_dot_rad": float(output["alpha_dot"]),
                    "alpha_from_down_deg": float(output["alpha_from_down_deg"]),
                    "energy_j": float(output["energy_j"]),
                    "energy_error_j": float(output["energy_error_j"]),
                    "pump_signal": float(output["pump_signal"]),
                    "accel_cmd": float(output["accel_cmd"]),
                    "arm_centering_voltage": float(output["arm_centering_voltage"]),
                    "dt_s": float(output["dt_s"]),
                    "voltage_cmd": voltage_cmd,
                    "voltage_applied": voltage_applied,
                    "safety_limited": safety_limited,
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
