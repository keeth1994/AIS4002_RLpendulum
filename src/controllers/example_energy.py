"""Course example energy swing-up controller adapted to simulator state."""

from __future__ import annotations

import math

import numpy as np

from src.envs.rotary_pendulum import wrap_angle


class ExampleEnergyPDController:
    """Fixed version of the course example energy controller.

    The original example uses angle=0 at upright and +/-180 deg at hanging down.
    The simulator already uses alpha=0 at upright, so this controller can use
    alpha directly while preserving the example's energy law and balance gains.
    """

    def __init__(
        self,
        dt: float,
        voltage_limit: float = 5.0,
        energy_reference: float = 0.015,
        energy_gain: float = 50.0,
        swingup_u_max: float = 2.5,
        balance_range_deg: float = 20.0,
        balance_exit_deg: float = 35.0,
        startup_kick_voltage: float = 0.0,
        startup_kick_seconds: float = 0.0,
    ) -> None:
        self.dt = dt
        self.voltage_limit = voltage_limit
        self.balance_range_rad = math.radians(balance_range_deg)
        self.balance_exit_rad = math.radians(balance_exit_deg)
        self.startup_kick_voltage = startup_kick_voltage
        self.startup_kick_seconds = max(0.0, startup_kick_seconds)

        self.m_p = 0.1
        self.l = 0.095
        self.l_com = self.l / 2.0
        self.J = (1.0 / 3.0) * self.m_p * self.l * self.l
        self.g = 9.81

        self.Er = energy_reference
        self.ke = energy_gain
        self.u_max = swingup_u_max

        s = 0.33
        self.kp_theta = 1.0 * s
        self.kd_theta = 0.125 * s
        self.kp_pos = 0.07 * s
        self.kd_pos = 0.06 * s

        self.wc3 = 500.0 / (2.0 * math.pi)
        self.y3_k_last = 0.0
        self.balance_mode = False

    def _filter_voltage(self, voltage: float) -> float:
        y3_k = self.y3_k_last + self.wc3 * self.dt * (voltage - self.y3_k_last)
        self.y3_k_last = y3_k
        return y3_k

    def _swingup_voltage(self, alpha_rad: float, alpha_dot_rad: float) -> float:
        energy = (
            0.5 * self.J * alpha_dot_rad**2
            + self.m_p * self.g * self.l_com * (1.0 - math.cos(alpha_rad))
        )
        u = self.ke * (energy - self.Er) * (-alpha_dot_rad * math.cos(alpha_rad))
        u_sat = float(np.clip(u, -self.u_max, self.u_max))
        voltage = u_sat * (8.4 * 0.095 * 0.085) / 0.042
        return self._filter_voltage(voltage)

    def _balance_voltage(
        self,
        theta_rad: float,
        alpha_rad: float,
        theta_dot_rad: float,
        alpha_dot_rad: float,
    ) -> float:
        theta_deg = math.degrees(theta_rad)
        alpha_deg = math.degrees(alpha_rad)
        theta_dot_deg = math.degrees(theta_dot_rad)
        alpha_dot_deg = math.degrees(alpha_dot_rad)
        return (
            self.kp_pos * theta_deg
            + self.kd_pos * theta_dot_deg
            + self.kp_theta * alpha_deg
            + self.kd_theta * alpha_dot_deg
        )

    def command(self, state: np.ndarray, time_s: float | None = None) -> np.ndarray:
        theta, alpha, theta_dot, alpha_dot = state
        alpha_error = float(wrap_angle(alpha))

        if (not self.balance_mode) and abs(alpha_error) < self.balance_range_rad:
            self.balance_mode = True
        elif self.balance_mode and abs(alpha_error) > self.balance_exit_rad:
            self.balance_mode = False

        if self.balance_mode:
            voltage = self._balance_voltage(theta, alpha_error, theta_dot, alpha_dot)
        elif (
            time_s is not None
            and time_s <= self.startup_kick_seconds
            and abs(abs(alpha_error) - math.pi) < math.radians(5.0)
        ):
            voltage = self.startup_kick_voltage
        else:
            voltage = self._swingup_voltage(alpha_error, alpha_dot)

        return np.array([np.clip(voltage, -self.voltage_limit, self.voltage_limit)], dtype=np.float32)
