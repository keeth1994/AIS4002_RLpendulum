"""Energy swing-up controller with voltage PD stabilization for QUBE-Servo 2."""

from __future__ import annotations

import numpy as np

from src.envs.rotary_pendulum import RotaryPendulumParams, wrap_angle


class EnergySwingUpPDController:
    """Classical baseline controller used to validate the simulator.

    This controller returns motor voltage, not torque. It is intentionally
    simple: far from upright it pumps energy using saturated voltage; near
    upright it switches to a PD balance controller.
    """

    def __init__(
        self,
        params: RotaryPendulumParams,
        swing_gain: float = 1.0,
        theta_gain: float = -1.93,
        alpha_gain: float = 33.40,
        theta_dot_gain: float = -1.40,
        alpha_dot_gain: float = 3.08,
        upright_threshold: float = np.deg2rad(25),
        balance_voltage_limit: float = 1.0,
        swingup_voltage_limit: float = 4.0,
        swingup_frequency_hz: float = 2.6,
    ) -> None:
        self.params = params
        self.swing_gain = swing_gain
        self.balance_voltage_limit = min(balance_voltage_limit, params.voltage_limit)
        self.swingup_voltage_limit = min(swingup_voltage_limit, params.voltage_limit)
        self.swingup_frequency_hz = swingup_frequency_hz
        self.k = np.array(
            [theta_gain, alpha_gain, theta_dot_gain, alpha_dot_gain],
            dtype=np.float64,
        )
        self.upright_threshold = upright_threshold

    def __call__(self, state: np.ndarray) -> np.ndarray:
        theta, alpha, theta_dot, alpha_dot = state
        alpha_error = float(wrap_angle(alpha))
        feedback_state = np.array([theta, alpha_error, theta_dot, alpha_dot], dtype=np.float64)

        if abs(alpha_error) < self.upright_threshold:
            voltage = -float(np.dot(self.k, feedback_state))
            voltage_limit = self.balance_voltage_limit
        else:
            pump_direction = np.sign(alpha_dot * np.cos(alpha_error) + 1e-6)
            voltage = self.swing_gain * self.swingup_voltage_limit * pump_direction
            voltage -= 0.2 * theta + 0.05 * theta_dot
            voltage_limit = self.swingup_voltage_limit

        voltage = np.clip(voltage, -voltage_limit, voltage_limit)
        return np.array([voltage], dtype=np.float32)

    def open_loop_swingup(self, time_s: float, state: np.ndarray) -> np.ndarray:
        theta, alpha, theta_dot, alpha_dot = state
        alpha_error = float(wrap_angle(alpha))
        if abs(alpha_error) < self.upright_threshold:
            return self(state)
        voltage = self.swingup_voltage_limit * np.sin(2.0 * np.pi * self.swingup_frequency_hz * time_s + np.pi / 2.0)
        voltage -= 0.2 * theta + 0.05 * theta_dot
        voltage = np.clip(voltage, -self.swingup_voltage_limit, self.swingup_voltage_limit)
        return np.array([voltage], dtype=np.float32)
