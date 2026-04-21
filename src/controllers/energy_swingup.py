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
    ) -> None:
        self.params = params
        self.swing_gain = swing_gain
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
        else:
            pump_direction = np.sign(alpha_dot * np.cos(alpha_error) + 1e-6)
            voltage = self.swing_gain * self.params.voltage_limit * pump_direction
            voltage -= 0.2 * theta + 0.05 * theta_dot

        voltage = np.clip(voltage, -self.params.voltage_limit, self.params.voltage_limit)
        return np.array([voltage], dtype=np.float32)
