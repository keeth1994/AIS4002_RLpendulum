"""Energy swing-up controller with voltage PD stabilization for QUBE-Servo 2."""

from __future__ import annotations

import numpy as np

from src.envs.rotary_pendulum import RotaryPendulumParams, wrap_angle


class EnergySwingUpPDController:
    """Classical baseline controller used to validate the simulator.

    The controller returns motor voltage, not torque. It uses a teacher-style
    hybrid state machine:
    - a swing-up mode that drives a resonant swing-up
    - a balance mode with hysteresis around the upright catch region
    """

    def __init__(
        self,
        params: RotaryPendulumParams,
        energy_gain: float = 50.0,
        theta_gain: float = -1.3235294117647058,
        alpha_gain: float = 18.90760723931717,
        theta_dot_gain: float = -1.134453781512605,
        alpha_dot_gain: float = 2.3634509049146462,
        upright_threshold: float = np.deg2rad(20),
        balance_exit_threshold: float = np.deg2rad(35),
        balance_voltage_limit: float = 5.0,
        swingup_voltage_limit: float = 2.5,
        swingup_frequency_hz: float = 1.5,
        swingup_amplitude: float = 10.0,
        swingup_accel_limit: float = 6.0,
        arm_centering_gain: float = 1.3235294117647058,
        arm_centering_rate_gain: float = 1.134453781512605,
        swingup_mode: str = "oscillatory",
        use_stateful_switching: bool = True,
    ) -> None:
        if swingup_mode not in ("oscillatory", "energy"):
            raise ValueError("swingup_mode must be 'oscillatory' or 'energy'")
        self.params = params
        self.energy_gain = energy_gain
        self.balance_voltage_limit = min(balance_voltage_limit, params.voltage_limit)
        self.swingup_voltage_limit = min(swingup_voltage_limit, params.voltage_limit)
        self.swingup_frequency_hz = swingup_frequency_hz
        self.swingup_amplitude = min(swingup_amplitude, self.swingup_voltage_limit)
        self.swingup_accel_limit = swingup_accel_limit
        self.arm_centering_gain = arm_centering_gain
        self.arm_centering_rate_gain = arm_centering_rate_gain
        self.swingup_mode = swingup_mode
        self.use_stateful_switching = use_stateful_switching
        self.k = np.array(
            [theta_gain, alpha_gain, theta_dot_gain, alpha_dot_gain],
            dtype=np.float64,
        )
        self.upright_threshold = upright_threshold
        self.balance_exit_threshold = balance_exit_threshold
        self.reference_energy = 2.0 * params.pendulum_mass * params.gravity * params.pendulum_com
        self.accel_to_voltage = (
            params.motor_resistance * params.arm_length * params.arm_mass
        ) / max(params.motor_torque_constant, 1e-9)
        self.balance_mode_active = False

    def _pendulum_energy(self, alpha_error: float, alpha_dot: float) -> float:
        alpha_from_down = float(wrap_angle(alpha_error - np.pi))
        kinetic = 0.5 * self.params.pendulum_inertia * alpha_dot**2
        potential = (
            self.params.pendulum_mass
            * self.params.gravity
            * self.params.pendulum_com
            * (1.0 - np.cos(alpha_from_down))
        )
        return kinetic + potential

    def _arm_centering_voltage(self, theta: float, theta_dot: float) -> float:
        return self.arm_centering_gain * theta + self.arm_centering_rate_gain * theta_dot

    def _energy_swingup_voltage(self, theta: float, alpha_error: float, theta_dot: float, alpha_dot: float) -> float:
        alpha_from_down = float(wrap_angle(alpha_error - np.pi))
        energy = self._pendulum_energy(alpha_error, alpha_dot)
        accel_cmd = -self.energy_gain * (energy - self.reference_energy) * alpha_dot * np.cos(alpha_from_down)
        accel_cmd = float(np.clip(accel_cmd, -self.swingup_accel_limit, self.swingup_accel_limit))
        voltage = self.accel_to_voltage * accel_cmd - self._arm_centering_voltage(theta, theta_dot)
        return float(np.clip(voltage, -self.swingup_voltage_limit, self.swingup_voltage_limit))

    def _oscillatory_swingup_voltage(self, time_s: float, theta: float, theta_dot: float) -> float:
        voltage = self.swingup_amplitude * np.sin(2.0 * np.pi * self.swingup_frequency_hz * time_s)
        voltage -= self._arm_centering_voltage(theta, theta_dot)
        return float(np.clip(voltage, -self.swingup_voltage_limit, self.swingup_voltage_limit))

    def _update_balance_mode(self, alpha_error: float) -> bool:
        if not self.use_stateful_switching:
            return abs(alpha_error) < self.upright_threshold
        if not self.balance_mode_active and abs(alpha_error) < self.upright_threshold:
            self.balance_mode_active = True
        elif self.balance_mode_active and abs(alpha_error) > self.balance_exit_threshold:
            self.balance_mode_active = False
        return self.balance_mode_active

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return self.command(state, time_s=None)

    def command(self, state: np.ndarray, time_s: float | None) -> np.ndarray:
        theta, alpha, theta_dot, alpha_dot = state
        alpha_error = float(wrap_angle(alpha))
        feedback_state = np.array([theta, alpha_error, theta_dot, alpha_dot], dtype=np.float64)

        if self._update_balance_mode(alpha_error):
            voltage = -float(np.dot(self.k, feedback_state))
            voltage_limit = self.balance_voltage_limit
        else:
            if self.swingup_mode == "energy":
                voltage = self._energy_swingup_voltage(theta, alpha_error, theta_dot, alpha_dot)
            else:
                if time_s is None:
                    time_s = 0.0
                voltage = self._oscillatory_swingup_voltage(time_s, theta, theta_dot)
            voltage_limit = self.swingup_voltage_limit

        voltage = np.clip(voltage, -voltage_limit, voltage_limit)
        return np.array([voltage], dtype=np.float32)

    def open_loop_swingup(self, time_s: float, state: np.ndarray) -> np.ndarray:
        theta, _, theta_dot, _ = state
        voltage = self._oscillatory_swingup_voltage(time_s, theta, theta_dot)
        return np.array([voltage], dtype=np.float32)
