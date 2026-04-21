"""Energy swing-up controller with PD stabilization near upright."""

from __future__ import annotations

import numpy as np

from src.envs.rotary_pendulum import RotaryPendulumParams, wrap_angle


class EnergySwingUpPDController:
    """Classical baseline controller used to validate the simulator."""

    def __init__(
        self,
        params: RotaryPendulumParams,
        swing_gain: float = 0.545,
        pd_alpha: float = 20, 
        #Controls how strongly the controller corrects pendulum angle error near upright.
        #Higher value means stronger correction toward upright.

        pd_alpha_dot: float = 10,
        #Controls damping of pendulum angular velocity near upright.
        #Higher value slows the pendulum more strongly.
        
        pd_theta: float = 0.1,
        #Controls how much the rotary arm is pulled back toward center.

        pd_theta_dot: float = 0.03,
        #Damps the arm velocity.

        upright_threshold: float = np.deg2rad(35),
        #Angle range where the controller switches from swing-up mode to stabilization mode.

    ) -> None:
        self.params = params
        self.swing_gain = swing_gain
        self.pd_alpha = pd_alpha
        self.pd_alpha_dot = pd_alpha_dot
        self.pd_theta = pd_theta
        self.pd_theta_dot = pd_theta_dot
        self.upright_threshold = upright_threshold

    def __call__(self, state: np.ndarray) -> np.ndarray:
        theta, alpha, theta_dot, alpha_dot = state
        alpha_error = wrap_angle(alpha)

        if abs(alpha_error) < self.upright_threshold:
            torque = self._inverse_dynamics_pd(theta, alpha_error, theta_dot, alpha_dot)
        else:
            pump_direction = np.sign(alpha_dot * np.cos(alpha_error) + 1e-6)
            torque = self.params.torque_limit * pump_direction
            torque -= self.pd_theta * theta + self.pd_theta_dot * theta_dot

        torque = np.clip(torque, -self.params.torque_limit, self.params.torque_limit)
        return np.array([torque], dtype=np.float32)

    def _inverse_dynamics_pd(
        self,
        theta: float,
        alpha_error: float,
        theta_dot: float,
        alpha_dot: float,
    ) -> float:
        p = self.params
        pendulum_inertia = p.pendulum_mass * p.pendulum_length**2
        desired_alpha_ddot = -self.pd_alpha * alpha_error - self.pd_alpha_dot * alpha_dot
        gravity_term = p.gravity / p.pendulum_length * np.sin(alpha_error)
        damping_term = p.pendulum_damping / pendulum_inertia * alpha_dot
        denominator = p.pendulum_coupling * np.cos(alpha_error)

        if abs(denominator) < 0.15:
            return 0.0

        desired_theta_ddot = (gravity_term - damping_term - desired_alpha_ddot) / denominator
        torque = (
            p.arm_inertia * desired_theta_ddot
            + p.arm_damping * theta_dot
            - p.pendulum_coupling * alpha_dot**2 * np.sin(alpha_error)
            - self.pd_theta * theta
            - self.pd_theta_dot * theta_dot
        )
        return float(torque)
