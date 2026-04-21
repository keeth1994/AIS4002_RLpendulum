"""Gymnasium environment for the Quanser QUBE-Servo 2 inverted pendulum.

The simulator uses a voltage-driven Furuta pendulum approximation with common
QUBE-Servo 2 parameters. It is not a vendor-certified hardware model, but the
state/action interface is chosen for sim-to-real: encoder states in, motor
voltage out.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class RotaryPendulumParams:
    """Nominal QUBE-Servo 2 rotary pendulum parameters.

    Parameter notes:
    - r/arm_length is the motor axis to pendulum pivot distance.
    - pendulum_length is the total link length.
    - pendulum_com is assumed to be Lp / 2.
    - motor voltage is clipped to +/-5 V for hardware-style safety.
    """

    gravity: float = 9.81
    arm_mass: float = 0.095
    arm_length: float = 0.085
    pendulum_mass: float = 0.024
    pendulum_length: float = 0.129
    arm_damping: float = 0.001
    pendulum_damping: float = 0.00005
    motor_resistance: float = 8.4
    motor_torque_constant: float = 0.042
    motor_back_emf_constant: float = 0.042
    voltage_limit: float = 5.0
    dt: float = 0.01

    @property
    def pendulum_com(self) -> float:
        return self.pendulum_length / 2.0

    @property
    def arm_inertia(self) -> float:
        return self.arm_mass * self.arm_length**2 / 3.0

    @property
    def pendulum_inertia(self) -> float:
        return self.pendulum_mass * self.pendulum_length**2 / 3.0


class RotaryPendulumEnv(gym.Env):
    """Voltage-controlled QUBE-Servo 2 inverted pendulum environment.

    Environment convention:
    - theta: rotary arm angle, where 0 is centered.
    - alpha: pendulum angle, where 0 is upright and pi is hanging downward.
    - action: motor voltage command in volts, clipped to +/-5 V.
    - observation: [theta, alpha, theta_dot, alpha_dot].

    The internal Furuta equations are expressed in the common Quanser modelling
    convention where the pendulum angle is zero downward, so alpha is converted
    internally by alpha_down = alpha + pi.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
        domain_randomization: bool = False,
        seed: int | None = None,
        start: str = "down",
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.domain_randomization = domain_randomization
        self.default_start = start
        self.base_params = RotaryPendulumParams()
        self.params = self.base_params
        self.state = np.zeros(4, dtype=np.float64)
        self.steps = 0
        self.np_random = np.random.default_rng(seed)

        high = np.array([np.pi, np.pi, 30.0, 40.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-self.base_params.voltage_limit], dtype=np.float32),
            high=np.array([self.base_params.voltage_limit], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.params = self._sample_params() if self.domain_randomization else self.base_params
        options = options or {}
        start = options.get("start", self.default_start)

        if start == "upright":
            alpha = self.np_random.normal(0.0, 0.05)
            alpha_dot = self.np_random.normal(0.0, 0.05)
        elif start == "random":
            alpha = self.np_random.uniform(-np.pi, np.pi)
            alpha_dot = self.np_random.uniform(-1.0, 1.0)
        else:
            alpha = np.pi + self.np_random.normal(0.0, 0.05)
            alpha_dot = self.np_random.normal(0.0, 0.05)

        theta = self.np_random.normal(0.0, 0.02)
        theta_dot = self.np_random.normal(0.0, 0.05)
        self.state = np.array([theta, alpha, theta_dot, alpha_dot], dtype=np.float64)
        self.state[:2] = wrap_angle(self.state[:2])
        self.steps = 0
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray | list[float] | float) -> tuple[np.ndarray, float, bool, bool, dict]:
        voltage = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        voltage = float(np.clip(voltage, -self.params.voltage_limit, self.params.voltage_limit))

        self.state = self._rk4(self.state, voltage, self.params.dt)
        self.state[0] = wrap_angle(self.state[0])
        self.state[1] = wrap_angle(self.state[1])
        self.state[2:] = np.clip(self.state[2:], [-30.0, -40.0], [30.0, 40.0])
        self.steps += 1

        theta, alpha, theta_dot, alpha_dot = self.state
        alpha_error = float(wrap_angle(alpha))
        reward = (
            4.0 * np.cos(alpha_error)
            - 0.8 * theta**2
            - 0.02 * theta_dot**2
            - 0.01 * alpha_dot**2
            - 0.001 * voltage**2
        )

        terminated = abs(theta) > np.deg2rad(60)
        truncated = self.steps >= self.max_episode_steps
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), self._get_info(voltage)

    def render(self) -> np.ndarray:
        """Return a simple top/side visualisation as an RGB frame."""
        frame = np.full((480, 640, 3), 245, dtype=np.uint8)
        theta, alpha, _, _ = self.state
        origin = np.array([320, 240], dtype=np.int32)
        arm_len = 125
        pend_len = 145

        arm_tip = origin + np.array([arm_len * np.cos(theta), arm_len * np.sin(theta)], dtype=np.int32)
        bob = arm_tip + np.array([pend_len * np.sin(alpha), -pend_len * np.cos(alpha)], dtype=np.int32)

        self._draw_line(frame, np.array([0, 240]), np.array([639, 240]), (220, 220, 220), width=1)
        self._draw_circle(frame, origin, 8, (30, 30, 30))
        self._draw_line(frame, origin, arm_tip, (180, 90, 20), width=5)
        self._draw_circle(frame, arm_tip, 7, (180, 90, 20))
        self._draw_line(frame, arm_tip, bob, (30, 80, 190), width=4)
        self._draw_circle(frame, bob, 13, (30, 80, 190))
        return frame

    def _dynamics(self, state: np.ndarray, voltage: float) -> np.ndarray:
        theta, alpha, theta_dot, alpha_dot = state
        p = self.params

        alpha_down = wrap_angle(alpha + np.pi)
        sin_a = np.sin(alpha_down)
        cos_a = np.cos(alpha_down)

        jr = p.arm_inertia
        jp = p.pendulum_inertia
        m = p.pendulum_mass
        r = p.arm_length
        l = p.pendulum_com
        torque = p.motor_torque_constant * (
            voltage - p.motor_back_emf_constant * theta_dot
        ) / p.motor_resistance

        mass_matrix = np.array(
            [
                [jr + jp * sin_a**2, m * l * r * cos_a],
                [m * l * r * cos_a, jp],
            ],
            dtype=np.float64,
        )
        rhs = np.array(
            [
                torque
                - p.arm_damping * theta_dot
                - 2.0 * jp * sin_a * cos_a * theta_dot * alpha_dot
                + m * l * r * sin_a * alpha_dot**2,
                -p.pendulum_damping * alpha_dot
                + jp * sin_a * cos_a * theta_dot**2
                - m * p.gravity * l * sin_a,
            ],
            dtype=np.float64,
        )
        theta_ddot, alpha_ddot = np.linalg.solve(mass_matrix, rhs)
        return np.array([theta_dot, alpha_dot, theta_ddot, alpha_ddot], dtype=np.float64)

    def _rk4(self, state: np.ndarray, voltage: float, dt: float) -> np.ndarray:
        k1 = self._dynamics(state, voltage)
        k2 = self._dynamics(state + 0.5 * dt * k1, voltage)
        k3 = self._dynamics(state + 0.5 * dt * k2, voltage)
        k4 = self._dynamics(state + dt * k3, voltage)
        return state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _get_obs(self) -> np.ndarray:
        theta, alpha, theta_dot, alpha_dot = self.state
        return np.array([theta, alpha, theta_dot, alpha_dot], dtype=np.float32)

    def _get_info(self, voltage: float = 0.0) -> dict:
        theta, alpha, theta_dot, alpha_dot = self.state
        return {
            "theta": float(theta),
            "alpha": float(alpha),
            "theta_dot": float(theta_dot),
            "alpha_dot": float(alpha_dot),
            "voltage": float(voltage),
            "is_upright": bool(abs(wrap_angle(alpha)) < np.deg2rad(10)),
        }

    def _sample_params(self) -> RotaryPendulumParams:
        p = self.base_params
        scale = self.np_random.uniform
        return RotaryPendulumParams(
            gravity=p.gravity,
            arm_mass=p.arm_mass * scale(0.85, 1.15),
            arm_length=p.arm_length * scale(0.9, 1.1),
            pendulum_mass=p.pendulum_mass * scale(0.85, 1.15),
            pendulum_length=p.pendulum_length * scale(0.9, 1.1),
            arm_damping=p.arm_damping * scale(0.5, 2.0),
            pendulum_damping=p.pendulum_damping * scale(0.5, 2.0),
            motor_resistance=p.motor_resistance * scale(0.9, 1.1),
            motor_torque_constant=p.motor_torque_constant * scale(0.9, 1.1),
            motor_back_emf_constant=p.motor_back_emf_constant * scale(0.9, 1.1),
            voltage_limit=p.voltage_limit,
            dt=p.dt,
        )

    def _draw_line(
        self,
        frame: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        color: tuple[int, int, int],
        width: int = 1,
    ) -> None:
        distance = int(np.linalg.norm(end - start))
        if distance <= 0:
            return
        xs = np.linspace(start[0], end[0], distance).astype(np.int32)
        ys = np.linspace(start[1], end[1], distance).astype(np.int32)
        for offset_x in range(-width, width + 1):
            for offset_y in range(-width, width + 1):
                x = np.clip(xs + offset_x, 0, frame.shape[1] - 1)
                y = np.clip(ys + offset_y, 0, frame.shape[0] - 1)
                frame[y, x] = color

    def _draw_circle(
        self,
        frame: np.ndarray,
        center: np.ndarray,
        radius: int,
        color: tuple[int, int, int],
    ) -> None:
        y_grid, x_grid = np.ogrid[: frame.shape[0], : frame.shape[1]]
        mask = (x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2 <= radius**2
        frame[mask] = color


QubeServo2PendulumEnv = RotaryPendulumEnv
