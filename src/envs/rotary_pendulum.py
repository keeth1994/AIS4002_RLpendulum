"""Gymnasium environment for a simplified Quanser QUBE rotary pendulum."""

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
    """Physical parameters for the simplified rotary pendulum simulator."""

    gravity: float = 9.81
    pendulum_mass: float = 0.095
    pendulum_length: float = 0.129
    pendulum_damping: float = 0.002
    arm_inertia: float = 0.002
    arm_damping: float = 0.02
    pendulum_coupling: float = 1.0
    torque_limit: float = 0.35
    dt: float = 0.02


class RotaryPendulumEnv(gym.Env):
    """Torque-controlled rotary inverted pendulum environment.

    State convention:
    - theta: rotary arm angle, wrapped to [-pi, pi]
    - alpha: pendulum angle, where 0 is upright and pi is hanging downward
    - theta_dot: rotary arm angular velocity
    - alpha_dot: pendulum angular velocity

    Observation uses sin/cos for the two angles to avoid discontinuities:
    [sin(theta), cos(theta), sin(alpha), cos(alpha), theta_dot, alpha_dot]
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
        domain_randomization: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.domain_randomization = domain_randomization
        self.base_params = RotaryPendulumParams()
        self.params = self.base_params
        self.state = np.zeros(4, dtype=np.float64)
        self.steps = 0
        self.np_random = np.random.default_rng(seed)

        high = np.array([1.0, 1.0, 1.0, 1.0, 30.0, 40.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-self.base_params.torque_limit], dtype=np.float32),
            high=np.array([self.base_params.torque_limit], dtype=np.float32),
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
        start = options.get("start", "down")

        if start == "upright":
            alpha = self.np_random.normal(0.0, 0.08)
            alpha_dot = self.np_random.normal(0.0, 0.05)
        elif start == "random":
            alpha = self.np_random.uniform(-np.pi, np.pi)
            alpha_dot = self.np_random.uniform(-1.0, 1.0)
        else:
            alpha = np.pi + self.np_random.normal(0.0, 0.08)
            alpha_dot = self.np_random.normal(0.0, 0.05)

        theta = self.np_random.normal(0.0, 0.05)
        theta_dot = self.np_random.normal(0.0, 0.05)
        self.state = np.array([theta, alpha, theta_dot, alpha_dot], dtype=np.float64)
        self.state[:2] = wrap_angle(self.state[:2])
        self.steps = 0
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray | list[float] | float) -> tuple[np.ndarray, float, bool, bool, dict]:
        torque = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        torque = float(np.clip(torque, -self.params.torque_limit, self.params.torque_limit))

        dt = self.params.dt
        self.state = self._rk4(self.state, torque, dt)
        self.state[0] = wrap_angle(self.state[0])
        self.state[1] = wrap_angle(self.state[1])
        self.state[2:] = np.clip(self.state[2:], [-30.0, -40.0], [30.0, 40.0])
        self.steps += 1

        theta, alpha, theta_dot, alpha_dot = self.state
        upright_error = float(wrap_angle(alpha))
        reward = (
            2.0 * np.cos(upright_error)
            - 0.08 * theta**2
            - 0.015 * theta_dot**2
            - 0.01 * alpha_dot**2
            - 0.08 * torque**2
        )

        terminated = abs(theta) > np.deg2rad(720)
        truncated = self.steps >= self.max_episode_steps
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), self._get_info(torque)

    def render(self) -> np.ndarray:
        """Return a simple top/side visualisation as an RGB frame."""
        frame = np.full((480, 640, 3), 245, dtype=np.uint8)
        theta, alpha, _, _ = self.state
        origin = np.array([320, 240], dtype=np.int32)
        arm_len = 125
        pend_len = 145

        arm_tip = origin + np.array([arm_len * np.cos(theta), arm_len * np.sin(theta)], dtype=np.int32)
        bob = arm_tip + np.array([pend_len * np.sin(alpha), pend_len * np.cos(alpha)], dtype=np.int32)

        self._draw_line(frame, np.array([0, 240]), np.array([639, 240]), (220, 220, 220), width=1)
        self._draw_circle(frame, origin, 8, (30, 30, 30))
        self._draw_line(frame, origin, arm_tip, (180, 90, 20), width=5)
        self._draw_circle(frame, arm_tip, 7, (180, 90, 20))
        self._draw_line(frame, arm_tip, bob, (30, 80, 190), width=4)
        self._draw_circle(frame, bob, 13, (30, 80, 190))
        return frame

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

    def _dynamics(self, state: np.ndarray, torque: float) -> np.ndarray:
        theta, alpha, theta_dot, alpha_dot = state
        p = self.params
        pendulum_inertia = p.pendulum_mass * p.pendulum_length**2
        coupling = p.pendulum_coupling * np.cos(alpha)

        theta_ddot = (
            torque
            - p.arm_damping * theta_dot
            + coupling * alpha_dot**2 * np.sin(alpha)
        ) / p.arm_inertia
        alpha_ddot = (
            p.gravity / p.pendulum_length * np.sin(alpha)
            - p.pendulum_damping / pendulum_inertia * alpha_dot
            - coupling * theta_ddot * np.cos(alpha)
        )
        return np.array([theta_dot, alpha_dot, theta_ddot, alpha_ddot], dtype=np.float64)

    def _rk4(self, state: np.ndarray, torque: float, dt: float) -> np.ndarray:
        k1 = self._dynamics(state, torque)
        k2 = self._dynamics(state + 0.5 * dt * k1, torque)
        k3 = self._dynamics(state + 0.5 * dt * k2, torque)
        k4 = self._dynamics(state + dt * k3, torque)
        return state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _get_obs(self) -> np.ndarray:
        theta, alpha, theta_dot, alpha_dot = self.state
        return np.array(
            [
                np.sin(theta),
                np.cos(theta),
                np.sin(alpha),
                np.cos(alpha),
                theta_dot,
                alpha_dot,
            ],
            dtype=np.float32,
        )

    def _get_info(self, torque: float = 0.0) -> dict:
        theta, alpha, theta_dot, alpha_dot = self.state
        return {
            "theta": float(theta),
            "alpha": float(alpha),
            "theta_dot": float(theta_dot),
            "alpha_dot": float(alpha_dot),
            "torque": float(torque),
            "is_upright": bool(abs(wrap_angle(alpha)) < np.deg2rad(12)),
        }

    def _sample_params(self) -> RotaryPendulumParams:
        p = self.base_params
        scale = self.np_random.uniform
        return RotaryPendulumParams(
            gravity=p.gravity,
            pendulum_mass=p.pendulum_mass * scale(0.85, 1.15),
            pendulum_length=p.pendulum_length * scale(0.9, 1.1),
            pendulum_damping=p.pendulum_damping * scale(0.5, 2.0),
            arm_inertia=p.arm_inertia * scale(0.75, 1.25),
            arm_damping=p.arm_damping * scale(0.5, 2.0),
            pendulum_coupling=p.pendulum_coupling * scale(0.8, 1.2),
            torque_limit=p.torque_limit,
            dt=p.dt,
        )
