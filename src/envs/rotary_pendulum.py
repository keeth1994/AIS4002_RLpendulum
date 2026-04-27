"""Gymnasium environment for the Quanser QUBE-Servo 2 inverted pendulum.

The simulator uses a CartPole-style surrogate model adapted to the QUBE rotary
arm. This is intentionally simpler than a full Furuta pendulum model and is
useful for learning/debugging swing-up behavior before moving toward hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

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
    motor_efficiency: float = 2.5
    motor_force_constant: float = 0.2
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
    - action: normalized motor command in [-1, 1], scaled to voltage_limit.
    - observation: [sin(theta), cos(theta), sin(alpha), cos(alpha), theta_dot, alpha_dot].

    Internally, the rotary arm is approximated as a cart moving along the local
    tangent direction, with x = arm_length * theta.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
        domain_randomization: bool = False,
        seed: int | None = None,
        arm_limit_rad: float = np.deg2rad(60),
        initial_perturbation: float = 0.25,
        voltage_limit: float | None = None,
        render_style: str = "qube",
    ) -> None:
        super().__init__()
        if render_style not in ("qube", "cartpole"):
            raise ValueError("render_style must be 'qube' or 'cartpole'")
        self.render_mode = render_mode
        self.render_style = render_style
        self.max_episode_steps = max_episode_steps
        self.domain_randomization = domain_randomization
        self.arm_limit_rad = arm_limit_rad
        self.initial_perturbation = initial_perturbation
        self.base_params = RotaryPendulumParams()
        if voltage_limit is not None:
            self.base_params = replace(self.base_params, voltage_limit=voltage_limit)
        self.params = self.base_params
        self.state = np.zeros(4, dtype=np.float64)
        self.steps = 0
        self.last_termination_reason = "running"
        self.np_random = np.random.default_rng(seed)

        high = np.array([1.0, 1.0, 1.0, 1.0, 30.0, 40.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
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
        side = self.np_random.choice([-1.0, 1.0])
        alpha = np.pi + side * self.np_random.uniform(0.03, self.initial_perturbation)
        alpha_dot = side * self.np_random.uniform(0.1, 1.0)

        theta = self.np_random.uniform(-0.05, 0.05)
        theta_dot = self.np_random.uniform(-0.2, 0.2)
        self.state = np.array([theta, alpha, theta_dot, alpha_dot], dtype=np.float64)
        self.state[:2] = wrap_angle(self.state[:2])
        self.steps = 0
        self.last_termination_reason = "running"
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray | list[float] | float) -> tuple[np.ndarray, float, bool, bool, dict]:
        motor_command = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        motor_command = float(np.clip(motor_command, -1.0, 1.0))
        voltage = motor_command * self.params.voltage_limit

        self.state = self._rk4(self.state, voltage, self.params.dt)
        self.state[0] = wrap_angle(self.state[0])
        self.state[1] = wrap_angle(self.state[1])
        self.state[2:] = np.clip(self.state[2:], [-30.0, -40.0], [30.0, 40.0])
        self.steps += 1

        theta, alpha, theta_dot, alpha_dot = self.state
        alpha_error = float(wrap_angle(alpha))
        is_upright = float(abs(alpha_error) < np.deg2rad(12))
        reward = (
            6.0 * np.cos(alpha_error)
            + 4.0 * is_upright
            - 0.15 * theta**2
            - 2.0 * (theta / self.arm_limit_rad) ** 2
            - 0.001 * theta_dot**2
            - 0.0005 * alpha_dot**2
            - 0.005 * voltage**2
        )

        terminated = abs(theta) > self.arm_limit_rad
        if terminated:
            reward -= 1000.0
            self.last_termination_reason = "arm_limit"
        truncated = self.steps >= self.max_episode_steps
        if truncated:
            self.last_termination_reason = "time_limit"
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), self._get_info(voltage)

    def render(self) -> np.ndarray:
        """Return a simple top/side visualisation as an RGB frame."""
        if self.render_style == "cartpole":
            return self._render_cartpole()
        return self._render_qube()

    def _render_qube(self) -> np.ndarray:
        """Return a QUBE-style rotary arm visualisation."""
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

    def _render_cartpole(self) -> np.ndarray:
        """Return a cart-pole view of the surrogate x = arm_length * theta model."""
        frame = np.full((480, 640, 3), 245, dtype=np.uint8)
        theta, alpha, _, _ = self.state
        rail_y = 305
        rail_margin = 70
        rail_start = np.array([rail_margin, rail_y], dtype=np.int32)
        rail_end = np.array([640 - rail_margin, rail_y], dtype=np.int32)
        usable_width = rail_end[0] - rail_start[0]
        theta_fraction = np.clip(theta / self.arm_limit_rad, -1.0, 1.0)
        cart_x = int(320 + theta_fraction * usable_width * 0.5)
        cart_center = np.array([cart_x, rail_y - 18], dtype=np.int32)
        cart_half_width = 34
        cart_half_height = 18
        pole_len = 160
        pole_tip = cart_center + np.array(
            [pole_len * np.sin(alpha), -pole_len * np.cos(alpha)],
            dtype=np.int32,
        )

        self._draw_line(frame, rail_start, rail_end, (150, 150, 150), width=2)
        self._draw_rect(
            frame,
            cart_center - np.array([cart_half_width, cart_half_height], dtype=np.int32),
            cart_center + np.array([cart_half_width, cart_half_height], dtype=np.int32),
            (190, 110, 35),
        )
        self._draw_line(frame, cart_center, pole_tip, (30, 80, 190), width=5)
        self._draw_circle(frame, cart_center, 7, (40, 40, 40))
        self._draw_circle(frame, pole_tip, 13, (30, 80, 190))
        return frame

    def _dynamics(self, state: np.ndarray, voltage: float) -> np.ndarray:
        theta, alpha, theta_dot, alpha_dot = state
        p = self.params

        # CartPole-style surrogate: x = r * theta and force = K_v * voltage.
        total_mass = p.arm_mass + p.pendulum_mass
        polemass_length = p.pendulum_mass * p.pendulum_com
        force = p.motor_force_constant * voltage - p.arm_damping * theta_dot
        sin_a = np.sin(alpha)
        cos_a = np.cos(alpha)

        temp = (
            force
            + polemass_length * alpha_dot**2 * sin_a
        ) / total_mass
        alpha_ddot = (
            p.gravity * sin_a
            - cos_a * temp
            - p.pendulum_damping * alpha_dot / max(polemass_length, 1e-9)
        ) / (
            p.pendulum_com
            * (4.0 / 3.0 - p.pendulum_mass * cos_a**2 / total_mass)
        )
        x_ddot = temp - polemass_length * alpha_ddot * cos_a / total_mass
        theta_ddot = x_ddot / p.arm_length
        return np.array([theta_dot, alpha_dot, theta_ddot, alpha_ddot], dtype=np.float64)

    def _rk4(self, state: np.ndarray, voltage: float, dt: float) -> np.ndarray:
        k1 = self._dynamics(state, voltage)
        k2 = self._dynamics(state + 0.5 * dt * k1, voltage)
        k3 = self._dynamics(state + 0.5 * dt * k2, voltage)
        k4 = self._dynamics(state + dt * k3, voltage)
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

    def _get_info(self, voltage: float = 0.0) -> dict:
        theta, alpha, theta_dot, alpha_dot = self.state
        return {
            "theta": float(theta),
            "alpha": float(alpha),
            "theta_dot": float(theta_dot),
            "alpha_dot": float(alpha_dot),
            "voltage": float(voltage),
            "is_upright": bool(abs(wrap_angle(alpha)) < np.deg2rad(10)),
            "termination_reason": self.last_termination_reason,
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
            motor_efficiency=p.motor_efficiency * scale(0.8, 1.2),
            motor_force_constant=p.motor_force_constant * scale(0.8, 1.2),
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

    def _draw_rect(
        self,
        frame: np.ndarray,
        top_left: np.ndarray,
        bottom_right: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        x0 = int(np.clip(top_left[0], 0, frame.shape[1] - 1))
        y0 = int(np.clip(top_left[1], 0, frame.shape[0] - 1))
        x1 = int(np.clip(bottom_right[0], 0, frame.shape[1] - 1))
        y1 = int(np.clip(bottom_right[1], 0, frame.shape[0] - 1))
        frame[y0 : y1 + 1, x0 : x1 + 1] = color


QubeServo2PendulumEnv = RotaryPendulumEnv
