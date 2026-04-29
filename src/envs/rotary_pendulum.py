"""Gymnasium environment for the Quanser QUBE-Servo 2 inverted pendulum.

The simulator uses a coupled rotary/Furuta pendulum grey-box model. It is still
compact enough for quick RL training, but the pivot follows the QUBE arm arc
rather than the older straight-line cart surrogate.
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
    arm_damping: float = 0.0001
    pendulum_damping: float = 0.000005
    motor_resistance: float = 4.2
    motor_torque_constant: float = 0.042
    motor_back_emf_constant: float = 0.042
    motor_efficiency: float = 3.0
    motor_force_constant: float = 0.2
    voltage_limit: float = 5.0
    dt: float = 1.0 / 300.0

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

    Internally, the rotary arm is modeled with a coupled Furuta-style mass
    matrix, so motor torque and arm acceleration affect the pendulum through
    the real rotary geometry.
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
        soft_arm_limit: bool = False,
        arm_limit_brake_gain: float = 5.0,
        arm_limit_damping: float = 0.25,
        terminate_on_arm_limit: bool = True,
        sensor_noise: bool = False,
        reset_mode: str = "down",
        reward_mode: str = "report_balance",
    ) -> None:
        super().__init__()
        if render_style not in ("qube", "cartpole"):
            raise ValueError("render_style must be 'qube' or 'cartpole'")
        if reset_mode not in ("down", "upright", "mixed"):
            raise ValueError("reset_mode must be 'down', 'upright', or 'mixed'")
        if reward_mode not in ("report_balance", "recovery"):
            raise ValueError("reward_mode must be 'report_balance' or 'recovery'")
        self.render_mode = render_mode
        self.render_style = render_style
        self.max_episode_steps = max_episode_steps
        self.domain_randomization = domain_randomization
        self.arm_limit_rad = arm_limit_rad
        self.initial_perturbation = initial_perturbation
        self.soft_arm_limit = soft_arm_limit
        self.arm_limit_brake_gain = arm_limit_brake_gain
        self.arm_limit_damping = arm_limit_damping
        self.terminate_on_arm_limit = terminate_on_arm_limit
        self.sensor_noise = sensor_noise
        self.reset_mode = reset_mode
        self.reward_mode = reward_mode
        self.base_params = RotaryPendulumParams()
        if voltage_limit is not None:
            self.base_params = replace(self.base_params, voltage_limit=voltage_limit)
        self.params = self.base_params
        self.state = np.zeros(4, dtype=np.float64)
        self.steps = 0
        self.last_termination_reason = "running"
        self.was_upright = False
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
        theta_dot = self.np_random.uniform(-0.2, 0.2)
        reset_mode = self.reset_mode
        if reset_mode == "mixed":
            reset_mode = str(self.np_random.choice(["down", "upright"]))
        if reset_mode == "upright":
            alpha = self.np_random.uniform(-self.initial_perturbation, self.initial_perturbation)
            alpha_dot = self.np_random.uniform(-0.6, 0.6)
        else:
            side = self.np_random.choice([-1.0, 1.0])
            alpha = np.pi + side * self.np_random.uniform(0.03, self.initial_perturbation)
            alpha_dot = side * self.np_random.uniform(0.1, 1.0)
        theta = self.np_random.uniform(-0.05, 0.05)
        self.state = np.array([theta, alpha, theta_dot, alpha_dot], dtype=np.float64)
        self.state[:2] = wrap_angle(self.state[:2])
        self.steps = 0
        self.last_termination_reason = "running"
        self.was_upright = bool(abs(wrap_angle(self.state[1])) < np.deg2rad(10.0))
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray | list[float] | float) -> tuple[np.ndarray, float, bool, bool, dict]:
        motor_command = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        motor_command = float(np.clip(motor_command, -1.0, 1.0))
        voltage = motor_command * self.params.voltage_limit
        voltage = self._apply_soft_arm_limit(voltage, self.state)

        self.state = self._rk4(self.state, voltage, self.params.dt)
        self.state[0] = wrap_angle(self.state[0])
        self.state[1] = wrap_angle(self.state[1])
        self.state[2:] = np.clip(self.state[2:], [-30.0, -40.0], [30.0, 40.0])
        self.steps += 1

        theta, alpha, theta_dot, alpha_dot = self.state
        alpha_error = float(wrap_angle(alpha))
        theta_ratio = theta / max(self.arm_limit_rad, 1e-9)
        near_limit = max(abs(theta_ratio) - 0.8, 0.0)
        over_limit = max(abs(theta) - self.arm_limit_rad, 0.0)

        if self.reward_mode == "report_balance":
            # Report-style quadratic reward for balance control:
            # r = F_k - 0.1 * (theta^2 + alpha^2 + theta_dot^2 + alpha_dot^2 + u^2)
            # where F_k = 1 while state/action constraints are satisfied.
            alpha_limit_rad = np.deg2rad(10.0)
            arm_ok = abs(theta) <= self.arm_limit_rad
            alpha_ok = abs(alpha_error) <= alpha_limit_rad
            voltage_ok = abs(voltage) <= self.params.voltage_limit + 1e-9
            constraints_ok = arm_ok and alpha_ok and voltage_ok

            reward = (
                (1.0 if constraints_ok else 0.0)
                - 0.1
                * (
                    theta**2
                    + alpha_error**2
                    + theta_dot**2
                    + alpha_dot**2
                    + voltage**2
                )
            )

            if not constraints_ok:
                reward -= 50.0

            terminated = self.terminate_on_arm_limit and (not constraints_ok)
            if terminated:
                reward -= 1000.0
                if not arm_ok:
                    self.last_termination_reason = "arm_limit"
                elif not alpha_ok:
                    self.last_termination_reason = "alpha_limit"
                else:
                    self.last_termination_reason = "voltage_limit"
        else:
            pendulum_energy = (
                0.5 * self.params.pendulum_inertia * alpha_dot**2
                + self.params.pendulum_mass
                * self.params.gravity
                * self.params.pendulum_com
                * (np.cos(alpha_error) + 1.0)
            )
            target_energy = (
                2.0
                * self.params.pendulum_mass
                * self.params.gravity
                * self.params.pendulum_com
            )
            energy_error = (pendulum_energy - target_energy) / max(target_energy, 1e-9)
            energy_score = np.exp(-energy_error**2)

            abs_alpha_error = abs(alpha_error)
            upright_score = np.exp(-(alpha_error / np.deg2rad(18.0)) ** 2)
            tight_upright_score = np.exp(-(alpha_error / np.deg2rad(8.0)) ** 2)
            balanced_score = (
                tight_upright_score
                * np.exp(-0.18 * alpha_dot**2)
                * np.exp(-0.10 * theta_dot**2)
                * np.exp(-4.0 * theta_ratio**2)
            )
            swing_weight = 1.0 - upright_score
            capture_bonus = float(abs_alpha_error < np.deg2rad(10.0)) * np.exp(
                -0.08 * theta_dot**2 - 0.12 * alpha_dot**2
            )
            recapture_bonus = 3.0 if abs_alpha_error < np.deg2rad(10.0) and not self.was_upright else 0.0

            reward = (
                3.0 * swing_weight * energy_score
                + 1.75 * upright_score
                + 4.0 * tight_upright_score
                + 16.0 * balanced_score
                + 3.0 * capture_bonus
                + recapture_bonus
                - 0.45 * theta_ratio**2
                - 10.0 * near_limit**2
                - 70.0 * over_limit**2
                - 0.0012 * theta_dot**2
                - 0.0012 * alpha_dot**2
                - 0.0010 * voltage**2
            )

            terminated = self.terminate_on_arm_limit and abs(theta) > self.arm_limit_rad
            if terminated:
                reward -= 1000.0
                self.last_termination_reason = "arm_limit"
            self.was_upright = abs_alpha_error < np.deg2rad(10.0)
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
        """Return a cart-pole-like debug view of the rotary state."""
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

        # Quanser's rotary-pendulum EOM uses alpha=0 at the hanging-down
        # position. The project state uses alpha=0 upright, so convert before
        # applying the nonlinear coupled equations.
        alpha_down = float(wrap_angle(alpha - np.pi))
        sin_alpha = np.sin(alpha_down)
        cos_alpha = np.cos(alpha_down)

        jr = p.arm_inertia + p.pendulum_mass * p.arm_length**2
        jp = p.pendulum_inertia
        coupling = p.pendulum_mass * p.arm_length * p.pendulum_com * cos_alpha
        rotary_inertia = jr + jp * sin_alpha**2

        motor_torque = (
            p.motor_efficiency
            *
            p.motor_torque_constant
            / max(p.motor_resistance, 1e-9)
            * (voltage - p.motor_back_emf_constant * theta_dot)
        )

        mass_matrix = np.array(
            [
                [rotary_inertia, coupling],
                [coupling, jp],
            ],
            dtype=np.float64,
        )

        coriolis = jp * sin_alpha * cos_alpha
        rhs = np.array(
            [
                motor_torque
                - p.arm_damping * theta_dot
                - 2.0 * coriolis * theta_dot * alpha_dot
                + p.pendulum_mass * p.arm_length * p.pendulum_com * sin_alpha * alpha_dot**2,
                -p.pendulum_damping * alpha_dot
                + coriolis * theta_dot**2
                - p.pendulum_mass * p.gravity * p.pendulum_com * sin_alpha
            ],
            dtype=np.float64,
        )

        det = float(np.linalg.det(mass_matrix))
        if abs(det) < 1e-9:
            theta_ddot = 0.0
            alpha_ddot = 0.0
        else:
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
        obs = np.array(
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
        if self.sensor_noise:
            obs += np.array(
                [
                    self.np_random.normal(0.0, 0.001),
                    self.np_random.normal(0.0, 0.001),
                    self.np_random.normal(0.0, 0.001),
                    self.np_random.normal(0.0, 0.001),
                    self.np_random.normal(0.0, 0.05),
                    self.np_random.normal(0.0, 0.075),
                ],
                dtype=np.float32,
            )
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

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

    def _apply_soft_arm_limit(self, voltage: float, state: np.ndarray) -> float:
        if not self.soft_arm_limit:
            return voltage

        theta, _, theta_dot, _ = state
        limited_voltage = voltage
        if theta > self.arm_limit_rad:
            brake_voltage = (
                -self.arm_limit_brake_gain * (theta - self.arm_limit_rad)
                - self.arm_limit_damping * theta_dot
            )
            limited_voltage = min(limited_voltage, brake_voltage)
        elif theta < -self.arm_limit_rad:
            brake_voltage = (
                -self.arm_limit_brake_gain * (theta + self.arm_limit_rad)
                - self.arm_limit_damping * theta_dot
            )
            limited_voltage = max(limited_voltage, brake_voltage)

        return float(np.clip(limited_voltage, -self.params.voltage_limit, self.params.voltage_limit))

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
