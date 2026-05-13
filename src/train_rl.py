"""Train a Stable-Baselines3 agent in the simulated pendulum environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from src.envs import RotaryPendulumEnv

REPORT_BALANCE_PRESET = {
    "algo": "sac",
    "domain_randomization": False,
    "arm_limit_deg": 60.0,
    "initial_perturbation": 0.08,
    "voltage_limit": 5.0,
    "soft_arm_limit": True,
    "sensor_noise": False,
    "max_episode_steps": 1500,
    "reset_mode": "upright",
    "reward_mode": "report_balance",
    "obs_mode": "base6",
    "action_filter_alpha": 1.0,
    "voltage_slew_rate": 0.0,
    "action_rate_penalty": 0.0,
    "motor_dead_voltage": 0.0,
    "learning_rate": 3e-4,
}


def make_env(
    seed: int,
    domain_randomization: bool,
    arm_limit_deg: float,
    initial_perturbation: float,
    voltage_limit: float,
    soft_arm_limit: bool,
    sensor_noise: bool,
    max_episode_steps: int,
    reset_mode: str,
    reward_mode: str,
    obs_mode: str,
    action_filter_alpha: float,
    voltage_slew_rate: float,
    action_rate_penalty: float,
    motor_dead_voltage: float,
) -> Monitor:
    env = RotaryPendulumEnv(
        max_episode_steps=max_episode_steps,
        domain_randomization=domain_randomization,
        seed=seed,
        arm_limit_rad=np.deg2rad(arm_limit_deg),
        initial_perturbation=initial_perturbation,
        voltage_limit=voltage_limit,
        soft_arm_limit=soft_arm_limit,
        terminate_on_arm_limit=not soft_arm_limit,
        sensor_noise=sensor_noise,
        reset_mode=reset_mode,
        reward_mode=reward_mode,
        obs_mode=obs_mode,
        action_filter_alpha=action_filter_alpha,
        voltage_slew_rate=voltage_slew_rate,
        action_rate_penalty=action_rate_penalty,
        motor_dead_voltage=motor_dead_voltage,
    )
    return Monitor(env)


class TextProgressCallback(BaseCallback):
    """Dependency-free training progress printer."""

    def __init__(self, total_timesteps: int, report_every: int = 2500) -> None:
        super().__init__()
        self.total_timesteps = max(1, total_timesteps)
        self.report_every = max(1, report_every)
        self._next_report = self.report_every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_report or self.num_timesteps >= self.total_timesteps:
            percent = 100.0 * min(self.num_timesteps, self.total_timesteps) / self.total_timesteps
            print(f"progress: {self.num_timesteps}/{self.total_timesteps} ({percent:.1f}%)")
            self._next_report += self.report_every
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=["custom", "report-balance"],
        default="custom",
        help="Use a named simulation training setup. 'report-balance' trains the upright RL balance policy used by the hardware residual controller.",
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model-path", type=Path, default=Path("models/sac_qube_classical_env_speed_obs"))
    parser.add_argument("--continue-from", type=Path, default=None)
    parser.add_argument("--domain-randomization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--arm-limit-deg", type=float, default=90.0)
    parser.add_argument("--initial-perturbation", type=float, default=0.25)
    parser.add_argument("--voltage-limit", type=float, default=10.0)
    parser.add_argument("--soft-arm-limit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sensor-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-episode-steps", type=int, default=5000)
    parser.add_argument("--reset-mode", choices=["down", "upright", "mixed"], default="down")
    parser.add_argument(
        "--reward-mode",
        choices=["report_balance", "recovery", "recovery_speed", "hardware_recovery"],
        default="recovery_speed",
    )
    parser.add_argument("--obs-mode", choices=["base6", "speed7", "speed8"], default="speed7")
    parser.add_argument("--action-filter-alpha", type=float, default=1.0)
    parser.add_argument("--voltage-slew-rate", type=float, default=0.0)
    parser.add_argument("--action-rate-penalty", type=float, default=0.0)
    parser.add_argument("--motor-dead-voltage", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    args = parser.parse_args()
    if args.preset == "report-balance":
        for name, value in REPORT_BALANCE_PRESET.items():
            setattr(args, name, value)

    env = make_env(
        args.seed,
        args.domain_randomization,
        args.arm_limit_deg,
        args.initial_perturbation,
        args.voltage_limit,
        args.soft_arm_limit,
        args.sensor_noise,
        args.max_episode_steps,
        args.reset_mode,
        args.reward_mode,
        args.obs_mode,
        args.action_filter_alpha,
        args.voltage_slew_rate,
        args.action_rate_penalty,
        args.motor_dead_voltage,
    )
    if args.check_env:
        check_env(
            RotaryPendulumEnv(
                max_episode_steps=args.max_episode_steps,
                seed=args.seed,
                arm_limit_rad=np.deg2rad(args.arm_limit_deg),
                initial_perturbation=args.initial_perturbation,
                voltage_limit=args.voltage_limit,
                soft_arm_limit=args.soft_arm_limit,
                terminate_on_arm_limit=not args.soft_arm_limit,
                sensor_noise=args.sensor_noise,
                reset_mode=args.reset_mode,
                reward_mode=args.reward_mode,
                obs_mode=args.obs_mode,
                action_filter_alpha=args.action_filter_alpha,
                voltage_slew_rate=args.voltage_slew_rate,
                action_rate_penalty=args.action_rate_penalty,
                motor_dead_voltage=args.motor_dead_voltage,
            ),
            warn=True,
        )

    tensorboard_log = "results/tensorboard" if args.tensorboard else None
    if args.continue_from is not None:
        model_cls = PPO if args.algo == "ppo" else SAC
        model = model_cls.load(
            args.continue_from,
            env=env,
            custom_objects={"learning_rate": args.learning_rate},
        )
    elif args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            tensorboard_log=tensorboard_log,
            learning_rate=args.learning_rate,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            ent_coef=0.01,
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            tensorboard_log=tensorboard_log,
            learning_rate=args.learning_rate,
            buffer_size=300_000,
            learning_starts=1_000,
            batch_size=256,
            gamma=0.995,
            tau=0.01,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
        )
    callback = TextProgressCallback(args.timesteps) if args.progress_bar else None
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    print(f"Saved model to {args.model_path}.zip")


if __name__ == "__main__":
    main()
