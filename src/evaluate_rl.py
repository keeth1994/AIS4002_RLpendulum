"""Evaluate a trained Stable-Baselines3 policy and optionally save a video."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.envs import RotaryPendulumEnv
from src.video import save_video_or_gif


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/ppo_rotary_pendulum.zip"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--video", type=str, default="videos/rl_policy.mp4")
    parser.add_argument("--start", choices=["upright", "down", "random"], default="upright")
    args = parser.parse_args()

    model = PPO.load(args.model_path)
    env = RotaryPendulumEnv(max_episode_steps=args.steps, seed=args.seed, start=args.start)
    frames = []
    returns = []

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode, options={"start": args.start})
        episode_return = 0.0
        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            if episode == 0 and step % 2 == 0:
                frames.append(env.render())
            if terminated or truncated:
                break
        returns.append(episode_return)

    video_path = None if args.video.lower() == "none" else Path(args.video)
    if video_path is not None and frames:
        save_video_or_gif(video_path, frames, fps=25)

    print(f"Mean return over {args.episodes} episodes: {np.mean(returns):.3f}")
    print(f"Episode returns: {[round(value, 3) for value in returns]}")


if __name__ == "__main__":
    main()
