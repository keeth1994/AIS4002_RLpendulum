"""Export policy metadata for later QUBE-Servo 2 hardware deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/ppo_qube_swingup.zip"))
    parser.add_argument("--output", type=Path, default=Path("models/qube_policy_export.json"))
    args = parser.parse_args()

    model = PPO.load(args.model_path)
    metadata = {
        "source_model": str(args.model_path),
        "algorithm": "PPO",
        "observation": [
            "sin(theta)",
            "cos(theta)",
            "sin(alpha)",
            "cos(alpha)",
            "theta_dot_rad_s",
            "alpha_dot_rad_s",
        ],
        "action": "normalized_motor_command",
        "action_range": [-1.0, 1.0],
        "hardware_scaling": "motor_voltage_v = normalized_motor_command * chosen_voltage_limit_v",
        "notes": (
            "For hardware, load the SB3 .zip model, build the same 6-value "
            "observation from QUBE encoders, call model.predict(..., deterministic=True), "
            "clip the normalized action to +/-1, multiply by a conservative voltage "
            "limit, and send the result to the motor amplifier."
        ),
        "policy_class": model.policy.__class__.__name__,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote hardware export metadata to {args.output}")


if __name__ == "__main__":
    main()
