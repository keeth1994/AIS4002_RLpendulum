"""Export policy metadata for later QUBE-Servo 2 hardware deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/ppo_rotary_pendulum.zip"))
    parser.add_argument("--output", type=Path, default=Path("models/qube_policy_export.json"))
    args = parser.parse_args()

    model = PPO.load(args.model_path)
    metadata = {
        "source_model": str(args.model_path),
        "algorithm": "PPO",
        "observation": ["theta_rad", "alpha_rad_upright_zero", "theta_dot_rad_s", "alpha_dot_rad_s"],
        "action": "motor_voltage_v",
        "action_limit_v": 5.0,
        "notes": (
            "For hardware, load the SB3 .zip model, build the same 4-value "
            "observation from QUBE encoders, call model.predict(..., deterministic=True), "
            "clip to +/-5 V, and send the result to the motor amplifier."
        ),
        "policy_class": model.policy.__class__.__name__,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote hardware export metadata to {args.output}")


if __name__ == "__main__":
    main()
