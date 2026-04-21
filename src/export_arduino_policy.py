"""Export a trained SB3 PPO policy to a small Arduino-compatible C++ header."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


def format_array(name: str, values: np.ndarray) -> str:
    flat = values.astype(np.float32).reshape(-1)
    body = ", ".join(f"{value:.9g}f" for value in flat)
    return f"const float {name}[{flat.size}] = {{{body}}};"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/ppo_qube_balance_500k.zip"))
    parser.add_argument("--output", type=Path, default=Path("arduino/qube_policy_runner/qube_policy.h"))
    parser.add_argument("--voltage-limit", type=float, default=2.0)
    args = parser.parse_args()

    model = PPO.load(args.model_path, device="cpu")
    state_dict = model.policy.state_dict()

    required = [
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.policy_net.0.bias",
        "mlp_extractor.policy_net.2.weight",
        "mlp_extractor.policy_net.2.bias",
        "action_net.weight",
        "action_net.bias",
    ]
    missing = [key for key in required if key not in state_dict]
    if missing:
        raise RuntimeError(f"Unsupported policy architecture. Missing keys: {missing}")

    arrays = {
        "W1": state_dict["mlp_extractor.policy_net.0.weight"].detach().cpu().numpy(),
        "B1": state_dict["mlp_extractor.policy_net.0.bias"].detach().cpu().numpy(),
        "W2": state_dict["mlp_extractor.policy_net.2.weight"].detach().cpu().numpy(),
        "B2": state_dict["mlp_extractor.policy_net.2.bias"].detach().cpu().numpy(),
        "W3": state_dict["action_net.weight"].detach().cpu().numpy(),
        "B3": state_dict["action_net.bias"].detach().cpu().numpy(),
    }

    if arrays["W1"].shape != (64, 4) or arrays["W2"].shape != (64, 64) or arrays["W3"].shape != (1, 64):
        raise RuntimeError(
            "Expected default SB3 PPO MlpPolicy architecture 4 -> 64 -> 64 -> 1. "
            f"Got W1={arrays['W1'].shape}, W2={arrays['W2'].shape}, W3={arrays['W3'].shape}."
        )

    header = f"""// Auto-generated from {args.model_path}
// Deterministic PPO actor for QUBE-Servo 2 balance.
// Observation order: theta, alpha, theta_dot, alpha_dot.
// Action output: motor voltage [V], clipped here to +/-{args.voltage_limit}.
#pragma once

#include <math.h>

const int QUBE_OBS_DIM = 4;
const int QUBE_HIDDEN_DIM = 64;
const float QUBE_POLICY_VOLTAGE_LIMIT = {args.voltage_limit:.9g}f;

{format_array("QUBE_W1", arrays["W1"])}
{format_array("QUBE_B1", arrays["B1"])}
{format_array("QUBE_W2", arrays["W2"])}
{format_array("QUBE_B2", arrays["B2"])}
{format_array("QUBE_W3", arrays["W3"])}
{format_array("QUBE_B3", arrays["B3"])}

inline float qubeClip(float value, float limit) {{
  if (value > limit) return limit;
  if (value < -limit) return -limit;
  return value;
}}

inline void qubeDenseTanh(
    const float* input,
    const float* weights,
    const float* bias,
    int in_dim,
    int out_dim,
    float* output) {{
  for (int row = 0; row < out_dim; ++row) {{
    float sum = bias[row];
    for (int col = 0; col < in_dim; ++col) {{
      sum += weights[row * in_dim + col] * input[col];
    }}
    output[row] = tanh(sum);
  }}
}}

inline float qubePolicyPredictVoltage(
    float theta,
    float alpha,
    float theta_dot,
    float alpha_dot) {{
  float obs[QUBE_OBS_DIM] = {{theta, alpha, theta_dot, alpha_dot}};
  float h1[QUBE_HIDDEN_DIM];
  float h2[QUBE_HIDDEN_DIM];

  qubeDenseTanh(obs, QUBE_W1, QUBE_B1, QUBE_OBS_DIM, QUBE_HIDDEN_DIM, h1);
  qubeDenseTanh(h1, QUBE_W2, QUBE_B2, QUBE_HIDDEN_DIM, QUBE_HIDDEN_DIM, h2);

  float voltage = QUBE_B3[0];
  for (int col = 0; col < QUBE_HIDDEN_DIM; ++col) {{
    voltage += QUBE_W3[col] * h2[col];
  }}
  return qubeClip(voltage, QUBE_POLICY_VOLTAGE_LIMIT);
}}
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header)
    print(f"Wrote Arduino policy header to {args.output}")


if __name__ == "__main__":
    main()
