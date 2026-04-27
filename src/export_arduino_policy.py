"""Export a trained SB3 PPO/SAC policy to a small Arduino-compatible C++ header."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC


def format_array(name: str, values: np.ndarray) -> str:
    flat = values.astype(np.float32).reshape(-1)
    body = ", ".join(f"{value:.9g}f" for value in flat)
    return f"const float {name}[{flat.size}] = {{{body}}};"


def format_float_literal(value: float) -> str:
    text = f"{value:.9g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return f"{text}f"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/ppo_qube_swingup.zip"))
    parser.add_argument("--output", type=Path, default=Path("arduino/qube_policy_runner/qube_policy.h"))
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--voltage-limit", type=float, default=2.0)
    args = parser.parse_args()

    model_cls = PPO if args.algo == "ppo" else SAC
    model = model_cls.load(args.model_path, device="cpu")
    state_dict = model.policy.state_dict()

    if args.algo == "ppo":
        required = [
            "mlp_extractor.policy_net.0.weight",
            "mlp_extractor.policy_net.0.bias",
            "mlp_extractor.policy_net.2.weight",
            "mlp_extractor.policy_net.2.bias",
            "action_net.weight",
            "action_net.bias",
        ]
        arrays = {
            "W1": state_dict["mlp_extractor.policy_net.0.weight"].detach().cpu().numpy(),
            "B1": state_dict["mlp_extractor.policy_net.0.bias"].detach().cpu().numpy(),
            "W2": state_dict["mlp_extractor.policy_net.2.weight"].detach().cpu().numpy(),
            "B2": state_dict["mlp_extractor.policy_net.2.bias"].detach().cpu().numpy(),
            "W3": state_dict["action_net.weight"].detach().cpu().numpy(),
            "B3": state_dict["action_net.bias"].detach().cpu().numpy(),
        }
    else:
        required = [
            "actor.latent_pi.0.weight",
            "actor.latent_pi.0.bias",
            "actor.latent_pi.2.weight",
            "actor.latent_pi.2.bias",
            "actor.mu.weight",
            "actor.mu.bias",
        ]
        arrays = {
            "W1": state_dict["actor.latent_pi.0.weight"].detach().cpu().numpy(),
            "B1": state_dict["actor.latent_pi.0.bias"].detach().cpu().numpy(),
            "W2": state_dict["actor.latent_pi.2.weight"].detach().cpu().numpy(),
            "B2": state_dict["actor.latent_pi.2.bias"].detach().cpu().numpy(),
            "W3": state_dict["actor.mu.weight"].detach().cpu().numpy(),
            "B3": state_dict["actor.mu.bias"].detach().cpu().numpy(),
        }
    missing = [key for key in required if key not in state_dict]
    if missing:
        raise RuntimeError(f"Unsupported policy architecture. Missing keys: {missing}")

    input_dim = arrays["W1"].shape[1]
    hidden_dim = arrays["W1"].shape[0]
    if input_dim not in (4, 6) or arrays["W2"].shape != (hidden_dim, hidden_dim) or arrays["W3"].shape != (1, hidden_dim):
        raise RuntimeError(
            "Expected a 4/6 -> hidden -> hidden -> 1 policy network. "
            f"Got W1={arrays['W1'].shape}, W2={arrays['W2'].shape}, W3={arrays['W3'].shape}."
        )
    algorithm_name = args.algo.upper()
    sac_output_line = "  command = tanh(command);\n" if args.algo == "sac" else ""

    header = f"""// Auto-generated from {args.model_path}
// Deterministic {algorithm_name} actor for QUBE-Servo 2 swing-up and balance.
// Observation order:
// - for 6-input policies: sin(theta), cos(theta), sin(alpha), cos(alpha), theta_dot, alpha_dot
// - for older 4-input policies: theta, alpha, theta_dot, alpha_dot
// Action output: normalized motor command in [-1, 1], scaled here to +/-{args.voltage_limit} V.
#pragma once

#include <math.h>

const int QUBE_OBS_DIM = {input_dim};
const int QUBE_HIDDEN_DIM = {hidden_dim};
const float QUBE_POLICY_VOLTAGE_LIMIT = {format_float_literal(args.voltage_limit)};

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
  float obs[QUBE_OBS_DIM];
  if (QUBE_OBS_DIM == 6) {{
    obs[0] = sin(theta);
    obs[1] = cos(theta);
    obs[2] = sin(alpha);
    obs[3] = cos(alpha);
    obs[4] = theta_dot;
    obs[5] = alpha_dot;
  }} else {{
    obs[0] = theta;
    obs[1] = alpha;
    obs[2] = theta_dot;
    obs[3] = alpha_dot;
  }}
  float h1[QUBE_HIDDEN_DIM];
  float h2[QUBE_HIDDEN_DIM];

  qubeDenseTanh(obs, QUBE_W1, QUBE_B1, QUBE_OBS_DIM, QUBE_HIDDEN_DIM, h1);
  qubeDenseTanh(h1, QUBE_W2, QUBE_B2, QUBE_HIDDEN_DIM, QUBE_HIDDEN_DIM, h2);

  float command = QUBE_B3[0];
  for (int col = 0; col < QUBE_HIDDEN_DIM; ++col) {{
    command += QUBE_W3[col] * h2[col];
  }}
{sac_output_line}  command = qubeClip(command, 1.0f);
  return qubeClip(command * QUBE_POLICY_VOLTAGE_LIMIT, QUBE_POLICY_VOLTAGE_LIMIT);
}}
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header)
    print(f"Wrote Arduino policy header to {args.output}")


if __name__ == "__main__":
    main()
