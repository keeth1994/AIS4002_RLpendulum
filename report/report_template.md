# AIS4002 Practical RL Assignment Report

## 1. Goal

Train and evaluate controllers for the Quanser QUBE-Servo 2 rotary inverted pendulum in simulation, then prepare for sim-to-real transfer to the physical platform.

## 2. Reproducibility

Recommended Python version: 3.11 or 3.12.

```bash
pip install -r requirements.txt
python main.py
python -m src.evaluate_classical --start upright
python -m src.train_rl --start upright --timesteps 100000
python -m src.evaluate_rl --start upright --model-path models/ppo_qube_balance.zip
```

## 3. Simulation Environment

The environment is a Gymnasium-compatible QUBE-Servo 2 rotary pendulum approximation.

Observation:

`[theta, alpha, theta_dot, alpha_dot]`

Action:

motor voltage command, clipped to +/-5 V.

Convention:

`alpha = 0` is upright and `alpha = pi` is hanging downward.

## 4. Classical Validation

The current classical baseline validates near-upright balance using state feedback. Full swing-up from the downward position is treated as a separate harder task.

Include metrics from:

```bash
python -m src.evaluate_classical --start upright --csv results/classical_baseline.csv
```

## 5. RL Agent

Describe the PPO setup, number of timesteps, seed, reward design, and final evaluation returns.

Start with near-upright balancing before attempting full swing-up:

```bash
python -m src.train_rl --start upright --timesteps 100000 --model-path models/ppo_qube_balance
```

## 6. Sim-to-Real

Describe how the simulator compares to the physical QUBE data. Mention voltage limits, encoder conventions, arm safety limits, and any domain randomization.

If hardware transfer is not completed, explain what remains: connecting the lab Quanser API, validating angle signs, limiting voltage, and testing with emergency stop procedures.

## 7. Group Contributions

- Person 1:
- Person 2:
- Person 3:

## 8. AI Usage

State exactly how AI was used, what was generated, and how the group tested or corrected it.
