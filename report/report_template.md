# AIS4002 Practical RL Assignment Report

## 1. Goal

Train an agent for the Quanser QUBE-Servo 2 that starts with the pendulum hanging down, swings it up, balances it, and can recover if it falls.

## 2. Reproducibility

Recommended Python version: 3.11 or 3.12.

```bash
pip install -r requirements.txt
python main.py
python -m src.evaluate_classical --arm-limit-deg 90
python -m src.train_rl --arm-limit-deg 90 --timesteps 1000000 --model-path models/ppo_qube_swingup
python -m src.evaluate_rl --arm-limit-deg 90 --model-path models/ppo_qube_swingup.zip
```

## 3. Simulation Environment

The Gymnasium environment approximates the QUBE-Servo 2 as a voltage-controlled rotary pendulum.

Observation:

`[sin(theta), cos(theta), sin(alpha), cos(alpha), theta_dot, alpha_dot]`

Action:

motor voltage command, clipped by the simulator.

Convention:

`alpha = 0` is the top balance position and `alpha = pi` is the hanging-down position.

## 4. Classical Baseline

The classical baseline uses a simple energy-pumping swing-up heuristic and switches to balance control near the top.

## 5. RL Agent

We use SAC from Stable-Baselines3 with an MLP policy because the task has a continuous voltage action and required reliable swing-up from the hanging-down initial condition. PPO was also tested during development, but SAC learned the simplified simulator more reliably.

Main simulation result to report:

```text
Model: models/sac_qube_swingup_90.zip
Mean return: 12948.604 over 10 episodes
Mean upright ratio: 0.895
Recovery time: about 1.45-2.01 s
Termination: time_limit for all 10 episodes
```

## 6. Sim-to-Real Attempt

The policy was exported to Arduino and tested on the physical QUBE-Servo 2. The hardware tests showed that the agent did not transfer zero-shot: it could move the arm and sometimes bring the pendulum near upright, but it did not reliably catch and balance. Several practical issues were observed:

- The physical arm center needed calibration. A trim value around `CENTER_TRIM_DEG = -20.0` was found experimentally.
- The real motor/pendulum behavior did not match the simplified simulation closely enough.
- The learned policy was too weak or mistimed for real swing-up unless additional swing-up assistance was added.
- When the pendulum reached the top, the balance handoff was unreliable.

This is a useful negative sim-to-real result: the simulation was learnable, but the reality gap was too large for direct transfer.

## 7. Results

Report return, upright ratio, recovery time, and video/demo observations.

Suggested result table:

| Experiment | Outcome |
| --- | --- |
| Classical baseline in simulation | Used to validate basic swing-up/balance logic. Include plot/video if successful. |
| SAC in simulation, +/-90 deg arm range | Successful in simplified simulator, upright ratio about 0.895. |
| Arduino export to physical QUBE | Motor commands executed, but zero-shot transfer did not reliably swing up and balance. |
| Hardware tuning | Center trim, voltage limits, swing-up pulse timing, and balance handoff were adjusted. |

Main conclusion: retraining is useful, but only after improving the simulator or adding domain randomization based on observed hardware behavior.

## 8. Group Contributions

- Person 1:
- Person 2:
- Person 3:

## 9. AI Usage

State how AI was used and how the group tested or corrected the generated code.
