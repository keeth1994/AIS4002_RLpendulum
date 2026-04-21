# AIS4002 RL Pendulum

Code for the AIS4002 practical reinforcement learning assignment using a simulated Quanser QUBE-Servo 2 rotary inverted pendulum and, if available, the physical QUBE-Servo 2 hardware.

The goal is to train an RL agent in simulation and investigate sim-to-real transfer. The simulator uses a QUBE-style voltage input and encoder-style state vector so trained policies can later be adapted to the physical platform.

## Run locally

Recommended Python version: Python 3.11 or 3.12.

1. Clone the repository:

```bash
git clone <your-repo-url>
cd AIS4002_RLpendulum
```

2. Create a virtual environment:

```bash
python -m venv .venv
```

If you have multiple Python versions installed on Windows, prefer:

```powershell
py -3.11 -m venv .venv
```

3. Activate the virtual environment.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install requirements:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Run the current entry point:

```bash
python main.py
```

## Useful Commands

Run the classical near-upright balance baseline and save CSV data:

```bash
python -m src.evaluate_classical --start upright --video none --plot none --csv results/classical_baseline.csv
```

Train a PPO balance agent in simulation:

```bash
python -m src.train_rl --start upright --timesteps 100000 --model-path models/ppo_qube_balance
```

TensorBoard logging and SB3's rich progress bar are optional. Enable them only if the matching packages are installed:

```bash
python -m src.train_rl --start upright --timesteps 100000 --model-path models/ppo_qube_balance --tensorboard --progress-bar
```

Train with domain randomization:

```bash
python -m src.train_rl --start upright --timesteps 100000 --domain-randomization
```

Evaluate a trained policy and save a demo video:

```bash
python -m src.evaluate_rl --start upright --model-path models/ppo_qube_balance.zip
```

Export hardware metadata for a trained policy:

```bash
python -m src.export_policy --model-path models/ppo_qube_balance.zip
```

If MP4 export fails because FFMPEG is unavailable, the scripts automatically save a `.gif` fallback in the same folder.

## Suggested Project Structure

The repository is expected to evolve toward this structure:

```text
AIS4002_RLpendulum/
|-- main.py
|-- requirements.txt
|-- README.md
|-- src/
|   |-- envs/
|   |-- controllers/
|   |-- train_rl.py
|   |-- evaluate_rl.py
|   `-- evaluate_classical.py
|-- models/
|-- results/
|-- videos/
`-- report/
```

## Dependencies

The `requirements.txt` file includes packages for:

- numerical simulation: `numpy`, `scipy`
- plotting and result analysis: `matplotlib`, `pandas`
- Gymnasium environments: `gymnasium`
- RL algorithms: `stable-baselines3`
- logging: `tensorboard`, `tqdm`
- video export: `imageio`

Quanser/QUBE hardware drivers are not included in `requirements.txt` because they normally depend on the course/lab installation instructions and vendor-specific software.

## Assignment Checklist

- Build a Gymnasium-compatible QUBE-Servo 2 rotary inverted pendulum simulation.
- Validate the simulator with a classical near-upright balance controller.
- Treat full swing-up from the downward position as a separate harder task.
- Train an RL agent in simulation, for example with Stable-Baselines3.
- Compare simulation behavior with real hardware data where available.
- Improve sim-to-real robustness, for example with domain randomization.
- Record a simple video demo for the final `.zip` attachment.
- Document reproducibility steps, results, problems encountered, group contributions, and AI usage in the report.

## Reproducibility Notes

- Keep random seeds fixed when comparing training runs.
- Save trained models in `models/`.
- Save plots, logs, and evaluation data in `results/`.
- Save short demo videos in `videos/`.
- If hardware transfer does not work, document what was tested, what failed, and what could be improved.
- If installation fails on a very new Python version, try Python 3.11 or 3.12 because the RL stack depends on packages such as PyTorch.
- The current classical controller is a starting baseline. If it does not fully swing up and stabilize in your experiments, include that result in the report and discuss tuning/model mismatch.

## QUBE-Servo 2 Simulation Convention

- Observation: `[theta, alpha, theta_dot, alpha_dot]`
- `theta`: rotary arm angle in radians, centered at 0
- `alpha`: pendulum angle in radians, where 0 is upright and pi is hanging downward
- Action: motor voltage command in volts
- Safety limit: action is clipped to +/-5 V and the episode stops if the rotary arm exceeds +/-60 degrees
- Hardware path: use the same observation ordering from encoders, call the trained policy, clip voltage conservatively, and send voltage to the QUBE API supplied by the course/lab setup
