# AIS4002 RL Pendulum

Code for the AIS4002 practical reinforcement learning assignment using a simulated rotary inverted pendulum and, if available, the Quanser QUBE Servo 2 Inverted Pendulum hardware.

The goal is to train an RL agent in simulation and investigate sim-to-real transfer.

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

Run the classical swing-up/PD baseline and save CSV data:

```bash
python -m src.evaluate_classical --video none --plot none --csv results/classical_baseline.csv
```

Train a PPO agent in simulation:

```bash
python -m src.train_rl --timesteps 100000 --model-path models/ppo_rotary_pendulum
```

Train with domain randomization:

```bash
python -m src.train_rl --timesteps 100000 --domain-randomization
```

Evaluate a trained policy and save a demo video:

```bash
python -m src.evaluate_rl --model-path models/ppo_rotary_pendulum.zip
```

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
- video export: `imageio`, `imageio-ffmpeg`

Quanser/QUBE hardware drivers are not included in `requirements.txt` because they normally depend on the course/lab installation instructions and vendor-specific software.

## Assignment Checklist

- Build a Gymnasium-compatible rotary inverted pendulum simulation.
- Validate the simulator with an energy-based swing-up controller and PD stabilizer.
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
