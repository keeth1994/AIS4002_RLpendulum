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

## Final Reproducible Workflow

The final reported hardware setup uses the course/example energy controller for
swing-up and a SAC policy trained in simulation as a bounded residual balance
controller. This was chosen because direct zero-shot RL swing-up transfer was
not repeatable on the physical QUBE, while the residual setup was able to
sustain upright balance after classical capture.

### Validate the Energy-Controller Example

Run the course/example energy swing-up plus PD balance controller on the QUBE,
then compare the new hardware log to the retained calibrated simulation log.
The latest tested port was `COM13`; change it if Windows assigns another port.

```powershell
python -m src.hardware.run_classical_on_qube --preset report-classical --port COM13 --csv results/classical_hw_report.csv
python -m src.compare_classical --sim-csv results/classical_sim_example_calibrated_defaults.csv --hw-csv results/classical_hw_report.csv --output results/report_figures/classical_report_sim_vs_hw.png --duration 6
```

For reference, the retained validation logs already in the repository are:

```powershell
python -m src.compare_classical --sim-csv results/classical_sim_example_calibrated_defaults.csv --hw-csv results/classical_hw_example_calm_kick_com13.csv --output results/report_figures/classical_existing_sim_vs_hw.png --duration 6
```

The comparison is meant to validate qualitative controller behavior and key
metrics, not exact point-by-point trajectory matching. Hardware timing,
friction, encoder quantization, and manual initial conditions make the real
trajectory diverge even when the controller and simulator are close enough for
the assignment.

If you specifically want to regenerate a fresh, uncalibrated simulator run of
the same controller implementation, use:

```powershell
python -m src.evaluate_classical --preset report-classical --csv results/classical_sim_report.csv
```

For the report comparison, prefer the retained calibrated simulation log above,
because it is the version already matched against the QUBE data.

### Train the SAC Balance Policy

This trains the upright balance policy used by the hardware residual controller:

```powershell
python -m src.train_rl --preset report-balance --timesteps 100000 --seed 81 --model-path models/sac_report_balance_centered_5v_100k --progress-bar
```

The retained final model is:

```text
models/sac_report_balance_centered_5v_100k.zip
```

### Evaluate the Model in Simulation

```powershell
python -m src.evaluate_rl --preset report-balance --model-path models/sac_report_balance_centered_5v_100k.zip --seed 40
```

### Test the Model on the QUBE

Connect the QUBE, verify the correct COM port, and place the arm near center
with the pendulum hanging down before the encoder reset countdown finishes.
The latest tested port was `COM13`; change `--port` if Windows assigns a
different port.

```powershell
python -m src.hardware.run_example_swingup_rl_balance_on_qube --preset report-residual --port COM13 --model-path models/sac_report_balance_centered_5v_100k.zip --csv results/hw_report_demo.csv
```

This command runs:

- `example_kick` / `example_swingup`: classical energy swing-up.
- `example_balance`: classical capture near upright.
- `rl_residual`: SAC policy blended in as a bounded residual voltage command.

Stop a long run manually with `Ctrl+C`; the script shuts the motor off in its
cleanup block.

### Analyze the Hardware Log

```powershell
python -m src.analyze_hardware_log results\hw_report_demo.csv
```

The script reports mode counts, upright ratio, longest upright interval, maximum
arm travel, voltage usage, and final state.

### Compare RL Balance Sim-To-Real

After a hardware RL-residual run, align the hardware trace at the settled
upright RL phase and compare it with the same RL balance policy in simulation:

```powershell
python -m src.compare_rl_sim_hw --hw-csv results/hw_rl_report_run.csv --model-path models/sac_report_balance_centered_5v_100k.zip --output results/report_figures/rl_report_sim_vs_hw.png
```

### Generate Report Figures

```powershell
python -m src.generate_report_graphs
```

Figures and metrics are written to:

```text
results/report_figures/
```

The retained evidence logs used for the report are:

- `results/classical_hw_example_calm_kick_com13.csv`
- `results/classical_sim_example_calibrated_defaults.csv`
- `results/hw_example_rl_residual_06.csv`
- `results/hw_example_rl_residual_06_repeat.csv`

## Useful Commands

The commands below are exploratory/development commands kept for reference. The
final reproducible report workflow is the preset-based workflow above.

Run the classical swing-up/balance baseline and save CSV data:

```bash
python -m src.evaluate_classical --video none --plot none --csv results/classical_baseline.csv
```

Sweep classical energy-controller parameters before retraining RL:

```bash
python -m src.sweep_classical --arm-limit-deg 90 --voltage-limit 5 --csv results/classical_sweep.csv
```

Train a swing-up agent in simulation. PPO is available, but SAC is usually the
better first choice for this continuous swing-up task:

```bash
python -m src.train_rl --algo sac --arm-limit-deg 90 --timesteps 1000000 --model-path models/sac_qube_swingup
```

TensorBoard logging and SB3's rich progress bar are optional. Enable them only if the matching packages are installed:

```bash
python -m src.train_rl --algo sac --arm-limit-deg 90 --timesteps 1000000 --model-path models/sac_qube_swingup --tensorboard --progress-bar
```

Train with domain randomization:

```bash
python -m src.train_rl --algo sac --arm-limit-deg 90 --timesteps 1000000 --domain-randomization --model-path models/sac_qube_swingup_dr
```

If the policy still never reaches upright, first prove the cart-pole surrogate
can learn with a wider arm range, then tighten back to the real +/-90 degree
hardware range:

```bash
python -m src.train_rl --algo sac --arm-limit-deg 180 --timesteps 500000 --model-path models/sac_qube_swingup_180
python -m src.evaluate_rl --algo sac --arm-limit-deg 180 --model-path models/sac_qube_swingup_180.zip --episodes 10 --steps 1500 --video videos/sac_swingup_180.mp4
python -m src.train_rl --algo sac --arm-limit-deg 90 --timesteps 500000 --continue-from models/sac_qube_swingup_180.zip --model-path models/sac_qube_swingup_90
python -m src.evaluate_rl --algo sac --arm-limit-deg 90 --model-path models/sac_qube_swingup_90.zip --episodes 10 --steps 1500 --video videos/sac_swingup_90.mp4
```

Evaluate the swing-up policy:

```bash
python -m src.evaluate_rl --algo sac --arm-limit-deg 90 --model-path models/sac_qube_swingup.zip --episodes 10 --steps 1500 --video videos/rl_swingup.mp4
```

The default video uses a QUBE-style rotary arm, which matches the physical
platform. To visualize the CartPole-style surrogate model instead, add
`--render-style cartpole`:

```bash
python -m src.evaluate_rl --algo sac --arm-limit-deg 90 --model-path models/sac_qube_swingup.zip --episodes 1 --steps 1500 --video videos/rl_swingup_cartpole.gif --render-style cartpole
```

Export a SAC swing-up policy to Arduino after it works in simulation:

```bash
python -m src.export_arduino_policy --algo sac --model-path models/sac_qube_swingup_90.zip --output arduino/qube_policy_runner/qube_policy.h --voltage-limit 2.0
```

The exported policy is only a starting point for hardware tests. If zero-shot
transfer fails, document the behavior and retrain with improved simulator
parameters or domain randomization instead of only tuning the Arduino sketch.

Use only a policy trained from the hanging-down initial condition as the main hardware controller.

Evaluate a trained policy and save a demo video:

```bash
python -m src.evaluate_rl --algo sac --arm-limit-deg 90 --model-path models/sac_qube_swingup.zip
```

Export hardware metadata for a trained policy:

```bash
python -m src.export_policy --model-path models/sac_qube_swingup_90.zip
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

- numerical simulation: `numpy`
- plotting and result analysis: `matplotlib`, `pandas`
- Gymnasium environments: `gymnasium`
- RL algorithms: `stable-baselines3`
- hardware serial communication: `pyserial`
- logging and progress bars: `tensorboard`, `tqdm`
- video export: `imageio`

Quanser/QUBE hardware drivers are not included in `requirements.txt` because they normally depend on the course/lab installation instructions and vendor-specific software.

## Assignment Checklist

- Build a Gymnasium-compatible QUBE-Servo 2 rotary inverted pendulum simulation.
- Validate the simulator with a classical swing-up/balance controller.
- Train and evaluate policies from the hanging-down initial condition.
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

- Observation: `[sin(theta), cos(theta), sin(alpha), cos(alpha), theta_dot, alpha_dot]`
- `theta`: rotary arm angle in radians, centered at 0
- `alpha`: pendulum angle in radians, where 0 is upright and pi is hanging downward
- Action: normalized motor command in `[-1, 1]`, scaled by `--voltage-limit` inside the simulator
- Safety limit: action is clipped to +/-5 V and the episode stops if the rotary arm exceeds +/-60 degrees
- Hardware path: the Arduino export computes the same sin/cos observation from encoder angles, calls the trained policy, clips voltage conservatively, and sends voltage to the QUBE API supplied by the course/lab setup
