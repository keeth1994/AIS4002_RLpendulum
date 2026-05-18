# Report Workflow

This repo has been cleaned down to the final sim2real workflow:

1. Validate the calibrated simulator with the classical example controller.
2. Train/evaluate a SAC upright-balance policy in simulation.
3. Run the hardware demo as classical swing-up plus bounded RL residual balance.
4. Generate report figures from the retained logs.

## Train Balance Policy

```powershell
python -m src.train_rl --preset report-balance --timesteps 100000 --seed 81 --model-path models/sac_report_balance_centered_5v_100k --progress-bar
```

## Evaluate Balance Policy In Simulation

```powershell
python -m src.evaluate_rl --preset report-balance --model-path models/sac_report_balance_centered_5v_100k.zip --seed 40
```

## Validate Energy-Controller Example

Run the course/example energy swing-up plus PD balance controller on hardware:

```powershell
python -m src.hardware.run_classical_on_qube --preset report-classical --port COM13 --csv results/classical_hw_report.csv
```

Generate the simulator-vs-hardware comparison plot using the retained
calibrated simulation log:

```powershell
python -m src.compare_classical --sim-csv results/classical_sim_example_calibrated_defaults.csv --hw-csv results/classical_hw_report.csv --output results/report_figures/classical_report_sim_vs_hw.png --duration 6
```

To regenerate the comparison from the retained evidence logs instead:

```powershell
python -m src.compare_classical --sim-csv results/classical_sim_example_calibrated_defaults.csv --hw-csv results/classical_hw_example_calm_kick_com13.csv --output results/report_figures/classical_existing_sim_vs_hw.png --duration 6
```

Optional fresh simulator run of the same example-controller implementation:

```powershell
python -m src.evaluate_classical --preset report-classical --csv results/classical_sim_report.csv
```

Use the retained calibrated simulation log for the main report comparison.

## Run Hardware Demo

```powershell
python -m src.hardware.run_example_swingup_rl_balance_on_qube --preset report-residual --port COM13 --model-path models/sac_report_balance_centered_5v_100k.zip --csv results/hw_report_demo.csv
```

## Analyze Hardware Log

```powershell
python -m src.analyze_hardware_log results\hw_report_demo.csv
```

## Compare RL Balance Sim-To-Real

After running the hardware RL-residual demo, generate the balance-phase
simulation-vs-hardware figure:

```powershell
python -m src.compare_rl_sim_hw --hw-csv results/hw_rl_report_run.csv --model-path models/sac_report_balance_centered_5v_100k.zip --output results/report_figures/rl_report_sim_vs_hw.png
```

## Generate Report Figures

```powershell
python -m src.generate_report_graphs
```

Outputs are written to `results/report_figures/`.

## Retained Evidence Logs

- `results/classical_hw_example_calm_kick_com13.csv`: hardware classical validation.
- `results/classical_sim_example_calibrated_defaults.csv`: calibrated simulation classical validation.
- `results/hw_example_rl_residual_06.csv`: successful hardware hybrid run.
- `results/hw_example_rl_residual_06_repeat.csv`: repeat hardware hybrid run.
- `models/sac_report_balance_centered_5v_100k.zip`: final SAC balance policy.
