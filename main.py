"""Quick smoke-test entry point for the AIS4002 rotary pendulum project."""

from src.evaluate_classical import run_baseline


if __name__ == "__main__":
    metrics = run_baseline(
        steps=1000,
        seed=1,
        video_path=None,
        plot_path=None,
        csv_path=None,
        arm_limit_deg=90.0,
        voltage_limit=5.0,
    )
    print("Rotary pendulum simulation smoke test")
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
