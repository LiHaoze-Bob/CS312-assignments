"""P1 experiments on the personal Modal Volume, one run at a time.

From the assignments directory:
  MODAL_PROFILE=haozeli2025 ~/.local/bin/modal run --env main -m \
    experiments.a1_basics.p1_hyperparameters::p1

Edit RUNS below. Each dictionary overrides TrainConfig defaults on Modal;
neither PyTorch nor the course workspace is required on this computer.
"""

import json

from modal_personal import app, download_training_result, run_training, volume


RUNS = [
    {"learning_rate": 1e-3},
    {"learning_rate": 9e-3},
]


# TODO: Design individual sweeps, paired sweeps, and scheduler comparisons.


def launch_training_jobs(runs, *, gpu="H100", download=True, output_dir=""):
    """Validate all overrides, then wait for each full run before starting the next."""
    if not isinstance(runs, list) or not runs:
        raise ValueError("RUNS must be a non-empty list of parameter dictionaries.")
    configs = []
    for index, overrides in enumerate(runs, start=1):
        if not isinstance(overrides, dict):
            raise ValueError(f"RUNS item {index} must be a parameter dictionary.")
        if overrides.get("wandb_online"):
            raise ValueError(f"RUNS item {index}: W&B is not configured in the personal runner.")
        if overrides.get("save_model") is False or overrides.get("latest_checkpoint_frequency", 1000) is None:
            raise ValueError(f"RUNS item {index}: model and latest checkpoint saving are required.")
        try:
            # Copy the JSON-compatible overrides; the remote runner constructs TrainConfig.
            configs.append(json.loads(json.dumps(overrides, allow_nan=False)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"RUNS item {index} must contain finite JSON-compatible values.") from exc

    # Match modal_personal.py's CPU-side data check before requesting a GPU.
    volume.listdir("/datasets/dclm_9p6m_ctx1024_ds42/train")
    runner = run_training.with_options(gpu=gpu, timeout=3600)
    results = []
    for index, config in enumerate(configs, start=1):
        print(f"\nExperiment {index}/{len(configs)}: {json.dumps(config)}", flush=True)
        result = runner.remote(config, False)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
        if download:
            download_training_result(result, output_dir or None)
    return results


@app.local_entrypoint()
def p1(gpu: str = "H100", download: bool = True, output_dir: str = ""):
    results = launch_training_jobs(RUNS, gpu=gpu, download=download, output_dir=output_dir)
    print("\nAll requested experiments completed.", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit(
        "Use: MODAL_PROFILE=haozeli2025 ~/.local/bin/modal run --env main -m "
        "experiments.a1_basics.p1_hyperparameters::p1"
    )
