"""Personal-workspace runner; only the Modal SDK is needed on the laptop."""

import json
import os
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
ENVIRONMENT = os.environ.get("MODAL_ENVIRONMENT", "main")
VOLUME_NAME = "cs312-personal"
MOUNT = "/root/data"
DATASETS = f"{MOUNT}/datasets"
MODELS = f"{MOUNT}/ckpts"
REMOTE_REPO = "/root/assignments"
DATA_REVISION = "8d066a12a614ce8641c661874960bf8567bf0f19"
IGNORED_PARTS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    "cache", "ckpts", "data", "plots", "temp_scripts", "slurmjobs", "wandb",
}


def ignore_source(path: Path) -> bool:
    return (
        any(part in IGNORED_PARTS for part in path.parts)
        or path.name.startswith(".env")
        or path.name == ".wandb.secret.env"
        or path.suffix in {".pyc", ".pyo"}
    )


app = modal.App("cs312-personal")
volume = modal.Volume.from_name(
    VOLUME_NAME, environment_name=ENVIRONMENT, create_if_missing=True, version=2,
)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(uv_project_dir=str(ROOT))
    .workdir(REMOTE_REPO)
    .env({
        "PYTHONPATH": REMOTE_REPO,
        "DL_ALCHEMY_DATA_DIR": DATASETS,
        "DL_ALCHEMY_DATA_REVISION": DATA_REVISION,
        "DL_ALCHEMY_MODEL_DIR": MODELS,
        "DL_ALCHEMY_EPHEMERAL_DATA_DIR": "/tmp/cs312-shuffled-data",
        "WANDB_DIR": "/tmp/wandb",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor",
        "TRITON_CACHE_DIR": "/tmp/triton",
    })
    .add_local_dir(ROOT, remote_path=REMOTE_REPO, ignore=ignore_source)
)


@app.function(image=image, volumes={MOUNT: volume}, cpu=2, memory=4096,
              timeout=3600, retries=0)
def prepare_data():
    """Download and verify the public release on CPU, then cache seed 42."""
    import shutil

    import numpy as np

    from data import (
        DEFAULT_DATASET_DIR_NAME, SHUFFLE_CHUNK_ROWS,
        global_shuffle_prefix_indices, load_preprocessed_token_dataset,
    )
    from download_data import _download_file, download_default_data, hf_file_url
    from data import DEFAULT_DATA_REPO_ID, MANIFEST_FILE

    base = Path(DATASETS) / DEFAULT_DATASET_DIR_NAME
    try:
        # Refresh this tiny manifest so cached files are always checked against
        # the same immutable release as the verified Kaggle baseline.
        _download_file(hf_file_url(DEFAULT_DATA_REPO_ID, DATA_REVISION, MANIFEST_FILE),
                       base / MANIFEST_FILE, force=True)
        release = download_default_data(output_dir=base, revision=DATA_REVISION)
        volume.commit()
        # Match the course's globally shuffled prefix without spending GPU time
        # materializing the same 600,000 rows for every experiment.
        seeded = Path(DATASETS) / f"{DEFAULT_DATASET_DIR_NAME}_ds42"
        cached_train = seeded / "train"
        expected_bytes = 600_000 * 1024 * 2
        metadata_path = cached_train / "metadata.json"
        token_path = cached_train / "tokens.bin"
        valid_cache = (
            metadata_path.is_file() and token_path.is_file()
            and token_path.stat().st_size == expected_bytes
            and json.loads(metadata_path.read_text()).get("data_seed") == 42
        )
        if not valid_cache:
            source = load_preprocessed_token_dataset(base / "train")
            cached_train.mkdir(parents=True, exist_ok=True)
            metadata_path.unlink(missing_ok=True)
            tmp = cached_train / "tokens.bin.tmp"
            tokens = np.memmap(tmp, dtype=source.dtype, mode="w+",
                               shape=(600_000, source.seq_len))
            indices = global_shuffle_prefix_indices(source.total_sequences, 600_000, 42)
            for start in range(0, 600_000, SHUFFLE_CHUNK_ROWS):
                stop = min(start + SHUFFLE_CHUNK_ROWS, 600_000)
                tokens[start:stop] = source.tokens[indices[start:stop]]
                print(f"Seed-42 cache: {stop:,}/600,000 rows", flush=True)
            tokens.flush()
            del tokens
            tmp.replace(token_path)
            metadata = {**source.metadata, "num_sequences": 600_000, "data_seed": 42}
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        (seeded / "val").mkdir(parents=True, exist_ok=True)
        for name in ("metadata.json", "tokens.bin"):
            shutil.copy2(base / "val" / name, seeded / "val" / name)
        source = load_preprocessed_token_dataset(base / "train")
        cached = load_preprocessed_token_dataset(cached_train)
        indices = global_shuffle_prefix_indices(source.total_sequences, 600_000, 42)
        for row in (0, 299_999, 599_999):
            if not np.array_equal(cached.tokens[row], source.tokens[indices[row]]):
                raise RuntimeError(f"Seed-42 cache disagrees with source at row {row}.")
        result = {"base_dataset": str(base), "seeded_dataset": str(seeded),
                  "seeded_train_sequences": 600_000, "revision": DATA_REVISION,
                  "sha256_verified": True, "seeded_sample_rows_verified": 3,
                  "download_bytes": sum(f["size_bytes"] for f in release["files"])}
        (Path(MOUNT) / "data-ready.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        volume.commit()


@app.function(image=image, volumes={MOUNT: volume}, gpu="H100", cpu=2,
              memory=8192, max_containers=1, timeout=600, retries=0)
def run_training(overrides: dict, smoke: bool):
    import math
    import re
    import time
    from dataclasses import replace

    import torch

    from data import dclm_val_dataset
    from model_io import load_model_metadata
    from train import TrainConfig, train, training_run_name

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU attached.")
    if not (Path(DATASETS) / "dclm_9p6m_ctx1024_ds42/train/metadata.json").is_file():
        raise RuntimeError("Run --action prepare before requesting a GPU training job.")
    config_values = dict(wandb_online=False, model_dir=MODELS,
                         run_name_suffix="modal-personal", metric_loggers=())
    config_values.update(overrides)
    config_values["model_dir"] = MODELS
    if smoke:
        config_values.update(
            model_name="d8", num_train_sequences=2048, batch_size=64,
            num_epochs=1.0, num_evals=4, compile_model=False,
            latest_checkpoint_frequency=16, run_name_suffix="modal-personal-smoke",
            save_model=True, force_run=False,
            val_dataset=replace(dclm_val_dataset(), sequence_limit=64),
        )
    config = TrainConfig(**config_values)
    started = time.perf_counter()
    run_dir = Path(MODELS) / training_run_name(config)
    log_path = run_dir / "console.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Keep metrics on the Volume even when dashboard log retention expires.
    import sys
    from contextlib import redirect_stdout

    original_stdout = sys.stdout

    class Tee:
        def write(self, text):
            original_stdout.write(text)
            log_file.write(text)
            return len(text)

        def flush(self):
            original_stdout.flush()
            log_file.flush()

    try:
        with log_path.open("a", buffering=1) as log_file, redirect_stdout(Tee()):
            model = train(config)
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError("Model contains non-finite parameters.")
        required = ("config.json", "model.pt", "metadata.json", "latest.pt")
        if any(not (run_dir / name).is_file() for name in required):
            raise RuntimeError("Training did not write the expected final artifacts.")
        checkpoint = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
        metadata = load_model_metadata(run_dir)
        completed = int(checkpoint["completed_steps"])
        total = int(metadata["total_steps"])
        if completed != total:
            raise RuntimeError(f"Incomplete training: {completed}/{total} steps.")
        logged_losses = re.findall(r", val Loss: ([0-9.eE+-]+)", log_path.read_text())
        final_val_loss = float(logged_losses[-1]) if logged_losses else None
        if final_val_loss is None or not math.isfinite(final_val_loss):
            raise RuntimeError("No finite validation loss was recorded.")
        result = {
            "run_name": run_dir.name, "gpu": torch.cuda.get_device_name(),
            "completed_steps": completed, "total_steps": total,
            "parameter_count": metadata["parameter_count"],
            "final_validation_loss": final_val_loss,
            "elapsed_seconds": time.perf_counter() - started, "finite_weights": True,
            "volume": VOLUME_NAME, "volume_path": f"/ckpts/{run_dir.name}",
        }
        (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        volume.commit()


@app.local_entrypoint()
def main(action: str = "smoke", gpu: str = "H100", config: str = ""):
    """prepare: CPU data download; smoke: 32 steps; train: full recipe."""
    if action not in {"prepare", "smoke", "train"}:
        raise ValueError("--action must be prepare, smoke, or train")
    if action == "prepare":
        result = prepare_data.remote()
    else:
        if action == "smoke" and config:
            raise ValueError("--config is only supported for --action train.")
        overrides = json.loads(Path(config).read_text()) if config else {}
        if not isinstance(overrides, dict):
            raise ValueError("--config must contain a JSON object of TrainConfig fields.")
        if overrides.get("wandb_online"):
            raise ValueError("W&B is not configured in this personal runner.")
        if overrides.get("save_model") is False or overrides.get("latest_checkpoint_frequency", 1000) is None:
            raise ValueError("This runner requires final model and latest checkpoint saving.")
        # Check data before allocating a GPU container.
        volume.listdir("/datasets/dclm_9p6m_ctx1024_ds42/train")
        runner = run_training.with_options(
            gpu=gpu, timeout=600 if action == "smoke" else 3600,
        )
        result = runner.remote(overrides, action == "smoke")
    print(json.dumps(result, indent=2))
