# Personal Modal workspace

Workspace: `haozeli2025`; environment: `main`.

This runner requires only the Modal CLI on your Mac. PyTorch and the course
dependencies are installed in the remote Linux image from `uv.lock`.

Setup completed on 2026-10-02: account/environment access, remote image build,
CPU data preparation, published SHA-256 verification, and seeded-cache sample
checks passed. A completed d8 baseline has since been downloaded to
`data/modal-results/model-d8-lr0.003-tok614M-modal-personal/`. Its `result.json`
records 9,375/9,375 steps, final validation loss 2.92579, and finite weights.

Run commands from this directory:

```bash
cd /Users/bob.li/Code/CS312-Stanford/code/assignments
```

## Prepare data once

```bash
MODAL_PROFILE=haozeli2025 modal run --env main modal_personal.py --action prepare
```

This CPU job downloads the public course dataset (about 18.31 GiB), checks its
published sizes and SHA-256 values, and creates a seed-42 cache of the globally
shuffled first 600,000 training rows. Validation uses the public validation
split, matching the previous Kaggle fallback. Both caches persist on your own
`cs312-personal` Volume. Subsequent preparation verifies and reuses the data.
The release is pinned to `8d066a12a614ce8641c661874960bf8567bf0f19`, the
same revision used for the verified Kaggle baseline. Preparation also compares
three seeded cache rows with the corresponding original rows and writes
`/data-ready.json` on the Volume.

## Check training

```bash
MODAL_PROFILE=haozeli2025 modal run --env main modal_personal.py --action smoke
```

The test uses d8, 2,048 sequences, batch size 64, 32 optimizer steps, 64
validation sequences, no compilation, and no W&B. It uses one H100, has a
10-minute function timeout, and has no automatic retries. A successful result
reports 32/32 completed steps, finite parameters, and saved native-model files
plus a training checkpoint. The test is setup evidence, not an A1 baseline.

## Full d8 baseline

After checking the smoke result and your remaining budget:

```bash
MODAL_PROFILE=haozeli2025 ~/.local/bin/modal run --env main modal_personal.py --action train
```

This uses the course's default 600,000 sequences, batch size 64, context length
1024, learning rate 0.003, seed 42, and compilation, with W&B disabled. It runs
9,375 optimizer steps on one H100, with a one-hour timeout and no retries.
Checkpoints allow the same config to resume; completed models are reused.
After a successful smoke or full run, the launcher automatically downloads
`model.pt`, `latest.pt`, `config.json`, `metadata.json`, `console.log`, and
`result.json` into `data/modal-results/RUN_NAME/` beside this script. Each file
is downloaded to a temporary file and replaces its previous local copy only
after the transfer finishes. Other local files and the cloud copies are kept.

Keep this local command running until the download completes. A cloud run can
outlive a disconnected detached client, but the automatic local download
cannot run after the client exits or while the laptop is asleep. Omit
`--detach` for the normal automatic-download workflow. Add `--no-download`
to keep results only on the Volume, or `--output-dir PATH` to choose a different
local parent directory. `--action prepare` does not download training results.

For experiments, put ordinary `TrainConfig` overrides in a JSON file, e.g.:

```json
{"learning_rate": 0.001, "run_name_suffix": "lr-1e-3"}
```

Pass it with `--config experiment.json`. Run one experiment at a time initially.
Nested custom model/dataset configs and custom metric loggers need a Python
launcher rather than this JSON interface. Other seeds can use the full base
dataset, but materialize their shuffled prefix during GPU startup.

## Read results and usage

```bash
MODAL_PROFILE=haozeli2025 modal volume ls --env main cs312-personal /ckpts
MODAL_PROFILE=haozeli2025 modal environment billing summary main --for 'this month'
```

Each run directory contains `console.log` with training and validation losses,
`result.json` with completion/weight checks, `config.json`, `metadata.json`,
`model.pt`, and `latest.pt`. Completed runs download automatically. To recover
a run after a disconnected client or a failed download, use its printed
Volume path without starting training again:

```bash
mkdir -p ./data/modal-results
MODAL_PROFILE=haozeli2025 ~/.local/bin/modal volume get --env main --force cs312-personal /ckpts/RUN_NAME ./data/modal-results/
```

Replace `RUN_NAME` with the directory name printed by the Volume listing. The
existing destination directory receives a `RUN_NAME/` subdirectory. `--force`
refreshes existing local files for that run. While a
run is still incomplete, only its logs and training checkpoint may be present;
the native final model requires `model.pt`, `config.json`, and `metadata.json`.

Set the Workspace usage limit to $30 in Usage & Billing if you want to stay
within the monthly included compute. The course workspace's GPU-hour policy
and shared data Volume do not apply to this personal runner.

## Run P1 experiments directly from RUNS

Edit `RUNS` in `experiments/a1_basics/p1_hyperparameters.py`. Each item is a
dictionary of `TrainConfig` overrides, constructed into the actual config on
Modal so the laptop still needs only the Modal SDK. For example:

```python
RUNS = [
    {"learning_rate": 1e-3},
    {"learning_rate": 9e-3},
]
```

Launch the list from the assignments directory:

```bash
MODAL_PROFILE=haozeli2025 ~/.local/bin/modal run --env main -m \
  experiments.a1_basics.p1_hyperparameters::p1
```

The launcher waits for each full training run before starting the next, using
the same personal Volume, output paths, W&B-disabled settings, one-hour per-run
timeout, and no-retry policy as `modal_personal.py`. Keep the terminal running
until the list finishes. A failed experiment stops the list. Each completed
run prints its result and downloads the six artifacts into
`data/modal-results/RUN_NAME/` before starting the next run. Files also remain
under `/ckpts/RUN_NAME` on `cs312-personal`. A download failure stops the list
and reports the retained cloud location. Add `--no-download` to skip local
downloads, or `--output-dir PATH` to change the local parent directory.
Identical completed runs are reused by the training code; rerunning the list
can resume interrupted checkpoints. No separate JSON files are needed.

## Inference with the trained d8 model

`modal_infer.py` reads the existing native model directly from the personal
Volume, reuses the training image and model loader, and caches the pinned
4096-token SentencePiece tokenizer in its image. It does not train a model.
Only the Modal SDK is needed locally. If `modal` is not on your shell's PATH,
use `~/.local/bin/modal` as below.

### Run once from the command line

```bash
cd /Users/bob.li/Code/CS312-Stanford/code/assignments
MODAL_PROFILE=haozeli2025 ~/.local/bin/modal run --env main modal_infer.py \
  --prompt "Once upon a time, there was" \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --top-k 40 \
  --seed 42 \
  --output ./data/modal-inference/generation.json
```

The JSON is printed and, when `--output` is supplied, saved locally at that
path (replacing a previous file at the same path). `text` is the decoded full
sequence and `completion` is the decoded generated suffix. The result also
includes token counts, stop reason, sampling options, device, and generation
time excluding cold start/model loading. Use `--temperature 0` for greedy
decoding or `--top-k 0` to sample across the vocabulary.

The model is an English text-continuation baseline, not an instruction-tuned
chat model. Generation recomputes the prefix at every step (no KV cache).
Start with 32–64 new tokens. The hard limit is 256 new tokens, and prompt
tokens (including BOS) plus requested new tokens must fit in 1024 tokens.
Invalid HTTP parameters return 422. Sampling uses a request-local seed;
identical seeds need not produce identical text across CPU/GPU platforms.

CPU is the default. Containers scale to zero with a 30-second idle window;
the maximum is one container, with no automatic retries. For optional GPU
inference, prefix the same run/serve/deploy command with `CS312_INFER_GPU=L4`:

```bash
CS312_INFER_GPU=L4 MODAL_PROFILE=haozeli2025 ~/.local/bin/modal run --env main \
  modal_infer.py --prompt "Once upon a time," --max-new-tokens 32
```

GPU configuration is chosen when the script is imported for deployment.
Redeploy with the environment variable unset to return a deployed endpoint
to CPU. Allocated containers consume resources while processing and waiting
to scale down, including the idle window.

### Serve or deploy an HTTP endpoint

```bash
# Temporary URL while this command is running; Ctrl-C stops the development app.
MODAL_PROFILE=haozeli2025 ~/.local/bin/modal serve --env main modal_infer.py

# Persistent URL, with compute still allowed to scale to zero.
MODAL_PROFILE=haozeli2025 ~/.local/bin/modal deploy --env main modal_infer.py
```

The app is named `cs312-personal-inference`. Use the `D8Inference.web` URL
printed by Modal. The endpoint requires **Modal Proxy Tokens**, which are
different from your SDK login credentials. Create a Proxy Token under Modal
Dashboard settings; put its ID and secret into your terminal's
`MODAL_PROXY_KEY` and `MODAL_PROXY_SECRET` environment variables. Set
`INFERENCE_URL` to the printed endpoint URL, then call it:

```bash
curl --fail-with-body "$INFERENCE_URL" \
  -H "Content-Type: application/json" \
  -H "Modal-Key: $MODAL_PROXY_KEY" \
  -H "Modal-Secret: $MODAL_PROXY_SECRET" \
  --data '{"prompt":"Once upon a time,","max_new_tokens":32,"temperature":0.8,"top_k":40,"seed":42}'
```

Python callers can use their existing Modal SDK authentication after deployment:

```python
import modal

D8 = modal.Cls.from_name(
    "cs312-personal-inference", "D8Inference", environment_name="main",
)
result = D8().generate.remote("Once upon a time,", max_new_tokens=32)
print(result["text"])
```

Run Python clients with `MODAL_PROFILE=haozeli2025`. Inference does not alter
the model files. To remove the deployed service:

```bash
MODAL_PROFILE=haozeli2025 ~/.local/bin/modal app stop --env main cs312-personal-inference
```

References: [Web Functions](https://modal.com/docs/guide/webhooks),
[Proxy Tokens](https://modal.com/docs/guide/webhook-proxy-auth),
[cold starts and scaling](https://modal.com/docs/guide/cold-start).
