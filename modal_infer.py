"""Serve the personal d8 baseline as a continuation model.

From this directory (only the Modal SDK is needed locally):
  MODAL_PROFILE=haozeli2025 modal run --env main modal_infer.py --prompt "Once upon a time,"
  MODAL_PROFILE=haozeli2025 modal serve --env main modal_infer.py
  MODAL_PROFILE=haozeli2025 modal deploy --env main modal_infer.py

HTTP POST accepts prompt, max_new_tokens, temperature, top_k, and seed.
HTTP requests require Modal Proxy Token headers (Modal-Key / Modal-Secret).
CPU is the default. Set CS312_INFER_GPU=L4 before run/serve/deploy to use a GPU.
This model performs text continuation; it has no chat template or KV cache.
"""

import json
import math
import os
import time
from pathlib import Path

import modal

from modal_personal import ENVIRONMENT, MOUNT, MODELS, REMOTE_REPO, ROOT, base_image, ignore_source


RUN_NAME = "model-d8-lr0.003-tok614M-modal-personal"
MODEL_PATH = f"{MODELS}/{RUN_NAME}"
TOKENIZER_REPO = "Ryukijano/parameter-golf-sp4096"
TOKENIZER_REVISION = "23dbd5ab0fcc1edfd25eedf1ae9d41a5d8b6b233"
TOKENIZER_PATH = "/opt/cs312-tokenizer/tokenizers/fineweb_4096_bpe.model"
GPU = os.environ.get("CS312_INFER_GPU", "").strip() or None

app = modal.App("cs312-personal-inference")
volume = modal.Volume.from_name("cs312-personal", environment_name=ENVIRONMENT)
# Cache the pinned tokenizer in the image, rather than download on each cold start.
image = base_image.uv_pip_install("fastapi[standard]").run_commands(
    "python -c \"from huggingface_hub import hf_hub_download; "
    f"hf_hub_download(repo_id='{TOKENIZER_REPO}', repo_type='dataset', "
    "filename='tokenizers/fineweb_4096_bpe.model', "
    f"revision='{TOKENIZER_REVISION}', local_dir='/opt/cs312-tokenizer')\""
).add_local_dir(ROOT, remote_path=REMOTE_REPO, ignore=ignore_source)


def validate_request(payload):
    """Validate JSON and CLI options before allocating inference work."""
    defaults = dict(max_new_tokens=64, temperature=0.8, top_k=40, seed=42)
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object.")
    unknown = payload.keys() - {"prompt", *defaults}
    if unknown:
        raise ValueError(f"Unknown request fields: {', '.join(sorted(unknown))}")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")
    if len(prompt) > 16384:
        raise ValueError("prompt must contain at most 16384 characters.")
    request = {**defaults, **payload}
    for name, lower, upper in (
        ("max_new_tokens", 1, 256), ("top_k", 0, 4096), ("seed", 0, 2**63 - 1),
    ):
        value = request[name]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer in [{lower}, {upper}].")
    temperature = request["temperature"]
    if (
        type(temperature) not in (int, float)
        or not 0 <= temperature <= 10
        or not math.isfinite(temperature)
    ):
        raise ValueError("temperature must be a finite number in [0, 10]; 0 is greedy.")
    request["temperature"] = float(temperature)
    return request


def generate_text(model, tokenizer, payload, device):
    """Use the course model's forward pass and request-local random generator."""
    import torch

    request = validate_request(payload)
    ids = [tokenizer.bos_id()] + tokenizer.encode(request["prompt"], out_type=int)
    prompt_tokens = len(ids)
    # Keep the complete prompt and continuation within the trained context.
    if prompt_tokens + request["max_new_tokens"] > model.config.context_length:
        raise ValueError(
            f"Prompt ({prompt_tokens} tokens including BOS) plus max_new_tokens "
            f"must fit within {model.config.context_length} tokens. "
            "Shorten the prompt or lower max_new_tokens."
        )
    generator = torch.Generator(device=device).manual_seed(request["seed"])
    # Use tensor/tensor division: CUDA scalar division can take a reciprocal
    # first, which overflows for an otherwise valid subnormal temperature.
    temperature = torch.tensor(request["temperature"], dtype=torch.float64, device=device)
    generated_ids = []
    stop_reason = "max_new_tokens"
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(request["max_new_tokens"]):
            inputs = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(input_ids=inputs)[0, -1].float()
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Model produced non-finite logits.")
            logits[tokenizer.pad_id()] = -torch.inf
            logits[tokenizer.bos_id()] = -torch.inf
            if request["temperature"] == 0:
                next_id = int(logits.argmax())
            else:
                # Float64 preserves tiny positive temperatures. Center first
                # so the maximum stays zero even if negative logits overflow.
                logits = logits.double()
                logits = (logits - logits.max()) / temperature
                if request["top_k"]:
                    cutoff = logits.topk(min(request["top_k"], model.vocab_size)).values[-1]
                    logits = logits.masked_fill(logits < cutoff, -torch.inf)
                next_id = int(torch.multinomial(logits.softmax(dim=-1), 1, generator=generator))
            ids.append(next_id)
            generated_ids.append(next_id)
            if next_id == tokenizer.eos_id():
                stop_reason = "eos"
                break
    return {
        "model": RUN_NAME, "prompt": request["prompt"],
        "text": tokenizer.decode(ids), "completion": tokenizer.decode(generated_ids),
        "prompt_tokens": prompt_tokens, "generated_tokens": len(generated_ids),
        "stop_reason": stop_reason, "seconds": time.perf_counter() - started,
        "device": device, "max_new_tokens": request["max_new_tokens"],
        "temperature": request["temperature"], "top_k": request["top_k"],
        "seed": request["seed"],
    }


@app.cls(
    image=image, volumes={MOUNT: volume}, gpu=GPU, cpu=2, memory=4096,
    min_containers=0, max_containers=1, buffer_containers=0,
    scaledown_window=30, timeout=300, retries=0,
)
class D8Inference:
    @modal.enter()
    def load(self):
        import sentencepiece as spm
        import torch

        from model_io import load_model

        for filename in ("model.pt", "config.json", "metadata.json"):
            if not (Path(MODEL_PATH) / filename).is_file():
                raise FileNotFoundError(f"Missing trained artifact: {MODEL_PATH}/{filename}")
        torch.set_num_threads(2)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model(MODEL_PATH, device=self.device, precision="fp32")
        self.tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
        if self.tokenizer.vocab_size() != self.model.vocab_size:
            raise RuntimeError("Tokenizer vocabulary does not match model vocabulary.")
        special_ids = (
            self.tokenizer.pad_id(), self.tokenizer.bos_id(),
            self.tokenizer.eos_id(), self.tokenizer.unk_id(),
        )
        if special_ids != (0, 1, 2, 3):
            raise RuntimeError("Tokenizer special IDs do not match the course data.")

    def _generate(self, payload):
        return generate_text(self.model, self.tokenizer, payload, self.device)

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int = 64,
                 temperature: float = 0.8, top_k: int = 40, seed: int = 42):
        return self._generate(dict(
            prompt=prompt, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, seed=seed,
        ))

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def web(self, payload: dict):
        from fastapi import HTTPException

        try:
            return self._generate(payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.local_entrypoint()
def main(prompt: str = "Once upon a time, there was", max_new_tokens: int = 64,
         temperature: float = 0.8, top_k: int = 40, seed: int = 42, output: str = ""):
    request = validate_request(dict(
        prompt=prompt, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k, seed=seed,
    ))
    result = D8Inference().generate.remote(**request)
    report = json.dumps(result, indent=2, ensure_ascii=False)
    if output:
        destination = Path(output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report + "\n", encoding="utf-8")
    print(report)
