"""Offline request-contract tests; no Modal connection or model download."""

import unittest
import importlib.util
from types import SimpleNamespace

try:
    from modal_infer import generate_text, validate_request
except ModuleNotFoundError as exc:
    if exc.name != "modal_infer":
        raise
    validate_request = None


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(validate_request, "modal_infer is not implemented")

    def test_defaults_preserve_prompt(self):
        self.assertEqual(validate_request({"prompt": "Once upon a time, "}), {
            "prompt": "Once upon a time, ", "max_new_tokens": 64,
            "temperature": 0.8, "top_k": 40, "seed": 42,
        })

    def test_accepts_greedy_and_full_vocabulary_sampling(self):
        result = validate_request({"prompt": "Hello", "temperature": 0, "top_k": 0})
        self.assertEqual(result["temperature"], 0.0)
        self.assertEqual(result["top_k"], 0)

    def test_rejects_missing_empty_and_oversized_prompts(self):
        for payload in ({}, {"prompt": "  "}, {"prompt": 123}, {"prompt": "a" * 16385}):
            with self.subTest(payload_type=type(payload.get("prompt"))):
                with self.assertRaises(ValueError):
                    validate_request(payload)

    def test_rejects_non_object_and_unknown_options(self):
        for payload in ([], None, {"prompt": "Hello", "max_tokens": 12}):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    validate_request(payload)

    def test_rejects_unbounded_generation_and_invalid_seeds(self):
        for field, value in (
            ("max_new_tokens", 0), ("max_new_tokens", 257),
            ("max_new_tokens", True), ("max_new_tokens", 1.5),
            ("top_k", -1), ("top_k", 4097), ("top_k", "40"),
            ("seed", -1), ("seed", 2**63), ("seed", False),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    validate_request({"prompt": "Hello", field: value})

    def test_rejects_nonfinite_or_invalid_temperatures(self):
        for temperature in (-0.1, 11, 10**400, float("nan"), float("inf"), "0.8", True):
            with self.subTest(temperature=temperature):
                with self.assertRaises(ValueError):
                    validate_request({"prompt": "Hello", "temperature": temperature})


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed locally")
class GenerationTests(unittest.TestCase):
    """Exercise real tensor sampling with hand-chosen next-token logits."""

    def setUp(self):
        import torch

        class FixedLogitsModel:
            config = SimpleNamespace(context_length=8)
            vocab_size = 6

            def __init__(self, logits):
                self.logits = logits
                self.calls = 0

            def __call__(self, input_ids):
                self.calls += 1
                return torch.tensor(self.logits, dtype=torch.float32, device=input_ids.device).view(1, 1, 6).expand(
                    1, input_ids.shape[1], 6,
                )

        class Tokenizer:
            def bos_id(self):
                return 1

            def pad_id(self):
                return 0

            def eos_id(self):
                return 2

            def encode(self, text, out_type):
                return [3]

            def decode(self, ids):
                return ",".join(str(token) for token in ids if token >= 3)

        self.make_model = FixedLogitsModel
        self.tokenizer = Tokenizer()

    def test_greedy_masks_special_tokens_and_stops_at_eos(self):
        model = self.make_model([1000, 1000, 9, 2, 1, 0])
        result = generate_text(model, self.tokenizer, {
            "prompt": "Hello", "temperature": 0, "max_new_tokens": 4,
        }, "cpu")
        self.assertEqual(result["stop_reason"], "eos")
        self.assertEqual(result["generated_tokens"], 1)
        self.assertEqual(model.calls, 1)
        self.assertEqual(result["completion"], "")

    def test_tiny_positive_temperature_produces_finite_sampling(self):
        import torch

        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            for temperature in (1e-300, 5e-324):
                with self.subTest(temperature=temperature, device=device):
                    result = generate_text(self.make_model([1000, 1000, -10, 0, 1, 2]),
                                           self.tokenizer, {
                        "prompt": "Hello", "temperature": temperature, "max_new_tokens": 3,
                    }, device)
                    self.assertEqual(result["completion"], "5,5,5")
                    self.assertEqual(result["stop_reason"], "max_new_tokens")

    def test_top_k_one_excludes_lower_logits(self):
        result = generate_text(self.make_model([1000, 1000, -10, 0, 1, 2]),
                               self.tokenizer, {
            "prompt": "Hello", "top_k": 1, "max_new_tokens": 3,
        }, "cpu")
        self.assertEqual(result["completion"], "5,5,5")

    def test_seeded_sampling_repeats_without_changing_global_rng(self):
        import torch

        state = torch.random.get_rng_state().clone()
        request = {"prompt": "Hello", "seed": 42, "top_k": 0, "max_new_tokens": 4}
        results = [generate_text(self.make_model([-100, -100, -100, 0, 0, 0]),
                                 self.tokenizer, request, "cpu") for _ in range(2)]
        self.assertEqual(results[0]["completion"], results[1]["completion"])
        self.assertEqual(results[0]["generated_tokens"], 4)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_context_overflow_is_rejected_before_forward(self):
        model = self.make_model([0] * 6)
        with self.assertRaisesRegex(ValueError, "fit within 8 tokens"):
            generate_text(model, self.tokenizer, {"prompt": "Hello", "max_new_tokens": 7}, "cpu")
        self.assertEqual(model.calls, 0)


if __name__ == "__main__":
    unittest.main()
