"""Verify the personal P1 launcher without connecting to Modal."""

import subprocess
import sys
import unittest
from importlib import import_module
from unittest.mock import patch


class ImportTests(unittest.TestCase):
    def test_import_does_not_require_local_pytorch(self):
        result = subprocess.run([
            sys.executable, "-c",
            "import sys; sys.modules['torch'] = None; "
            "import experiments.a1_basics.p1_hyperparameters",
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.launcher = import_module("experiments.a1_basics.p1_hyperparameters")

    def test_forwards_each_configuration_to_full_training_in_order(self):
        runs = [{"learning_rate": 0.001}, {"learning_rate": 0.009}]

        def remote(config, smoke):
            self.assertIs(smoke, False)
            return {"learning_rate": config["learning_rate"], "completed_steps": 9375}

        with patch.object(self.launcher, "volume"), patch.object(self.launcher, "run_training") as training:
            training.with_options.return_value.remote.side_effect = remote
            results = self.launcher.launch_training_jobs(runs, download=False)
        self.assertEqual(results, [
            {"learning_rate": 0.001, "completed_steps": 9375},
            {"learning_rate": 0.009, "completed_steps": 9375},
        ])
        self.assertEqual(runs, [{"learning_rate": 0.001}, {"learning_rate": 0.009}])

    def test_validates_entire_list_before_any_training(self):
        for invalid in (
            [], [{"learning_rate": 0.001}, {"wandb_online": True}],
            [{"save_model": False}], [{"latest_checkpoint_frequency": None}],
            [{"learning_rate": float("nan")}], [object()],
        ):
            with self.subTest(invalid=invalid), patch.object(self.launcher, "volume"), \
                    patch.object(self.launcher, "run_training") as training:
                training.with_options.side_effect = AssertionError("Training must not start")
                with self.assertRaises(ValueError):
                    self.launcher.launch_training_jobs(invalid)

    def test_failure_stops_remaining_experiments(self):
        completed = []

        def remote(config, smoke):
            if config["learning_rate"] == 0.009:
                raise RuntimeError("Training failed")
            completed.append(config["learning_rate"])
            return {"learning_rate": config["learning_rate"]}

        with patch.object(self.launcher, "volume"), patch.object(self.launcher, "run_training") as training:
            training.with_options.return_value.remote.side_effect = remote
            with self.assertRaisesRegex(RuntimeError, "Training failed"):
                self.launcher.launch_training_jobs([
                    {"learning_rate": 0.001}, {"learning_rate": 0.009},
                    {"learning_rate": 0.006},
                ], download=False)
        self.assertEqual(completed, [0.001])


if __name__ == "__main__":
    unittest.main()
