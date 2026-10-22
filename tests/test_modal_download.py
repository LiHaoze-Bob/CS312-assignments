"""Exercise auto-download using real files and a fake remote Volume boundary."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import modal_personal as personal
from experiments.a1_basics import p1_hyperparameters as p1


ARTIFACTS = {
    "model.pt": b"model weights", "latest.pt": b"optimizer checkpoint",
    "config.json": b'{"name":"d8"}', "metadata.json": b'{"total_steps":9375}',
    "console.log": b"Step 9375/9375\n", "result.json": b'{"completed_steps":9375}',
}


def result_for(name="d8-test"):
    return {"run_name": name, "volume": "cs312-personal", "volume_path": f"/ckpts/{name}"}


class RemoteVolume:
    def __init__(self, failed_file=None):
        self.failed_file = failed_file

    def read_file_into_fileobj(self, remote_path, destination):
        filename = remote_path.rsplit("/", 1)[-1]
        if filename == self.failed_file:
            destination.write(b"partial")
            raise OSError("Connection interrupted")
        payload = ARTIFACTS[filename]
        destination.write(payload)
        return len(payload)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(personal, "download_training_result"), "Auto-download is not implemented")

    def test_downloads_artifacts_into_a_run_subdirectory(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(personal, "volume", RemoteVolume()):
            existing = Path(directory) / "d8-test"
            existing.mkdir()
            (existing / "model.pt").write_bytes(b"older weights")
            (existing / "notes.txt").write_bytes(b"my experiment notes")
            destination = personal.download_training_result(result_for(), directory)
            self.assertEqual(destination, Path(directory).resolve() / "d8-test")
            self.assertEqual({p.name: p.read_bytes() for p in destination.iterdir()}, {
                **ARTIFACTS, "notes.txt": b"my experiment notes",
            })

    def test_default_destination_is_relative_to_the_script_not_working_directory(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(personal, "ROOT", Path(directory)), \
                patch.object(personal, "volume", RemoteVolume()):
            destination = personal.download_training_result(result_for())
            self.assertEqual(destination, Path(directory) / "data/modal-results/d8-test")
            self.assertEqual((destination / "model.pt").read_bytes(), b"model weights")

    def test_failed_transfer_preserves_existing_file_and_removes_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "d8-test"
            target.mkdir()
            (target / "model.pt").write_bytes(b"previous complete model")
            (target / "notes.txt").write_text("Keep my notes")
            with patch.object(personal, "volume", RemoteVolume(failed_file="model.pt")):
                with self.assertRaisesRegex(RuntimeError, "Training.*saved.*download failed"):
                    personal.download_training_result(result_for(), directory)
            self.assertEqual((target / "model.pt").read_bytes(), b"previous complete model")
            self.assertEqual((target / "notes.txt").read_text(), "Keep my notes")
            self.assertEqual(sorted(p.name for p in target.iterdir()), ["model.pt", "notes.txt"])

    def test_rejects_paths_outside_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("..", "../outside", "/absolute", "nested/run"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    personal.download_training_result(result_for(name), directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


class EntrypointTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(personal, "download_training_result"), "Auto-download is not implemented")

    def test_single_run_downloads_after_training_returns(self):
        events = []
        def remote(*args):
            events.append("trained")
            return result_for()
        def download(result, output_dir=None):
            self.assertEqual(result, result_for())
            events.append("downloaded")
        with patch.object(personal, "volume"), patch.object(personal, "run_training") as training, \
                patch.object(personal, "download_training_result", side_effect=download):
            training.with_options.return_value.remote.side_effect = remote
            personal.main(action="train")
        self.assertEqual(events, ["trained", "downloaded"])

    def test_p1_downloads_each_run_before_training_the_next(self):
        events = []
        def remote(config, smoke):
            name = str(config["learning_rate"])
            events.append(("trained", name))
            return result_for(name)
        def download(result, output_dir=None):
            events.append(("downloaded", result["run_name"]))
        with patch.object(p1, "volume"), patch.object(p1, "run_training") as training, \
                patch.object(p1, "download_training_result", side_effect=download):
            training.with_options.return_value.remote.side_effect = remote
            p1.launch_training_jobs([{"learning_rate": 0.001}, {"learning_rate": 0.009}])
        self.assertEqual(events, [
            ("trained", "0.001"), ("downloaded", "0.001"),
            ("trained", "0.009"), ("downloaded", "0.009"),
        ])

    def test_download_failure_stops_the_next_training(self):
        completed = []
        def remote(config, smoke):
            completed.append(config["learning_rate"])
            return result_for()
        with patch.object(p1, "volume"), patch.object(p1, "run_training") as training, \
                patch.object(p1, "download_training_result", side_effect=RuntimeError("download failed")):
            training.with_options.return_value.remote.side_effect = remote
            with self.assertRaisesRegex(RuntimeError, "download failed"):
                p1.launch_training_jobs([{"learning_rate": 0.001}, {"learning_rate": 0.009}])
        self.assertEqual(completed, [0.001])

    def test_download_can_be_disabled(self):
        with patch.object(personal, "volume"), patch.object(personal, "run_training") as training, \
                patch.object(personal, "download_training_result") as download:
            training.with_options.return_value.remote.return_value = result_for()
            personal.main(action="train", download=False)
        download.assert_not_called()

    def test_failed_training_does_not_download_a_result(self):
        with patch.object(personal, "volume"), patch.object(personal, "run_training") as training, \
                patch.object(personal, "download_training_result") as download:
            training.with_options.return_value.remote.side_effect = RuntimeError("Training failed")
            with self.assertRaisesRegex(RuntimeError, "Training failed"):
                personal.main(action="train")
        download.assert_not_called()

    def test_preparing_data_does_not_download_a_training_result(self):
        with patch.object(personal, "prepare_data") as prepare, \
                patch.object(personal, "download_training_result") as download:
            prepare.remote.return_value = {"sha256_verified": True}
            personal.main(action="prepare")
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
