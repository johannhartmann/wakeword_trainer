"""Offline regression checks for the updated training dependencies."""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch

from wakeword_trainer import trainer


class CompatibilityTests(unittest.TestCase):
    def test_help_without_loading_training_stack(self):
        result = subprocess.run(
            [sys.executable, "-c", (
                "import runpy, sys; sys.argv = ['wakeword-trainer', '--help']; "
                "\ntry: runpy.run_module('wakeword_trainer', run_name='__main__')"
                "\nexcept SystemExit as exc: assert exc.code == 0"
                "\nassert 'wakeword_trainer.trainer' not in sys.modules"
            )],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GEMINI_MODEL", result.stdout)

    def test_both_phrase_generators_use_selected_model(self):
        self.assertEqual(trainer.DEFAULT_GEMINI_MODEL, "gemini-4-hydrogen")
        for selected in (None, "custom-gemini-model"):
            with self.subTest(selected=selected), patch.dict(os.environ, {"GEMINI_API_KEY": "test"}), \
                    patch.object(trainer.genai, "Client") as client, \
                    patch("dotenv.load_dotenv"):
                os.environ.pop("GEMINI_MODEL", None)
                if selected:
                    os.environ["GEMINI_MODEL"] = selected
                generate = client.return_value.models.generate_content
                generate.return_value.text = '{"phrases": ["jarvis", "hey jarvis"]}'
                self.assertEqual(trainer.generate_phrases_with_llm("jarvis"), ["jarvis", "hey jarvis"])
                self.assertEqual(generate.call_args.kwargs["model"], selected or trainer.DEFAULT_GEMINI_MODEL)
                self.assertEqual(generate.call_args.kwargs["config"]["response_json_schema"]["type"], "object")

                generate.return_value.text = '["harvest", "jars"]'
                instance = trainer.WakeWordTrainer.__new__(trainer.WakeWordTrainer)
                self.assertEqual(instance._generate_adversarial_with_llm("jarvis"), ["harvest", "jars"])
                self.assertEqual(generate.call_args.kwargs["model"], selected or trainer.DEFAULT_GEMINI_MODEL)
                self.assertEqual(generate.call_args.kwargs["config"]["response_json_schema"]["type"], "array")

    def test_audio_metadata_and_mixing_with_reverberation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            foreground = root / "foreground.wav"
            background = root / "background.wav"
            rir = root / "rir.wav"
            sf.write(foreground, 0.3 * np.sin(np.arange(16000) * 0.1), 16000)
            sf.write(background, np.random.default_rng(42).normal(0, 0.01, 40000), 16000)
            sf.write(rir, np.r_[1.0, np.zeros(1599)], 16000)
            self.assertAlmostEqual(trainer.openwakeword.data.get_clip_duration(str(foreground)), 1.0)
            batch, labels, _ = next(trainer.openwakeword.data.mix_clips_batch(
                foreground_clips=[str(foreground)],
                background_clips=[str(background)],
                combined_size=32000, batch_size=1, labels=[1],
                rirs=[str(rir)], rir_probability=1.0, seed=42,
            ))
            self.assertEqual(batch.shape, (1, 32000))
            self.assertEqual(batch.dtype, np.int16)
            self.assertGreater(np.abs(batch.astype(np.float32)).max(), 0)
            np.testing.assert_array_equal(labels, [1])

    def test_training_and_single_file_onnx_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            instance = trainer.WakeWordTrainer.__new__(trainer.WakeWordTrainer)
            instance.output_dir = Path(directory)
            instance.wake_word = "test"
            instance.steps = 20
            instance.batch_size = 32
            instance.layer_dim = 8
            instance.target_lr = 0.001
            rng = np.random.default_rng(42)
            positive = rng.normal(1, 0.1, (10, 16, 96)).astype(np.float32)
            negative = rng.normal(0, 0.1, (10, 16, 96)).astype(np.float32)
            with patch.object(torch.cuda, "is_available", return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()):
                model, history = instance.train_model(positive, negative)
                path = instance.export_onnx(model, (16, 96))
            self.assertEqual(len(history["loss"]), 20)
            self.assertTrue(np.isfinite(history["loss"]).all())
            onnx.checker.check_model(str(path))
            self.assertEqual(list(Path(directory).iterdir()), [path])
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.assertEqual(session.get_outputs()[0].name, "test")
            for size in (1, 3):
                inputs = positive[:size]
                with torch.no_grad():
                    expected = model(torch.from_numpy(inputs)).numpy()
                actual = session.run(None, {"input": inputs})[0]
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
