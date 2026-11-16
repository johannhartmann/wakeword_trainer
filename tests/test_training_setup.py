"""Pronunciation and OpenRouter routing checks without live API calls."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from wakeword_trainer import trainer


class TrainingSetupTests(unittest.TestCase):
    def test_saturated_wrong_predictions_still_receive_gradients(self):
        sequential = trainer.nn.Sequential

        def saturated_model(*layers):
            model = sequential(*layers)
            torch.nn.init.zeros_(model[-2].weight)
            torch.nn.init.constant_(model[-2].bias, 30.0)
            return model

        instance = trainer.WakeWordTrainer.__new__(trainer.WakeWordTrainer)
        instance.steps, instance.batch_size, instance.layer_dim = 4, 32, 8
        instance.target_lr = 0.01
        features = np.zeros((2, 16, 96), dtype=np.float32)
        with patch.object(trainer.nn, "Sequential", side_effect=saturated_model), \
                patch.object(torch.cuda, "is_available", return_value=False), \
                contextlib.redirect_stdout(io.StringIO()):
            model, _ = instance.train_model(features, features)
        self.assertLess(model[-2].bias.item(), 29.999)

    def test_perfect_batch_recall_does_not_restore_an_early_checkpoint(self):
        sequential = trainer.nn.Sequential

        def positive_model(*layers):
            model = sequential(*layers)
            torch.nn.init.zeros_(model[-2].weight)
            torch.nn.init.constant_(model[-2].bias, 2.0)
            return model

        instance = trainer.WakeWordTrainer.__new__(trainer.WakeWordTrainer)
        instance.steps, instance.batch_size, instance.layer_dim = 20, 32, 8
        instance.target_lr = 0.01
        features = np.zeros((2, 16, 96), dtype=np.float32)
        with patch.object(trainer.nn, "Sequential", side_effect=positive_model), \
                patch.object(torch.cuda, "is_available", return_value=False), \
                contextlib.redirect_stdout(io.StringIO()):
            model, history = instance.train_model(features, features)
        self.assertEqual(history["recall"], [1.0] * 20)
        # The negative-weighted objective keeps reducing this bias after the
        # first perfect-recall checkpoint. Restoring that early state fails.
        self.assertLess(model[-2].bias.item(), 1.95)

    def test_augmentation_keeps_target_near_end_of_window(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            instance = trainer.WakeWordTrainer.__new__(trainer.WakeWordTrainer)
            instance.output_dir = root
            instance.clips_dir = root / "clips"
            instance.clips_dir.mkdir()
            instance.clips_augmented_dir = root / "augmented"
            instance.rir_dir = root / "missing_rirs"
            instance.clip_length = 2.0
            instance.augmentation_rounds = 1
            sf.write(instance.clips_dir / "target.wav", np.sin(np.arange(8000) * 0.1) * 0.3, 16000)
            instance.augment_clips()
            audio, rate = sf.read(next(instance.clips_augmented_dir.glob("*.wav")))
            active = np.flatnonzero(np.abs(audio) > 0.001)
            self.assertEqual((len(audio), rate), (32000, 16000))
            self.assertGreaterEqual(active[0], 20800)
            self.assertGreaterEqual(active[-1], 28800)

    def test_identical_sounds_are_not_used_as_negatives(self):
        self.assertEqual(
            trainer._exclude_homophones(["may stack"], ["mace tack", "my stack", "haystack"]),
            ["my stack", "haystack"],
        )

    def test_openrouter_routes_both_generators_to_gemini(self):
        responses = [
            {"phrases": ["may stack", "hey may stack"]},
            ["my stack", "haystack"],
        ]
        streams = [io.BytesIO(json.dumps({
            "choices": [{"message": {"content": json.dumps(value)}}],
        }).encode()) for value in responses]
        with patch.object(trainer, "_get_openrouter_api_key", return_value="fake-test-key"), \
                patch.object(trainer, "urlopen", side_effect=streams) as request, \
                patch.object(trainer.genai, "Client") as google, \
                patch.dict(os.environ, {"OPENROUTER_MODEL": "google/gemini-4-hydrogen"}), \
                patch("dotenv.load_dotenv"):
            self.assertEqual(trainer.generate_phrases_with_llm("may stack"), responses[0]["phrases"])
            instance = trainer.WakeWordTrainer.__new__(trainer.WakeWordTrainer)
            self.assertEqual(instance._generate_adversarial_with_llm("may stack"), responses[1])
            for call in request.call_args_list:
                req = call.args[0]
                body = json.loads(req.data)
                self.assertEqual(req.full_url, "https://openrouter.ai/api/v1/chat/completions")
                self.assertEqual(req.get_header("Authorization"), "Bearer fake-test-key")
                self.assertEqual(body["model"], "google/gemini-4-hydrogen")
                self.assertEqual(body["response_format"]["type"], "json_schema")
                self.assertNotIn("fake-test-key", req.data.decode())
            google.assert_not_called()

    def test_home_credential_does_not_export_unrelated_secrets(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch("dotenv.dotenv_values", return_value={"OPENROUTER_API_KEY": "fake-home-key", "UNRELATED_SECRET": "other"}) as read:
            self.assertEqual(trainer._get_openrouter_api_key(), "fake-home-key")
            read.assert_called_once_with(Path.home() / ".env")
            self.assertNotIn("UNRELATED_SECRET", os.environ)

    def test_pronunciation_does_not_change_model_name(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(trainer, "generate_phrases_with_llm", return_value=["may stack"]) as positive, \
                patch.object(trainer, "generate_adversarial_from_phonemes", return_value=["my stack"]) as negative, \
                contextlib.redirect_stdout(io.StringIO()):
            instance = trainer.WakeWordTrainer("maistack", output_dir=directory, pronunciation="may stack")
            self.assertEqual(instance.wake_word, "maistack")
            self.assertEqual(instance.clips_dir.name, "maistack_clips")
            positive.assert_called_once_with("may stack")
            negative.assert_called_once_with("may stack", n_samples=500)


if __name__ == "__main__":
    unittest.main()
