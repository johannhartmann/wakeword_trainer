"""Check asset refresh and recovery without network access."""

import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wakeword_trainer import model_assets
from wakeword_trainer.__main__ import main


class ModelAssetTests(unittest.TestCase):
    def test_refresh_replaces_stale_voice_and_repairs_missing_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, target = root / "snapshot", root / "voices"
            name, filename = model_assets.PIPER_VOICES[0]
            source = snapshot / filename
            source.parent.mkdir(parents=True)
            source.write_bytes(b"current weights")
            Path(str(source) + ".json").write_text('{"current": true}')
            (source.parent / "MODEL_CARD").write_text("Voice model license")
            target.mkdir()
            (target / f"{name}.onnx").write_bytes(b"old weights")
            with patch.object(model_assets, "PIPER_VOICES", [(name, filename)]), \
                    patch("huggingface_hub.snapshot_download", return_value=str(snapshot)) as download:
                model_assets.download_piper_voices(target)
                self.assertEqual((target / f"{name}.onnx").read_bytes(), b"current weights")
                self.assertEqual((target / f"{name}.onnx.json").read_text(), '{"current": true}')
                self.assertEqual(download.call_args.kwargs["revision"], model_assets.PIPER_REVISION)
                config = target / f"{name}.onnx.json"
                config.unlink()
                model_assets.download_piper_voices(target)
                self.assertTrue(config.exists())

    def test_failed_download_preserves_existing_voice(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            name, _ = model_assets.PIPER_VOICES[0]
            existing = target / f"{name}.onnx"
            existing.write_bytes(b"existing usable weights")
            with patch("huggingface_hub.snapshot_download", side_effect=OSError("offline")):
                with self.assertRaises(OSError):
                    model_assets.download_piper_voices(target)
            self.assertEqual(existing.read_bytes(), b"existing usable weights")

    def test_frontend_checksum_refresh_and_failed_download(self):
        payload = b"current frontend weights"
        checksum = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "embedding_model.onnx"
            target.write_bytes(b"old or truncated weights")
            with patch.object(model_assets, "OPENWAKEWORD_MODELS", {target.name: checksum}), \
                    patch.object(model_assets, "urlopen", return_value=io.BytesIO(payload)) as download:
                model_assets.download_openwakeword_models(root)
                self.assertEqual(target.read_bytes(), payload)
                self.assertEqual(download.call_count, 1)
                model_assets.download_openwakeword_models(root)
                self.assertEqual(download.call_count, 1)
            with patch.object(model_assets, "OPENWAKEWORD_MODELS", {target.name: "0" * 64}), \
                    patch.object(model_assets, "urlopen", return_value=io.BytesIO(b"truncated")):
                with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                    model_assets.download_openwakeword_models(root)
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(list(root.iterdir()), [target])

    def test_update_models_without_wake_word_or_llm(self):
        with patch("sys.argv", ["wakeword-trainer", "--update-models", "-o", "output"]), \
                patch.object(model_assets, "download_piper_voices") as piper, \
                patch.object(model_assets, "download_openwakeword_models") as features, \
                patch("wakeword_trainer.trainer.genai.Client") as llm:
            self.assertEqual(main(), 0)
            piper.assert_called_once_with(Path("output/piper_voices"))
            features.assert_called_once_with()
            llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
