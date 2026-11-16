"""Versioned upstream models used by the training pipeline."""

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen


DEFAULT_GEMINI_MODEL = "gemini-4-hydrogen"
PIPER_REPOSITORY = "rhasspy/piper-voices"
# Current upstream snapshot, checked 2026-11-16. Pin weights and configs together.
PIPER_REVISION = "e9ab671f2a0e50a701b34361136ba3e6a3a9eb0c"
PIPER_VOICE_NAMES = (
    "en_US-lessac-high", "en_US-libritts-high", "en_US-amy-medium",
    "en_US-joe-medium", "en_US-kusal-medium", "en_US-l2arctic-medium",
    "en_US-ljspeech-high", "en_GB-alba-medium", "en_GB-aru-medium",
    "en_GB-cori-high",
)


def _voice_path(name: str) -> str:
    locale, speaker, quality = name.split("-")
    return f"{locale[:2]}/{locale}/{speaker}/{quality}/{name}.onnx"


PIPER_VOICES = [(name, _voice_path(name)) for name in PIPER_VOICE_NAMES]

# These remain the latest published openWakeWord feature weights, even though
# the Python package release is 0.6.0. They also define the ACAV100M embeddings.
OPENWAKEWORD_RELEASE = "v0.5.1"
OPENWAKEWORD_MODELS = {
    "melspectrogram.onnx": "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    "embedding_model.onnx": "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_atomic(source: Path, target: Path) -> None:
    """Replace stale assets without exposing partially downloaded files."""
    if target.exists() and _sha256(target) == _sha256(source):
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            with source.open("rb") as content:
                shutil.copyfileobj(content, output)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_piper_voices(voice_dir: Path) -> None:
    """Synchronize each voice and config from the same checked revision."""
    from huggingface_hub import snapshot_download

    voice_dir.mkdir(parents=True, exist_ok=True)
    filenames = [
        filename
        for _, model in PIPER_VOICES
        for filename in (model, f"{model}.json", str(Path(model).parent / "MODEL_CARD"))
    ]
    snapshot = Path(snapshot_download(
        repo_id=PIPER_REPOSITORY, revision=PIPER_REVISION,
        allow_patterns=filenames,
    ))
    for name, model in PIPER_VOICES:
        for filename, target in (
            (model, f"{name}.onnx"),
            (f"{model}.json", f"{name}.onnx.json"),
            (str(Path(model).parent / "MODEL_CARD"), f"{name}.MODEL_CARD"),
        ):
            _copy_atomic(snapshot / filename, voice_dir / target)


def download_openwakeword_models(model_dir: Path | None = None) -> None:
    """Install and verify the current compatible ONNX feature models."""
    if model_dir is None:
        import openwakeword
        model_dir = Path(openwakeword.__file__).parent / "resources" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, checksum in OPENWAKEWORD_MODELS.items():
        target = model_dir / name
        if target.exists() and _sha256(target) == checksum:
            continue
        url = f"https://github.com/dscripka/openWakeWord/releases/download/{OPENWAKEWORD_RELEASE}/{name}"
        temporary = None
        try:
            with urlopen(url, timeout=60) as source, tempfile.NamedTemporaryFile(
                dir=model_dir, delete=False,
            ) as output:
                temporary = Path(output.name)
                shutil.copyfileobj(source, output)
            if _sha256(temporary) != checksum:
                raise ValueError(f"Checksum mismatch for {name}")
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
