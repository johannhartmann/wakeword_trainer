"""Streaming inference with consecutive-frame confirmation."""

from pathlib import Path

import numpy as np


class ScoreGate:
    """Emit one event per consecutive run of sufficiently confident frames."""

    def __init__(self, threshold: float = 0.5, confirmation_frames: int = 4):
        if not 0 <= threshold <= 1 or confirmation_frames < 1:
            raise ValueError("Threshold must be in [0, 1] and confirmation_frames must be positive")
        self.threshold = threshold
        self.confirmation_frames = confirmation_frames
        self.reset()

    def reset(self) -> None:
        self._consecutive = 0

    def update(self, score: float) -> bool:
        if score < self.threshold:
            self.reset()
            return False
        self._consecutive += 1
        return self._consecutive == self.confirmation_frames


class WakeWordDetector:
    """Process 80 ms frames of mono 16 kHz int16 PCM audio."""

    def __init__(self, model_path: str, threshold: float = 0.5, confirmation_frames: int = 4):
        import openwakeword

        self.name = Path(model_path).stem
        self.model = openwakeword.Model(wakeword_models=[model_path], inference_framework="onnx")
        self.gate = ScoreGate(threshold, confirmation_frames)
        self.last_score = 0.0

    def reset(self) -> None:
        self.model.reset()
        self.gate.reset()
        self.last_score = 0.0

    def process(self, pcm: np.ndarray) -> bool:
        if pcm.shape != (1280,) or pcm.dtype != np.int16:
            raise ValueError("Expected exactly 1280 mono int16 samples at 16 kHz")
        self.last_score = float(np.asarray(self.model.predict(pcm)[self.name]).item())
        return self.gate.update(self.last_score)
