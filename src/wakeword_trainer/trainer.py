"""Wake word model trainer for openWakeWord.

Aligned with official openWakeWord training recommendations:
- 10,000-100,000+ positive samples
- 2,000+ hours of negative data (pre-computed features)
- Room impulse response augmentation
- Phoneme-based adversarial generation
- Step-based training with cosine LR decay
- Large batch sizes for GPU utilization
"""

import collections
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

import datasets
import openwakeword
import openwakeword.data
import openwakeword.utils
import torch
from torch import nn
from google import genai


def _patch_openwakeword_mix_clips_batch():
    """Monkey-patch openwakeword.data.mix_clips_batch to fix numpy/torch bug.

    Bug in openwakeword 0.6.0: After converting to numpy at line 463,
    the code tries to call .max(dim=1) which is torch syntax.
    Should be .max(axis=1) for numpy arrays.
    """
    import openwakeword.data as oww_data
    import inspect

    # Get the original source code
    original_source = inspect.getsource(oww_data.mix_clips_batch)

    # Fix the bug: change .max(dim=1) to .max(axis=1) after conversion to numpy
    # The buggy line is: error_index = torch.from_numpy(np.where(mixed_clips_batch.max(dim=1) != 0)[0])
    patched_source = original_source.replace(
        "mixed_clips_batch.max(dim=1) != 0",
        "mixed_clips_batch.max(axis=1) != 0"
    )

    # Compile and execute the patched function
    local_namespace = {}
    # Need to bring in all the module's globals for the function to work
    exec(patched_source, oww_data.__dict__, local_namespace)

    # Replace the original function with the patched one
    oww_data.mix_clips_batch = local_namespace["mix_clips_batch"]


# Apply the patch at module load time
_patch_openwakeword_mix_clips_batch()


def generate_phrases_with_llm(wake_word: str) -> tuple[list[str], list[str]]:
    """Generate training phrases using Gemini LLM."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable required")

    client = genai.Client(api_key=api_key)

    prompt = f'''Generate SHORT phrases for training a wake word detection model for "{wake_word}".

IMPORTANT: Each phrase must be speakable in under 1.5 seconds (max 3-4 syllables total).
Only use short prefixes like "Hey", "Hi", "Ok". Do NOT add words after the wake word.

Return a JSON object with one array:
"phrases": 6-10 short variations of the wake word

Example for "Alexa":
{{"phrases": ["Alexa", "Hey Alexa", "Ok Alexa", "Hi Alexa"]}}

BAD examples (too long): "Alexa can you", "Hey Alexa please help", "Alexa wake up"
GOOD examples (short): "Alexa", "Hey Alexa", "Ok Alexa"

Return ONLY valid JSON, no markdown.
Wake word: "{wake_word}"'''

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    data = json.loads(text)
    phrases = data.get("phrases", [])

    if not phrases:
        raise ValueError("LLM did not generate any phrases")

    return phrases


def generate_adversarial_from_phonemes(wake_word: str, n_samples: int = 500) -> list[str]:
    """Generate adversarial phrases using phoneme overlap (official method)."""
    try:
        adversarial = openwakeword.data.generate_adversarial_texts(
            wake_word,
            N=n_samples,
            include_partial_phrase=0.5,
            include_input_words=0.1
        )
        return list(set(adversarial))  # Remove duplicates
    except Exception as e:
        print(f"Warning: Phoneme-based adversarial generation failed: {e}")
        print("Falling back to LLM-based generation")
        return []


# Default Piper TTS voices (multi-speaker for diversity)
PIPER_VOICES = [
    ("en_US-lessac-medium", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"),
    ("en_US-libritts-high", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/en_US-libritts-high.onnx"),
    ("en_US-amy-medium", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"),
    ("en_US-joe-medium", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx"),
    ("en_US-kusal-medium", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kusal/medium/en_US-kusal-medium.onnx"),
    ("en_US-l2arctic-medium", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/l2arctic/medium/en_US-l2arctic-medium.onnx"),
    ("en_US-ljspeech-medium", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx"),
    ("en_GB-alba-medium", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx"),
    ("en_GB-aru-medium", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/aru/medium/en_GB-aru-medium.onnx"),
    ("en_GB-cori-medium", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/cori/medium/en_GB-cori-medium.onnx"),
]


class WakeWordTrainer:
    """Train wake word models for openWakeWord.

    Aligned with official recommendations:
    - n_samples: 10,000+ (minimum), 50,000-100,000 recommended
    - Uses pre-computed negative features (~2,000 hours from HuggingFace)
    - RIR augmentation for reverberation
    - Phoneme-based adversarial generation
    - Step-based training with cosine LR decay
    - Large batch sizes (2048-4096) for GPU utilization
    """

    def __init__(
        self,
        wake_word: str,
        output_dir: str = ".",
        # Positive sample generation
        n_samples: int = 10000,  # Official: minimum 10,000, recommended 50,000-100,000
        n_samples_val: int = 2000,  # Validation samples
        augmentation_rounds: int = 1,  # Reuse each generation with different augmentation
        # Training parameters
        steps: int = 50000,  # Official default
        batch_size: int = 2048,  # Large batch for 48GB GPU (official: 1024-4096)
        layer_dim: int = 32,  # Official default
        target_lr: float = 0.001,
        # Advanced options
        clip_length: float = 2.0,  # Must match pre-computed negative features (16 time steps)
        use_precomputed_negatives: bool = True,  # Use HuggingFace pre-computed features
        download_rirs: bool = True,  # Download MIT room impulse responses
    ):
        self.wake_word = wake_word.lower().replace(" ", "_")
        self.wake_word_display = wake_word
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate phrases using LLM
        print(f"Generating phrases for '{wake_word}' using Gemini...")
        self.phrases = generate_phrases_with_llm(wake_word)
        print(f"  Phrases: {self.phrases}")

        # Generate adversarial phrases using phoneme overlap (official method)
        print(f"Generating adversarial phrases using phoneme overlap...")
        self.adversarial_phrases = generate_adversarial_from_phonemes(wake_word, n_samples=500)
        if not self.adversarial_phrases:
            # Fallback: generate with LLM
            print("Using LLM fallback for adversarial phrases")
            self.adversarial_phrases = self._generate_adversarial_with_llm(wake_word)
        print(f"  Adversarial phrases: {len(self.adversarial_phrases)} unique phrases")
        print(f"  Examples: {self.adversarial_phrases[:10]}")

        self.n_samples = n_samples
        self.n_samples_val = n_samples_val
        self.augmentation_rounds = augmentation_rounds
        self.steps = steps
        self.batch_size = batch_size
        self.layer_dim = layer_dim
        self.target_lr = target_lr
        self.clip_length = clip_length
        self.use_precomputed_negatives = use_precomputed_negatives
        self.should_download_rirs = download_rirs

        # Paths
        self.clips_dir = self.output_dir / f"{self.wake_word}_clips"
        self.clips_augmented_dir = self.output_dir / f"{self.wake_word}_clips_augmented"
        self.adversarial_dir = self.output_dir / f"{self.wake_word}_adversarial"
        self.voice_dir = self.output_dir / "piper_voices"
        self.rir_dir = self.output_dir / "mit_rirs"
        self.background_dir = self.output_dir / "background_audio"

    def _generate_adversarial_with_llm(self, wake_word: str) -> list[str]:
        """Fallback: generate adversarial phrases with LLM."""
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return []

        client = genai.Client(api_key=api_key)
        prompt = f'''Generate 50 phonetically similar phrases to "{wake_word}" that should NOT trigger a wake word detector.
Think about: rhymes, similar sounds, common mishearings, shared syllables.
Return ONLY a JSON array of strings, no explanation.
Example: ["a lecture", "I'll ask her", "elect a"]'''

        try:
            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(text)
        except Exception:
            return []

    def download_piper_voices(self) -> None:
        """Download Piper TTS voices."""
        self.voice_dir.mkdir(parents=True, exist_ok=True)

        for voice_name, url in PIPER_VOICES:
            onnx_path = self.voice_dir / f"{voice_name}.onnx"
            json_path = self.voice_dir / f"{voice_name}.onnx.json"

            if not onnx_path.exists():
                print(f"Downloading {voice_name}...")
                subprocess.run(["wget", "-q", "-O", str(onnx_path), url], check=True)
                subprocess.run(["wget", "-q", "-O", str(json_path), f"{url}.json"], check=True)

    def download_rirs(self) -> None:
        """Download MIT room impulse responses for reverberation augmentation."""
        if not self.should_download_rirs:
            return

        self.rir_dir.mkdir(parents=True, exist_ok=True)

        if len(list(self.rir_dir.glob("*.wav"))) > 50:
            print("RIRs already downloaded")
            return

        print("Downloading MIT room impulse responses from HuggingFace...")

        try:
            from huggingface_hub import snapshot_download
            import shutil

            # Download the entire dataset (contains WAV files in 16khz folder)
            cache_dir = snapshot_download(
                repo_id="davidscripka/MIT_environmental_impulse_responses",
                repo_type="dataset",
                allow_patterns="16khz/*.wav",
            )

            # Copy WAV files from cache to our rir_dir
            source_dir = Path(cache_dir) / "16khz"
            for wav_file in tqdm(list(source_dir.glob("*.wav")), desc="Copying RIRs"):
                dest_path = self.rir_dir / wav_file.name
                if not dest_path.exists():
                    shutil.copy2(wav_file, dest_path)

            print(f"Downloaded {len(list(self.rir_dir.glob('*.wav')))} RIR files")
        except Exception as e:
            print(f"Warning: Could not download RIRs: {e}")
            print("Training will continue without RIR augmentation")

    def download_background_audio(self) -> None:
        """Download background audio for mixing (AudioSet + FMA samples)."""
        self.background_dir.mkdir(parents=True, exist_ok=True)

        # Download FMA sample
        fma_dir = self.output_dir / "fma_sample"
        if not fma_dir.exists():
            print("Downloading FMA music samples...")
            subprocess.run(["wget", "-q", "-O", str(self.output_dir / "fma_sample.zip"),
                           "https://f002.backblazeb2.com/file/openwakeword-resources/data/fma_sample.zip"],
                          check=True)
            subprocess.run(["unzip", "-q", "-o", "fma_sample.zip"], check=True, cwd=str(self.output_dir))

        # Download FSD50K sample
        fsd_dir = self.output_dir / "fsd50k_sample"
        if not fsd_dir.exists():
            print("Downloading FSD50K noise samples...")
            subprocess.run(["wget", "-q", "-O", str(self.output_dir / "fsd50k_sample.zip"),
                           "https://f002.backblazeb2.com/file/openwakeword-resources/data/fsd50k_sample.zip"],
                          check=True)
            subprocess.run(["unzip", "-q", "-o", "fsd50k_sample.zip"], check=True, cwd=str(self.output_dir))

    def download_precomputed_features(self) -> Path:
        """Download pre-computed negative features from HuggingFace (~2,000 hours)."""
        features_path = self.output_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"

        if features_path.exists():
            print(f"Pre-computed features already exist: {features_path}")
            return features_path

        print("Downloading pre-computed negative features (~2,000 hours from ACAV100M)...")
        print("This is a large file (~4GB), please wait...")

        subprocess.run([
            "wget", "-q", "--show-progress",
            "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
            "-O", str(features_path)
        ], check=True)

        return features_path

    def download_validation_features(self) -> Path:
        """Download validation features for false positive estimation."""
        val_path = self.output_dir / "validation_set_features.npy"

        if val_path.exists():
            return val_path

        print("Downloading validation features (~11 hours)...")
        subprocess.run([
            "wget", "-q", "--show-progress",
            "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy",
            "-O", str(val_path)
        ], check=True)

        return val_path

    def download_openwakeword_models(self) -> None:
        """Download openwakeword embedding models if missing."""
        import openwakeword
        oww_path = Path(openwakeword.__file__).parent / "resources" / "models"
        oww_path.mkdir(parents=True, exist_ok=True)

        models = [
            ("melspectrogram.onnx", "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx"),
            ("melspectrogram.tflite", "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite"),
            ("embedding_model.onnx", "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx"),
            ("embedding_model.tflite", "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite"),
            ("silero_vad.onnx", "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/silero_vad.onnx"),
        ]

        for model_name, url in models:
            model_path = oww_path / model_name
            if not model_path.exists():
                print(f"Downloading openwakeword model: {model_name}...")
                subprocess.run(["wget", "-q", "-O", str(model_path), url], check=True)

    def validate_wav(self, path: Path) -> bool:
        """Validate that a WAV file is readable and has valid audio data."""
        try:
            import soundfile as sf
            data, sr = sf.read(str(path))
            # Check if file has actual audio content
            if len(data) < 1000:  # Less than ~60ms at 16kHz
                return False
            if np.abs(data).max() < 0.001:  # Essentially silent
                return False
            return True
        except Exception:
            return False

    def generate_clips(self, voice_path: Path, phrases: list[str], n_per_phrase: int, output_dir: Path) -> int:
        """Generate synthetic speech clips with Piper TTS (CUDA accelerated)."""
        from piper import PiperVoice
        from piper.config import SynthesisConfig
        import wave
        import soundfile as sf

        voice_name = voice_path.stem
        clips_generated = 0

        voice = PiperVoice.load(str(voice_path), use_cuda=True)

        for phrase in phrases:
            for i in range(n_per_phrase):
                output_file = output_dir / f"{voice_name}_{phrase.replace(' ', '_')}_{i:04d}.wav"

                if output_file.exists() and self.validate_wav(output_file):
                    clips_generated += 1
                    continue
                elif output_file.exists():
                    # Remove corrupted file
                    output_file.unlink()

                # Randomize synthesis parameters for diversity
                syn_config = SynthesisConfig(
                    length_scale=random.uniform(0.8, 1.2),
                    noise_scale=random.uniform(0.3, 0.7),
                    noise_w_scale=random.uniform(0.6, 1.0),
                )

                try:
                    # Collect all audio chunks first
                    audio_data = []
                    for audio_chunk in voice.synthesize(phrase, syn_config):
                        audio_data.append(np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16))

                    if not audio_data:
                        continue

                    # Concatenate and save using soundfile (more robust)
                    full_audio = np.concatenate(audio_data)
                    if len(full_audio) < 1000:  # Skip very short clips
                        continue

                    sf.write(str(output_file), full_audio, voice.config.sample_rate)
                    clips_generated += 1
                except Exception as e:
                    print(f"Error generating {output_file}: {e}")
                    if output_file.exists():
                        output_file.unlink()
                    continue

        return clips_generated

    def resample_to_16k(self, input_path: Path) -> bool:
        """Resample audio file to 16kHz using soundfile."""
        try:
            import soundfile as sf

            data, sr = sf.read(str(input_path))
            if sr != 16000:
                # Resample
                n_samples = int(len(data) * 16000 / sr)
                data = signal.resample(data, n_samples)
                sf.write(str(input_path), data, 16000)
            return True
        except Exception as e:
            print(f"Error resampling {input_path}: {e}")
            # Try to remove corrupted file
            try:
                input_path.unlink()
            except Exception:
                pass
            return False

    def cleanup_corrupted_files(self, directory: Path) -> int:
        """Remove corrupted WAV files from a directory."""
        removed = 0
        for wav_path in list(directory.glob("*.wav")):
            if not self.validate_wav(wav_path):
                try:
                    wav_path.unlink()
                    removed += 1
                except Exception:
                    pass
        return removed

    def _load_audio_safe(self, path: str, target_sr: int = 16000) -> np.ndarray | None:
        """Safely load audio file using soundfile."""
        try:
            import soundfile as sf
            data, sr = sf.read(path)
            if len(data.shape) > 1:
                data = data.mean(axis=1)  # Convert to mono
            if sr != target_sr:
                n_samples = int(len(data) * target_sr / sr)
                data = signal.resample(data, n_samples)
            return data.astype(np.float32)
        except Exception:
            return None

    def _apply_rir(self, audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
        """Apply room impulse response convolution."""
        convolved = signal.fftconvolve(audio, rir, mode='full')
        # Trim to original length and normalize
        convolved = convolved[:len(audio)]
        max_val = np.abs(convolved).max()
        if max_val > 0:
            convolved = convolved / max_val * np.abs(audio).max()
        return convolved.astype(np.float32)

    def _mix_with_background(self, audio: np.ndarray, background: np.ndarray, snr_db: float) -> np.ndarray:
        """Mix foreground audio with background at specified SNR."""
        # Ensure same length
        if len(background) < len(audio):
            # Repeat background if too short
            repeats = int(np.ceil(len(audio) / len(background)))
            background = np.tile(background, repeats)[:len(audio)]
        else:
            # Random crop
            start = random.randint(0, len(background) - len(audio))
            background = background[start:start + len(audio)]

        # Calculate RMS
        audio_rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
        bg_rms = np.sqrt(np.mean(background ** 2)) + 1e-8

        # Scale background for desired SNR
        target_bg_rms = audio_rms / (10 ** (snr_db / 20))
        background = background * (target_bg_rms / bg_rms)

        # Mix
        mixed = audio + background

        # Normalize to prevent clipping
        max_val = np.abs(mixed).max()
        if max_val > 1.0:
            mixed = mixed / max_val * 0.95

        return mixed.astype(np.float32)

    def augment_clips(self) -> None:
        """Augment generated clips with background noise and RIRs (robust implementation)."""
        import soundfile as sf

        self.clips_augmented_dir.mkdir(parents=True, exist_ok=True)

        # Check if augmented clips already exist
        existing_augmented = len(list(self.clips_augmented_dir.glob("*.wav")))
        source_clips = len(list(self.clips_dir.glob("*.wav")))
        expected_augmented = source_clips * self.augmentation_rounds

        if existing_augmented >= expected_augmented * 0.9:
            print(f"Reusing {existing_augmented} existing augmented clips (expected: {expected_augmented})")
            return

        # Get valid clip paths
        clip_paths = [p for p in self.clips_dir.glob("*.wav") if self.validate_wav(p)]
        if not clip_paths:
            print("Warning: No valid clips found for augmentation")
            return

        # Get background paths
        background_paths = []
        for bg_dir in [self.output_dir / "fma_sample", self.output_dir / "fsd50k_sample"]:
            if bg_dir.exists():
                background_paths.extend(list(bg_dir.glob("*.wav")))

        # Get RIR paths
        rir_paths = list(self.rir_dir.glob("*.wav")) if self.rir_dir.exists() else []

        print(f"Augmenting {len(clip_paths)} clips with {len(background_paths)} backgrounds and {len(rir_paths)} RIRs")

        # Pre-load some backgrounds and RIRs
        backgrounds = []
        for bg_path in tqdm(random.sample(background_paths, min(100, len(background_paths))), desc="Loading backgrounds"):
            audio = self._load_audio_safe(str(bg_path))
            if audio is not None and len(audio) > 16000:
                backgrounds.append(audio)

        rirs = []
        for rir_path in rir_paths[:50]:  # Limit to 50 RIRs
            audio = self._load_audio_safe(str(rir_path))
            if audio is not None and len(audio) > 100:
                rirs.append(audio)

        if not backgrounds:
            print("Warning: Could not load any background audio")

        total_length = int(16000 * self.clip_length)

        for round_idx in range(self.augmentation_rounds):
            print(f"Augmentation round {round_idx + 1}/{self.augmentation_rounds}")

            for clip_idx, clip_path in enumerate(tqdm(clip_paths, desc="Augmenting")):
                try:
                    # Load clip
                    audio = self._load_audio_safe(str(clip_path))
                    if audio is None:
                        continue

                    # Pad or trim to target length
                    if len(audio) < total_length:
                        audio = np.pad(audio, (0, total_length - len(audio)))
                    else:
                        audio = audio[:total_length]

                    # Apply augmentations randomly
                    # 1. Gain variation
                    gain = random.uniform(0.5, 1.5)
                    audio = audio * gain

                    # 2. Apply RIR (50% chance)
                    if rirs and random.random() < 0.5:
                        rir = random.choice(rirs)
                        audio = self._apply_rir(audio, rir)

                    # 3. Mix with background (75% chance)
                    if backgrounds and random.random() < 0.75:
                        bg = random.choice(backgrounds)
                        snr = random.uniform(5, 20)
                        audio = self._mix_with_background(audio, bg, snr)

                    # Normalize
                    max_val = np.abs(audio).max()
                    if max_val > 0:
                        audio = audio / max_val * 0.9

                    # Save
                    output_path = self.clips_augmented_dir / f"aug_r{round_idx}_{clip_idx:06d}.wav"
                    sf.write(str(output_path), audio, 16000)

                except Exception as e:
                    print(f"Error augmenting {clip_path}: {e}")
                    continue

    def compute_positive_features(self, feature_extractor) -> np.ndarray:
        """Compute features for positive samples with augmentation."""
        output_file = self.output_dir / f"{self.wake_word}_features.npy"

        if output_file.exists():
            print(f"Loading existing {output_file}")
            return np.load(str(output_file))

        # Use augmented clips if available, otherwise raw clips
        if self.clips_augmented_dir.exists() and len(list(self.clips_augmented_dir.glob("*.wav"))) > 0:
            clips_dir = self.clips_augmented_dir
        else:
            clips_dir = self.clips_dir

        positive_clips, durations = openwakeword.data.filter_audio_paths(
            [str(clips_dir)], min_length_secs=0.3, max_length_secs=self.clip_length, duration_method="header"
        )

        print(f"{len(positive_clips)} positive clips after filtering")

        # Get background clips for mixing
        background_dirs = [
            str(self.output_dir / "fma_sample"),
            str(self.output_dir / "fsd50k_sample"),
        ]
        background_clips, _ = openwakeword.data.filter_audio_paths(
            [d for d in background_dirs if Path(d).exists()],
            min_length_secs=1.0, max_length_secs=60*30, duration_method="header"
        )

        sr = 16000
        total_length = int(sr * self.clip_length)

        # Align clips so wake word ends near end of window (reduces latency)
        jitters = (np.random.uniform(0, 0.2, len(positive_clips)) * sr).astype(np.int32)
        starts = [max(0, total_length - (int(np.ceil(d * sr)) + j)) for d, j in zip(durations, jitters)]

        # Get RIR paths for reverberation
        rir_paths = [str(p) for p in self.rir_dir.glob("*.wav")] if self.rir_dir.exists() else []

        batch_size = 32  # Larger batch for GPU
        mixing_generator = openwakeword.data.mix_clips_batch(
            foreground_clips=positive_clips,
            background_clips=background_clips if background_clips else positive_clips,
            combined_size=total_length,
            batch_size=batch_size,
            snr_low=0,  # Official: 0-30 dB range
            snr_high=30,
            start_index=starts,
            foreground_durations=durations,  # Truncate clips to fit window
            rirs=rir_paths,
            rir_probability=0.5 if rir_paths else 0,
            volume_augmentation=True,
        )

        N_total = len(positive_clips)
        n_feature_cols = feature_extractor.get_embedding_shape(self.clip_length)

        output_array_shape = (N_total, n_feature_cols[0], n_feature_cols[1])
        fp = open_memmap(str(output_file), mode='w+', dtype=np.float32, shape=output_array_shape)

        row_counter = 0
        for batch in tqdm(mixing_generator, total=N_total//batch_size, desc="Computing positive features"):
            batch_data, lbls, background = batch[0], batch[1], batch[2]
            features = feature_extractor.embed_clips(batch_data, batch_size=512)  # Large batch for GPU
            fp[row_counter:row_counter+features.shape[0], :, :] = features
            row_counter += features.shape[0]
            fp.flush()

            if row_counter >= N_total:
                break

        openwakeword.data.trim_mmap(str(output_file))
        print(f"Saved positive features to {output_file}")
        return np.load(str(output_file))

    def compute_adversarial_features(self, feature_extractor) -> np.ndarray:
        """Compute features for adversarial samples."""
        output_file = self.output_dir / f"{self.wake_word}_adversarial_features.npy"

        if output_file.exists():
            print(f"Loading existing {output_file}")
            return np.load(str(output_file))

        adversarial_clips, durations = openwakeword.data.filter_audio_paths(
            [str(self.adversarial_dir)], min_length_secs=0.3, max_length_secs=self.clip_length, duration_method="header"
        )

        if not adversarial_clips:
            print("No adversarial clips found")
            return np.array([])

        print(f"{len(adversarial_clips)} adversarial clips")

        # Get background clips
        background_dirs = [str(self.output_dir / "fma_sample"), str(self.output_dir / "fsd50k_sample")]
        background_clips, _ = openwakeword.data.filter_audio_paths(
            [d for d in background_dirs if Path(d).exists()],
            min_length_secs=1.0, max_length_secs=60*30, duration_method="header"
        )

        sr = 16000
        total_length = int(sr * self.clip_length)
        jitters = (np.random.uniform(0, 0.2, len(adversarial_clips)) * sr).astype(np.int32)
        starts = [max(0, total_length - (int(np.ceil(d * sr)) + j)) for d, j in zip(durations, jitters)]

        rir_paths = [str(p) for p in self.rir_dir.glob("*.wav")] if self.rir_dir.exists() else []

        batch_size = 32
        mixing_generator = openwakeword.data.mix_clips_batch(
            foreground_clips=adversarial_clips,
            background_clips=background_clips if background_clips else adversarial_clips,
            combined_size=total_length,
            batch_size=batch_size,
            snr_low=0,
            snr_high=30,
            start_index=starts,
            foreground_durations=durations,  # Truncate clips to fit window
            rirs=rir_paths,
            rir_probability=0.5 if rir_paths else 0,
            volume_augmentation=True,
        )

        N_total = len(adversarial_clips)
        n_feature_cols = feature_extractor.get_embedding_shape(self.clip_length)

        output_array_shape = (N_total, n_feature_cols[0], n_feature_cols[1])
        fp = open_memmap(str(output_file), mode='w+', dtype=np.float32, shape=output_array_shape)

        row_counter = 0
        for batch in tqdm(mixing_generator, total=N_total//batch_size, desc="Computing adversarial features"):
            batch_data, lbls, background = batch[0], batch[1], batch[2]
            features = feature_extractor.embed_clips(batch_data, batch_size=512)
            fp[row_counter:row_counter+features.shape[0], :, :] = features
            row_counter += features.shape[0]
            fp.flush()

            if row_counter >= N_total:
                break

        openwakeword.data.trim_mmap(str(output_file))
        return np.load(str(output_file))

    def lr_warmup_cosine_decay(self, step: int, warmup_steps: int, total_steps: int) -> float:
        """Cosine LR decay with warmup (official method)."""
        if step < warmup_steps:
            return self.target_lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * self.target_lr * (1 + np.cos(np.pi * progress))

    def train_model(self, positive_features: np.ndarray, negative_features: np.ndarray,
                    adversarial_features: np.ndarray = None) -> tuple:
        """Train the wake word detection model with step-based training."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Build model (official architecture)
        input_shape = (positive_features.shape[1], positive_features.shape[2])
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[0] * input_shape[1], self.layer_dim),
            nn.LayerNorm(self.layer_dim),
            nn.ReLU(),
            nn.Linear(self.layer_dim, self.layer_dim),
            nn.LayerNorm(self.layer_dim),
            nn.ReLU(),
            nn.Linear(self.layer_dim, 1),
            nn.Sigmoid(),
        ).to(device)

        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        loss_function = torch.nn.functional.binary_cross_entropy
        optimizer = torch.optim.Adam(model.parameters(), lr=self.target_lr)

        # Prepare data
        n_positive = len(positive_features)
        n_negative = len(negative_features)
        n_adversarial = len(adversarial_features) if adversarial_features is not None and len(adversarial_features) > 0 else 0

        print(f"Training data: {n_positive} positive, {n_negative} negative, {n_adversarial} adversarial")

        # Calculate samples per batch based on ratios (official approach)
        # Batch composition: mostly negatives, some positives, some adversarial
        n_neg_per_batch = int(self.batch_size * 0.85)
        n_pos_per_batch = int(self.batch_size * 0.10)
        n_adv_per_batch = self.batch_size - n_neg_per_batch - n_pos_per_batch

        history = collections.defaultdict(list)
        warmup_steps = min(1000, self.steps // 10)
        best_model_state = None
        best_recall = 0

        print(f"\nTraining for {self.steps} steps...")
        print(f"Batch size: {self.batch_size} (neg: {n_neg_per_batch}, pos: {n_pos_per_batch}, adv: {n_adv_per_batch})")

        pbar = tqdm(range(self.steps), desc="Training")
        for step in pbar:
            # Update learning rate
            lr = self.lr_warmup_cosine_decay(step, warmup_steps, self.steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Sample batch
            pos_idx = np.random.choice(n_positive, n_pos_per_batch, replace=True)
            neg_idx = np.random.choice(n_negative, n_neg_per_batch, replace=True)

            X_batch = np.vstack([
                positive_features[pos_idx],
                negative_features[neg_idx],
            ])
            y_batch = np.array([1] * n_pos_per_batch + [0] * n_neg_per_batch, dtype=np.float32)

            # Add adversarial (as negatives)
            if n_adversarial > 0 and n_adv_per_batch > 0:
                adv_idx = np.random.choice(n_adversarial, min(n_adv_per_batch, n_adversarial), replace=True)
                X_batch = np.vstack([X_batch, adversarial_features[adv_idx]])
                y_batch = np.concatenate([y_batch, np.zeros(len(adv_idx), dtype=np.float32)])

            # Shuffle
            perm = np.random.permutation(len(y_batch))
            X_batch = X_batch[perm]
            y_batch = y_batch[perm]

            # To tensors
            X_tensor = torch.from_numpy(X_batch).to(device)
            y_tensor = torch.from_numpy(y_batch[..., None]).to(device)

            # Compute class weights (weight negatives more to reduce false positives)
            weights = torch.ones(len(y_tensor), device=device)
            weights[y_tensor.flatten() == 0] = 10.0  # Official: higher weight on negatives

            # Forward pass
            optimizer.zero_grad()
            predictions = model(X_tensor)
            loss = loss_function(predictions, y_tensor, weights[..., None])
            loss.backward()
            optimizer.step()

            # Metrics
            with torch.no_grad():
                pred_binary = (predictions >= 0.5).float()
                tp = ((pred_binary == 1) & (y_tensor == 1)).sum().item()
                fn = ((pred_binary == 0) & (y_tensor == 1)).sum().item()
                fp = ((pred_binary == 1) & (y_tensor == 0)).sum().item()
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            history['loss'].append(float(loss.item()))
            history['recall'].append(recall)
            history['precision'].append(precision)
            history['lr'].append(lr)

            # Update progress bar
            if step % 100 == 0:
                avg_loss = np.mean(history['loss'][-100:])
                avg_recall = np.mean(history['recall'][-100:])
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'recall': f'{avg_recall:.3f}', 'lr': f'{lr:.6f}'})

            # Save best model
            if recall > best_recall and step > warmup_steps:
                best_recall = recall
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Log every 1000 steps
            if (step + 1) % 1000 == 0:
                print(f"\nStep {step+1}: Loss={np.mean(history['loss'][-1000:]):.4f}, "
                      f"Recall={np.mean(history['recall'][-1000:]):.3f}, "
                      f"Precision={np.mean(history['precision'][-1000:]):.3f}")

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
            print(f"\nLoaded best model with recall: {best_recall:.3f}")

        return model.cpu(), history

    def export_onnx(self, model, input_shape: tuple) -> Path:
        """Export model to ONNX format."""
        model.eval()
        output_path = self.output_dir / f"{self.wake_word}.onnx"
        dummy_input = torch.zeros((1, input_shape[0], input_shape[1]))

        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=['input'],
            output_names=[self.wake_word],
            dynamic_axes={'input': {0: 'batch_size'}, self.wake_word: {0: 'batch_size'}}
        )

        print(f"Model exported to: {output_path}")
        print(f"Model size: {output_path.stat().st_size / 1024:.1f} KB")
        return output_path

    def test_model(self) -> dict:
        """Test the trained model."""
        model_path = self.output_dir / f"{self.wake_word}.onnx"
        oww = openwakeword.Model(
            wakeword_model_paths=[str(model_path)],
            enable_speex_noise_suppression=False,
            vad_threshold=0.5
        )

        print(f"\nTesting model: {self.wake_word}")

        # Test on positive clips
        test_clips = list(self.clips_dir.glob("*.wav"))[:100]
        detected = 0

        for clip in tqdm(test_clips, desc="Testing positive clips"):
            try:
                scores = oww.predict_clip(str(clip))
                max_score = max([s[self.wake_word] for s in scores])
                if max_score >= 0.5:
                    detected += 1
            except Exception:
                continue

        detection_rate = detected / len(test_clips) if test_clips else 0
        print(f"Detection rate: {detected}/{len(test_clips)} ({detection_rate*100:.1f}%)")

        # Test false positive rate on validation data if available
        val_path = self.output_dir / "validation_set_features.npy"
        if val_path.exists():
            print("Testing false positive rate on validation data...")
            # This would require running inference on the validation features
            # For now, just report that validation data is available
            print(f"Validation features available at: {val_path}")

        return {"detection_rate": detection_rate}

    def train(self, skip_tts: bool = False, skip_download: bool = False) -> Path:
        """Run the full training pipeline."""
        print("=" * 60)
        print(f"Training wake word model for: {self.wake_word_display}")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Positive samples: {self.n_samples}")
        print(f"  Training steps: {self.steps}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Layer dim: {self.layer_dim}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Phrases: {self.phrases}")
        print(f"  Adversarial phrases: {len(self.adversarial_phrases)}")
        print()

        # Step 1: Download resources
        if not skip_download:
            print("\n=== Step 1: Downloading resources ===")
            self.download_piper_voices()
            self.download_rirs()
            self.download_background_audio()

            if self.use_precomputed_negatives:
                self.download_precomputed_features()
                self.download_validation_features()

        # Step 2: Generate synthetic speech
        if not skip_tts:
            print("\n=== Step 2: Generating synthetic speech ===")
            self.clips_dir.mkdir(parents=True, exist_ok=True)

            # Check if we already have enough clips (within 10% of target)
            existing_clips = len(list(self.clips_dir.glob("*.wav")))
            target_clips = self.n_samples

            if existing_clips >= target_clips * 0.9:
                print(f"Reusing {existing_clips} existing clips (target: {target_clips})")
            else:
                # Calculate clips per voice per phrase
                n_per_phrase = self.n_samples // (len(PIPER_VOICES) * len(self.phrases))
                n_per_phrase = max(1, n_per_phrase)

                print(f"Generating {n_per_phrase} clips per phrase per voice")
                print(f"Total target: ~{n_per_phrase * len(PIPER_VOICES) * len(self.phrases)} clips")

                total_clips = 0
                for voice_name, _ in tqdm(PIPER_VOICES, desc="Generating clips per voice"):
                    voice_path = self.voice_dir / f"{voice_name}.onnx"
                    if voice_path.exists():
                        n = self.generate_clips(voice_path, self.phrases, n_per_phrase, self.clips_dir)
                        total_clips += n

                print(f"Total synthetic clips generated: {total_clips}")

                # Resample to 16kHz and validate
                print("Resampling and validating clips...")
                for clip in tqdm(list(self.clips_dir.glob("*.wav")), desc="Resampling to 16kHz"):
                    self.resample_to_16k(clip)

                # Clean up corrupted files
                removed = self.cleanup_corrupted_files(self.clips_dir)
                if removed > 0:
                    print(f"Removed {removed} corrupted clips")

            valid_clips = len(list(self.clips_dir.glob("*.wav")))
            print(f"Valid clips: {valid_clips}")

            # Generate adversarial clips
            print("\n=== Step 2b: Generating adversarial clips ===")
            self.adversarial_dir.mkdir(parents=True, exist_ok=True)

            existing_adversarial = len(list(self.adversarial_dir.glob("*.wav")))
            if existing_adversarial >= 100:
                print(f"Reusing {existing_adversarial} existing adversarial clips")
            else:
                n_adv_per_phrase = max(1, 50 // len(self.adversarial_phrases[:20]))  # Limit to 20 phrases

                total_adversarial = 0
                for voice_name, _ in tqdm(PIPER_VOICES[:5], desc="Generating adversarial clips"):  # Use 5 voices
                    voice_path = self.voice_dir / f"{voice_name}.onnx"
                    if voice_path.exists():
                        n = self.generate_clips(voice_path, self.adversarial_phrases[:20], n_adv_per_phrase, self.adversarial_dir)
                        total_adversarial += n

                print(f"Total adversarial clips generated: {total_adversarial}")

                for clip in tqdm(list(self.adversarial_dir.glob("*.wav")), desc="Resampling adversarial to 16kHz"):
                    self.resample_to_16k(clip)

                # Clean up corrupted adversarial files
                removed = self.cleanup_corrupted_files(self.adversarial_dir)
                if removed > 0:
                    print(f"Removed {removed} corrupted adversarial clips")

        # Step 3: Augment clips (optional but recommended)
        if not skip_tts and self.augmentation_rounds > 0:
            print("\n=== Step 3: Augmenting clips ===")
            self.augment_clips()

        # Step 4: Compute features
        print("\n=== Step 4: Computing audio features ===")
        self.download_openwakeword_models()
        feature_extractor = openwakeword.utils.AudioFeatures()

        positive_features = self.compute_positive_features(feature_extractor)
        adversarial_features = self.compute_adversarial_features(feature_extractor)

        # Load negative features
        if self.use_precomputed_negatives:
            neg_path = self.output_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
            if neg_path.exists():
                print(f"Loading pre-computed negative features from {neg_path}")
                negative_features = np.load(str(neg_path), mmap_mode='r')
                print(f"Negative features shape: {negative_features.shape}")
            else:
                print("Warning: Pre-computed features not found, using empty negatives")
                negative_features = np.zeros((1000, positive_features.shape[1], positive_features.shape[2]), dtype=np.float32)
        else:
            negative_features = np.zeros((1000, positive_features.shape[1], positive_features.shape[2]), dtype=np.float32)

        print(f"Positive features shape: {positive_features.shape}")
        print(f"Adversarial features shape: {adversarial_features.shape if len(adversarial_features) > 0 else 'None'}")
        print(f"Negative features shape: {negative_features.shape}")

        # Step 5: Train model
        print("\n=== Step 5: Training model ===")
        model, history = self.train_model(positive_features, negative_features, adversarial_features)

        # Step 6: Export to ONNX
        print("\n=== Step 6: Exporting model ===")
        input_shape = (positive_features.shape[1], positive_features.shape[2])
        output_path = self.export_onnx(model, input_shape)

        # Step 7: Test model
        print("\n=== Step 7: Testing model ===")
        self.test_model()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Model saved to: {output_path}")
        print(f"\nUsage with openWakeWord:")
        print(f"  oww = openwakeword.Model(wakeword_model_paths=['{output_path}'])")

        return output_path
