# wakeword-trainer

A CLI tool to train custom wake word models for [openWakeWord](https://github.com/dscripka/openWakeWord), aligned with official training recommendations.

## Features

- **LLM-powered phrase generation** - Uses Gemini to generate natural wake word phrases
- **Phoneme-based adversarial generation** - Creates similar-sounding negative examples using CMU pronouncing dictionary
- **Multi-voice TTS synthesis** - 10 different Piper TTS voices (US/UK English, male/female)
- **Pre-computed negative features** - Downloads ~2,000 hours from ACAV100M dataset
- **Room impulse response augmentation** - Realistic reverberation from MIT RIR dataset
- **Step-based training with cosine LR decay** - Following official openWakeWord recommendations
- **GPU-optimized** - Large batch sizes for 48GB+ GPUs
- **ONNX export** - Ready for deployment with openWakeWord

## Installation

```bash
# Install from source with the tested dependency lockfile (recommended)
git clone https://github.com/johannhartmann/wakeword_trainer.git
cd wakeword_trainer
uv sync --python 3.12 --locked
source .venv/bin/activate
```

Use `uv sync --locked` to install updates after pulling repository changes.
For dependency maintenance, run `uv lock --upgrade`, `uv sync --locked`, and
`uv run --locked python -m unittest discover -s tests -v`.

The uv configuration selects ONNX Runtime GPU instead of installing both the
CPU and GPU packages, which share the `onnxruntime` module. It also skips the
unused TFLite runtime. Use `uv sync` so these dependency overrides are applied;
`pip` and `uv pip install` do not apply them.

SciPy stays below 1.17 because `acoustics`, used by openWakeWord, imports the
removed `scipy.special.sph_harm` function. Librosa stays on 0.11 because of
Audiomentations' dependency constraint.

## Model versions

Upstream model selections were checked on **2026-11-16**:

| Component | Selected model |
|-----------|----------------|
| Phrase generation and adversarial fallback | [`gemini-4-hydrogen`](https://aistudio.google.com/docs/models/gemini-4-hydrogen), with JSON response schemas |
| Piper voices | [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices/tree/e9ab671f2a0e50a701b34361136ba3e6a3a9eb0c), snapshot `e9ab671f2a0e50a701b34361136ba3e6a3a9eb0c` |
| Mel spectrogram and speech embeddings | [openWakeWord v0.5.1 weights](https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1), still the current published feature models |
| Out-of-dictionary phonemes | [DeepPhonemizer `en_us_cmudict_forward`](https://github.com/axelspringer/DeepPhonemizer#pretrained-models), still the published ARPABET checkpoint used by openWakeWord; Gemini handles failures |

All ten Piper voices use matching weights and configs from the pinned snapshot.
Lessac, LJSpeech, and Cori now use their high-quality variants; LibriTTS already
used high quality. The other six voices use their available medium variants.

Download or refresh model assets without starting training or requiring a Gemini key:

```bash
uv run --locked wakeword-trainer --update-models --output-dir .
```

This replaces stale voice files, repairs missing configs, and verifies the
openWakeWord feature model checksums. Normal training also synchronizes the
voice snapshot unless `--skip-download` is supplied. Only the ONNX feature
models used by this pipeline are downloaded; unused TFLite and legacy Silero
VAD downloads have been removed. VAD filtering remains disabled.

The feature weights must match the precomputed ACAV100M embeddings. A newer
Python package version does not imply newer feature weights.

Previously generated clips and feature caches are retained. Use a fresh output
directory to train entirely with the refreshed voices. The bundled
`models/kitt.onnx` and `models/glados.onnx` are trained outputs, with no upstream
replacement to download; updating them requires retraining.

## Requirements

- Python 3.11+ (Python 3.12 recommended)
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- ~15GB disk space for training data and pre-computed features
- Gemini API key for phrase generation

Python 3.10 is no longer supported because Pillow 13's security fixes require
Python 3.11 or newer. Pillow 13 is an explicit minimum so dependency resolution
cannot fall back to vulnerable Pillow 12 versions.

## Quick Start

```bash
# Set your API key
export GEMINI_API_KEY=your_api_key_here

# Train a wake word model (quick test)
wakeword-trainer jarvis --n-samples 1000 --steps 5000

# The trained model will be saved as jarvis.onnx
```

## Recommended Configurations

### Quick Test Run
For testing the pipeline (~5 minutes):
```bash
wakeword-trainer myword --n-samples 1000 --steps 5000 --batch-size 1024
```

### Standard Training
Balanced quality and speed (~30 minutes):
```bash
wakeword-trainer myword --n-samples 10000 --steps 25000 --batch-size 2048
```

### Production Quality (Recommended)
Best results, following official recommendations (~2 hours):
```bash
wakeword-trainer myword --n-samples 50000 --steps 50000 --batch-size 2048
```

### Maximum Quality
For challenging wake words or maximum accuracy:
```bash
wakeword-trainer myword --n-samples 100000 --steps 100000 --batch-size 4096
```

## Usage

```bash
wakeword-trainer <wake_word> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `wake_word` | The wake word to train (e.g., 'jarvis', 'computer', 'hey assistant') |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir`, `-o` | `.` | Output directory for model and data |
| `--n-samples` | `10000` | Number of positive samples to generate (recommended: 50,000-100,000) |
| `--augmentation-rounds` | `1` | Number of augmentation rounds per clip |
| `--steps` | `50000` | Number of training steps |
| `--batch-size` | `2048` | Training batch size (recommended: 1024-4096) |
| `--layer-dim` | `32` | Model hidden layer dimension |
| `--lr` | `0.001` | Target learning rate |
| `--clip-length` | `2.0` | Clip length in seconds (must match pre-computed features) |
| `--no-precomputed-negatives` | - | Don't download pre-computed negative features |
| `--no-rirs` | - | Don't download room impulse responses |
| `--skip-tts` | - | Skip TTS generation (use existing clips) |
| `--skip-download` | - | Skip downloading data (voices, RIRs, negative features) |
| `--update-models` | - | Refresh model assets only; omit `wake_word` |

### Examples

```bash
# Train "hey jarvis" wake word with production settings
wakeword-trainer "hey jarvis" --n-samples 50000 --steps 50000

# Train with custom output directory
wakeword-trainer glados --output-dir ./models --n-samples 25000

# Retrain with different parameters (reuse existing TTS clips)
wakeword-trainer jarvis --skip-tts --skip-download --steps 100000

# Quick iteration during development
wakeword-trainer test --n-samples 500 --steps 2000 --batch-size 512
```

## How It Works

### 1. Phrase Generation
- **LLM phrases**: Gemini generates natural variations ("Kit", "Hey Kit", "Ok Kit")
- **Adversarial phrases**: Phoneme-based generation finds similar-sounding words ("kit" → "skit", "quit", "kits")

### 2. TTS Synthesis (Piper)
Generates audio clips using 10 Piper TTS voices:
- Multiple US and UK English accents
- Male and female voices
- Speed and noise variation for diversity

### 3. Data Preparation
- **Pre-computed negatives**: ~2,000 hours from ACAV100M (HuggingFace)
- **Background audio**: FMA music and FSD50K environmental sounds
- **RIR augmentation**: MIT room impulse responses for reverberation

### 4. Feature Extraction
Converts audio to openWakeWord embeddings:
- Mel spectrogram → Audio embedding (96-dim)
- 2-second clips → 16 time steps
- Mixed with background at 0-30 dB SNR

### 5. Model Training
Neural network with cosine LR decay:
- Architecture: Flatten → Dense(32) → LayerNorm → ReLU → Dense(32) → LayerNorm → ReLU → Dense(1) → Sigmoid
- Optimizer: AdamW with 1000-step warmup
- Balanced sampling: ~85% negative, ~10% positive, ~5% adversarial

### 6. Export & Test
- Exports to ONNX format
- Tests with openWakeWord inference

## Using the Trained Model

```python
import openwakeword

# Load your custom model
model = openwakeword.Model(
    wakeword_models=['jarvis.onnx'],
    inference_framework="onnx",
)

# Process audio (16kHz, 16-bit PCM)
prediction = model.predict(audio_chunk)
if prediction['jarvis'] > 0.5:
    print("Wake word detected!")
```

## Output Files

```
output_dir/
├── jarvis.onnx                                    # Trained model (use this!)
├── jarvis_clips/                                  # Positive TTS clips
├── jarvis_clips_augmented/                        # Augmented positive clips
├── jarvis_adversarial/                            # Adversarial TTS clips
├── jarvis_features.npy                            # Cached positive features
├── jarvis_adversarial_features.npy                # Cached adversarial features
├── openwakeword_features_ACAV100M_2000_hrs_16bit.npy  # Pre-computed negatives
├── piper_voices/                                  # Downloaded TTS models
├── mit_rirs/                                      # Room impulse responses
├── fma_sample/                                    # Music samples
└── fsd50k_sample/                                 # Noise samples
```

## Performance Guidelines

| Samples | Steps | Expected Quality | Training Time* |
|---------|-------|------------------|----------------|
| 1,000 | 5,000 | Testing only | ~5 min |
| 10,000 | 25,000 | Basic | ~15 min |
| 50,000 | 50,000 | Good (recommended) | ~1 hour |
| 100,000 | 100,000 | Best | ~2 hours |

*With NVIDIA RTX 3090/4090 or A6000

## Typical Results

With recommended settings (50k samples, 50k steps):
- **Detection rate (recall)**: 98-100%
- **Precision**: 99%+
- **Model size**: ~200 KB
- **Inference latency**: <1ms

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Required. Gemini API key for phrase generation |
| `GOOGLE_API_KEY` | Alternative to GEMINI_API_KEY |
| `GEMINI_MODEL` | Optional. Phrase generation model (default: `gemini-4-hydrogen`) |

Create a `.env` file in your working directory:
```
GEMINI_API_KEY=your_api_key_here
```

## Troubleshooting

### Out of GPU memory
Reduce batch size:
```bash
wakeword-trainer myword --batch-size 512
```

### Training takes too long
Reduce samples and steps:
```bash
wakeword-trainer myword --n-samples 10000 --steps 25000
```

### High false positive rate
Increase training data and steps:
```bash
wakeword-trainer myword --n-samples 100000 --steps 100000
```

### Model not detecting wake word
- Ensure wake word phrases are short (under 2 seconds when spoken)
- Increase positive samples
- Check that TTS clips sound correct

## License

MIT License

## Credits

- [openWakeWord](https://github.com/dscripka/openWakeWord) - Wake word detection framework
- [Piper TTS](https://github.com/OHF-Voice/piper1-gpl) - Text-to-speech synthesis
- [Google Gemini](https://ai.google.dev/) - Phrase generation
- [ACAV100M](https://huggingface.co/datasets/davidscripka/openwakeword_features) - Pre-computed negative features
