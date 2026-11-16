"""CLI entry point for wake word trainer."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="wakeword-trainer",
        description="Train custom wake word models for openWakeWord (aligned with official recommendations).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run (smaller dataset)
  wakeword-trainer glados --n-samples 1000 --steps 5000

  # Production training (recommended settings)
  wakeword-trainer jarvis --n-samples 50000 --steps 50000

  # Full production with maximum samples
  wakeword-trainer alexa --n-samples 100000 --steps 100000 --batch-size 4096

Environment variables:
  OPENROUTER_API_KEY                 Gemini through OpenRouter (also read from ~/.env).
  OPENROUTER_MODEL                   Optional. OpenRouter model slug.
  GEMINI_API_KEY or GOOGLE_API_KEY    Alternative: direct Gemini API access.
  GEMINI_MODEL                       Optional. Gemini model for phrase generation.

Official recommendations:
  - n_samples: 10,000 minimum, 50,000-100,000 for best results
  - steps: 50,000 default
  - batch_size: 1024-4096 (larger is better with more GPU memory)
  - Automatically downloads ~2,000 hours of pre-computed negative features
  - Uses room impulse responses (RIRs) for reverberation augmentation
  - Uses phoneme-based adversarial generation
        """
    )

    parser.add_argument(
        "wake_word",
        type=str,
        nargs="?",
        help="Wake word to train (e.g., 'glados', 'jarvis', 'alexa')"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=".",
        help="Output directory for model and data (default: current directory)"
    )

    # Data generation
    parser.add_argument(
        "--pronunciation",
        help="Spoken form for synthesis and adversarial generation (e.g. 'may stack')"
    )
    parser.add_argument(
        "--phrases-file",
        help="JSON file with positive and adversarial phrase arrays; avoids Gemini when both are supplied"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of positive samples to generate (default: 10000, recommended: 50000-100000)"
    )
    parser.add_argument(
        "--augmentation-rounds",
        type=int,
        default=1,
        help="Number of augmentation rounds per clip (default: 1)"
    )

    # Training parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=50000,
        help="Number of training steps (default: 50000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Training batch size (default: 2048, recommended: 1024-4096)"
    )
    parser.add_argument(
        "--layer-dim",
        type=int,
        default=32,
        help="Model hidden layer dimension (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Target learning rate (default: 0.001)"
    )

    # Advanced options
    parser.add_argument(
        "--clip-length",
        type=float,
        default=2.0,
        help="Clip length in seconds (default: 2.0, must match pre-computed negative features)"
    )
    parser.add_argument(
        "--no-precomputed-negatives",
        action="store_true",
        help="Don't download pre-computed negative features from HuggingFace"
    )
    parser.add_argument(
        "--no-rirs",
        action="store_true",
        help="Don't download room impulse responses for augmentation"
    )

    # Skip options
    parser.add_argument(
        "--update-models",
        action="store_true",
        help="Download the current pinned Piper and openWakeWord models without training or an API key"
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Skip TTS generation (use existing clips)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading data (voices, RIRs, negative features)"
    )

    args = parser.parse_args()

    if args.update_models:
        if args.wake_word:
            parser.error("--update-models runs without a wake_word; run training separately")
        from pathlib import Path
        from .model_assets import download_openwakeword_models, download_piper_voices
        download_piper_voices(Path(args.output_dir) / "piper_voices")
        download_openwakeword_models()
        print("Model assets are up to date.")
        return 0
    if not args.wake_word:
        parser.error("wake_word is required unless --update-models is used")

    from .trainer import WakeWordTrainer

    phrase_options = {}
    if args.phrases_file:
        import json
        from pathlib import Path
        phrases = json.loads(Path(args.phrases_file).read_text())
        phrase_options = {
            "positive_phrases": phrases["positive"],
            "adversarial_phrases": phrases["adversarial"],
        }

    trainer = WakeWordTrainer(
        wake_word=args.wake_word,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        augmentation_rounds=args.augmentation_rounds,
        steps=args.steps,
        batch_size=args.batch_size,
        layer_dim=args.layer_dim,
        target_lr=args.lr,
        clip_length=args.clip_length,
        use_precomputed_negatives=not args.no_precomputed_negatives,
        download_rirs=not args.no_rirs,
        pronunciation=args.pronunciation,
        **phrase_options,
    )

    try:
        output_path = trainer.train(
            skip_tts=args.skip_tts,
            skip_download=args.skip_download,
        )
        return 0
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
