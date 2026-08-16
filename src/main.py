
import argparse
from pathlib import Path

from .preprocess.audio_loader import AudioLoader
from .preprocess.config import TARGET_SAMPLE_RATE
from .preprocess.feature_extractor import FeatureExtractor


def process_audio(path: Path):
    """Load an audio file and return its waveform and log-mel features."""
    loader = AudioLoader(target_sample_rate=TARGET_SAMPLE_RATE)
    extractor = FeatureExtractor(sample_rate=TARGET_SAMPLE_RATE)

    waveform, sample_rate = loader.process(path)
    return waveform, sample_rate, extractor.process(waveform)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess an audio file.")
    parser.add_argument(
        "audio",
        nargs="?",
        type=Path,
        default=Path("asset/voice-sample.wav"),
        help="audio file to process (default: %(default)s)",
    )
    args = parser.parse_args()

    if not args.audio.is_file():
        parser.error(f"audio file not found: {args.audio}")

    waveform, sample_rate, log_mel = process_audio(args.audio)
    print(f"Processed: {args.audio}")
    print(f"  waveform: {tuple(waveform.shape)}")
    print(f"  sample rate: {sample_rate} Hz")
    print(f"  log-mel: {tuple(log_mel.shape)}")


if __name__ == "__main__":
    main()
