import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .audio_loader import AudioLoader
from .feature_extractor import FeatureExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio",
        nargs="?",
        type=Path,
        default=Path("asset/voice-sample.wav"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mel_spectrogram.png"),
    )
    args = parser.parse_args()

    if not args.audio.is_file():
        parser.error(f"audio file not found: {args.audio}")

    waveform, sample_rate = AudioLoader().process(args.audio)
    mel_spectrogram = FeatureExtractor().process(waveform)

    print(f"Waveform shape: {tuple(waveform.shape)}")
    print(f"Mel-spectrogram shape: {tuple(mel_spectrogram.shape)}")

    figure, axis = plt.subplots(figsize=(10, 4))
    image = axis.imshow(
        mel_spectrogram.detach().cpu().numpy().T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_title(f"Log-Mel Spectrogram ({sample_rate} Hz)")
    axis.set_xlabel("Time frames")
    axis.set_ylabel("Mel bins")
    figure.colorbar(image, ax=axis, label="Log power")
    figure.tight_layout()
    figure.savefig(args.output, dpi=150)
    plt.show()
    print(f"Plot saved to: {args.output}")


if __name__ == "__main__":
    main()
