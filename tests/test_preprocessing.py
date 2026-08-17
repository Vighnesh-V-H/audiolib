from pathlib import Path

from src.preprocess.audio_loader import AudioLoader
from src.preprocess.feature_extractor import FeatureExtractor


def test_preprocessing_pipeline():
    waveform, sample_rate = AudioLoader().process(Path("asset/voice-sample.wav"))
    log_mel = FeatureExtractor().process(waveform)

    assert sample_rate == 16_000
    assert waveform.shape[0] == 1
    assert log_mel.shape[-1] == 80
