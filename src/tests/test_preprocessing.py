from ..preprocess.audio_loader import AudioLoader
from ..preprocess.feature_extractor import FeatureExtractor


audio_loader = AudioLoader(
    target_sample_rate=16000
)

feature_extractor = FeatureExtractor(
    sample_rate=16000,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=80
)

waveform, sample_rate = audio_loader.process(
    "asset/voice-sample.wav"
)

log_mel = feature_extractor.process(
    waveform
)

print("Waveform:", waveform.shape)
print("Sample rate:", sample_rate)
print("Log-Mel:", log_mel.shape)
