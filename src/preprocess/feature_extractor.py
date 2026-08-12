import torch
import torchaudio
from .config import F_MAX, F_MIN, HOP_LENGTH, N_FFT, N_MELS, TARGET_SAMPLE_RATE, WIN_LENGTH


class FeatureExtractor:

    def __init__(
        self,
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        self.window = torch.hann_window(self.win_length)

        self.mel_filterbank = torchaudio.functional.melscale_fbanks(
            n_freqs=(self.n_fft // 2) + 1,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            norm="slaney",
            mel_scale="htk"
        )

    def frame(self, waveform):
        waveform = waveform.squeeze(0)

        frames = waveform.unfold(
            dimension=0,
            size=self.win_length,
            step=self.hop_length
        )

        return frames

    def apply_window(self, frames):
        return frames * self.window

    def fft(self, frames):
        return torch.fft.rfft(
            frames,
            n=self.n_fft,
            dim=-1
        )

    def power_spectrum(self, spectrum):
        return spectrum.abs() ** 2

    def mel(self, power_spectrum):
        return torch.matmul(
            power_spectrum,
            self.mel_filterbank
        )

    def log(self, mel_spectrogram):
        return torch.log(
            mel_spectrogram + 1e-10
        )

    def process(self, waveform):
        frames = self.frame(waveform)
        frames = self.apply_window(frames)
        spectrum = self.fft(frames)
        power = self.power_spectrum(spectrum)
        mel = self.mel(power)
        log_mel = self.log(mel)

        return log_mel
