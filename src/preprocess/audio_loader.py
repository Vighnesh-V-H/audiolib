import os
import subprocess
import tempfile

import soundfile as sf
import torch
import torch.nn.functional as F


class AudioLoader:

    def __init__(self, target_sample_rate=16000):
        self.target_sample_rate = target_sample_rate

    def load(self, path):
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as temp:
                temp_path = temp.name

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(path),
                    "-acodec",
                    "pcm_s16le",
                    temp_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            waveform, sample_rate = sf.read(
                temp_path,
                dtype="float32"
            )

            waveform = torch.from_numpy(waveform)

            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.transpose(0, 1)

            return waveform, sample_rate

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def to_mono(self, waveform):
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform

    def resample(self, waveform, sample_rate):
        if sample_rate == self.target_sample_rate:
            return waveform, sample_rate

        new_length = int(
            waveform.shape[-1]
            * self.target_sample_rate
            / sample_rate
        )

        waveform = waveform.unsqueeze(0)

        waveform = F.interpolate(
            waveform,
            size=new_length,
            mode="linear",
            align_corners=False,
        )

        waveform = waveform.squeeze(0)

        return waveform, self.target_sample_rate

    def normalize(self, waveform):
        max_value = waveform.abs().max()

        if max_value > 0:
            waveform = waveform / max_value

        return waveform

    def process(self, path):
        waveform, sample_rate = self.load(path)
        waveform = self.to_mono(waveform)
        waveform, sample_rate = self.resample(
            waveform,
            sample_rate
        )
        waveform = self.normalize(waveform)

        return waveform, sample_rate
