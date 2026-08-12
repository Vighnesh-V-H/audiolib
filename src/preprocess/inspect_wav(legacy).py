import struct


def inspect_wav(filename):
    with open(filename, "rb") as f:
        riff = f.read(4)
        if riff != b"RIFF":
            raise ValueError("Not a RIFF WAV file")

        file_size = struct.unpack("<I", f.read(4))[0]

        wave = f.read(4)
        if wave != b"WAVE":
            raise ValueError("Not a WAV file")

        channels = None
        sample_rate = None
        bits_per_sample = None
        data_size = None

        while True:
            chunk_id = f.read(4)

            if not chunk_id:
                break

            chunk_size = struct.unpack("<I", f.read(4))[0]

            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)

                audio_format, channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack(
                    "<HHIIHH", fmt_data[:16])

                if audio_format != 1:
                    raise ValueError(
                        f"Not PCM audio. Format code: {audio_format}"
                    )

            elif chunk_id == b"data":
                data_size = chunk_size
                break

            else:
                f.seek(chunk_size, 1)

        if channels is None:
            raise ValueError("fmt chunk not found")

        if data_size is None:
            raise ValueError("data chunk not found")

        bytes_per_sample = bits_per_sample // 8

        bytes_per_frame = channels * bytes_per_sample

        num_samples = data_size // bytes_per_frame

        duration = num_samples / sample_rate

        return {
            "sample_rate": sample_rate,
            "channels": channels,
            "bit_depth": bits_per_sample,
            "number_of_samples": num_samples,
            "duration_seconds": duration,
        }


info = inspect_wav("./voice-sample.wav")

for key, value in info.items():
    print(f"{key}: {value}")
