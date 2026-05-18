import numpy as np


def to_mono_int16(audio: np.ndarray, channels: int) -> np.ndarray:
    flat = audio.reshape(-1)
    if channels <= 1:
        return flat
    frames = flat.reshape(-1, channels)
    return frames.mean(axis=1).astype(np.int16)


def resample_int16(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or len(audio) == 0:
        return audio
    out_len = max(1, int(len(audio) * dst_rate / src_rate))
    x_old = np.arange(len(audio), dtype=np.float64)
    x_new = np.linspace(0, len(audio) - 1, out_len)
    return np.interp(x_new, x_old, audio.astype(np.float64)).astype(np.int16)


def frames_to_mono_int16(
    audio: np.ndarray,
    channels: int,
    src_rate: int,
    dst_rate: int,
) -> np.ndarray:
    mono = to_mono_int16(audio, channels)
    return resample_int16(mono, src_rate, dst_rate)
