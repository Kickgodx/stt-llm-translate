"""Единый поток ввода: микрофон или системный звук."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from ai_audio_transcription.audio_convert import frames_to_mono_int16
from ai_audio_transcription.audio_devices import (
    AudioSource,
    CaptureConfig,
    resolve_stereo_mix_device,
)
from ai_audio_transcription.logging_config import get_logger

log = get_logger("audio_input")


class AudioInputStream:
    def __init__(
        self,
        *,
        config: CaptureConfig,
        sample_rate: int,
        on_audio: Callable[[np.ndarray], None],
    ) -> None:
        self._config = config
        self._target_rate = sample_rate
        self._on_audio = on_audio
        self._active = threading.Event()
        self._stream_sd: sd.InputStream | None = None
        self._stream_pa = None
        self._pyaudio = None
        self._input_rate = sample_rate
        self._input_channels = 1

    @property
    def is_running(self) -> bool:
        return self._stream_sd is not None or self._stream_pa is not None

    def start(self) -> None:
        if self.is_running:
            raise RuntimeError("Ввод уже запущен")
        if self._config.source == AudioSource.MICROPHONE:
            self._start_microphone()
        else:
            self._start_system()

    def stop(self) -> None:
        self._active.clear()
        if self._stream_sd is not None:
            self._stream_sd.stop()
            self._stream_sd.close()
            self._stream_sd = None
        if self._stream_pa is not None:
            self._stream_pa.stop_stream()
            self._stream_pa.close()
            self._stream_pa = None
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None

    def _emit(self, audio: np.ndarray, *, channels: int | None = None) -> None:
        ch = channels or self._input_channels
        mono = frames_to_mono_int16(audio, ch, self._input_rate, self._target_rate)
        self._on_audio(mono.reshape(-1, 1))

    def _start_microphone(self) -> None:
        device = self._config.device
        if device is None:
            device = sd.default.device[0]
        self._input_rate = self._target_rate
        self._input_channels = 1
        self._active.set()

        def callback(indata, _frames, _time, status) -> None:
            if status:
                log.warning("sounddevice: %s", status)
            if self._active.is_set():
                self._on_audio(indata.copy())

        self._stream_sd = sd.InputStream(
            samplerate=self._target_rate,
            channels=1,
            dtype="int16",
            device=device,
            callback=callback,
        )
        self._stream_sd.start()
        log.info("Микрофон: device=%s, sr=%s", device, self._target_rate)

    def _start_system(self) -> None:
        if sys.platform == "win32":
            try:
                self._start_system_wasapi()
                return
            except Exception as exc:
                log.warning("WASAPI loopback недоступен: %s", exc)
        self._start_system_stereo_mix()

    def _start_system_wasapi(self) -> None:
        import pyaudiowpatch as pyaudio

        self._pyaudio = pyaudio.PyAudio()
        if self._config.device is not None:
            info = self._pyaudio.get_device_info_by_index(self._config.device)
            if not info.get("isLoopbackDevice"):
                raise ValueError(f"Устройство #{self._config.device} не является loopback.")
        else:
            info = self._pyaudio.get_default_wasapi_loopback()

        self._input_rate = int(info["defaultSampleRate"])
        self._input_channels = int(info["maxInputChannels"])
        device_index = int(info["index"])
        chunk = 1024
        self._active.set()

        def callback(in_data, _frame_count, _time_info, status) -> tuple[None, int]:
            if status:
                log.warning("pyaudio: %s", status)
            if self._active.is_set():
                raw = np.frombuffer(in_data, dtype=np.int16)
                self._emit(raw, channels=self._input_channels)
            return None, pyaudio.paContinue

        self._stream_pa = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=self._input_channels,
            rate=self._input_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk,
            stream_callback=callback,
        )
        self._stream_pa.start_stream()
        log.info(
            "Системный звук (WASAPI): %s, sr=%s, ch=%s",
            info["name"],
            self._input_rate,
            self._input_channels,
        )

    def _start_system_stereo_mix(self) -> None:
        device = resolve_stereo_mix_device(self._config.device)
        info = sd.query_devices(device)
        self._input_rate = int(info["default_samplerate"])
        self._input_channels = min(2, int(info["max_input_channels"]))
        self._active.set()

        def callback(indata, _frames, _time, status) -> None:
            if status:
                log.warning("sounddevice: %s", status)
            if self._active.is_set():
                self._emit(indata.copy(), channels=self._input_channels)

        self._stream_sd = sd.InputStream(
            samplerate=self._input_rate,
            channels=self._input_channels,
            dtype="int16",
            device=device,
            callback=callback,
        )
        self._stream_sd.start()
        log.info(
            "Системный звук (Стерео микшер): %s, sr=%s, ch=%s",
            info["name"],
            self._input_rate,
            self._input_channels,
        )
