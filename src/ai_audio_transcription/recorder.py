import io
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_audio_transcription.audio_devices import AudioSource, CaptureConfig
from ai_audio_transcription.audio_input import AudioInputStream
from ai_audio_transcription.logging_config import get_logger

log = get_logger("recorder")


@dataclass
class RecordingResult:
    """Ровно одно из полей заполнено после stop()."""

    audio: np.ndarray | None = None
    path: Path | None = None
    duration_seconds: float = 0.0
    used_file: bool = False

    def __post_init__(self) -> None:
        if (self.audio is None) == (self.path is None):
            raise ValueError("Нужно audio или path, но не оба сразу.")


def audio_to_wav_bytes(audio: np.ndarray, *, sample_rate: int) -> bytes:
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.astype(np.int16).tobytes())
    return buf.getvalue()


def write_wav(path: Path, audio: np.ndarray, *, sample_rate: int) -> None:
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.astype(np.int16).tobytes())


class AudioCaptureSession:
    """
    Запись с микрофона или системного звука.
    До ram_max_seconds — буфер в RAM, затем потоковая запись во временный WAV.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        capture: CaptureConfig | None = None,
        device: int | None = None,
        ram_max_seconds: float = 300.0,
    ) -> None:
        self.sample_rate = sample_rate
        if capture is not None:
            self._capture = capture
        else:
            self._capture = CaptureConfig(
                source=AudioSource.MICROPHONE,
                device=device,
            )
        self.ram_max_seconds = ram_max_seconds
        self._max_ram_frames = int(ram_max_seconds * sample_rate)
        self._input: AudioInputStream | None = None
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._use_file = False
        self._wav_path: Path | None = None
        self._wave_file: wave.Wave_write | None = None
        self._frames_written = 0

    @property
    def is_recording(self) -> bool:
        return self._input is not None and self._input.is_running

    def set_capture(self, capture: CaptureConfig) -> None:
        if self.is_recording:
            raise RuntimeError("Нельзя сменить источник во время записи")
        self._capture = capture

    def _on_audio(self, indata: np.ndarray) -> None:
        with self._lock:
            self._frames_written += len(indata)
            if self._use_file:
                if self._wave_file is not None:
                    self._wave_file.writeframes(indata.tobytes())
            else:
                self._chunks.append(indata.copy())
                if self._frames_written > self._max_ram_frames:
                    self._switch_to_file_locked()

    def _switch_to_file_locked(self) -> None:
        if self._use_file:
            return
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self._wav_path = Path(tmp.name)
        tmp.close()
        self._wave_file = wave.open(str(self._wav_path), "wb")
        self._wave_file.setnchannels(1)
        self._wave_file.setsampwidth(2)
        self._wave_file.setframerate(self.sample_rate)
        if self._chunks:
            merged = np.concatenate(self._chunks, axis=0).reshape(-1)
            self._wave_file.writeframes(merged.astype(np.int16).tobytes())
            self._chunks.clear()
        self._use_file = True
        log.info(
            "Порог %.0f с достигнут — запись в temp-файл: %s",
            self.ram_max_seconds,
            self._wav_path.name,
        )

    def start(self) -> None:
        with self._lock:
            if self.is_recording:
                raise RuntimeError("Запись уже идёт")
            self._chunks.clear()
            self._use_file = False
            self._wav_path = None
            self._wave_file = None
            self._frames_written = 0
            self._input = AudioInputStream(
                config=self._capture,
                sample_rate=self.sample_rate,
                on_audio=self._on_audio,
            )
            self._input.start()
            source = "микрофон" if self._capture.source == AudioSource.MICROPHONE else "система"
            log.info(
                "Запись начата (%s): RAM до %.0f с, sr=%s",
                source,
                self.ram_max_seconds,
                self.sample_rate,
            )

    def stop(self) -> RecordingResult:
        with self._lock:
            if self._input is None or not self._input.is_running:
                raise RuntimeError("Запись не запущена")
            self._input.stop()
            self._input = None
            if self._wave_file is not None:
                self._wave_file.close()
                self._wave_file = None

        if self._frames_written == 0:
            if self._wav_path:
                self._wav_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Запись пуста. Проверьте источник звука или запишите дольше."
            )

        duration = self._frames_written / self.sample_rate
        if self._use_file:
            path = self._wav_path
            self._wav_path = None
            if path is None:
                raise RuntimeError("Внутренняя ошибка: режим файла без пути.")
            size_kb = path.stat().st_size / 1024
            log.info(
                "Запись остановлена: %.2f с, файл %.1f KB (%s)",
                duration,
                size_kb,
                path.name,
            )
            return RecordingResult(path=path, duration_seconds=duration, used_file=True)

        audio = np.concatenate(self._chunks, axis=0).reshape(-1)
        self._chunks.clear()
        size_kb = audio.nbytes / 1024
        log.info("Запись остановлена: %.2f с, в RAM %.1f KB", duration, size_kb)
        return RecordingResult(audio=audio, duration_seconds=duration, used_file=False)

    def stop_to_wav(self) -> Path:
        result = self.stop()
        if result.path is not None:
            return result.path
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        path = Path(tmp.name)
        tmp.close()
        write_wav(path, result.audio, sample_rate=self.sample_rate)
        return path

    def stop_to_array(self) -> np.ndarray:
        result = self.stop()
        if result.audio is not None:
            return result.audio
        with wave.open(str(result.path), "rb") as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).copy()
        result.path.unlink(missing_ok=True)
        return audio.reshape(-1)


MicrophoneSession = AudioCaptureSession
