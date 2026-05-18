import threading
from collections.abc import Callable

import numpy as np

from ai_audio_transcription.live.events import Segment
from ai_audio_transcription.logging_config import get_logger

log = get_logger("live.segmenter")


class PhraseSegmenter:
    """Сегментация по паузе (RMS VAD, без лишних зависимостей)."""

    def __init__(
        self,
        *,
        sample_rate: int,
        pause_ms: float = 2000,
        min_segment_ms: float = 400,
        max_segment_ms: float = 30_000,
        speech_rms_threshold: float = 400.0,
        on_segment: Callable[[Segment], None],
    ) -> None:
        self.sample_rate = sample_rate
        self.pause_frames = max(1, int(sample_rate * pause_ms / 1000))
        self.min_segment_frames = max(1, int(sample_rate * min_segment_ms / 1000))
        self.max_segment_frames = max(
            self.min_segment_frames, int(sample_rate * max_segment_ms / 1000)
        )
        self.speech_rms_threshold = speech_rms_threshold
        self._on_segment = on_segment
        self._lock = threading.Lock()

        self._segment_id = 0
        self._in_speech = False
        self._silence_frames = 0
        self._buffer: list[np.ndarray] = []
        self._buffer_frames = 0

    def _rms(self, frame: np.ndarray) -> float:
        samples = frame.astype(np.float32)
        return float(np.sqrt(np.mean(samples * samples)))

    def _is_speech(self, frame: np.ndarray) -> bool:
        return self._rms(frame) >= self.speech_rms_threshold

    def _emit_segment(self, *, force: bool = False) -> bool:
        if self._buffer_frames == 0:
            return False
        if not force and self._buffer_frames < self.min_segment_frames:
            self._buffer.clear()
            self._buffer_frames = 0
            return False
        audio = np.concatenate(self._buffer, axis=0).reshape(-1)
        self._segment_id += 1
        duration = len(audio) / self.sample_rate
        log.info("Сегмент #%s: %.2f с%s", self._segment_id, duration, " (вручную)" if force else "")
        self._on_segment(
            Segment(id=self._segment_id, audio=audio, duration_seconds=duration),
        )
        self._buffer.clear()
        self._buffer_frames = 0
        return True

    def feed(self, frame: np.ndarray) -> None:
        frame = frame.reshape(-1)
        with self._lock:
            speech = self._is_speech(frame)

            if speech:
                if not self._in_speech:
                    self._in_speech = True
                self._silence_frames = 0
                self._buffer.append(frame.copy())
                self._buffer_frames += len(frame)
                if self._buffer_frames >= self.max_segment_frames:
                    self._emit_segment()
                    self._in_speech = False
                    self._silence_frames = 0
                return

            if not self._in_speech:
                return

            self._silence_frames += len(frame)
            self._buffer.append(frame.copy())
            self._buffer_frames += len(frame)

            if self._silence_frames >= self.pause_frames:
                self._emit_segment()
                self._in_speech = False
                self._silence_frames = 0

    def emit_now(self) -> bool:
        """Принудительно отправить накопленное аудио (кнопка в UI)."""
        with self._lock:
            emitted = self._emit_segment(force=True)
            self._in_speech = False
            self._silence_frames = 0
            return emitted

    def flush(self) -> None:
        with self._lock:
            if self._buffer_frames >= self.min_segment_frames:
                self._emit_segment()
            else:
                self._buffer.clear()
                self._buffer_frames = 0
            self._in_speech = False
            self._silence_frames = 0
