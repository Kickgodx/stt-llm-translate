import queue
import threading
from collections.abc import Callable

from ai_audio_transcription.audio_devices import CaptureConfig
from ai_audio_transcription.audio_input import AudioInputStream
from ai_audio_transcription.config import Settings
from ai_audio_transcription.live.events import LiveEvent, Segment
from ai_audio_transcription.live.segmenter import PhraseSegmenter
from ai_audio_transcription.logging_config import get_logger
from ai_audio_transcription.processor import AudioProcessor
from ai_audio_transcription.prompts import resolve_live_post_prompt
from ai_audio_transcription.service import ProcessOptions

log = get_logger("live.session")


class LiveSession:
    def __init__(
        self,
        *,
        settings: Settings,
        processor: AudioProcessor,
        on_event: Callable[[LiveEvent], None],
        capture: CaptureConfig | None = None,
        device: int | None = None,
    ) -> None:
        self._settings = settings
        self._processor = processor
        self._on_event = on_event
        self._capture = capture or CaptureConfig(device=device)

        self._running = False
        self._generation = 0
        self._input: AudioInputStream | None = None
        self._segment_queue: queue.Queue[Segment | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._segmenter: PhraseSegmenter | None = None
        self._options: ProcessOptions | None = None

    @property
    def is_active(self) -> bool:
        return self._running

    def start(self, options: ProcessOptions) -> None:
        if self._running:
            raise RuntimeError("Лайв-сессия уже запущена")

        self._generation += 1
        gen = self._generation
        self._options = options
        self._running = True
        self._segment_queue = queue.Queue()

        def on_segment(segment: Segment) -> None:
            self._segment_queue.put(segment)
            self._on_event(
                LiveEvent(
                    type="segment_queued",
                    segment_id=segment.id,
                    queue_size=self._segment_queue.qsize(),
                )
            )

        self._segmenter = PhraseSegmenter(
            sample_rate=self._settings.record_sample_rate,
            pause_ms=self._settings.live_pause_ms,
            min_segment_ms=self._settings.live_min_segment_ms,
            max_segment_ms=self._settings.live_max_segment_ms,
            speech_rms_threshold=self._settings.live_speech_rms_threshold,
            on_segment=on_segment,
        )

        self._worker = threading.Thread(
            target=self._worker_loop,
            args=(gen,),
            daemon=True,
        )
        self._worker.start()

        def on_audio(indata) -> None:
            if not self._running or self._segmenter is None:
                return
            self._segmenter.feed(indata[:, 0])

        self._input = AudioInputStream(
            config=self._capture,
            sample_rate=self._settings.record_sample_rate,
            on_audio=on_audio,
        )
        self._input.start()
        log.info("Лайв-сессия запущена (gen=%s)", gen)
        self._on_event(LiveEvent(type="listening"))

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        if self._input is not None:
            self._input.stop()
            self._input = None

        if self._segmenter is not None:
            self._segmenter.flush()
            self._segmenter = None

        self._segment_queue.put(None)

        if self._worker is not None:
            self._worker.join(timeout=30)
            self._worker = None

        log.info("Лайв-сессия остановлена")
        self._on_event(LiveEvent(type="stopped"))

    def _worker_loop(self, generation: int) -> None:
        while True:
            segment = self._segment_queue.get()
            if segment is None:
                break
            if generation != self._generation:
                continue
            self._process_segment(segment)

    def _process_segment(self, segment: Segment) -> None:
        options = self._options
        if options is None:
            return

        try:
            stt_model = options.model or self._settings.openrouter_model
            language = options.language or self._settings.openrouter_language

            transcript = self._processor.transcribe_segment(
                segment.audio,
                sample_rate=self._settings.record_sample_rate,
                stt_model=stt_model,
                language=language,
                temperature=options.temperature,
            )
            if not transcript.strip():
                self._on_event(
                    LiveEvent(
                        type="stt_done",
                        segment_id=segment.id,
                        text="(не распознано)",
                    )
                )
                self._on_event(LiveEvent(type="segment_done", segment_id=segment.id))
                return

            self._on_event(LiveEvent(type="stt_done", segment_id=segment.id, text=transcript))

            post_prompt = resolve_live_post_prompt(
                prompt=options.prompt,
                translate=options.translate,
            )
            if not post_prompt:
                self._on_event(LiveEvent(type="segment_done", segment_id=segment.id))
                return

            chat_model = options.chat_model or self._settings.openrouter_chat_model
            for delta in self._processor.stream_live_post(
                transcript,
                post_prompt,
                model=chat_model,
                max_tokens=self._settings.live_llm_max_tokens,
            ):
                self._on_event(
                    LiveEvent(
                        type="translation_delta",
                        segment_id=segment.id,
                        text=delta,
                    )
                )

            self._on_event(LiveEvent(type="segment_done", segment_id=segment.id))
        except Exception as exc:
            log.exception("Ошибка сегмента #%s", segment.id)
            self._on_event(
                LiveEvent(type="error", segment_id=segment.id, error=str(exc)),
            )
