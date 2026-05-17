from dataclasses import dataclass
from typing import Literal

LiveEventType = Literal[
    "listening",
    "speech",
    "segment_queued",
    "stt_done",
    "translation_delta",
    "segment_done",
    "error",
    "stopped",
]


@dataclass
class Segment:
    id: int
    audio: object  # np.ndarray int16
    duration_seconds: float


@dataclass
class LiveEvent:
    type: LiveEventType
    segment_id: int = 0
    text: str = ""
    queue_size: int = 0
    error: str | None = None
