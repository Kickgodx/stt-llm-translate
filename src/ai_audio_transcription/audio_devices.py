"""Список устройств ввода: микрофоны и системный звук (loopback)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum

import sounddevice as sd

from ai_audio_transcription.logging_config import get_logger

log = get_logger("audio_devices")

LOOPBACK_NAME_KEYWORDS = (
    "stereo mix",
    "стерео микшер",
    "loopback",
    "wave out",
    "what u hear",
    "динамик пк",
    "speaker",
    "перехват",
    "monitor of",
)


class AudioSource(str, Enum):
    MICROPHONE = "microphone"
    SYSTEM = "system"


@dataclass(frozen=True)
class CaptureConfig:
    source: AudioSource = AudioSource.MICROPHONE
    device: int | None = None


@dataclass(frozen=True)
class AudioDeviceOption:
    index: int | None
    label: str


def is_loopback_device_name(name: str) -> bool:
    lower = name.lower()
    if "[loopback]" in lower:
        return True
    return any(keyword in lower for keyword in LOOPBACK_NAME_KEYWORDS)


def list_microphone_devices() -> list[AudioDeviceOption]:
    options = [AudioDeviceOption(index=None, label="По умолчанию")]
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] < 1:
            continue
        if is_loopback_device_name(info["name"]):
            continue
        options.append(AudioDeviceOption(index=index, label=f"{info['name']} (#{index})"))
    return options


def list_stereo_mix_devices() -> list[AudioDeviceOption]:
    options: list[AudioDeviceOption] = []
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] < 1:
            continue
        if not is_loopback_device_name(info["name"]):
            continue
        options.append(AudioDeviceOption(index=index, label=f"{info['name']} (#{index})"))
    return options


def list_wasapi_loopback_devices() -> list[AudioDeviceOption]:
    if sys.platform != "win32":
        return []
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        return []

    options: list[AudioDeviceOption] = []
    pa = pyaudio.PyAudio()
    try:
        for info in pa.get_loopback_device_info_generator():
            index = int(info["index"])
            name = str(info["name"])
            options.append(AudioDeviceOption(index=index, label=f"{name} (#{index})"))
    finally:
        pa.terminate()
    return options


def list_system_audio_devices() -> list[AudioDeviceOption]:
    wasapi = list_wasapi_loopback_devices()
    if wasapi:
        return [AudioDeviceOption(index=None, label="По умолчанию (WASAPI)"), *wasapi]
    stereo = list_stereo_mix_devices()
    if stereo:
        return [AudioDeviceOption(index=None, label="По умолчанию"), *stereo]
    return [AudioDeviceOption(index=None, label="(устройства не найдены)")]


def system_audio_available() -> bool:
    if sys.platform != "win32":
        return False
    return bool(list_wasapi_loopback_devices()) or bool(list_stereo_mix_devices())


def resolve_stereo_mix_device(device: int | None) -> int:
    if device is not None:
        return device
    stereo = list_stereo_mix_devices()
    if not stereo:
        raise RuntimeError(
            "Не найден «Стерео микшер». Включите его в параметрах звука Windows "
            "(Запись → Стерео микшер) или обновите драйвер."
        )
    if stereo[0].index is None:
        raise RuntimeError("Нет доступного устройства системного звука.")
    return stereo[0].index
