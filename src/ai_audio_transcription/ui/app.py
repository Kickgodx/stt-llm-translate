import queue
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from ai_audio_transcription.audio_devices import (
    AudioDeviceOption,
    AudioSource,
    CaptureConfig,
    list_microphone_devices,
    list_system_audio_devices,
    system_audio_available,
)
from ai_audio_transcription.live import LiveEvent, LiveSession
from ai_audio_transcription.logging_config import get_logger
from ai_audio_transcription.model_catalog import (
    chat_options_for_ui,
    id_from_chat_label,
    id_from_stt_label,
    stt_options_for_ui,
)
from ai_audio_transcription.recorder import AudioCaptureSession
from ai_audio_transcription.service import AudioService, ProcessOptions, format_result

log = get_logger("ui")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

MODE_BATCH = "Пакетная запись"
MODE_LIVE = "Лайв (по паузе)"

SOURCE_MIC = "Микрофон"
SOURCE_SYSTEM = "Системный звук"

TRANSLATE_OPTIONS = {
    "Без перевода": None,
    "Перевод на EN": "en",
    "Перевод на RU": "ru",
}


class TranscriptionApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AI Audio Transcription")
        self.geometry("860x760")
        self.minsize(680, 580)

        try:
            self.service = AudioService()
        except Exception as exc:
            messagebox.showerror(
                "Ошибка конфигурации",
                f"Не удалось загрузить настройки.\n\n{exc}\n\nПроверьте файл .env",
            )
            raise

        self.capture = AudioCaptureSession(
            sample_rate=self.service.settings.record_sample_rate,
            ram_max_seconds=self.service.settings.record_ram_max_seconds,
        )
        self._device_options: list[AudioDeviceOption] = []
        self._live_session: LiveSession | None = None
        self._processing = False
        self._selected_file: Path | None = None
        self._mode = ctk.StringVar(value=MODE_BATCH)
        self._ui_events: queue.Queue[LiveEvent] = queue.Queue()
        self._live_has_llm = False

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_ui_events()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 8))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="AI Audio Transcription",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        self.status_label = ctk.CTkLabel(
            header,
            text="Готов",
            font=ctk.CTkFont(size=13),
            text_color=("gray50", "gray60"),
        )
        self.status_label.grid(row=1, column=0, sticky="w", pady=(4, 0))

        mode_row = ctk.CTkFrame(self, fg_color="transparent")
        mode_row.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 4))
        ctk.CTkLabel(mode_row, text="Режим").pack(side="left", padx=(0, 8))
        self.mode_switch = ctk.CTkSegmentedButton(
            mode_row,
            values=[MODE_BATCH, MODE_LIVE],
            variable=self._mode,
            command=self._on_mode_changed,
        )
        self.mode_switch.pack(side="left", fill="x", expand=True)

        input_row = ctk.CTkFrame(self, fg_color="transparent")
        input_row.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 4))
        input_row.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(input_row, text="Источник").grid(row=0, column=0, sticky="w", padx=(0, 8))
        source_values = [SOURCE_MIC, SOURCE_SYSTEM]
        if not system_audio_available():
            source_values = [SOURCE_MIC]
        self._source_var = ctk.StringVar(value=SOURCE_MIC)
        self.source_switch = ctk.CTkSegmentedButton(
            input_row,
            values=source_values,
            variable=self._source_var,
            command=self._on_source_changed,
        )
        self.source_switch.grid(row=0, column=1, sticky="ew", padx=(0, 12))
        ctk.CTkLabel(input_row, text="Устройство").grid(row=0, column=2, sticky="w", padx=(0, 8))
        self._device_var = ctk.StringVar(value="По умолчанию")
        self.device_menu = ctk.CTkOptionMenu(
            input_row,
            variable=self._device_var,
            values=["По умолчанию"],
            command=self._on_device_selected,
        )
        self.device_menu.grid(row=0, column=3, sticky="ew")
        self._refresh_device_menu()

        controls = ctk.CTkFrame(self)
        controls.grid(row=3, column=0, sticky="ew", padx=20, pady=4)
        controls.grid_columnconfigure(0, weight=1)

        self.record_btn = ctk.CTkButton(
            controls,
            text="Запись",
            height=52,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color="#2d8a4e",
            hover_color="#246e3f",
            command=self._toggle_record,
        )
        self.record_btn.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        prompt_frame = ctk.CTkFrame(controls, fg_color="transparent")
        prompt_frame.grid(row=1, column=0, sticky="ew")
        prompt_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(prompt_frame, text="Промпт LLM").grid(
            row=0, column=0, sticky="nw", padx=(0, 8)
        )
        self.prompt_box = ctk.CTkTextbox(prompt_frame, height=72)
        self.prompt_box.grid(row=0, column=1, sticky="ew")

        opts = ctk.CTkFrame(controls, fg_color="transparent")
        opts.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        for i in range(4):
            opts.grid_columnconfigure(i, weight=1)

        ctk.CTkLabel(opts, text="Перевод").grid(row=0, column=0, sticky="w")
        self.translate_var = ctk.StringVar(value="Перевод на EN")
        ctk.CTkOptionMenu(
            opts,
            variable=self.translate_var,
            values=list(TRANSLATE_OPTIONS),
        ).grid(row=1, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkLabel(opts, text="Язык STT").grid(row=0, column=1, sticky="w")
        self.language_entry = ctk.CTkEntry(opts, placeholder_text="ru, en…")
        if self.service.settings.openrouter_language:
            self.language_entry.insert(0, self.service.settings.openrouter_language)
        self.language_entry.grid(row=1, column=1, sticky="ew", padx=(0, 8))

        self.show_transcript_var = ctk.BooleanVar(value=True)
        self.show_transcript_cb = ctk.CTkCheckBox(
            opts,
            text="Показать транскрипцию",
            variable=self.show_transcript_var,
        )
        self.show_transcript_cb.grid(row=1, column=2, sticky="w", padx=(0, 8))

        self.show_usage_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            opts,
            text="Usage / cost",
            variable=self.show_usage_var,
        ).grid(row=1, column=3, sticky="w")

        settings = ctk.CTkFrame(controls)
        settings.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        settings.grid_columnconfigure(1, weight=1)
        settings.grid_columnconfigure(3, weight=1)

        self._stt_labels, self._stt_ids, stt_default = stt_options_for_ui(
            self.service.settings.openrouter_model
        )
        self._chat_labels, self._chat_ids, chat_default = chat_options_for_ui(
            self.service.settings.openrouter_chat_model
        )

        ctk.CTkLabel(settings, text="STT модель").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        self.stt_model_var = ctk.StringVar(value=stt_default)
        ctk.CTkOptionMenu(
            settings,
            variable=self.stt_model_var,
            values=self._stt_labels,
        ).grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=8)

        ctk.CTkLabel(settings, text="LLM модель").grid(row=0, column=2, sticky="w", padx=8, pady=8)
        self.chat_model_var = ctk.StringVar(value=chat_default)
        ctk.CTkOptionMenu(
            settings,
            variable=self.chat_model_var,
            values=self._chat_labels,
        ).grid(row=0, column=3, sticky="ew", padx=(0, 8), pady=8)

        file_row = ctk.CTkFrame(controls, fg_color="transparent")
        file_row.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        file_row.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(file_row, text="Выбрать файл", width=140, command=self._pick_file).grid(
            row=0, column=0, padx=(0, 8)
        )
        self.file_label = ctk.CTkLabel(
            file_row,
            text="Файл не выбран",
            anchor="w",
            text_color=("gray40", "gray55"),
        )
        self.file_label.grid(row=0, column=1, sticky="ew")
        self.process_file_btn = ctk.CTkButton(
            file_row,
            text="Обработать файл",
            width=160,
            command=self._process_file_clicked,
        )
        self.process_file_btn.grid(row=0, column=2, padx=(8, 0))

        output_outer = ctk.CTkFrame(self)
        output_outer.grid(row=4, column=0, sticky="nsew", padx=20, pady=(8, 12))
        output_outer.grid_rowconfigure(1, weight=1)
        output_outer.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        actions = ctk.CTkFrame(output_outer, fg_color="transparent")
        actions.grid(row=0, column=0, sticky="ew", pady=(12, 8), padx=12)
        self.output_title = ctk.CTkLabel(
            actions, text="Ответ", font=ctk.CTkFont(size=15, weight="bold")
        )
        self.output_title.pack(side="left")
        ctk.CTkButton(actions, text="Копировать", width=110, command=self._copy_output).pack(
            side="right", padx=(8, 0)
        )
        ctk.CTkButton(actions, text="Очистить", width=110, command=self._clear_output).pack(
            side="right"
        )

        self.batch_output_frame = ctk.CTkFrame(output_outer, fg_color="transparent")
        self.batch_output_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.batch_output_frame.grid_rowconfigure(0, weight=1)
        self.batch_output_frame.grid_columnconfigure(0, weight=1)
        self.output_box = ctk.CTkTextbox(self.batch_output_frame, font=ctk.CTkFont(size=14))
        self.output_box.grid(row=0, column=0, sticky="nsew")

        self.live_output_frame = ctk.CTkFrame(output_outer, fg_color="transparent")
        self.live_output_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.live_output_frame.grid_rowconfigure(0, weight=1)
        self.live_output_frame.grid_columnconfigure(0, weight=1)
        self.live_box = ctk.CTkTextbox(self.live_output_frame, font=ctk.CTkFont(size=14))
        self.live_box.grid(row=0, column=0, sticky="nsew")

        self._show_batch_output()

    def _is_live_mode(self) -> bool:
        return self._mode.get() == MODE_LIVE

    def _is_busy(self) -> bool:
        return (
            self._processing
            or self.capture.is_recording
            or (self._live_session is not None and self._live_session.is_active)
        )

    def _on_mode_changed(self, _value: str) -> None:
        if self._is_busy():
            messagebox.showwarning("Режим", "Сначала остановите запись.")
            self._mode.set(MODE_LIVE if self._is_live_mode() else MODE_BATCH)
            return
        if self._is_live_mode():
            self._show_live_output()
            self._set_status("Лайв: нажмите кнопку для старта")
            self.record_btn.configure(text="Старт лайв")
            self.show_transcript_cb.configure(state="disabled")
        else:
            self._show_batch_output()
            self._set_status("Готов к записи")
            self.record_btn.configure(text="Запись")
            self.show_transcript_cb.configure(state="normal")

    def _show_batch_output(self) -> None:
        self.live_output_frame.grid_remove()
        self.batch_output_frame.grid()
        self.output_title.configure(text="Ответ")

    def _show_live_output(self) -> None:
        self.batch_output_frame.grid_remove()
        self.live_output_frame.grid()
        self.output_title.configure(text="Лайв")

    def _collect_options(self) -> ProcessOptions:
        prompt = self.prompt_box.get("1.0", "end").strip() or None
        translate = TRANSLATE_OPTIONS[self.translate_var.get()]
        language = self.language_entry.get().strip() or None
        return ProcessOptions(
            model=id_from_stt_label(self.stt_model_var.get(), self._stt_ids, self._stt_labels),
            chat_model=id_from_chat_label(
                self.chat_model_var.get(), self._chat_ids, self._chat_labels
            ),
            language=language,
            prompt=prompt,
            translate=translate,
        )

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def _set_busy(self, busy: bool) -> None:
        self._processing = busy
        state = "disabled" if busy else "normal"
        if not self.capture.is_recording and not (self._live_session and self._live_session.is_active):
            self.record_btn.configure(state=state)
        self.mode_switch.configure(state=state)
        self.source_switch.configure(state=state)
        self.device_menu.configure(state=state)
        self.process_file_btn.configure(state=state)

    def _toggle_record(self) -> None:
        if self._processing:
            return
        if self._is_live_mode():
            self._toggle_live()
        else:
            self._toggle_batch()

    def _is_system_source(self) -> bool:
        return self._source_var.get() == SOURCE_SYSTEM

    def _refresh_device_menu(self) -> None:
        if self._is_system_source():
            self._device_options = list_system_audio_devices()
        else:
            self._device_options = list_microphone_devices()
        labels = [opt.label for opt in self._device_options]
        if not labels:
            labels = ["По умолчанию"]
        self.device_menu.configure(values=labels)
        if self._device_var.get() not in labels:
            self._device_var.set(labels[0])

    def _on_source_changed(self, _value: str) -> None:
        if self._is_busy():
            messagebox.showwarning("Источник", "Сначала остановите запись.")
            self._source_var.set(SOURCE_MIC if self._is_system_source() else SOURCE_SYSTEM)
            return
        self._refresh_device_menu()

    def _on_device_selected(self, _value: str) -> None:
        pass

    def _get_capture_config(self) -> CaptureConfig:
        label = self._device_var.get()
        device: int | None = None
        for opt in self._device_options:
            if opt.label == label:
                device = opt.index
                break
        source = AudioSource.SYSTEM if self._is_system_source() else AudioSource.MICROPHONE
        return CaptureConfig(source=source, device=device)

    def _toggle_batch(self) -> None:
        if self.capture.is_recording:
            self._stop_batch()
        else:
            self._start_batch()

    def _start_batch(self) -> None:
        if self._is_system_source() and sys.platform != "win32":
            messagebox.showerror(
                "Системный звук",
                "Захват системного звука поддерживается только на Windows.",
            )
            return
        try:
            self.capture.set_capture(self._get_capture_config())
            self.capture.start()
        except Exception as exc:
            messagebox.showerror("Запись", str(exc))
            return
        self.record_btn.configure(
            text="Стоп и отправить",
            fg_color="#b83232",
            hover_color="#962828",
        )
        self._set_status("Идёт запись…")

    def _stop_batch(self) -> None:
        try:
            recording = self.capture.stop()
        except Exception as exc:
            self._reset_record_button()
            messagebox.showerror("Запись", str(exc))
            return

        self._reset_record_button()
        self._set_status("Обработка…")
        self._set_busy(True)
        options = self._collect_options()

        if recording.used_file:
            threading.Thread(
                target=self._run_process_file,
                args=(recording.path, options, True),
                daemon=True,
            ).start()
        else:
            threading.Thread(
                target=self._run_process_audio,
                args=(recording.audio, options),
                daemon=True,
            ).start()

    def _toggle_live(self) -> None:
        if self._live_session and self._live_session.is_active:
            self._stop_live()
        else:
            self._start_live()

    def _start_live(self) -> None:
        options = self._collect_options()
        if not options.prompt and not options.translate:
            if not messagebox.askyesno(
                "Лайв",
                "Без перевода/промпта будет только транскрипт.\nПродолжить?",
            ):
                return

        self._clear_live_output()
        self._live_has_llm = bool(options.prompt or options.translate)

        def on_event(event: LiveEvent) -> None:
            self._ui_events.put(event)

        if self._is_system_source() and sys.platform != "win32":
            messagebox.showerror(
                "Системный звук",
                "Захват системного звука поддерживается только на Windows.",
            )
            return

        self._live_session = LiveSession(
            settings=self.service.settings,
            processor=self.service.processor,
            on_event=on_event,
            capture=self._get_capture_config(),
        )
        try:
            self._live_session.start(options)
        except Exception as exc:
            self._live_session = None
            messagebox.showerror("Лайв", str(exc))
            return

        self.record_btn.configure(
            text="Стоп лайв",
            fg_color="#b83232",
            hover_color="#962828",
        )
        self.mode_switch.configure(state="disabled")
        self.source_switch.configure(state="disabled")
        self.device_menu.configure(state="disabled")
        self._set_status("Слушаю… говорите, пауза = новая реплика")

    def _stop_live(self) -> None:
        if self._live_session:
            self._live_session.stop()
            self._live_session = None
        self._reset_record_button()
        self.mode_switch.configure(state="normal")
        self.source_switch.configure(state="normal")
        self.device_menu.configure(state="normal")
        self._set_status("Лайв остановлен")

    def _reset_record_button(self) -> None:
        label = "Старт лайв" if self._is_live_mode() else "Запись"
        self.record_btn.configure(
            text=label,
            fg_color="#2d8a4e",
            hover_color="#246e3f",
            state="normal",
        )

    def _poll_ui_events(self) -> None:
        while True:
            try:
                event = self._ui_events.get_nowait()
            except queue.Empty:
                break
            self._handle_live_event(event)
        self.after(100, self._poll_ui_events)

    def _handle_live_event(self, event: LiveEvent) -> None:
        if event.type == "listening":
            self._set_status("Слушаю…")
        elif event.type == "segment_queued":
            self._set_status(f"В очереди: {event.queue_size}")
        elif event.type == "stt_done":
            self.live_box.insert("end", f"\n[{event.segment_id}]\n{event.text}\n")
            if self._live_has_llm:
                self.live_box.insert("end", "→ ")
            self.live_box.see("end")
        elif event.type == "translation_delta":
            self.live_box.insert("end", event.text)
            self.live_box.see("end")
        elif event.type == "segment_done":
            self.live_box.insert("end", "\n")
            self.live_box.see("end")
            self._set_status("Слушаю…")
        elif event.type == "error":
            messagebox.showerror("Лайв", event.error or "Ошибка")
            self._set_status("Ошибка сегмента")
        elif event.type == "stopped":
            self._set_status("Лайв остановлен")

    def _pick_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите аудио",
            filetypes=[
                ("Аудио", "*.wav *.mp3 *.flac *.m4a *.ogg *.webm *.aac"),
                ("Все файлы", "*.*"),
            ],
        )
        if path:
            self._selected_file = Path(path)
            self.file_label.configure(text=self._selected_file.name)

    def _process_file_clicked(self) -> None:
        if self._is_busy():
            return
        if not self._selected_file:
            messagebox.showwarning("Файл", "Сначала выберите аудиофайл.")
            return
        self._set_status("Обработка файла…")
        self._set_busy(True)
        options = self._collect_options()
        threading.Thread(
            target=self._run_process_file,
            args=(self._selected_file, options),
            daemon=True,
        ).start()

    def _run_process_audio(self, audio, options: ProcessOptions) -> None:
        error: str | None = None
        text = ""
        try:
            result = self.service.process_audio(audio, options)
            text = format_result(
                result,
                show_transcript=self.show_transcript_var.get(),
                show_usage=self.show_usage_var.get(),
            )
        except Exception as exc:
            log.exception("Ошибка batch RAM")
            error = str(exc)
        self.after(0, lambda: self._on_batch_done(text, error))

    def _run_process_file(
        self,
        audio_path: Path,
        options: ProcessOptions,
        delete_after: bool = False,
    ) -> None:
        error: str | None = None
        text = ""
        try:
            result = self.service.process_file(audio_path, options)
            text = format_result(
                result,
                show_transcript=self.show_transcript_var.get(),
                show_usage=self.show_usage_var.get(),
            )
        except Exception as exc:
            log.exception("Ошибка batch file")
            error = str(exc)
        finally:
            if delete_after:
                audio_path.unlink(missing_ok=True)
        self.after(0, lambda: self._on_batch_done(text, error))

    def _on_batch_done(self, text: str, error: str | None) -> None:
        self._set_busy(False)
        if error:
            self._set_status("Ошибка")
            messagebox.showerror("Ошибка", error)
            return
        self.output_box.delete("1.0", "end")
        self.output_box.insert("1.0", text)
        self._set_status("Готов")

    def _copy_output(self) -> None:
        if self._is_live_mode():
            content = self.live_box.get("1.0", "end").strip()
        else:
            content = self.output_box.get("1.0", "end").strip()
        if content:
            self.clipboard_clear()
            self.clipboard_append(content)
            self._set_status("Скопировано")

    def _clear_output(self) -> None:
        if self._is_live_mode():
            self._clear_live_output()
        else:
            self.output_box.delete("1.0", "end")
        self._set_status("Очищено")

    def _clear_live_output(self) -> None:
        self.live_box.delete("1.0", "end")

    def _on_close(self) -> None:
        if self._live_session and self._live_session.is_active:
            self._live_session.stop()
        if self.capture.is_recording:
            try:
                self.capture.stop()
            except Exception:
                pass
        self.destroy()


def main() -> None:
    app = TranscriptionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
