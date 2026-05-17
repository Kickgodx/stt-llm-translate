import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from ai_audio_transcription.logging_config import get_logger
from ai_audio_transcription.recorder import MicrophoneSession
from ai_audio_transcription.service import AudioService, ProcessOptions, format_result

log = get_logger("ui")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

TRANSLATE_OPTIONS = {
    "Без перевода": None,
    "Перевод на EN": "en",
    "Перевод на RU": "ru",
}


class TranscriptionApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AI Audio Transcription")
        self.geometry("820x720")
        self.minsize(640, 560)

        try:
            self.service = AudioService()
        except Exception as exc:
            messagebox.showerror(
                "Ошибка конфигурации",
                f"Не удалось загрузить настройки.\n\n{exc}\n\nПроверьте файл .env",
            )
            raise

        self.mic = MicrophoneSession(
            sample_rate=self.service.settings.record_sample_rate,
            ram_max_seconds=self.service.settings.record_ram_max_seconds,
        )
        self._processing = False
        self._selected_file: Path | None = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

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
            text="Готов к записи",
            font=ctk.CTkFont(size=13),
            text_color=("gray50", "gray60"),
        )
        self.status_label.grid(row=1, column=0, sticky="w", pady=(4, 0))

        controls = ctk.CTkFrame(self)
        controls.grid(row=1, column=0, sticky="ew", padx=20, pady=8)
        controls.grid_columnconfigure(0, weight=1)

        self.record_btn = ctk.CTkButton(
            controls,
            text="Запись",
            height=52,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color="#2d8a4e",
            hover_color="#246e3f",
            command=self._toggle_recording,
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
        self.prompt_box.insert("1.0", "")

        opts = ctk.CTkFrame(controls, fg_color="transparent")
        opts.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        for i in range(4):
            opts.grid_columnconfigure(i, weight=1)

        ctk.CTkLabel(opts, text="Перевод").grid(row=0, column=0, sticky="w")
        self.translate_var = ctk.StringVar(value="Без перевода")
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
        ctk.CTkCheckBox(
            opts,
            text="Показать транскрипцию",
            variable=self.show_transcript_var,
        ).grid(row=1, column=2, sticky="w", padx=(0, 8))

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

        ctk.CTkLabel(settings, text="STT модель").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        self.stt_model_entry = ctk.CTkEntry(settings)
        self.stt_model_entry.insert(0, self.service.settings.openrouter_model)
        self.stt_model_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=8)

        ctk.CTkLabel(settings, text="LLM модель").grid(row=0, column=2, sticky="w", padx=8, pady=8)
        self.chat_model_entry = ctk.CTkEntry(settings)
        self.chat_model_entry.insert(0, self.service.settings.openrouter_chat_model)
        self.chat_model_entry.grid(row=0, column=3, sticky="ew", padx=(0, 8), pady=8)

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
        ctk.CTkButton(
            file_row,
            text="Обработать файл",
            width=160,
            command=self._process_file_clicked,
        ).grid(row=0, column=2, padx=(8, 0))

        output_frame = ctk.CTkFrame(self)
        output_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(8, 12))
        output_frame.grid_rowconfigure(1, weight=1)
        output_frame.grid_columnconfigure(0, weight=1)

        actions = ctk.CTkFrame(output_frame, fg_color="transparent")
        actions.grid(row=0, column=0, sticky="ew", pady=(12, 8), padx=12)
        ctk.CTkLabel(actions, text="Ответ", font=ctk.CTkFont(size=15, weight="bold")).pack(
            side="left"
        )
        ctk.CTkButton(actions, text="Копировать", width=110, command=self._copy_output).pack(
            side="right", padx=(8, 0)
        )
        ctk.CTkButton(actions, text="Очистить", width=110, command=self._clear_output).pack(
            side="right"
        )

        self.output_box = ctk.CTkTextbox(output_frame, font=ctk.CTkFont(size=14))
        self.output_box.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

    def _collect_options(self) -> ProcessOptions:
        prompt = self.prompt_box.get("1.0", "end").strip() or None
        translate = TRANSLATE_OPTIONS[self.translate_var.get()]
        language = self.language_entry.get().strip() or None
        return ProcessOptions(
            model=self.stt_model_entry.get().strip() or None,
            chat_model=self.chat_model_entry.get().strip() or None,
            language=language,
            prompt=prompt,
            translate=translate,
        )

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def _set_busy(self, busy: bool) -> None:
        self._processing = busy
        state = "disabled" if busy else "normal"
        self.record_btn.configure(state=state)
        if not self.mic.is_recording:
            self.record_btn.configure(state=state)

    def _toggle_recording(self) -> None:
        if self._processing:
            return
        if self.mic.is_recording:
            self._stop_and_send()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        try:
            self.mic.start()
        except Exception as exc:
            messagebox.showerror("Микрофон", str(exc))
            return
        self.record_btn.configure(
            text="Стоп и отправить",
            fg_color="#b83232",
            hover_color="#962828",
        )
        self._set_status("Идёт запись…")

    def _stop_and_send(self) -> None:
        try:
            recording = self.mic.stop()
        except Exception as exc:
            self._reset_record_button()
            messagebox.showerror("Запись", str(exc))
            return

        self._reset_record_button()
        self._set_status("Обработка…")
        self._set_busy(True)
        options = self._collect_options()

        if recording.used_file:
            log.info(
                "Отправка из файла (%.1f с > RAM-порог): %s",
                recording.duration_seconds,
                recording.path.name,
            )
            threading.Thread(
                target=self._run_process_file,
                args=(recording.path, options, True),
                daemon=True,
            ).start()
        else:
            log.info(
                "Отправка из RAM (%.1f с): %.1f KB",
                recording.duration_seconds,
                recording.audio.nbytes / 1024,
            )
            threading.Thread(
                target=self._run_process_audio,
                args=(recording.audio, options),
                daemon=True,
            ).start()

    def _reset_record_button(self) -> None:
        self.record_btn.configure(
            text="Запись",
            fg_color="#2d8a4e",
            hover_color="#246e3f",
        )

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
        if self._processing or self.mic.is_recording:
            return
        if not self._selected_file:
            messagebox.showwarning("Файл", "Сначала выберите аудиофайл.")
            return
        self._set_status("Обработка файла…")
        log.info("Обработка выбранного файла: %s", self._selected_file)
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
            log.exception("Ошибка обработки записи (RAM)")
            error = str(exc)
        self.after(0, lambda: self._on_process_done(text, error))

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
            log.exception("Ошибка обработки")
            error = str(exc)
        finally:
            if delete_after:
                audio_path.unlink(missing_ok=True)
                log.info("Временный файл удалён: %s", audio_path.name)

        self.after(0, lambda: self._on_process_done(text, error))

    def _on_process_done(self, text: str, error: str | None) -> None:
        self._set_busy(False)
        if error:
            self._set_status("Ошибка")
            messagebox.showerror("Ошибка", error)
            return
        self.output_box.delete("1.0", "end")
        self.output_box.insert("1.0", text)
        self._set_status("Готов к записи")

    def _copy_output(self) -> None:
        content = self.output_box.get("1.0", "end").strip()
        if content:
            self.clipboard_clear()
            self.clipboard_append(content)
            self._set_status("Скопировано")

    def _clear_output(self) -> None:
        self.output_box.delete("1.0", "end")
        self._set_status("Готов к записи")

    def _on_close(self) -> None:
        if self.mic.is_recording:
            try:
                self.mic.stop()
            except Exception:
                pass
        self.destroy()


def main() -> None:
    app = TranscriptionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
