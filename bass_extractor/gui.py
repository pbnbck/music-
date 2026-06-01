from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .core import PROFILES, SeparationOptions, diagnose_environment, separate_bass


class BassExtractorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Bass Extractor Pro")
        self.geometry("900x660")
        self.minsize(780, 600)
        self._messages: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._build_ui()
        self.after(120, self._drain_messages)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        header = ttk.Frame(self, padding=(18, 16, 18, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Bass Extractor Pro", font=("Segoe UI", 18, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(header, text="Doctor", command=self._run_doctor).grid(row=0, column=1, padx=(12, 0))

        paths = ttk.LabelFrame(self, text="Files", padding=14)
        paths.grid(row=1, column=0, sticky="ew", padx=18, pady=8)
        paths.columnconfigure(1, weight=1)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        ttk.Label(paths, text="Input song").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=5)
        ttk.Entry(paths, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", pady=5)
        ttk.Button(paths, text="Browse", command=self._choose_input).grid(
            row=0, column=2, padx=(10, 0), pady=5
        )

        ttk.Label(paths, text="Bass output").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=5)
        ttk.Entry(paths, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", pady=5)
        ttk.Button(paths, text="Save as", command=self._choose_output).grid(
            row=1, column=2, padx=(10, 0), pady=5
        )

        settings = ttk.LabelFrame(self, text="Separation Settings", padding=14)
        settings.grid(row=2, column=0, sticky="ew", padx=18, pady=8)
        for column in range(8):
            settings.columnconfigure(column, weight=1)

        self.profile_var = tk.StringVar(value="studio")
        self.device_var = tk.StringVar(value="auto")
        self.format_var = tk.StringVar(value="wav")
        self.no_bass_var = tk.BooleanVar(value=False)
        self.keep_work_var = tk.BooleanVar(value=False)
        self.kick_clean_var = tk.BooleanVar(value=False)
        self.kick_strength_var = tk.DoubleVar(value=0.65)
        self.score_var = tk.BooleanVar(value=False)

        ttk.Label(settings, text="Quality").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.profile_var,
            values=sorted(PROFILES),
            state="readonly",
            width=12,
        ).grid(row=1, column=0, sticky="ew", padx=(0, 10), pady=(4, 0))

        ttk.Label(settings, text="Device").grid(row=0, column=1, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.device_var,
            values=["auto", "cpu", "cuda"],
            state="readonly",
            width=10,
        ).grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(4, 0))

        ttk.Label(settings, text="Format").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.format_var,
            values=["wav", "flac", "mp3"],
            state="readonly",
            width=10,
        ).grid(row=1, column=2, sticky="ew", padx=(0, 10), pady=(4, 0))

        ttk.Checkbutton(settings, text="Export no_bass", variable=self.no_bass_var).grid(
            row=1, column=3, sticky="w", padx=(0, 10)
        )
        ttk.Checkbutton(settings, text="Keep work dir", variable=self.keep_work_var).grid(
            row=1, column=4, sticky="w", padx=(0, 10)
        )
        ttk.Checkbutton(settings, text="Kick clean", variable=self.kick_clean_var).grid(
            row=2, column=0, sticky="w", pady=(12, 0), padx=(0, 10)
        )
        ttk.Label(settings, text="Kick strength").grid(row=2, column=1, sticky="e", pady=(12, 0))
        ttk.Scale(
            settings,
            variable=self.kick_strength_var,
            from_=0.2,
            to=0.9,
            orient="horizontal",
        ).grid(row=2, column=2, columnspan=3, sticky="ew", pady=(12, 0), padx=(10, 10))
        ttk.Checkbutton(settings, text="Make score", variable=self.score_var).grid(
            row=2, column=5, sticky="w", pady=(12, 0), padx=(0, 10)
        )

        self.start_button = ttk.Button(settings, text="Extract Bass", command=self._start)
        self.start_button.grid(row=1, column=7, sticky="e")

        log_frame = ttk.LabelFrame(self, text="Log", padding=10)
        log_frame.grid(row=3, column=0, sticky="nsew", padx=18, pady=(8, 18))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log = tk.Text(log_frame, wrap="word", height=18)
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)

        self._log("Select a song, choose a quality profile, then extract. First model run may download weights.")

    def _choose_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose song",
            filetypes=[
                ("Audio", "*.wav *.mp3 *.flac *.m4a *.aac *.ogg"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.input_var.set(path)
        input_path = Path(path)
        suffix = "." + self.format_var.get()
        self.output_var.set(str(input_path.parent / "bass-extractor-output" / f"{input_path.stem}_bass{suffix}"))

    def _choose_output(self) -> None:
        suffix = "." + self.format_var.get()
        path = filedialog.asksaveasfilename(
            title="Save bass stem",
            defaultextension=suffix,
            filetypes=[(self.format_var.get().upper(), f"*{suffix}"), ("All files", "*.*")],
        )
        if path:
            self.output_var.set(path)

    def _run_doctor(self) -> None:
        self._log("Running environment diagnostics...")
        for check in diagnose_environment():
            mark = "OK" if check["pass"] else "MISSING"
            self._log(f"{mark:8} {check['name']}: {check['detail']}")

    def _start(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Busy", "The current job is still running.")
            return
        if not self.input_var.get().strip():
            messagebox.showwarning("Missing input", "Choose a song first.")
            return
        self.start_button.configure(state="disabled")
        self._log("Queued extraction job.")
        self._worker = threading.Thread(target=self._extract_worker, daemon=True)
        self._worker.start()

    def _extract_worker(self) -> None:
        try:
            options = SeparationOptions(
                profile=self.profile_var.get(),
                device=self.device_var.get(),
                output_format=self.format_var.get(),
                keep_work_dir=self.keep_work_var.get(),
                export_no_bass=self.no_bass_var.get(),
                kick_clean=self.kick_clean_var.get(),
                kick_strength=float(self.kick_strength_var.get()),
                make_score=self.score_var.get(),
            )
            result = separate_bass(
                Path(self.input_var.get()),
                Path(self.output_var.get()) if self.output_var.get().strip() else None,
                options=options,
                progress=self._messages.put,
            )
            self._messages.put(f"DONE bass: {result.bass_path}")
            if result.score_path is not None:
                self._messages.put(f"DONE score: {result.score_path}")
            if result.score_pdf_path is not None:
                self._messages.put(f"DONE score PDF: {result.score_pdf_path}")
            self._messages.put(f"DONE report: {result.report_path}")
        except Exception as exc:
            self._messages.put(f"ERROR: {exc}")
        finally:
            self._messages.put("__ENABLE_START__")

    def _drain_messages(self) -> None:
        while True:
            try:
                message = self._messages.get_nowait()
            except queue.Empty:
                break
            if message == "__ENABLE_START__":
                self.start_button.configure(state="normal")
            else:
                self._log(message)
        self.after(120, self._drain_messages)

    def _log(self, message: str) -> None:
        self.log.insert("end", message + "\n")
        self.log.see("end")


def main() -> None:
    app = BassExtractorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
