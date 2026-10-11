"""A dependency-free terminal bar and readable redirected logs."""

import sys
import threading
import time


STAGES = {
    "select": "отбор положений",
    "source_review": "проверка источников",
    "formalize": "формализация",
    "reconstruct": "повторная формализация",
    "align": "сопоставление правил",
    "blind_solve": "слепое решение",
    "disagreement_review": "разбор расхождений",
    "language_review": "итоговый аудит",
}


class Progress:
    def __init__(self, enabled=True, stream=None):
        self.stream = stream or sys.stderr
        self.enabled = enabled
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.total = 0
        self.done = 0
        self.counts = {"test": 0, "train": 0}
        self.active = {}

    def show(self, message):
        if not self.enabled:
            return
        filled = int(20 * self.done / self.total) if self.total else 0
        bar = "#" * filled + "-" * (20 - filled)
        elapsed = int(time.monotonic() - self.started)
        line = (f"[{bar}] статьи {self.done}/{self.total} | "
                f"test {self.counts['test']} / few-shot {self.counts['train']} | "
                f"{elapsed // 60:02d}:{elapsed % 60:02d} | {message}")
        if self.stream.isatty():
            self.stream.write("\r\033[2K" + line)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def stage(self, label, stage):
        with self.lock:
            self.active[label] = STAGES.get(stage, stage)
            self.show("; ".join(f"{key}: {value}" for key, value in sorted(self.active.items())))

    def advance(self, label, status, counts):
        with self.lock:
            self.done += 1
            self.counts = counts
            self.active.pop(label, None)
            self.show(f"{label}: {status}")

    def close(self, status):
        with self.lock:
            self.show(status)
            if self.enabled and self.stream.isatty():
                self.stream.write("\n")
