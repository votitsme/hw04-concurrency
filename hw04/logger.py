from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from queue import Queue
from typing import TextIO

# консьюмер выходит по сентинелу, а не по флагу: флаг он увидит только после
# следующего сообщения, а сентинел лежит в очереди последним
_SENTINEL = None


class ThreadedLogger:
    def __init__(
        self,
        log_path: Path | str,
        max_lines: int = 100,
        n_producers: int = 3,
        messages_per_producer: int = 10,
        lock: AbstractContextManager[object] | None = None,
    ) -> None:
        if max_lines <= 0:
            raise ValueError("max_lines must be positive")
        self._path = Path(log_path)
        self._max_lines = max_lines
        self._n_producers = n_producers
        self._messages_per_producer = messages_per_producer
        self._queue: Queue[str | None] = Queue()
        # RLock, а не Lock: _write под локом зовёт _rotate, а тот берёт лок опять -
        # с обычным Lock это мгновенный самоблок. лок можно подменить на nullcontext
        # и посмотреть гонку, см. STUDENT.md
        self._lock: AbstractContextManager[object] = lock if lock is not None else threading.RLock()
        self._started = threading.Event()
        self._stopping = threading.Event()
        self._producers: list[threading.Thread] = []
        self._consumer_thread = threading.Thread(target=self._consumer, name="log-consumer")
        self._stream: TextIO | None = None
        self._lines = 0
        self._rotations = 0
        self._failure: BaseException | None = None

    @property
    def paths(self) -> list[Path]:
        with self._lock:
            rotated = [self._rotated_path(number) for number in range(1, self._rotations + 1)]
            return [*rotated, self._path]

    def start(self) -> None:
        if self._started.is_set():
            raise RuntimeError("logger is already started")
        self._started.set()
        self._stream = self._open()
        self._producers = [
            threading.Thread(target=self._producer, args=(index,), name=f"producer-{index}")
            for index in range(self._n_producers)
        ]
        self._consumer_thread.start()
        for thread in self._producers:
            thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if not self._started.is_set() or self._stopping.is_set():
            return
        self._stopping.set()
        deadline = time.monotonic() + timeout
        _join_all(self._producers, deadline)
        self._queue.put(_SENTINEL)
        _join_all([self._consumer_thread], deadline)
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        self._reraise()

    def _producer(self, producer_id: int) -> None:
        for number in range(self._messages_per_producer):
            self._queue.put(f"[{producer_id}] {number}: message from producer {producer_id}")

    def _consumer(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is _SENTINEL:
                        return
                    self._write(item)
                finally:
                    self._queue.task_done()
        except Exception as error:
            # поток не должен глохнуть молча: ошибку отдаём в stop
            self._failure = error
            self._stopping.set()
            self._drain()

    def _write(self, line: str) -> None:
        with self._lock:
            if self._lines >= self._max_lines:
                self._rotate()
            if self._stream is None:
                raise RuntimeError("logger is not started")
            self._stream.write(f"{line}\n")
            self._lines += 1

    def _rotate(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.close()
            self._rotations += 1
            self._path.replace(self._rotated_path(self._rotations))
            self._stream = self._open()
            self._lines = 0

    def _rotated_path(self, number: int) -> Path:
        return self._path.with_name(f"{self._path.name}.{number}")

    def _open(self) -> TextIO:
        # buffering=1 - строка уходит в файл сразу, а не после закрытия логгера
        return self._path.open("w", encoding="utf-8", buffering=1)

    def _drain(self) -> None:
        while not self._queue.empty():
            self._queue.get()
            self._queue.task_done()

    def _reraise(self) -> None:
        failure = self._failure
        if failure is not None:
            self._failure = None
            raise failure


def _join_all(threads: Sequence[threading.Thread], deadline: float) -> None:
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            raise TimeoutError(f"поток {thread.name} не завершился вовремя")
