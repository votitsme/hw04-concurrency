# задание 4.1
# гил - один мьютекс на интерпретатор, байткод крутит только тот, кто его держит.
# потоки идут по очереди: работы столько же плюс переключения, поэтому на счёте
# ускорения нет. потоки вывозят только ввод-вывод, там гил отпускают.
# процессы: у каждого свой гил, платим за старт и за пикл.
# субинтерпретаторы: PEP 684 сделал гил отдельным на интерпретатор (3.12), PEP 734
# и InterpreterPoolExecutor (3.14) дали публичный апи. считают параллельно и без
# процессов, но передать между ними можно только то, что пиклится.

from __future__ import annotations

import multiprocessing
import os
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import InterpreterPoolExecutor, ProcessPoolExecutor

JOIN_TIMEOUT = 60.0

Task = Callable[..., int]
ArgsList = Sequence[tuple[int, ...]]
Runner = Callable[[Task, ArgsList], list[int]]


def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


def run_sequential(fn: Task, args_list: ArgsList) -> list[int]:
    return [fn(*args) for args in args_list]


def run_threading(fn: Task, args_list: ArgsList, timeout: float = JOIN_TIMEOUT) -> list[int]:
    results: list[int] = [0] * len(args_list)

    def worker(index: int, args: tuple[int, ...]) -> None:
        results[index] = fn(*args)

    threads = [
        threading.Thread(target=worker, args=(index, args), name=f"fib-{index}")
        for index, args in enumerate(args_list)
    ]
    for thread in threads:
        thread.start()
    _join_all(threads, timeout)
    return results


def run_multiprocessing_pool(fn: Task, args_list: ArgsList) -> list[int]:
    if not args_list:
        return []
    with multiprocessing.Pool(processes=_workers(len(args_list))) as pool:
        return pool.starmap(fn, args_list)


def run_executor(fn: Task, args_list: ArgsList) -> list[int]:
    if not args_list:
        return []
    with ProcessPoolExecutor(max_workers=_workers(len(args_list))) as pool:
        futures = [pool.submit(fn, *args) for args in args_list]
        return [future.result() for future in futures]


def run_interpreters(fn: Task, args_list: ArgsList) -> list[int]:
    if not args_list:
        return []
    with InterpreterPoolExecutor(max_workers=_workers(len(args_list))) as pool:
        futures = [pool.submit(fn, *args) for args in args_list]
        return [future.result() for future in futures]


METHODS: tuple[tuple[str, Runner], ...] = (
    ("sequential", run_sequential),
    ("threading", run_threading),
    ("multiprocessing", run_multiprocessing_pool),
    ("executor", run_executor),
    ("interpreters", run_interpreters),
)


def compare_all(n: int = 35, repeat: int = 8) -> dict[str, dict]:
    args_list = [(n,)] * repeat
    baseline = 0.0
    table: dict[str, dict] = {}
    for name, runner in METHODS:
        started = time.perf_counter()
        results = runner(fib, args_list)
        seconds = time.perf_counter() - started
        if name == "sequential":
            baseline = seconds
        table[name] = {
            "results": results,
            "time": seconds,
            "speedup": baseline / seconds,
        }
    return table


def format_table(table: dict[str, dict]) -> str:
    header = f"{'Метод':<18}{'Время (сек)':<14}{'Ускорение'}"
    rows = [
        f"{name:<18}{data['time']:<14.2f}{data['speedup']:.1f}x" for name, data in table.items()
    ]
    return "\n".join([header, "-" * len(header), *rows])


def _workers(count: int) -> int:
    return max(1, min(count, os.cpu_count() or 1))


def _join_all(threads: Sequence[threading.Thread], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            raise TimeoutError(f"поток {thread.name} не завершился за {timeout} с")


if __name__ == "__main__":
    print(format_table(compare_all()))
