from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from random import Random

DEFAULT_SAMPLES: tuple[int, ...] = (10**5, 10**6, 10**7)


def monte_carlo_pi(n_samples: int, seed: int = 0) -> float:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    return 4.0 * count_hits((n_samples, seed)) / n_samples


def monte_carlo_pi_parallel(n_samples: int, n_workers: int | None = None, seed: int = 0) -> float:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    chunks = split_samples(n_samples, n_workers or os.cpu_count() or 1)
    tasks = [(size, seed + index) for index, size in enumerate(chunks)]
    with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
        hits = sum(pool.map(count_hits, tasks))
    return 4.0 * hits / n_samples


def count_hits(task: tuple[int, int]) -> int:
    n_samples, seed = task
    next_value = Random(seed).random
    hits = 0
    for _ in range(n_samples):
        x = next_value()
        y = next_value()
        if x * x + y * y <= 1.0:
            hits += 1
    return hits


def split_samples(n_samples: int, parts: int) -> list[int]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    parts = max(1, min(parts, n_samples))
    size, remainder = divmod(n_samples, parts)
    return [size + (1 if index < remainder else 0) for index in range(parts)]


def compare_performance(
    samples_list: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    sizes: Sequence[int] = DEFAULT_SAMPLES if samples_list is None else samples_list
    table: dict[int, dict[str, float]] = {}
    for n_samples in sizes:
        started = time.perf_counter()
        sequential_pi = monte_carlo_pi(n_samples)
        sequential_time = time.perf_counter() - started

        started = time.perf_counter()
        parallel_pi = monte_carlo_pi_parallel(n_samples)
        parallel_time = time.perf_counter() - started

        table[n_samples] = {
            "sequential_time": sequential_time,
            "parallel_time": parallel_time,
            "sequential_pi": sequential_pi,
            "parallel_pi": parallel_pi,
            "speedup": sequential_time / parallel_time,
        }
    return table


def format_table(table: dict[int, dict[str, float]]) -> str:
    header = (
        f"{'n_samples':>10}  {'pi (seq)':<11}{'ошибка':<11}{'сек (seq)':<11}"
        f"{'pi (par)':<11}{'ошибка':<11}{'сек (par)':<11}{'Ускорение'}"
    )
    rows = []
    for n_samples, data in table.items():
        sequential_error = abs(data["sequential_pi"] - math.pi)
        parallel_error = abs(data["parallel_pi"] - math.pi)
        rows.append(
            f"{n_samples:>10}  "
            f"{data['sequential_pi']:<11.5f}{sequential_error:<11.5f}"
            f"{data['sequential_time']:<11.2f}"
            f"{data['parallel_pi']:<11.5f}{parallel_error:<11.5f}"
            f"{data['parallel_time']:<11.2f}{data['speedup']:.1f}x"
        )
    return "\n".join([header, "-" * len(header), *rows])


if __name__ == "__main__":
    print(format_table(compare_performance()))
