import time
from collections.abc import Callable
from typing import Literal


class Timer:
    def __init__(
        self,
        message: str | None = None,
        unit: Literal["s", "ms"] = "ms",
    ):
        self.message = message

        assert unit in ["s", "ms"], f"invalid unit {unit}, only allowed choices are {('s', 'ms')}"
        self.unit = unit

        self.start_time = None

    def __enter__(self, *args, **kwargs):
        self.start_time = time.time()

    def __exit__(self, *args, **kwargs):
        elapsed = (time.time() - self.start_time) * (1000 if self.unit == "ms" else 1)
        print(self.message + f" - {elapsed:.3f}{self.unit}")
