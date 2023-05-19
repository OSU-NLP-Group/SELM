from typing import Callable


def make_linear_reg_scheduler(warmup: int) -> Callable[[int], float]:
    def scheduler_fn(step: int) -> float:
        if step > warmup:
            return 1.0

        return step / warmup

    return scheduler_fn
