from pathlib import Path
from enum import Enum
import numpy as np
import typing as tp
import time
from collections import defaultdict

import torch
from rich.console import Console
from rich.table import Table

from torch.profiler import record_function, ProfilerActivity


class Opt(Enum):
    JIT = "jit"
    NONE = "none"
    COMPILE = "compile"


class Reduction(Enum):
    MEAN = "mean"
    SUM = "sum"
    MEDIAN = "median"


class Timer:
    def __init__(
        self,
        modules_and_fns: tp.List[tp.Tuple[torch.nn.Module, str]],
        nvtx: bool = False,
        sync: bool = True,
    ) -> None:
        self.modules_and_fns = modules_and_fns
        self.saved_fns = [getattr(m, f) for m, f in modules_and_fns]
        self.nvtx = nvtx
        self.sync = sync
        self.timers: tp.Dict[str, tp.List[float]] = defaultdict(list)
        self.is_profiling = False

    def start_profiling(self) -> None:
        self.is_profiling = True
        for m, f in self.modules_and_fns:
            self.time_fn_in_module(m, f)
        if self.nvtx:
            torch.cuda.profiler.start()

    def start_range(self, label: str) -> None:
        if not self.is_profiling:
            return
        if self.sync:
            torch.cuda.synchronize()
        if self.nvtx:
            torch.cuda.nvtx.range_push(label)
        self.timers[label].append(time.perf_counter_ns())

    def end_range(self, label: str) -> None:
        if not self.is_profiling:
            return
        if self.sync:
            torch.cuda.synchronize()
        if self.nvtx:
            torch.cuda.nvtx.range_pop()
        self.timers[label][-1] = (time.perf_counter_ns() - self.timers[label][-1]) / 1e6

    def stop_profiling(self) -> None:
        self.is_profiling = False
        if self.nvtx:
            torch.cuda.profiler.stop()
        # Restore the wrapped functions
        for (m, f), sf in zip(self.modules_and_fns, self.saved_fns):
            setattr(m, f, sf)

    def dump_csv(self, path: Path) -> None:
        with open(path, mode="wt", encoding="utf-8") as f:
            f.write(",".join(("timing", "num", "median", "mean", "std", "total\n")))
            for k, v in self.timers.items():
                fw = np.array(v) if v else np.array([0.0])
                f.write(
                    ",".join(
                        (
                            k,
                            f"{len(fw)}",
                            f"{np.median(fw):.5f}",
                            f"{np.mean(fw):.5f}",
                            f"{np.std(fw):.5f}",
                            f"{np.sum(fw):.5f}\n",
                        )
                    )
                )

    def display(self) -> None:
        console = Console()
        table = Table(
            box=None,
        )
        table.add_column("timing", style="magenta")
        table.add_column("num", style="blue")
        table.add_column("median (ms)", style="green")
        table.add_column("mean (ms)", style="green")
        table.add_column("std (ms)", style="green")
        table.add_column("total (ms)", style="green")
        for k, v in self.timers.items():
            fw = np.array(v) if v else np.array([0.0])
            table.add_row(
                k,
                f"{len(fw)}",
                f"{np.median(fw):.5f}",
                f"{np.mean(fw):.5f}",
                f"{np.std(fw):.5f}",
                f"{np.sum(fw):.5f}",
            )
        if self.modules_and_fns:
            console.print(
                "WARNING: Callbacks and sync create overhead",
                style="yellow",
            )
        console.print(table)
        console.print()

    def time_fn_in_module(self, module, fn_name: str) -> None:
        fn = getattr(module, fn_name)
        qualname = fn.__qualname__

        def wrapper(*args, **kwargs):
            if self.sync:
                torch.cuda.synchronize()
            if self.nvtx:
                torch.cuda.nvtx.range_push(qualname)
            start = time.perf_counter_ns()
            ret = fn(*args, **kwargs)
            if self.sync:
                torch.cuda.synchronize()
            if self.nvtx:
                torch.cuda.nvtx.range_pop()
            end = time.perf_counter_ns()
            self.timers[qualname].append((end - start) / 1e6)
            return ret

        setattr(module, fn_name, wrapper)

    def reset_timers(self) -> None:
        self.timers = defaultdict(list)


def time_func(
    key: str,
    func: tp.Callable[..., tp.Any],
    timers: tp.Dict[str, float],
    sync: bool = False,
    nvtx: bool = False,
):
    timers[key] = 0.0

    def wrapper(*args, **kwargs):
        if sync:
            torch.cuda.synchronize()
        if nvtx:
            torch.cuda.nvtx.range_push(key)
        start = time.perf_counter()
        ret = func(*args, **kwargs)
        if sync:
            torch.cuda.synchronize()
        if nvtx:
            torch.cuda.nvtx.range_pop()
        end = time.perf_counter()
        timers[key] += end - start
        return ret

    return wrapper


def time_functions(
    names_models: tp.Sequence[
        tp.Tuple[tp.Union[str, tp.Tuple[str, ...]], torch.nn.Module]
    ],
    timers: tp.Dict[str, float],
    sync: bool = False,
    nvtx: bool = False,
):
    for fn_names, model in names_models:
        if isinstance(fn_names, str):
            fn_names = (fn_names,)
        for fn_name in fn_names:
            setattr(
                model,
                fn_name,
                time_func(
                    ".".join((model.__class__.__name__, fn_name)),
                    getattr(model, fn_name),
                    timers,
                    sync,
                    nvtx,
                ),
            )


def timeit(
    func,
    *args,
    steps=100,
    warmup=10,
    run_profile=False,
    verbose=True,
    label=None,
    label_padding=35,
    cpu_timing=False,
):
    """
    Returns time/step in ms.
    If run_profile is True, then return (time/step in ms, a captured cuda events table)
    """
    if label is None:
        assert func.__name__, "please provide a label for this benchmark"
        label = func.__name__

    # warmup
    torch.cuda.nvtx.range_push(f"{label}-warmup")
    for _ in range(warmup):
        func(*args)
    torch.cuda.nvtx.range_pop()

    # start timer
    if cpu_timing:
        torch.cuda.synchronize()
        start = time.perf_counter()
    else:
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    torch.cuda.nvtx.range_push(f"{label}")
    if run_profile:
        if verbose:
            print("\n" + "=" * 70 + " " + label + " " + "=" * 70)
        with torch.profiler.profile(activities=[ProfilerActivity.CUDA]) as prof:
            with record_function("run_total"):
                for i in range(steps):
                    torch.cuda.nvtx.range_push(f"iteration-{i}")
                    func(*args)
                    torch.cuda.nvtx.range_pop()
        events = prof.key_averages()
        if verbose:
            for i, e in enumerate(events):
                print(i, ":", e.key)
            print(
                events.table(
                    sort_by="self_cuda_time_total",
                    max_src_column_width=200,
                    row_limit=15,
                )
            )
    else:
        events = None
        for i in range(steps):
            torch.cuda.nvtx.range_push(f"iteration-{i}")
            func(*args)
            torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()  # pop label

    # stop timer
    if cpu_timing:
        torch.cuda.synchronize()
        time_ms = ((time.perf_counter() - start) / steps) * 1000
    else:
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        end_event.synchronize()
        time_ms = start_event.elapsed_time(end_event) / steps

    if verbose:
        print(f"{label.ljust(label_padding)}: {time_ms:.3f} ms/step")

    if run_profile:
        return time_ms, events
    return time_ms
