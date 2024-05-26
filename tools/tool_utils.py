from enum import Enum
import numpy as np
from copy import deepcopy
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


class Timer:
    def __init__(
        self,
        modules: tp.List[torch.nn.Module],
        nvtx: bool = False,
        sync: bool = True,
        extra_title: str = "",
        reduction: str = "median",
        units: str = "ms",
    ) -> None:
        self.modules = modules
        self.module_names = tuple(m.__class__.__name__ for m in modules)
        self.nvtx = nvtx
        self.sync = sync
        self.timers: tp.Dict[str, tp.List[float]] = defaultdict(list)
        self._last_timers: tp.Dict[str, tp.List[float]] = defaultdict(list)
        self.batch_counter = 0
        self.is_profiling = False
        if units == "ns":
            self.factor = 1.0
        elif units == "mus":
            self.factor = 1e-3
        elif units == "ms":
            self.factor = 1e-6
        elif units == "s":
            self.factor = 1e-9
        else:
            raise ValueError("Unsupported units")
        self.units = units
        reduction_fn: tp.Callable[..., tp.Any]
        if reduction == "sum":
            reduction_fn = np.sum
        elif reduction == "mean":
            reduction_fn = np.mean
        elif reduction == "median":
            reduction_fn = np.median
        else:
            raise ValueError(f"Unknown reduction {reduction}")
        self.reduction_fn = reduction_fn
        self.reduction_title = reduction.replace("sum", "total").capitalize()
        if extra_title:
            extra_title = f", {extra_title}"
        self.extra_title = extra_title

    def start_profiling(self) -> None:
        self.batch_counter = 0
        self.is_profiling = True
        for m in self.modules:
            self.time_module(m)
        if self.nvtx:
            torch.cuda.profiler.start()

    def start_loss(self) -> None:
        if not self.is_profiling:
            return
        if self.sync:
            torch.cuda.synchronize()
        if self.nvtx:
            torch.cuda.nvtx.range_push("loss-bw")
        self.timers["loss-bw"].append(time.perf_counter_ns())

    def end_loss(self) -> None:
        if not self.is_profiling:
            return
        if self.sync:
            torch.cuda.synchronize()
        if self.nvtx:
            torch.cuda.nvtx.range_pop()
        self.timers["loss-bw"][-1] = (
            time.perf_counter_ns() - self.timers["loss-bw"][-1]
        ) * self.factor

    def start_opt(self) -> None:
        if not self.is_profiling:
            return
        if self.sync:
            torch.cuda.synchronize()
        if self.nvtx:
            torch.cuda.nvtx.range_push("opt-step")
        self.timers["opt-step"].append(time.perf_counter_ns())

    def end_opt(self) -> None:
        if not self.is_profiling:
            return
        if self.sync:
            torch.cuda.synchronize()
        if self.nvtx:
            torch.cuda.nvtx.range_pop()
        self.timers["opt-step"][-1] = (
            time.perf_counter_ns() - self.timers["opt-step"][-1]
        ) * self.factor

    def start_batch(self) -> None:
        if not self.is_profiling:
            return
        if self.sync:
            torch.cuda.synchronize()
        if self.nvtx:
            torch.cuda.nvtx.range_push(f"batch-{self.batch_counter}")
        self.batch_counter += 1
        self.timers["batch"].append(time.perf_counter_ns())

    def end_batch(self) -> None:
        if not self.is_profiling:
            return
        if self.sync:
            torch.cuda.synchronize()
        if self.nvtx:
            torch.cuda.nvtx.range_pop()
        self.timers["batch"][-1] = (
            time.perf_counter_ns() - self.timers["batch"][-1]
        ) * self.factor

    def stop_profiling(self) -> None:
        self.batch_counter = 0
        self.is_profiling = False
        if self.nvtx:
            torch.cuda.profiler.stop()
        # Reset timers
        self._last_timers = deepcopy(self.timers)
        self.timers = defaultdict(list)

    def display(self) -> None:
        console = Console()
        table = Table(
            title="".join(
                (
                    f"{self.reduction_title} times",
                    self.extra_title,
                )
            ),
            box=None,
        )
        table.add_column("module", style="magenta")
        table.add_column(f"forward ({self.units})", style="green")
        table.add_column(f"backward ({self.units})", style="blue")
        for name in self.module_names:
            fw_values = self._last_timers[".".join((name, "forward"))]
            fw = self.reduction_fn(np.array(fw_values)) if fw_values else 0.0
            bw_values = self._last_timers[".".join((name, "backward"))]
            bw = self.reduction_fn(np.array(bw_values)) if bw_values else 0.0
            table.add_row(name, f"{fw:.5f}", f"{bw:.5f}")
        if self.modules:
            console.print(
                "WARNING: Backward times are unreliable. Hooks, sync create overhead",
                style="yellow",
            )
            console.print(table)
        if "batch" in self._last_timers:
            value = self._last_timers["batch"]
            _time = self.reduction_fn(np.array(value))
            console.print(
                f"Batch {self.reduction_title} time ({self.units}): {_time:.5f}"
            )
        if "opt-step" in self._last_timers:
            value = self._last_timers["opt-step"]
            _time = self.reduction_fn(np.array(value))
            console.print(
                f"Optimizer {self.reduction_title} time ({self.units}): {_time:.5f}"
            )
        if "loss-bw" in self._last_timers:
            value = self._last_timers["loss-bw"]
            _time = self.reduction_fn(np.array(value))
            console.print(
                f"Loss bw {self.reduction_title} time ({self.units}): {_time:.5f}"
            )

    def start_fw(self, module, args=None, kwargs=None, output=None) -> None:
        self._start(module, label="forward")

    def end_fw(self, module, args=None, kwargs=None, output=None) -> None:
        self._end(module, label="forward")

    def start_bw(self, module, grad_output=None) -> None:
        self._start(module, label="backward")

    def end_bw(self, module, grad_input=None, grad_output=None) -> None:
        self._end(module, label="backward")

    def _start(self, module, label) -> None:
        name = ".".join((module.__class__.__name__, label))
        if self.nvtx:
            torch.cuda.nvtx.range_push(name)
        if self.sync:
            torch.cuda.synchronize()
        self.timers[name].append(time.perf_counter_ns())

    def _end(self, module, label) -> None:
        name = ".".join((module.__class__.__name__, label))
        if self.nvtx:
            torch.cuda.nvtx.range_pop()
        if self.sync:
            torch.cuda.synchronize()
        self.timers[name][-1] = (
            time.perf_counter_ns() - self.timers[name][-1]
        ) * self.factor

    def time_module(self, module) -> None:
        module.register_forward_pre_hook(self.start_fw)
        module.register_forward_hook(self.end_fw)

        module.register_full_backward_pre_hook(self.start_bw)
        module.register_full_backward_hook(self.end_bw)


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
    torch.cuda.nvtx.range_push(f"{label}_warmup")
    for _ in range(warmup):
        func(*args)
    torch.cuda.nvtx.range_pop()  # pop label_warmup

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
                    torch.cuda.nvtx.range_push(f"{i}th_iteration")
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
            torch.cuda.nvtx.range_push(f"{i}th_iteration")
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
