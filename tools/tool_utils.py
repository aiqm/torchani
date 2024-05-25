import typing as tp
import time

import torch

from torch.profiler import record_function, ProfilerActivity


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
