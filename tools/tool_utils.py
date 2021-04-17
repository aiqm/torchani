import torch
import timeit


def time_func(key, func, timers=None, synchronize=False, nvtx=False):
    if nvtx:
        def wrapper(*args, **kwargs):
            torch.cuda.nvtx.range_push(key)
            ret = func(*args, **kwargs)
            torch.cuda.nvtx.range_pop()
            return ret
    else:
        assert timers is not None
        timers[key] = 0.0

        def wrapper(*args, **kwargs):
            if synchronize:
                torch.cuda.synchronize()
            start = timeit.default_timer()
            ret = func(*args, **kwargs)
            if synchronize:
                torch.cuda.synchronize()
            end = timeit.default_timer()
            timers[key] += end - start
            return ret

    return wrapper


def time_functions_in_model(model, function_names_list, timers=None, synchronize=False, nvtx=False):
    # Wrap all the functions from "function_names_list" from the model
    # "model" with a timer
    for n in function_names_list:
        setattr(model, n, time_func(model.__class__.__name__ + '.' + n, getattr(model, n), timers, synchronize, nvtx))
