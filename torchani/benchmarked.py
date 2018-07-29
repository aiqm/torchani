import torch
import timeit
import functools


class BenchmarkedModule(torch.nn.Module):
    """Module with member function benchmarking support.

    The benchmarking is done by wrapping the original member function with
    a wrapped function. The wrapped function will call the original function,
    and accumulate its running time into `self.timers`. Different accumulators
    are distinguished by different keys. All times should have unit seconds.

    To enable benchmarking for member functions in a subclass, simply
    call the `__init__` of this class with `benchmark=True`, and add the
    following code to your subclass's `__init__`:

    ```
    if self.benchmark:
        self._enable_benchmark(self.function_to_be_benchmarked, 'key1', 'key2')
    ```

    Example
    -------
    The following code implements a subclass for timing the running time of
    member function `f` and `g` and the total of these two::
    ```
    class BenchmarkFG(BenchmarkedModule):
        def __init__(self, benchmark=False)
            super(BenchmarkFG, self).__init__(benchmark)
            if benchmark:
                self.f = self._enable_benchmark(self.f, 'function f', 'total')
                self.g = self._enable_benchmark(self.g, 'function g', 'total')

        def f(self):
            print('in function f')

        def g(self):
            print('in function g')
    ```

    Attributes
    ----------
    benchmark : boolean
        Whether benchmark is enabled
    timers : dict
        Dictionary storing the the benchmark result.
    """

    def _enable_benchmark(self, fun, *keys):
        """Wrap a function to automatically benchmark it, and assign a key
        for it.

        Parameters
        ----------
        keys
            The keys in `self.timers` assigned. If multiple keys are specified,
            then the time will be accumulated to all the keys.
        func : function
            The function to be benchmarked.

        Returns
        -------
        function
            Wrapped function that time the original function and update the
            corresponding value in `self.timers` automatically.
        """
        for key in keys:
            self.timers[key] = 0

        @functools.wraps(fun)
        def wrapped(*args, **kwargs):
            start = timeit.default_timer()
            ret = fun(*args, **kwargs)
            end = timeit.default_timer()
            for key in keys:
                self.timers[key] += end - start
            return ret
        return wrapped

    def reset_timers(self):
        """Reset all timers. If benchmark is not enabled, a `ValueError`
        will be raised."""
        if not self.benchmark:
            raise ValueError('Can not reset timers, benchmark not enabled')
        for i in self.timers:
            self.timers[i] = 0

    def __init__(self, benchmark=False):
        super(BenchmarkedModule, self).__init__()
        self.benchmark = benchmark
        if benchmark:
            self.timers = {}
