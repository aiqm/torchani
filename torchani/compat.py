import warnings

# We try to install tqdm, if this doesn't work we fall back to pkbar,
# if all fails we just use a dummy object that does nothing
try:
    from tqdm.auto import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

try:
    from pkbar import Pbar
    _PKBAR_AVAILABLE = True
except ImportError:
    _PKBAR_AVAILABLE = False

if not _TQDM_AVAILABLE and _PKBAR_AVAILABLE:
    # wrapper to make pkbar behave like tqdm
    class tqdm:  # type: ignore  # noqa
        def __init__(self, iterable=None, desc=None, total=None, disable=False, **kwargs):
            self.pbar = Pbar(name=desc, target=total) if total is not None else None
            self.iterable = iterable
            self.disable = disable

        def update(self, n=1):
            if self.disable or self.pbar is None:
                return
            self.pbar.update(n)

        def __iter__(self):
            iterable = self.iterable
            if self.disable or self.pbar is None:
                yield from iterable
                return

            pbar = self.pbar
            for obj in iterable:
                yield obj
                pbar.update(1)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass
elif not _TQDM_AVAILABLE and not _PKBAR_AVAILABLE:
    # wrapper that does nothing
    class tqdm:  # type: ignore  # noqa
        def __init__(self, iterable=None, desc=None, total=None, disable=False, **kwargs):
            self.iterable = iterable
            pass

        def update(self, n=1):
            pass

        def __iter__(self):
            iterable = self.iterable
            yield from iterable

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass
    warnings.warn("tqdm or pkbar were not be found, for progress bars support install tqdm or pkbar")
