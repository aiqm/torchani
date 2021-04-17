import sys
# This is needed for compatibility with python3.6, in python 3.6 torch.jit
# Final doesn't work correctly

if sys.version_info[:2] < (3, 7):

    class FakeFinal:
        def __getitem__(self, x):
            return x

    Final = FakeFinal()
else:
    from torch.jit import Final # noqa
