import torch
from torch.utils import cpp_extension

d = torch.utils.cpp_extension.include_paths(cuda=True)
print(' -I'.join([''] + d))