import torch
from torch.utils import cpp_extension
from sysconfig import get_paths

pt = torch.utils.cpp_extension.include_paths(cuda=True)
py = get_paths()
dirs = ('', *pt, py['include'])
print(' -I'.join(dirs))
