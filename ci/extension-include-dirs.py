from torch.utils import cpp_extension
from sysconfig import get_paths

pt = cpp_extension.include_paths(cuda=True)
py = get_paths()
dirs = ('', *pt, py['include'])
print(' -I'.join(dirs))
