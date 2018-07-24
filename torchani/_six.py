# file for python 2 compatibility

import math

if not hasattr(math, 'inf'):
    math.inf = float('inf')
