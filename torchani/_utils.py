import torch
import itertools


def meshgrid(x, y=None):
    if y is None:
        y = x
    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    m, n = x.size(0), y.size(0)
    grid_x = x[None].expand(n, m)
    grid_y = y[:, None].expand(n, m)
    return grid_x, grid_y


def combinations(tensor, dim=0):
    n = tensor.shape[dim]
    r = torch.arange(n).type(torch.long).to(tensor.device)
    grid_x, grid_y = meshgrid(r)
    index1 = grid_y[torch.triu(torch.ones(n, n), diagonal=1) == 1]
    index2 = grid_x[torch.triu(torch.ones(n, n), diagonal=1) == 1]
    if torch.numel(index1) == 0:
        # TODO: pytorch are unable to handle size 0 tensor well. Is this an expected behavior?
        # See: https://github.com/pytorch/pytorch/issues/5014
        return None
    return tensor.index_select(dim, index1), tensor.index_select(dim, index2)
