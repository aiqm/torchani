import torch
import itertools


def cartesian_prod(*tensors, dim=0, newdim=0):
    """Compute the cartesian product along a given dimension.

    If, for example, there are two input tensors having shape (a,m,b), (a,n,b),
    and `dim=1`, `newdim=0`, then the resulting tensor would have shape (2,a,m*n,b)
    where the value at (0,i,j,k) of the resulting tensor would be the same as the value
    at (i,j/n,k) at the first tensor, and the value at (1,i,j,k) of the resulting tensor
    would be the same as the value at (i,j%n,k) at the second tensor.

    Example
    -------
        >>> a = torch.tensor([[1,2],
        >>>                   [3,4]])
        >>> b = torch.tensor([[5,6],
        >>>                   [7,8]])
        >>> cartesian_prod(a, b, dim=1, newdim=0)

    """
    total_dims = len(tensors[0].shape)
    if newdim < 0:
        newdim += total_dims
    if dim < 0:
        dim += total_dims
    tensors = [torch.unbind(t, dim) for t in tensors]
    tensors = [torch.stack(t, (newdim - 1) if newdim > dim else newdim)
               for t in itertools.product(*tensors)]
    return torch.stack(tensors, (dim + 1) if newdim <= dim else dim)


def combinations(tensor, repeat, dim=0, newdim=0):
    total_dims = len(tensor.shape)
    if newdim < 0:
        newdim += total_dims
    if dim < 0:
        dim += total_dims
    tensors = torch.unbind(tensor, dim)
    tensors = [torch.stack(t, (newdim - 1) if newdim > dim else newdim)
               for t in itertools.combinations(t, repeat)]
    return torch.stack(tensors, (dim + 1) if newdim <= dim else dim)


def combinations_with_replacement(tensor, repeat, dim=0, newdim=0):
    total_dims = len(tensor.shape)
    if newdim < 0:
        newdim += total_dims
    if dim < 0:
        dim += total_dims
    tensors = torch.unbind(tensor, dim)
    tensors = [torch.stack(t, (newdim - 1) if newdim > dim else newdim)
               for t in itertools.combinations_with_replacement(t, repeat)]
    return torch.stack(tensors, (dim + 1) if newdim <= dim else dim)
