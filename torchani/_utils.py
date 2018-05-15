import torch
import itertools


def cartesian_prod(*tensors, dim=0, newdim=0):
    """Compute the cartesian product along a given dimension and add a new dimension to specify input index.

    If, for example, there are two input tensors having shape (a,m,b), (a,n,b),
    and `dim=1`, `newdim=0`, then the resulting tensor would have shape (2,a,m*n,b)
    where the value at (0,i,j,k) of the resulting tensor would be the same as the value
    at (i,j//n,k) at the first tensor, and the value at (1,i,j,k) of the resulting tensor
    would be the same as the value at (i,j%n,k) at the second tensor.

    Example
    -------
        >>> a = torch.tensor([[1,2],
        >>>                   [3,4]])
        >>> b = torch.tensor([[5,6],
        >>>                   [7,8]])
        >>> cartesian_prod(a, b, dim=1, newdim=0)

        (0 ,.,.) = 
        1  1  2  2
        3  3  4  4

        (1 ,.,.) = 
        5  6  5  6
        7  8  7  8
        [torch.LongTensor of size (2,2,4)]


    Parameters
    ----------
    *tensors
        Input tensors. The shape of input tensors except at the dimension
        specified by `dim` must be the same.
    dim: int
        The dimension along which the cartesian product will be taken.
    newdim: int
        The dimension that specifies the input index

    Returns
    -------
    torch.Tensor
        The cartesian product of input tensors along specified dimension.
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
    """Compute the combination of a tensor along a given dimension and add a new dimension to specify input index.

    If, for example, if the input tensors having shape (a,m,b) and `repeat=2`, `dim=1`, `newdim=0`,
    then the resulting tensor would have shape (2,a,m*(m-1)/2,b) where the value at (0,i,j,k)
    of the resulting tensor would be the same as the value at (i,j',k) at the input tensor,
    and the value at (1,i,j,k) of the resulting tensor would be the same as the value at (i,j'',k)
    at the input tensor where (j',j'') would be the j-th element of itertools.combinations(range(m),2).

    Example
    -------
        >>> a = torch.tensor([[1,2],
        >>>                   [3,4]])
        >>> combinations(a, 2, dim=1, newdim=0)

        (0 ,.,.) = 
        1
        3

        (1 ,.,.) = 
        2
        4
        [torch.LongTensor of size (2,2,1)]

    Parameters
    ----------
    *tensor: pytorch tensor
        The input tensor.
    dim: int
        The dimension along which the combination will be taken.
    newdim: int
        The dimension that specifies the input index

    Returns
    -------
    torch.Tensor
        The combination of input tensors along specified dimension.
    """
    total_dims = len(tensor.shape)
    if newdim < 0:
        newdim += total_dims
    if dim < 0:
        dim += total_dims
    tensors = torch.unbind(tensor, dim)
    tensors = [torch.stack(t, (newdim - 1) if newdim > dim else newdim)
               for t in itertools.combinations(tensors, repeat)]
    return torch.stack(tensors, (dim + 1) if newdim <= dim else dim)


def combinations_with_replacement(tensor, repeat, dim=0, newdim=0):
    """Compute the combination with replacement of a tensor along a given dimension and add a new dimension to specify input index.

    If, for example, if the input tensors having shape (a,m,b) and `repeat=2`, `dim=1`, `newdim=0`,
    then the resulting tensor would have shape (2,a,m*(m+1)/2,b) where the value at (0,i,j,k)
    of the resulting tensor would be the same as the value at (i,j',k) at the input tensor,
    and the value at (1,i,j,k) of the resulting tensor would be the same as the value at (i,j'',k)
    at the input tensor where (j',j'') would be the j-th element of itertools.combinations_with_replacement(range(m),2).

    Example
    -------
        >>> a = torch.tensor([[1,2],
        >>>                   [3,4]])
        >>> combinations(a, 2, dim=1, newdim=0)

        (0 ,.,.) = 
        1  1  2
        3  3  4

        (1 ,.,.) = 
        1  2  2
        3  4  4
        [torch.LongTensor of size (2,2,3)]

    Parameters
    ----------
    *tensor: pytorch tensor
        The input tensor.
    dim: int
        The dimension along which the combination will be taken.
    newdim: int
        The dimension that specifies the input index

    Returns
    -------
    torch.Tensor
        The combination with replacement of input tensors along specified dimension.
    """
    total_dims = len(tensor.shape)
    if newdim < 0:
        newdim += total_dims
    if dim < 0:
        dim += total_dims
    tensors = torch.unbind(tensor, dim)
    tensors = [torch.stack(t, (newdim - 1) if newdim > dim else newdim)
               for t in itertools.combinations_with_replacement(tensors, repeat)]
    return torch.stack(tensors, (dim + 1) if newdim <= dim else dim)
