from typing import Callable
import torch
import torch.nn.functional as F
from typing import Callable

def _boolrelextrema(data: torch.Tensor, comparator: Callable, dim=0, order=1, mode='clip'):
    """
    Calculate the relative extrema of `data`.
    Relative extrema are calculated by finding locations where
    ``comparator(data[n], data[n+1:n+order+1])`` is True.
    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    dim : int, optional
        Axis over which to select from `data`. Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n,n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated. 'wrap' (wrap around) or
        'clip' (treat overflow as the same as the last (or first) element).
        Default 'clip'. See numpy.take.
    Returns
    -------
    extrema : ndarray
        Boolean array of the same shape as `data` that is True at an extrema,
        False otherwise.
    See also
    --------
    argrelmax, argrelmin
    Examples
    --------
    >>> testdata = np.array([1,2,3,2,1])
    >>> _boolrelextrema(testdata, np.greater, axis=0)
    array([False, False,  True, False, False], dtype=bool)
    """
    if((int(order) != order) or (order < 1)):
        raise ValueError('Order must be an int >= 1')

    datalen = data.shape[dim]
    locs = torch.arange(0, datalen, dtype=torch.int32, device=data.device)
    results = torch.ones_like(data, dtype=torch.bool)
    main = data.index_select(dim=dim, index=locs)
    for shift in range(1, order + 1):
        if mode.lower() == 'clip':
            locs_plus = torch.where(
                locs + shift > locs[-1],
                locs[-1],
                locs + shift
                )
            locs_minus = torch.where(
                locs - shift < 0,
                locs[0],
                locs - shift
            )
        elif mode.lower() == 'wrap':
            locs_plus = torch.where(
                locs + shift > locs[-1],
                (locs + shift) % datalen,
                locs + shift
                )
            locs_minus = torch.where(
                locs - shift < 0,
                (locs - shift) + datalen,
                locs - shift
            )
        plus = data.index_select(dim=dim, index=locs_plus)
        minus = data.index_select(dim=dim, index=locs_minus)
        torch.logical_and(results, comparator(main, plus), out=results)
        torch.logical_and(results, comparator(main, minus), out=results)
       
    return results

def relextrema2d(data: torch.Tensor, comparator: Callable, dims: tuple = (-2, -1), order=1, mode='clip'):
    mask = torch.ones_like(data, dtype=torch.bool)
    for d in dims:
        torch.logical_and(_boolrelextrema(data, comparator, d, order, mode), mask, out=mask)
    return mask