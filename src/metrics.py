import numpy as np
import torch
import itertools

def spearman_footrule_distance(s, t, normalized=True):
    """
    Computes the Spearman footrule distance between two full lists of ranks:
    
    F(s,t) = sum[ |s(i) - t(i)| ]/S,
    
    the normalized sum over all elements in a set of the absolute difference between
    the rank according to s and t. As defined, 0 <= F(s,t) <= 1.
    
    S is a normalizer which is equal to 0.5*len(s)^2 for even length ranklists and
    0.5*(len(s)^2 - 1) for odd length ranklists.
    
    If s,t are *not* full, this function should not be used. s,t should be array-like
    (lists are OK).
    From: https://github.com/thelahunginjeet/kbutil/blob/master/statistics.py
          @author: Kevin S. Brown, University of Connecticut
    """
    # print("*"*20)
    # print(s)
    # print(t)
    # print("*"*20)
    if isinstance(s, list) and isinstance(t, list):
        # check that size of intersection = size of s,t?
        assert len(s) == len(t)
        s_len = len(s)
        sdist = sum(abs(np.asarray(s) - np.asarray(t)))
    elif isinstance(s, np.ndarray) and isinstance(t, np.ndarray):
        assert s.size == t.size
        s_len = s.size
        #sdist = sum(abs(s - t))
        sdist = np.abs(s - t).sum()
    elif isinstance(s, torch.Tensor) and isinstance(t, torch.Tensor):
        assert s.size == t.size
        s_len = s.size()[0]
        sdist = (s - t).abs().sum().item()
    else:
        raise TypeError(
            "Boot inputs should be of type 'list', 'array' or 'tensor'."
        )
    # c will be 1 for odd length lists and 0 for even ones
    if normalized:
        c = s_len % 2
        normalizer = 0.5 * (s_len ** 2 - c)
        sdist = sdist / normalizer

    return sdist


def kendall_tau_distance(s, t, normalized=True, both=False):
    """
    Computes the Kendall's tau distance bertween two full list of ranks:
    K(s,t) = 
    """
    assert len(s) == len(t)
    pairs = itertools.combinations(range(0, len(s)), 2)
    kt_distance = 0
    for x, y in pairs:
        a = s.index(x) - s.index(y)
        b = t.index(x) - t.index(y)
        if (a*b < 0):
            kt_distance += 1

    normalizer = len(s)*(len(s)-1)/2
    if normalized and both:
        return (kt_distance,  kt_distance/normalizer)
    elif normalized:
        return kt_distance/normalizer

    return kt_distance
