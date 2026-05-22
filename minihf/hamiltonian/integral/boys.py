import numpy as np
import scipy.special as sc
from functools import lru_cache


@lru_cache(maxsize=8192)
def boys(n, x):
    """
    Boys function

        F_n(x)

    using incomplete gamma function.

    Parameters
    ----------
    n : int
    x : float

    Returns
    -------
    float
    """

    if x < 1e-12:

        return 1.0 / (2 * n + 1)

    return (
        0.5
        * x**(-(n + 0.5))
        * sc.gammainc(
            n + 0.5,
            x,
        )
        * sc.gamma(n + 0.5)
    )


def boys_recursion(n, x, fn):
    """ Downward recursion F_{n-1}(x) from F_n(x)
    
    Parameters
    ----------
    n : int
    x : float
    fn : callable

    Returns
    -------
    float
    """
    return (np.exp(-x)+ 2.0 * x * fn) / (2.0 * n - 1.0)