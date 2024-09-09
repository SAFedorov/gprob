import numpy as np

from .arrayops import resolve


def solve(a, b):
    """Solves a system of linear equations where the coefficient 
    matrix is ``a`` and the constant vector is ``b``.

    Args:
        a (numeric matrix):
            A two-dimensional full-rank matrix with the shape (m, m).
        b (random variable):
            The random variable to be solved for. Must have the shape (m,) 
            or (m, k).

    Returns:
        A new random variable, ``a^-1 @ b``, with the same shape as ``b``.

    Note:
        This function operates similarly to `numpy.linalg.solve`, but
        does not support arrays of system matrices. For solving arrays 
        of equations, see `asolve`.
    """

    if np.ndim(a) != 2:
        raise ValueError("The system matrix `a` must have 2 dimensions. "
                         f"Now it has {np.ndim(a)}.")
    
    if np.ndim(b) != 1 and np.ndim(b) != 2:
        raise ValueError("The constant `b` must have 1 or 2 dimensions. "
                         f"Now it has {np.ndim(b)}.")
    
    mod, cls = resolve([b])
    b = cls._mod.lift(cls, b)
    return mod.solve(cls, a, b)


# The reason for having a separate array-solve function is that the broadcasting 
# rules for `b` operands of `solve` changed between numpy versions 1.26 and 2.0.
# The dedicated batch function allows supporting numpy versions older than 2.0. 

def asolve(a, b):
    """Solves an array of systems of linear equations where the coefficient 
    matrices are ``a`` and the constant vectors are ``b``.

    Args:
        a (array of numeric matrix):
            An array of two-dimensional full-rank matrices with the shape 
            (..., m, m).
        b (random variable):
            The random variable to be solved for with the shape (..., m).

    Returns:
        A new random variable with the same shape as ``b``.
    """

    if np.ndim(a) < 2:
        raise ValueError("The array of system matrices `a` must have at least "
                         f"2 dimensions. Now it has {np.ndim(a)}.")
    
    if np.ndim(a) != np.ndim(b) + 1:
        raise ValueError("The array of constants `b` must have one less "
                         f"dimension than `a`. Now `a` and `b` have "
                         f"{np.ndim(a)} and {np.ndim(b)} dimensions, "
                         "repectively.")

    mod, cls = resolve([b])
    b = cls._mod.lift(cls, b)
    return mod.asolve(cls, a, b)