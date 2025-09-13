import numpy as np
import scipy as sp

from .normal_ import Normal, lift
from .arrayops import concatenate


def sde(x0, t, a, df):
    """Solves an initial value problem for a system of linear stochastic 
    differential equations (SDE),
    
    ``dx = a(t) @ x * dt + df,  x[0] = x0,``

    where ``x`` is a scalar or vector sulution of the SDE, ``x0`` is the initial
    condition, and ``df`` is the Wiener increments of the driving force over 
    the intervals between the points of the time grid ``t``. 
    The error of the solution is second order in dt. 

    Args:
        x0 (numeric or Normal):
            The initial condition, a scalar or a 1D vector with the shape (k,).
        t (array):
            The times at which the solution is evaluated, shape (n,). Must
            be sorted in ascending order.
        a (callable):
            The deterministic evolution factor as function of time. 
            The value returned by the function ``a`` should be a scalar 
            if ``x`` is scalar or a (k, k) matrix if ``x`` is a (k,) vector.
        df (numeric or Normal):
            The force increments. When ``x`` is scalar, the shape of ``df`` 
            is (n-1,) and ``df[i]`` is the integral of ``f(t)`` from ``t[i]`` 
            to ``t[i+1]``. When ``x`` is a vector of the length k, the shape 
            of ``df`` is (k, n-1), and ``df[j, i]`` is the integral 
            of ``f[j](t)`` from ``t[i]`` to ``t[i+1]``.
    
    Returns:
        Normal: A solution of the SDE, ``x(t[i])``, with the shape (n,) or 
        (k, n) depending on the dimension of ``x``.
    """

    x0 = lift(Normal, x0)
    shx0 = x0.shape

    if x0.ndim > 1:
        raise ValueError(f"x0 has {x0.ndim} dimensions, while it must be "
                         "a scalar or a vector.")
    
    t = np.asanyarray(t)
    n = len(t)

    if n < 2:
        raise ValueError("The time grid must have at least two points. "
                         f"Now it has {n}.")

    df = lift(Normal, df)

    if df.shape != shx0 + (n-1,):
        raise ValueError(f"The shape of df is inconsistent "
                         f"with the shape of x0, {shx0}, and t, {t.shape}, "
                         f"- expecting {shx0 + (n-1,)}, got {df.shape}.")
    
    a_ = np.array([a(tp) for tp in t])

    if a_[0].shape != shx0 + shx0:
        raise ValueError(f"The shape of a is inconsistent "
                         f"with the shape of x0, {shx0}, "
                         f"- expecting {shx0 + shx0}, got {a_[0].shape}.")

    if x0.ndim == 0:
        return _sde_scalar(x0, t, a_, df)

    # x0 is a vector in the following

    dta = 0.5 * (a_[1:] + a_[:-1]) * np.reshape(t[1:] - t[:-1], (n-1, 1, 1))
    dta_ = np.roll(dta, -1, axis=0)

    k = len(x0)
    e = np.eye(k)

    v0 = (e + dta[0] / 2) @ x0
    v = concatenate([v0, np.zeros(((n - 2) * k,))])

    dia = np.concatenate([(e - dta / 2), (-e - dta_ / 2)], axis=-2)
    dia = np.transpose(dia, (0, 2, 1))
    zp = np.zeros((n-1, k, k-1))

    dia = np.reshape(np.concatenate([zp, dia], axis=-1), (n-1, -1))
    dia = np.concatenate([dia, np.zeros((n-1, k))], axis=-1)
    dia = np.reshape(dia, ((n-1)*k, 3 * k))[:, :-1].T

    x = _solve_banded((2*k - 1, k-1), dia, df.T.flatten() + v)
    x = x.reshape((n-1, k)).T

    return concatenate([x0.reshape((k, 1)), x], axis=1)


def _sde_scalar(x0, t, a, df):
    """Solves an initial value problem for a scalar SDE."""

    dta = 0.5 * (t[1:] - t[:-1]) * (a[1:] + a[:-1])
    dta_ = np.roll(dta, -1)

    dia = np.stack([(1 - dta / 2), (-1 - dta_ / 2)])
    v0 = (1 + dta[0] / 2) * x0
    v = concatenate([v0.reshape((1,)), np.zeros(shape=(len(t)-2,))])

    x_ = _solve_banded((1, 0), dia, df + v)
    return concatenate([x0.reshape((1,)), x_])


def _solve_banded(sig, x, y):
    b = sp.linalg.solve_banded(sig, x, y.b, check_finite=False)
    a = sp.linalg.solve_banded(sig, x, y.a.T, check_finite=False).T
    return Normal(a, b, y.lat)