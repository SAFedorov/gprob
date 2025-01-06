import numpy as np
import scipy as sp

from .normal_ import Normal, lift
from .arrayops import concatenate


def sde(t, a, df, x0):
    """Solves an initial value problem for a system of linear stochastic 
    differential equations (SDE),
    
    ``dx = a @ x * dt + df,  x[0] = x0,``

    where x can be a scalar or a vector of the shape (k,). The shapes of ``a``, 
    ``df`` and ``x0`` must be consistent with the shapes of x and ``t``. 
    The solution error is second order in dt. 

    Args:
        t (array):
            The times at which the solution is evaluated, shape (n,).
        a (array):
            The deterministic evolution factor(s). When x is scalar, ``a`` 
            can be: 
            1. A scalar, giving the time-independent anti-damping constant.
            2. An array of the shape (n,), giving the anti-damping constant at  
            each point in time.
            When x is a vector of the length k, ``a`` can be:
            1. An array of the shape (k, k), giving the time-independent 
            system evolution matrix.
            2. An array of the shape (k, k, n), giving the system 
            matrix at each point in time.   
        df (Normal):
            The integral input noises: df[i] is the integral of f(t) between 
            t[i] and t[i+1]. The shape is (n,) when x is scalar and 
            (k, n-1) when x is a vector of the length k.
        x0 (Normal):
            The initial condition.
    
    Returns:
        Normal: A solution of the SDE x(t[i]) with the shape (n,) or 
        (k, n) depending on the dimensionality of x.
    """

    t = np.asanyarray(t)
    a = np.asanyarray(a)
    df = lift(Normal, df)
    x0 = lift(Normal, x0)

    if x0.ndim > 1:
        raise ValueError(f"x0 has {x0.ndim} dimensions, while it must be "
                         "a scalar or a vector.")

    n = len(t)
    shx0 = x0.shape

    if df.shape != shx0 + (n-1,):
        raise ValueError(f"The shape of df {df.shape} is inconsistent "
                         f"with the shape of x0 {shx0} and t {t.shape} "
                         f"- expecting {shx0 + (n-1,)}.")
    
    if (a.shape != shx0 + shx0) and (a.shape != shx0 + shx0 + (n,)):
        raise ValueError(f"The shape of a {a.shape} is inconsistent "
                         f"with the shape of x0 {shx0} and t {t.shape} "
                         f"- expecting {shx0 + shx0} or {shx0 + shx0 + (n,)}.")

    if x0.ndim == 0:
        return _sde_scalar(t, a, df, x0)

    k = len(x0)

    if a.ndim == 2:
        a_ = np.broadcast_to(a, (n, k, k))
    else:  
        # a.ndim == 3
        a_ = np.transpose(a, (2, 0, 1))

    dta = 0.5 * (a_[1:] + a_[:-1]) * np.reshape(t[1:] - t[:-1], (n-1, 1, 1))
    dta_ = np.roll(dta, -1, axis=0)

    e = np.eye(k)

    v0 = (e + dta[0] / 2) @ x0
    v = concatenate([v0, np.zeros(((n - 2) * k, ))])

    dia = np.concatenate([(e - dta / 2), (-e - dta_ / 2)], axis=-2)
    dia = np.transpose(dia, (0, 2, 1))
    zp = np.zeros((n-1, k, k-1))

    dia = np.reshape(np.concatenate([zp, dia], axis=-1), (n-1, -1))
    dia = np.concatenate([dia, np.zeros((n-1, k))], axis=-1)
    dia = np.reshape(dia, ((n-1)*k, 3 * k))[:, :-1].T

    x = _solve_banded((2*k - 1, k-1), dia, df.T.flatten() + v)
    x = x.reshape((n-1, k)).T

    return concatenate([x0.reshape((k, 1)), x], axis=1)


def _sde_scalar(t, a, df, x0):
    """Solves an initial value problem for a scalar SDE."""

    a_ = np.broadcast_to(a, (len(t),))
    dta = 0.5 * (t[1:] - t[:-1]) * (a_[1:] + a_[:-1])
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