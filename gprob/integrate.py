import numpy as np
import scipy as sp

from .normal_ import Normal, lift
from .arrayops import concatenate, einsum


def sde(x0, t, a, df, dz=None):
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
    
    a_trial = np.array(a((t[0] + t[1]) / 2))

    if a_trial.shape != shx0 + shx0:
        raise ValueError(f"The shape of a is inconsistent "
                         f"with the shape of x0, {shx0}, "
                         f"- expecting {shx0 + shx0}, got {a_trial.shape}.")

    if dz is None:
        if x0.ndim == 0:
            return _sde_scalar2(x0, t, a, df)
        
        return  _sde2(x0, t, a, df)
    
    if x0.ndim == 0:
        return _sde_scalar4(x0, t, a, df, dz)
    
    return _sde4(x0, t, a, df, dz)


def _sde2(x0, t, a, df):
    n = len(t)

    # x0 is a vector in the following

    a_ = np.array([a((t1 + t2) / 2) for t1, t2 in zip(t[:-1], t[1:])])

    dta = a_ * np.reshape(t[1:] - t[:-1], (n-1, 1, 1))
    dta_ = np.roll(dta, -1, axis=0)

    k = len(x0)
    e = np.eye(k)

    v0 = (e + dta[0] / 2) @ x0
    v = concatenate([v0, np.zeros(((n - 2) * k,))])  # TODO: this line is the reason for importing concatenate from arrayops

    dia = np.concatenate([(e - dta / 2), (-e - dta_ / 2)], axis=-2)
    dia = np.transpose(dia, (0, 2, 1))
    zp = np.zeros((n-1, k, k-1))

    dia = np.reshape(np.concatenate([zp, dia], axis=-1), (n-1, -1))
    dia = np.concatenate([dia, np.zeros((n-1, k))], axis=-1)
    dia = np.reshape(dia, ((n-1)*k, 3 * k))[:, :-1].T

    x = _solve_banded((2*k - 1, k-1), dia, df.T.flatten() + v)
    x = x.reshape((n-1, k)).T

    return concatenate([x0.reshape((k, 1)), x], axis=1)


def _sde_scalar2(x0, t, a, df):
    """Solves an initial value problem for a scalar SDE."""

    a_ = np.array([a((t1 + t2) / 2) for t1, t2 in zip(t[:-1], t[1:])])

    dta = (t[1:] - t[:-1]) * a_
    dta_ = np.roll(dta, -1)

    dia = np.stack([(1 - dta / 2), (-1 - dta_ / 2)])
    v0 = (1 + dta[0] / 2) * x0
    v = concatenate([v0.reshape((1,)), np.zeros(shape=(len(t)-2,))])

    x_ = _solve_banded((1, 0), dia, df + v)
    return concatenate([x0.reshape((1,)), x_])


def _sde4(x0, t, a, df, dz):

    # The coefficients for the Butcher-Kuntzmann method of order 6.
    a_co = np.array([[5/36, 2/9 - np.sqrt(15)/15, 5/36 - np.sqrt(15)/30], 
                     [5/36 + np.sqrt(15)/24, 2/9, 5/36 - np.sqrt(15)/24], 
                     [5/36 + np.sqrt(15)/30, 2/9 + np.sqrt(15)/15, 5/36]])
    b_co = np.array([5/18, 4/9, 5/18])
    c_co = np.array([1/2 - np.sqrt(15)/10, 1/2, 1/2 + np.sqrt(15)/10])

    s = len(b_co)
    n = len(t)
    k = len(x0)
    e = np.eye(k)

    at = [[a(t1 + c*(t2 - t1)) for c in c_co] for t1, t2 in zip(t[:-1], t[1:])]
    
    # Transposing and reversing for finding the inverse Green's function.
    ar = -np.transpose(at, (0, 1, 3, 2))  # shape (n-1, s, k, k)

    dt = np.reshape(t[1:] - t[:-1], (n-1, 1, 1))
    ar_ = np.reshape(ar, (n-1, s, k, 1, k))
    a_co_ = np.reshape(a_co, (s, 1, s, 1))

    # The shape of ar_ * a_co_ is (n-1, s, k, s, k).
    
    m = np.eye(s * k) - dt * np.reshape(ar_ * a_co_, (n-1, s * k, s * k))

    sol = np.linalg.solve(m, np.reshape(ar, (n-1, s * k, k)))
    ka = np.reshape(sol, (n-1, s, k, k))

    kb = e + np.einsum("ijkl, imj -> imkl", ka, dt * a_co)  # (n-1, s, k, k)
    
    g_inv = e + dt * np.einsum("ijkl, j -> ilk", ka, b_co)
    g_inv_i0 = np.einsum("ijkl, j -> ilk", kb, b_co)
    g_inv_i1 = np.einsum("ijkl, j -> ilk", kb, b_co * (c_co - 1/2)) / (dt / 12)

    v0 = concatenate([x0, np.zeros(((n - 2) * k,))])

    dia = np.concatenate([g_inv, -np.broadcast_to(e, (n-1, k, k))], axis=-2)
    dia = np.transpose(dia, axes=(0, 2, 1))
    zp = np.zeros((n-1, k, k-1))

    dia = np.reshape(np.concatenate([zp, dia], axis=-1), (n-1, -1))
    dia = np.concatenate([dia, np.zeros((n-1, k))], axis=-1)
    dia = np.reshape(dia, ((n-1)*k, 3 * k))[:, :-1].T

    rhs = (einsum("ikj, ji -> ik", g_inv_i0, df) 
           + einsum("ikj, ji -> ik", g_inv_i1, dz))

    x = _solve_banded((2*k - 1, k-1), dia, rhs.flatten() + v0)
    x = x.reshape((n-1, k)).T

    return concatenate([x0.reshape((k, 1)), x], axis=1)


def _sde_scalar4(x0, t, a, df, dz):
    x0 = x0.reshape((1,))
    a_ = lambda t: np.reshape(a(t), (1, 1))
    df = df.reshape((1, -1))
    dz = dz.reshape((1, -1))
    sol = _sde4(x0, t, a_, df, dz)
    return sol.squeeze()


def fmul(g, t, df, dz):
    """Multiplies the stochastic function f(t) given by its increments df and dz 
    on the grid t by a deterministic twice differentiable function g(t)"""
    
    # Using Legendre polynomial decomposition. 6th order.

    b_co = np.array([5/18, 4/9, 5/18])
    c_co = np.array([1/2 - np.sqrt(15)/10, 1/2, 1/2 + np.sqrt(15)/10])

    gt = np.array([[b * g(t1 + c * (t2 - t1)) for b, c in zip(b_co, c_co)] 
                   for t1, t2 in zip(t[:-1], t[1:])])

    gi0 = np.sum(gt, axis=1)
    gi1 = np.sum(gt * (c_co - 1/2), axis=1)
    gi2 = np.sum(gt * 12 * (c_co - 1/2)**2, axis=1)

    dt = t[1:] - t[:-1]
    df_ = df * gi0 + dz * gi1 / (dt / 12)
    dz_ = dz * gi2 + df * gi1 * dt

    return df_, dz_


def _solve_banded(sig, x, y):
    b = sp.linalg.solve_banded(sig, x, y.b, check_finite=False)
    a = sp.linalg.solve_banded(sig, x, y.a.T, check_finite=False).T
    return Normal(a, b, y.lat)