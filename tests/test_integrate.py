import pytest
import numpy as np
import gprob as gp

from utils import get_message


def _true_cov_ou(t, x0):
    """Covariance matrix of an Ornstein-Uhlenbeck process with unit damping 
    rate and unit diffusion constant. The stationary variance is 0.5. """

    t1 = np.reshape(t, (len(t), 1))
    t2 = np.reshape(t, (1, len(t)))
    e1 = np.exp(-(t1 + t2))
    e2 = np.exp(-np.abs(t1 - t2))
    return gp.var(x0) * e1 + (e2 - e1) / 2


def _true_cov_ou_td(t, x0):
    # Time-dependent damping and diffusion constants.
    t1 = np.reshape(t, (len(t), 1))
    t2 = np.reshape(t, (1, len(t)))
    e1 = np.exp((t1**2 + t2**2) / 2)
    e2 = np.minimum(t1, t2)
    return e1 * (gp.var(x0) + e2)


def _true_cov_ou_cn(t, x0):
    # Non-white input noise.
    t1 = np.reshape(t, (len(t), 1))
    t2 = np.reshape(t, (1, len(t)))
    e1 = np.exp((t1**2 + t2**2) / 2)
    e2 = np.exp(-t1) + np.exp(-t2) - np.exp(-np.abs(t1 - t2)) + 2 * np.minimum(t1, t2) - 1
    return e1 * (gp.var(x0) + e2)


def _noise_cov_cn(t):
    # The covariance matrix for the input noise in _true_cov_ou_cn.
    t1 = np.reshape(t, (len(t), 1))
    t2 = np.reshape(t, (1, len(t)))
    return np.exp((t1**2 + t2**2) / 2 - np.abs(t1-t2))


def _error(tcov, cov):
    """The maximum relative deviation of the covariance ``cov`` from the 
    true covariance, ``tcov``"""
    return np.max(np.abs(1 - cov/tcov))


def test_sde():
    # Tests for input/output formats and errors.

    # Non-array inputs, degenerate cases.

    sz = 100
    t = [(i / sz)**2 for i in range(sz)]

    # Deterministic force.

    t_ = np.array(t)
    g = 1.5 + 1.1j
    f = np.exp(g * t_)
    df = 0.5 * (f[1:] + f[:-1]) * (t_[1:] - t_[:-1])
    sol_ref =  np.exp(-t_) + (np.exp(g *t_) - np.exp(-t_)) / (g + 1)

    # (2,) vector.
    x = gp.integrate.sde(t, [[-1, 0], [0, -1]], [1.5 * df, df], [1, 2])
    sol_ref =  [np.exp(-t_) + 1.5 * (np.exp(g *t_) - np.exp(-t_)) / (g + 1),
                2. * np.exp(-t_) + (np.exp(g *t_) - np.exp(-t_)) / (g + 1)]
    assert np.max(np.abs(x.mean() / sol_ref - 1)) < 4.9e-5  # 4.812e-05
    assert np.max(np.abs(x.cov())) < 1e-15

    # Scalar.
    x = gp.integrate.sde(t, -1, df, 1)
    sol_ref =  np.exp(-t_) + (np.exp(g *t_) - np.exp(-t_)) / (g + 1)
    assert np.max(np.abs(x.mean() / sol_ref - 1)) < 4.7e-5  # 4.62e-05
    assert np.max(np.abs(x.cov())) < 1e-15

    del f, df, g, t_, sol_ref

    # Zero force - free decay.

    # (1,) vector - I.
    df = [0 for _ in range(sz-1)]
    x = gp.integrate.sde(t, [[-1]], [df], [1]).squeeze(0)

    assert np.max(np.abs(x.mean() * np.exp(t) - 1)) < 1.7e-5  # 1.60e-05
    assert np.max(np.abs(x.cov())) < 1e-15

    # (1,) vector - II.
    df = [0 for _ in range(sz-1)]
    x = gp.integrate.sde(t, [[[-1 for _ in range(sz)]]], [df], [1]).squeeze(0)

    assert np.max(np.abs(x.mean() * np.exp(t) - 1)) < 1.7e-5  # 1.60e-05
    assert np.max(np.abs(x.cov())) < 1e-15
    
    # Scalar.
    df = [0 for _ in range(sz-1)]
    x = gp.integrate.sde(t, -1, df, 1)

    assert np.max(np.abs(x.mean() * np.exp(t) - 1)) < 1.7e-5  # 1.60e-05
    assert np.max(np.abs(x.cov())) < 1e-15

    # Wrong dimension of x0:
    with pytest.raises(ValueError) as e:
        gp.integrate.sde(t, -1, df, [[1]])

    assert "2 dimensions" in get_message(e)

    # Incompatible a and t
    with pytest.raises(ValueError) as e:
        gp.integrate.sde(t, [-1, -1], df, 1)

    # Checks that the size hint is right.
    assert (f"()" in get_message(e)) and (f"({sz},)" in get_message(e))

    # Incompatible a and x0
    with pytest.raises(ValueError) as e:
        gp.integrate.sde(t, -1, [df], [1])
    
    assert ("(1, 1)" in get_message(e)) and (f"(1, 1, {sz})" in get_message(e))

    with pytest.raises(ValueError) as e:
        gp.integrate.sde(t, [[-1]], df, 1)
    
    assert (f"()" in get_message(e)) and (f"({sz},)" in get_message(e))

    # Incompatible df and t
    with pytest.raises(ValueError) as e:
        gp.integrate.sde(t, -1, df[:-1], 1)

    assert f"({sz-1},)" in get_message(e)

    # Incompatible df and x0
    with pytest.raises(ValueError) as e:
        gp.integrate.sde(t, [[-1]], df, [1])

    assert f"(1, {sz-1})" in get_message(e)


def test_scalar_sde():
    # Only use non-uniform time grid.
    
    sz = 2000
    x0 = gp.normal(0, 1e-2)
    t = (np.linspace(0, 1, sz) - 1)**3 + 1  # Temporal grid.
    dt = t[1:] - t[:-1]

    # Time-independent case.
    tcov = _true_cov_ou(t, x0)

    # (1,) vector.
    df = gp.normal(size=(1, len(dt))) * np.sqrt(dt)
    x = gp.integrate.sde(t, [[-1]], df, x0.flatten()).squeeze()
    assert _error(tcov, x.cov()) < 1.5e-7  # 1.4424086458575403e-07

    # Scalar.
    df = gp.normal(size=len(dt)) * np.sqrt(dt)
    x = gp.integrate.sde(t, -1, df, x0)
    assert _error(tcov, x.cov()) < 1.5e-7

    # Time-dependent diffusion constant.
    csq = np.exp(t**2)
    dcsq = np.sqrt((csq[1:] + csq[:-1]) / 2)
    tcov = _true_cov_ou_td(t, x0)
    
    # (1,) vector.
    a = np.reshape(t, (1, 1, sz))
    df = gp.normal(size=(1, len(dt))) * np.sqrt(dt) * dcsq
    x = gp.integrate.sde(t, a, df, x0.flatten()).squeeze()
    assert _error(tcov, x.cov()) < 1.2e-7  # 1.1717172809788678e-07

    # Scalar.
    a = t
    df = gp.normal(size=len(dt)) * np.sqrt(dt) * dcsq
    x = gp.integrate.sde(t, a, df, x0)
    assert _error(tcov, x.cov()) < 1.2e-7

    # Time-depent non-white driving noise.
    tcov = _true_cov_ou_cn(t, x0)

    # (1,) vector.
    a = np.reshape(t, (1, 1, sz))
    f = gp.normal(0, _noise_cov_cn(t))
    df = gp.reshape((f[1:] + f[:-1]) * dt / 2, (1, len(dt)))
    x = gp.integrate.sde(t, a, df, x0.flatten()).squeeze()
    assert _error(tcov, x.cov()) < 1.7e-6  # 1.629043777873207e-06

    # Scalar.
    a = t
    f = gp.normal(0, _noise_cov_cn(t))
    df = (f[1:] + f[:-1]) * dt / 2
    x = gp.integrate.sde(t, a, df, x0)
    assert _error(tcov, x.cov()) < 1.7e-6


def test_real_vector_sde():
    # A simple check between the independent solutions of three scalar 
    # equations and their vector solution.

    sz = 200
    t = (np.linspace(0, 1, sz) - 1)**3 + 1  # Temporal grid.
    dt = t[1:] - t[:-1]

    al = [1.3, -0.8, 0.5]
    dfl = [gp.normal(0, 1, size=sz-1) * np.sqrt(dt) + 0.9 * dt,
           gp.normal(0, 1.2, size=sz-1)  * np.sqrt(dt) - 1.3 * dt,
           gp.normal(0, 0.8, size=sz-1) * np.sqrt(dt) - 0.6 * dt]
    x0l = [gp.normal(1, 2), gp.normal(-1, 0.5), gp.normal(-0.4, 1.5)]

    sols = [gp.integrate.sde(t, a, df, x0) for a, df, x0 in zip(al, dfl, x0l)]
    solv = gp.integrate.sde(t, np.diag(al), dfl, x0l)

    assert solv.shape == (3, sz)

    tol = 1e-14

    for i in range(3):
        assert np.max(np.abs(solv[i].mean() / sols[i].mean() - 1)) < tol
        assert np.max(np.abs(solv[i].cov() / sols[i].cov() - 1)) < tol


def test_complex_vector_sde():
    # A test for complex SDE.

    sz = 200
    t = (np.linspace(0, 1, sz) - 1)**3 + 1  # Temporal grid.
    dt = t[1:] - t[:-1]

    al = [1j + 0.2, -0.5j -0.8, 2j + 1.]
    dfl = [(gp.normal(1, 1, size=sz-1) 
            + 1j * gp.normal(0.2, 2.2, size=sz-1)) * np.sqrt(dt),
           (gp.normal(-1, 1.2, size=sz-1) 
            + 1j * gp.normal(-0.4, 1.5, size=sz-1)) * np.sqrt(dt),
           (gp.normal(-0.5, 0.8, size=sz-1) 
            + 1j * gp.normal(1.9, 0.8, size=sz-1)) * np.sqrt(dt)]
    x0l = [gp.normal(1, 2) + 1j * gp.normal(0.9, 1.),
           gp.normal(-1, 0.5),
           gp.normal(-0.4, 1.5) + 1j * gp.normal(0.35, 0.1)]

    # The solutions for the three scalar equations for the principle components
    # are the reference.
    sols = [gp.integrate.sde(t, a, df, x0) for a, df, x0 in zip(al, dfl, x0l)]

    # A transformation matrix.
    trmat = np.array([[1, 0.2, 1.3], 
                      [-2.3, 1j * 0.3, 0.1 + 2j],
                      [-0.2 - 0.3j, -1.1 + 1.1j, 1.2]])
    
    assert np.linalg.matrix_rank(trmat) == 3
    trmati = np.linalg.inv(trmat)

    sols_ = trmat @ gp.stack(sols)

    amat = trmat @ np.diag(al) @ trmati
    v0 = trmat @ gp.stack(x0l)
    dfv = trmat @ gp.stack(dfl)
    solv = gp.integrate.sde(t, amat, dfv, v0)

    assert solv.shape == (3, sz)
    assert sols_.shape == (3, sz)
    assert np.max(np.abs(solv.mean() / sols_.mean() - 1)) < 1e-8
    assert np.max(np.abs(solv.cov() / sols_.cov() - 1)) < 1e-8


def test_convergence_order():
    # Test for the convergence order on a non-uniform grid.
    
    g = 1.5
    x0 = gp.normal(1, 1e-2) + 0.5j * gp.normal(1, 1)

    szs = [100, 200]

    err_s = []
    err_v = []

    for sz in szs:
        t = (np.linspace(0, 1, sz) - 1)**3 + 1  # Temporal grid.
        dt = t[1:] - t[:-1]

        csq = np.exp(t**2)
        dcsq = np.sqrt((csq[1:] + csq[:-1]) / 2)
        dfr = gp.normal(size=len(dt)) * np.sqrt(dt) * dcsq  # Random force.
        fd = np.exp(g * t + t**2 / 2)  # Deterministic force.
        df = dfr + 0.5 * (fd[1:] + fd[:-1]) * dt  # Total force.
        
        tcov = _true_cov_ou_td(t, x0)
        tmean = np.exp(t**2 / 2) * (x0.mean() + (np.exp(g * t) - 1) / g)

        # Scalar.
        a = t
        x = gp.integrate.sde(t, a, df, x0)
        
        err_s.append(np.max(np.abs(x.mean() / tmean) - 1) 
                     + _error(tcov, x.cov()))

        # (2,) vector.
        z = np.zeros(shape=(len(t),))
        a = np.array([[t, z], [z, t]])
        x = gp.integrate.sde(t, a, [df, 2*df], [x0, 2*x0])
        
        err_v.append(np.max(np.abs(x.mean() / [tmean, 2 * tmean]) - 1) 
                     + _error(tcov, x[0].cov())+ _error(4 * tcov, x[1].cov()))
    
    assert err_s[-1] < 3.2e-5  # 3.098e-05
    assert err_v[-1] < 4.1e-5  # 4.042e-05

    assert np.log(err_s[0] / err_s[1]) / np.log(szs[1] / szs[0]) >= 2
    assert np.log(err_v[0] / err_v[1]) / np.log(szs[1] / szs[0]) >= 2