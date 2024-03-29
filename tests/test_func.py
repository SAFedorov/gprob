import sys
sys.path.append('..')  # Until there is a package structure.

import pytest
import numpy as np
from plaingaussian.normal import N, join
from plaingaussian.func import logp, dlogp, d2logp, fisher, logp_batch


def num_dlogp(x, m, cov, dm, dcov, delta=1e-7):
    """Finite-difference implementation of the gradient of the log probability 
    density."""

    npar = dcov.shape[0]

    g = []
    for i in range(npar):
        m_plus = m + dm[i] * delta
        m_minus = m - dm[i] * delta
        cov_plus = cov + dcov[i] * delta
        cov_minus = cov - dcov[i] * delta

        fp = logp(x, m_plus, cov_plus)
        fm = logp(x, m_minus, cov_minus)

        g.append((fp-fm) / (2 * delta))

    return np.array(g)


def num_d2logp(x, m, cov, dm, dcov, d2m, d2cov, delta=1e-10):
    """Finite-difference implementation of the Hessian of the log probability 
    density."""

    npar = dcov.shape[0]

    h = []
    for i in range(npar):
        m_plus = m + dm[i] * delta
        m_minus = m - dm[i] * delta
        cov_plus = cov + dcov[i] * delta
        cov_minus = cov - dcov[i] * delta
        dm_plus = dm + d2m[i] * delta
        dm_minus = dm - d2m[i] * delta
        dcov_plus = dcov + d2cov[i] * delta
        dcov_minus = dcov - d2cov[i] * delta

        fp = dlogp(x, m_plus, cov_plus, dm_plus, dcov_plus)
        fm = dlogp(x, m_minus, cov_minus, dm_minus, dcov_minus)

        h.append((fp-fm) / (2 * delta))

    return np.array(h)


def random_d1(sz, npar):
    """Prepares random matrices for testing formulas using 1st derivatives."""

    mat1 = np.random.rand(sz, sz)
    msq1 = mat1 @ mat1.T

    mat2 = np.random.rand(npar, sz, sz)
    msq2 = np.einsum('ijk, ilk -> ijl', mat2, mat2)

    v = np.random.rand(sz)
    v1 = np.random.rand(sz)
    v2 = np.random.rand(npar, sz)

    return v, v1, msq1, v2, msq2  # x, m, cov, dm, dcov


def random_d2(sz, npar):
    """Prepares random matrices for testing formulas using 2nd derivatives."""

    mat3 = np.random.rand(npar, npar, sz, sz)
    msq3 = np.einsum('ijkl, ijrl -> ijkr', mat3, mat3)
    msq3 = msq3.transpose(1, 0, 2, 3) + msq3  # Symmetrizes the Hessian of m

    v3 = np.random.rand(npar, npar, sz)
    v3 = v3.transpose(1, 0, 2) + v3  # Symmetrizes the Hessian of cov

    return random_d1(sz, npar) + (v3, msq3)  # x, m, cov, dm, dcov, d2m, d2cov


def test_dlogp():

    tol = 5e-5  # The actual errors should be in 1e-6 - 1e-5 range

    v, v1, msq1, v2, msq2 = random_d1(200, 10)
    g = dlogp(v, v1, msq1, v2, msq2)
    num_g = num_dlogp(v, v1, msq1, v2, msq2, delta=1e-9)

    assert np.abs((g - num_g)/num_g).max() < tol

    v, v1, msq1, v2, msq2 = random_d1(20, 3)
    g = dlogp(v, v1, msq1, v2, msq2)
    num_g = num_dlogp(v, v1, msq1, v2, msq2, delta=1e-9)

    assert np.abs((g - num_g)/num_g).max() < tol

    v, v1, msq1, v2, msq2 = random_d1(40, 100)
    g = dlogp(v, v1, msq1, v2, msq2)
    num_g = num_dlogp(v, v1, msq1, v2, msq2, delta=1e-9)

    assert np.abs((g - num_g)/num_g).max() < tol

    v, v1, msq1, v2, msq2 = random_d1(400, 1)
    g = dlogp(v, v1, msq1, v2, msq2)
    num_g = num_dlogp(v, v1, msq1, v2, msq2, delta=1e-9)

    assert np.abs((g - num_g)/num_g).max() < tol


def test_d2logp():

    tol = 1e-4  # The actual errors are typically around or below 1e-5

    v, v1, msq, v2, msq2, v3, msq3 = random_d2(200, 10)
    h = d2logp(v, v1, msq, v2, msq2, v3, msq3)
    num_h = num_d2logp(v, v1, msq, v2, msq2, v3, msq3, delta=1e-10)
    
    assert np.abs((h - num_h)/num_h).max() < tol

    v, v1, msq, v2, msq2, v3, msq3 = random_d2(20, 3)
    h = d2logp(v, v1, msq, v2, msq2, v3, msq3)
    num_h = num_d2logp(v, v1, msq, v2, msq2, v3, msq3, delta=1e-10)
    
    assert np.abs((h - num_h)/num_h).max() < tol

    v, v1, msq, v2, msq2, v3, msq3 = random_d2(40, 100)
    h = d2logp(v, v1, msq, v2, msq2, v3, msq3)
    num_h = num_d2logp(v, v1, msq, v2, msq2, v3, msq3, delta=1e-10)
    
    assert np.abs((h - num_h)/num_h).max() < tol

    v, v1, msq, v2, msq2, v3, msq3 = random_d2(400, 1)
    h = d2logp(v, v1, msq, v2, msq2, v3, msq3)
    num_h = num_d2logp(v, v1, msq, v2, msq2, v3, msq3, delta=1e-10)
    
    assert np.abs((h - num_h)/num_h).max() < tol


def test_fisher():
    # Do the testing via random sampling based on the formulas
    # FI[i, j] = <dlogp/dtheta_i * dlogp/dtheta_j>
    # and
    # FI[i, j] = - <d2logp/dtheta_i dtheta_j>

    pass


def test_logp():
    # TODO: add
    pass


def test_logp_batch():
    # Validation of the log likelihood calculation, batch version.

    nc = np.log(1/np.sqrt(2 * np.pi))  # Normalization constant.
    
    # Scalar variables
    xi = N()
    m, cov = xi.mean(), xi.cov()
    assert logp_batch(0, m, cov) == nc
    assert (logp_batch([0], m, cov) == np.array([nc])).all()
    assert (logp_batch([0, 0], m, cov) == np.array([nc, nc])).all()
    assert (logp_batch([[0], [0]], m, cov) == np.array([nc, nc])).all()
    assert logp_batch(2, m, cov) == -4/2 + nc
    assert (logp_batch([0, 1.1], m, cov) == [nc, -1.1**2/2 + nc]).all()

    with pytest.raises(ValueError):
        logp_batch([[0, 1]], m, cov)

    xi = N(0.9, 3.3)
    m, cov = xi.mean(), xi.cov()
    assert logp_batch(2, m, cov) == (-(2-0.9)**2/(2 * 3.3)
                                     + np.log(1/np.sqrt(2 * np.pi * 3.3)))

    # Vector variables
    xi = N(0.9, 3.3, dim=2)
    m, cov = xi.mean(), xi.cov()

    res = (-(2-0.9)**2/(2 * 3.3)-(1-0.9)**2/(2 * 3.3) 
           + 2 * np.log(1/np.sqrt(2 * np.pi * 3.3)))
    
    assert logp_batch([2, 1], m, cov) == res
    
    res = [-(3.2-0.9)**2/(2 * 3.3)-(1.2-0.9)**2/(2 * 3.3) 
           + 2 * np.log(1/np.sqrt(2 * np.pi * 3.3)), 
           -(-1-0.9)**2/(2 * 3.3)-(-2.2-0.9)**2/(2 * 3.3) 
           + 2 * np.log(1/np.sqrt(2 * np.pi * 3.3))]

    assert (logp_batch([[3.2, 1.2], [-1., -2.2]], m, cov) == np.array(res)).all()

    xi = N(0.9, 3.3, dim=2)
    m, cov = xi.mean(), xi.cov()
    with pytest.raises(ValueError):
        logp_batch(0, m, cov)
    with pytest.raises(ValueError):
        logp_batch([0, 0, 0], m, cov)
    with pytest.raises(ValueError):
        logp_batch([[0], [0]], m, cov)
    with pytest.raises(ValueError):
        logp_batch([[0, 0, 0]], m, cov)

    # Degenerate cases.

    # Deterministic variables.
    xi = join(N(), 1)
    m, cov = xi.mean(), xi.cov()
    assert logp_batch([0, 1.1], m, cov) == float("-inf")
    assert (logp_batch([[0, 1.1]], m, cov) == np.array([float("-inf")])).all()
    assert logp_batch([0, 1.], m, cov) == nc
    assert (logp_batch([[0, 1], [1.1, 1], [0, 2]], m, cov) == 
            [nc, -(1.1)**2/(2) + nc, float("-inf")]).all()
    
    # Degenerate covariance matrix. 
    xi1 = N()
    xi2 = 0 * N()
    xi12 = xi1 & xi2
    m, cov = xi12.mean(), xi12.cov()
    assert logp_batch([1.2, 0], m, cov) == -(1.2)**2/(2) + nc
    assert logp_batch([1.2, 0.1], m, cov) == float("-inf")
    assert (logp_batch([[1, 0.1]], m, cov) == np.array([float("-inf")])).all()
    assert (logp_batch([[1, 0.1], [1.2, 0]], m, cov) == 
            [float("-inf"), -(1.2)**2/(2) + nc]).all()

    # TODO: add higher-dimensional examples
    
    # Integrals of the probability density
    xi = N(0, 3.3)
    m, cov = xi.mean(), xi.cov()
    npt = 200000
    ls = np.linspace(-10, 10, npt)
    err = np.abs(1 - np.sum(np.exp(logp_batch(ls, m, cov))) * (20)/ npt)
    assert err < 6e-6  # should be 5.03694e-06

    xi = N(0, [[2.1, 0.5], [0.5, 1.3]])
    m, cov = xi.mean(), xi.cov()
    npt = 1000
    ls = np.linspace(-7, 7, npt)
    points = [(x, y) for x in ls for y in ls]
    err = np.abs(1 - np.sum(np.exp(logp_batch(points, m, cov))) * ((14)/ npt)**2)
    assert err < 2.5e-3  # should be 0.00200