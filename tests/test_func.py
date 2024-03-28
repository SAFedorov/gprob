import numpy as np
from func import logp, dlogp, d2logp, fisher, cholesky_inv

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