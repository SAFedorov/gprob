import numpy as np
import scipy as sp
from scipy.linalg import LinAlgError


# Note: the choice of the parameter dimension order is made to fascilitate 
# broadcasting in matrix multiplication.

def cholesky_inv(mat):
    """Inverts the positive-definite symmetric matrix `mat` using Cholesky 
    decomposition. Marginally faster than `linalg.inv`. """
    
    ltr, _ = sp.linalg.cho_factor(mat, check_finite=False, lower=True)
    ltinv = sp.linalg.solve_triangular(ltr, np.eye(ltr.shape[0]), 
                                       check_finite=False, lower=True)
    return ltinv.T @ ltinv


def fisher(cov, dm, dcov):
    """Calculates the Fisher information matrix of an `n`-dimensional normal
    distribution depending on `k` parameters.
    
    Args:
        cov: Covariance matrix (n, n), non-degenerate.
        dm: Derivatives of the mean with respect to the parameters, (k, n).
        dcov: Derivatives of the covariance matrix with respect to 
            the parameters, (k, n, n).

    Returns:
        Fisher information matrix, (k, k).
    """

    # See Eq. 6. in https://arxiv.org/abs/1206.0730v1
    # "Theoretical foundation for CMA-ES from information geometric perspective"

    cov_inv = cholesky_inv(cov)
    prod1 = cov_inv @ dcov

    # Does the same as prod2 = np.einsum('kij, lji -> kl', prod1, prod1), 
    # but faster for large numbers of parameters.
    k, n, _ = dcov.shape
    prod1_flat = np.reshape(prod1, (k, n**2))
    prod1tr_flat = np.reshape(np.transpose(prod1, axes=(0, 2, 1)), (k, n**2))
    prod2 = prod1tr_flat @ prod1_flat.T

    return dm @ cov_inv @ dm.T + 0.5 * prod2


def dlogp(x, m, cov, dm, dcov):
    """Calculates the derivatives of the logarithmic probability density with 
    respect to the parameters.

    Args:
        x: Sample value.
        m: Mean vector (n).
        cov: Covariance matrix (n, n), non-degenerate.
        dm: Derivatives of the mean with respect to the parameters, (k, n).
        dcov: Derivatives of the covariance matrix with respect to 
            the parameters, (k, n, n).

    Returns:
        Gradient vector of the natural logarithm of the probability at x, (k).
    """

    cov_inv = cholesky_inv(cov)
    dnorm = -0.5 * np.einsum("ij, kij -> k", cov_inv, dcov)
    y = cov_inv @ (x - m)
    
    return 0.5 * y.T @ dcov @ y + dm @ y + dnorm


def d2logp(x, m, cov, dm, dcov, d2m, d2cov):

    k, n, _ = dcov.shape

    cov_inv = cholesky_inv(cov)
    y = cov_inv @ (x - m)

    term1 = -dm @ cov_inv @ dm.T
    term2 = d2m @ y

    prod1 = cov_inv @ dcov

    # The three lines below are an optimized version of
    # prod2 = prod1 @ (0.5 * np.eye(n) - np.outer(y, (x - m)))
    # term3 = np.einsum('kij, lji -> kl', prod1, prod2)
    prod1_flat = np.reshape(prod1, (k, n**2))
    prod1tr_flat = np.reshape(np.transpose(prod1, axes=(0, 2, 1)), (k, n**2))
    term3 = 0.5 * prod1tr_flat @ prod1_flat.T - ((x-m) @ prod1) @ (prod1 @ y).T

    term4 = 0.5 * np.einsum('klij, ij -> kl', d2cov, np.outer(y, y) - cov_inv)
    term5 = - (dm @ cov_inv) @ (dcov @ y).T

    return term1 + term2 + term3 + term4 + term5 + term5.T


# ---------- versions supporting batching ----------

def logp_sc(x, mu, sigmasq):
    """The scalar version of logp."""

    x = np.array(x)
    eps = np.finfo(float).eps
    
    if sigmasq < eps:
        # Degenerate case. By our convention, 
        # logp is 1 if x==mu, and -inf otherwise.

        match_idx = np.abs(x - mu) < eps
        llk = np.ones_like(x)
        return np.where(match_idx, llk, float("-inf"))

    return -(x-mu)**2 / (2 * sigmasq) - 0.5 * np.log(2 * np.pi * sigmasq)


def logp_cho(x, m, cov):
    """

    Args:
        x: Numpy array (n,) or (m, n).
    
    Fast for positive-definite covariance matrices.
    Supports batching.
    """

    ltr, _ = sp.linalg.cho_factor(cov, check_finite=False, lower=True)
    z = sp.linalg.solve_triangular(ltr, (x - m).T, 
                                   check_finite=False, lower=True)
    
    rank = cov.shape[0]  # Since the factorization suceeded, the rank is full.
    log_sqrt_det = np.sum(np.log(np.diagonal(ltr)))
    norm = np.log(np.sqrt(2 * np.pi)) * rank + log_sqrt_det

    return -0.5 * np.einsum("i..., i... -> ...", z, z) - norm


def logp_lstsq(x, m, cov):
    """Log likelihood of a sample.

    Works for degenerate covariance matrices.
    Supports batching.
    
    Args:
        x: Sample value or a sequence of sample values. Numpy array (n,) or (m, n)

    Returns:
        Natural logarithm of the probability density at the sample value - 
        a single number for a single sample, and an array for a sequence 
        of samples.
    """

    y, _, rank, sv = np.linalg.lstsq(cov, (x - m).T, rcond=None)
    sv = sv[:rank]  # Selects only non-zero singular values.

    norm = np.log(np.sqrt(2 * np.pi)) * rank + np.sum(np.log(sv))
    llk = -0.5 * np.einsum("...i, i... -> ...", x, y) - norm  # log likelihoods

    if rank == cov.shape[0]:
        # The covariance matrix has full rank, all solutions must be good.
        return llk
    
    # Otherwise checks the residual errors.
    delta = (y.T - (x - m)) 
    res = np.einsum("...i, ...i -> ...", delta, delta)
    eps = np.finfo(float).eps * cov.shape[0] 
    valid_idx = np.abs(res) < eps

    return np.where(valid_idx, llk, float("-inf"))


def logp(x, m, cov):

    x = np.array(x)
    m = np.array(m)
    dd = 1 if m.ndim == 0 else len(m)  # Distribution dimension.

    # Sample dimension.
    if dd == 1:
        sd = 1 if x.ndim <= 1 else x.shape[-1]
    else:
        sd = 1 if x.ndim == 0 else x.shape[-1]
    
    if sd != dd:
        raise ValueError(f"The dimension of the sample vector ({sd}) does not "
                         f"match the dimension of the distribution ({dd}).")
    
    if dd == 1:
        return logp_sc(x, m, cov)
    
    try:
        return logp_cho(x, m, cov)
    except LinAlgError:
        return logp_lstsq(x, m, cov)