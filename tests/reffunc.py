import numpy as np
import scipy as sp

# Reference implementations. Can be a bit slower than the main ones,
# especially at large parameter numbers, but avoid opaque 
# optimization tricks.


def fisher(cov, dm, dcov):

    cov_inv = np.linalg.inv(cov)
    prod1 = cov_inv @ dcov
    prod2 = np.einsum('kij, lji -> kl', prod1, prod1)
    
    return dm @ cov_inv @ dm.T + 0.5 * prod2


def dkl(m1, cov1, m2, cov2):
    term1   = np.trace(np.linalg.solve(cov2, cov1))

    dm = m2 - m1
    term2 = dm.T @ np.linalg.solve(cov2, dm)

    _, logdet1 = np.linalg.slogdet(cov1)
    _, logdet2 = np.linalg.slogdet(cov2)

    return (term1 + term2 + logdet2 - logdet1 - len(dm)) / 2


def dkl_qr(m1, a1, m2, a2):
    # KL divergence using the qr decomposition of the map matrices rather than
    # the Cholesky decomposition of the covariances. About 2 times slower than 
    # the Cholesky-based approach, but can be more precise sometimes. 

    dm = m1 - m2

    _, r1 = sp.linalg.qr(a1, mode="economic", check_finite=False)
    _, r2 = sp.linalg.qr(a2, mode="economic", check_finite=False)

    z = sp.linalg.solve_triangular(r2.T, dm, check_finite=False, lower=True)

    s = sp.linalg.solve_triangular(r2.T, r1.T, check_finite=False, lower=True)
    strace = np.einsum("ij, ij -> ", s, s)
    
    log_det = 2 * np.sum(np.log(np.abs(np.diagonal(r1))) 
                         - np.log(np.abs(np.diagonal(r2))))
    return 0.5 * (strace + z @ z - log_det - len(dm))


def logp(x, m, cov):
    
    # A simple implementation with no batching

    ltr, _ = sp.linalg.cho_factor(cov, check_finite=False, lower=True)
    z = sp.linalg.solve_triangular(ltr, x - m, check_finite=False, lower=True)
    rank = cov.shape[0]  # Since the solution suceeded, the rank is full.
    log_sqrt_det = np.sum(np.log(np.diagonal(ltr)))

    norm = np.log(np.sqrt(2 * np.pi)) * rank + log_sqrt_det
    return - z @ z / 2 - norm


def dlogp_eigh(x, m, cov, dm, dcov):

    eigvals, eigvects = np.linalg.eigh(cov)
    dlambda = np.einsum('ij, kji -> ki', eigvects.T, dcov @ eigvects)
    dnorm = - 0.5 * np.sum(dlambda / eigvals, axis=1)
    y = (eigvects / eigvals) @ eigvects.T @ (x - m)
    
    return 0.5 * y.T @ dcov @ y + dm @ y + dnorm 


def d2logp(x, m, cov, dm, dcov, d2m, d2cov):

    k, n, _ = dcov.shape

    cov_inv = np.linalg.inv(cov)
    y = cov_inv @ (x - m)

    term1 = -dm @ cov_inv @ dm.T
    term2 = d2m @ y

    prod1 = cov_inv @ dcov
    prod2 = prod1 @ (0.5 * np.eye(n) - np.outer(y, (x - m)))

    term3 = np.einsum('kij, lji -> kl', prod1, prod2)
    term4 = 0.5 * np.einsum('klij, ji -> kl', d2cov, np.outer(y, y) - cov_inv)
    term5 = - np.einsum("ik, jk -> ij", dm @ cov_inv, dcov @ y)

    return term1 + term2 + term3 + term4 + term5 + term5.T