import pytest
import numpy as np
from scipy.stats import multivariate_normal as mvn
from numpy.linalg import LinAlgError
from numpy.exceptions import ComplexWarning
from gprob import hstack, vstack, iid_copy, broadcast_to, cov
from gprob.normal_ import normal, Normal, safer_cholesky
from gprob.sparse import iid_repeat
from utils import random_normal, random_correlate, asnormal

np.random.seed(0)


def test_construction():
    v = Normal(np.array([1]), np.array(0))
    assert v.nlat == 1

    a = np.ones(shape=(8, 2, 3))
    b = np.ones(shape=(2, 3))

    v = Normal(a, b)
    assert v.nlat == 8

    v1 = Normal(v.a, v.b, v.lat)
    assert v1.lat is v.lat

    # Mismatching latent dimension of a and the number of latent ids.
    with pytest.raises(ValueError):
        Normal(np.ones(shape=(7, 2, 3)), b, v.lat)

    # Mismatching variable shapes in a and b.
    with pytest.raises(ValueError):
        Normal(a, np.ones(shape=(3, 3)))

    # The shapes must match exactly, broadcastable shapes are not allowed.
    with pytest.raises(ValueError):
        Normal(a, np.ones(shape=(3,)))


def test_creation():
    # tests for normal() function, which is the main way how a user can create
    # normal variables.

    xi = normal()
    assert (xi.a == np.array([1.])).all()
    assert (xi.b == np.array(0.)).all()

    xi = normal(1.3, 4)
    assert (xi.a == np.array([2.])).all()
    assert (xi.b == np.array(1.3)).all()

    xi = normal(0, 4, size=3)
    assert (xi.a == 2 * np.eye(3)).all()
    assert (xi.b == np.zeros(3)).all()

    with pytest.raises(ValueError):
        normal(0, -4)
    
    xi = normal(0, 0)
    assert (xi.a == np.array([0.])).all()
    assert (xi.b == np.array(0.)).all()

    # Creation from a full-rank real covariance matrix.
    cov = [[2.1, 0.5], [0.5, 1.3]]
    xi = normal(0, cov)
    tol = 1e-14
    assert (np.abs(xi.cov() - np.array(cov)) < tol).all()
    assert (xi.b == 0).all()

    # Covariance matrices with negative eigenvalues are not allowed
    with pytest.raises(ValueError):
        normal(0, [[0, 1], [1, 0]])

    # But covariance matrices with zero eigenvalues are.
    cov = [[1., 0.], [0., 0.]]
    xi = normal(0, cov)
    assert (xi.cov() == np.array(cov)).all()

    mu = [1.3, 2.5]
    cov = [[1., 0.], [0., 0.]]
    xi = normal(mu, cov)
    assert (xi.cov() == np.array(cov)).all()
    assert (xi.b == mu).all()

    # High-dimensional covaraince arrays.

    mu = 0.1 
    sh = (3, 7, 2)
    vsz = int(np.prod(sh))

    a = np.random.rand(vsz * 2, *sh)
    cov = np.einsum("ijkl, imno -> jklmno", a, a)

    tol = 10 * vsz * np.finfo(cov.dtype).eps

    xi = normal(mu, cov)
    assert xi.shape == sh
    assert (xi.mean() == mu).all()
    assert np.allclose(xi.cov(), cov, rtol=tol, atol=tol)

    # Complex
    a = np.random.rand(vsz * 2, *sh) + 1j * np.random.rand(vsz * 2, *sh)
    cov = np.einsum("ijkl, imno -> jklmno", a, a.conj())
    xi = normal(mu, cov)
    assert xi.shape == sh
    assert (xi.mean() == mu).all()
    assert np.allclose(xi.cov(), cov, rtol=tol, atol=tol)

    # Degenerate complex
    a = np.random.rand(vsz // 2, *sh) + 1j * np.random.rand(vsz // 2, *sh)
    cov = np.einsum("ijkl, imno -> jklmno", a, a.conj())

    with pytest.raises(LinAlgError):  # confirms the degeneracy
        safer_cholesky(cov.reshape((vsz, vsz)))

    xi = normal(mu, cov)
    assert xi.shape == sh
    assert (xi.mean() == mu).all()
    assert np.allclose(xi.cov(), cov, rtol=tol, atol=tol)

    # Inputs for the covaraince array with inappropriate shapes.
    with pytest.raises(ValueError):
        normal(0, np.zeros((3,)))

    with pytest.raises(ValueError):
        normal(0, np.zeros((1, 2, 3)))

    with pytest.raises(ValueError):
        normal(0, np.zeros((2, 3, 2, 4)))


def test_creation_w_dtype():
    # The propagation of data types in various creation branches.

    real_dtypes = [np.float16, np.float32, np.float64]
    for dt in real_dtypes:
        tol = 100 * np.finfo(dt).eps

        mu = np.array(0.1, dtype=dt)
        sigmasq = np.array(1, dtype=dt)

        xi = normal(mu, sigmasq)
        assert xi.mean().dtype == dt
        assert xi.mean() == mu
        assert xi.var().dtype == dt
        assert np.allclose(xi.var(), sigmasq, rtol=tol, atol=tol)

        xi = normal(mu, sigmasq, size=7)
        assert xi.mean().dtype == dt
        assert (xi.mean() == mu).all()
        assert xi.var().dtype == dt
        assert np.allclose(xi.var(), sigmasq, rtol=tol, atol=tol)

        xi = normal(mu, sigmasq, size=(2, 3))
        assert xi.mean().dtype == dt
        assert (xi.mean() == mu).all()
        assert xi.var().dtype == dt
        assert np.allclose(xi.var(), sigmasq, rtol=tol, atol=tol)

        mu = np.array([0.1, 0.2], dtype=dt)
        xi = normal(mu, sigmasq)
        assert xi.mean().dtype == dt
        assert (xi.mean() == mu).all()
        assert xi.var().dtype == dt
        assert np.allclose(xi.var(), sigmasq, rtol=tol, atol=tol)

        if dt is np.float16:
            continue # numpy linalg does not support float16

        # Non-degenerate covariance matrix.
        sz = 11
        esz = 17
        a = 2. * np.random.rand(esz, sz) - 1.
        cov = (a.T @ a).astype(dt)
        mu = np.random.rand(sz).astype(dt)
        xi = normal(mu, cov)
        assert xi.mean().dtype == dt
        assert (xi.mean() == mu).all()
        assert xi.cov().dtype == dt
        assert np.allclose(xi.cov(), cov, rtol=tol, atol=tol)

        # Singly degenerate covariance matrix.
        a = (2. * np.random.rand(esz, sz) - 1.)
        cov_ = (a.T @ a).astype(dt)
        evals, evects = np.linalg.eigh(cov_)
        evals[1] = 0.

        cov = (evects * evals) @ evects.T
        with pytest.raises(LinAlgError):  # confirms the degeneracy
            safer_cholesky(cov)

        mu = np.random.rand(sz).astype(dt)
        xi = normal(mu, cov)
        assert xi.mean().dtype == dt
        assert (xi.mean() == mu).all()
        assert xi.cov().dtype == dt
        assert np.allclose(xi.cov(), cov, rtol=tol, atol=tol)

        # Doubly degenerate covariance matrix.
        evals[2] = 0.
        cov = (evects * evals) @ evects.T
        with pytest.raises(LinAlgError):
            safer_cholesky(cov)

        mu = np.random.rand(sz).astype(dt)
        xi = normal(mu, cov)
        assert xi.mean().dtype == dt
        assert (xi.mean() == mu).all()
        assert xi.cov().dtype == dt
        assert np.allclose(xi.cov(), cov, rtol=tol, atol=tol)

    complex_dtypes = [np.complex64, np.complex128]
    for dt in complex_dtypes:
        tol = 100 * np.finfo(dt).eps

        # Non-degenerate covariance matrix.
        sz = 11
        esz = 17
        a = ((2. * np.random.rand(esz, sz) - 1.) 
             + 1j * (2. * np.random.rand(esz, sz) - 1.))
        cov = (a.T @ a.conj()).astype(dt)
        mu = (np.random.rand(sz) + 1j * np.random.rand(sz)).astype(dt)
        xi = normal(mu, cov)
        assert xi.mean().dtype == dt
        assert (xi.mean() == mu).all()
        assert xi.cov().dtype == dt
        assert np.allclose(xi.cov(), cov, rtol=tol, atol=tol)

        # Singly degenerate covariance matrix.
        a = ((2. * np.random.rand(esz, sz) - 1.) 
             + 1j * (2. * np.random.rand(esz, sz) - 1.))
        cov_ = (a.T @ a.conj()).astype(dt)
        evals, evects = np.linalg.eigh(cov_)
        evals[1] = 0.

        cov = (evects * evals) @ evects.T.conj()
        with pytest.raises(LinAlgError):  # confirms the degeneracy
            safer_cholesky(cov)

        mu = (np.random.rand(sz) + 1j * np.random.rand(sz)).astype(dt)
        xi = normal(mu, cov)
        assert xi.mean().dtype == dt
        assert (xi.mean() == mu).all()
        assert xi.cov().dtype == dt
        assert np.allclose(xi.cov(), cov, rtol=tol, atol=tol)

        # Doubly degenerate covariance matrix.
        evals[2] = 0.
        cov = (evects * evals) @ evects.T.conj()
        with pytest.raises(LinAlgError):
            safer_cholesky(cov)

        mu = (np.random.rand(sz) + 1j * np.random.rand(sz)).astype(dt)
        xi = normal(mu, cov)
        assert xi.mean().dtype == dt
        assert (xi.mean() == mu).all()
        assert xi.cov().dtype == dt
        assert np.allclose(xi.cov(), cov, rtol=tol, atol=tol)


def test_properties():
    # Checks the descriptive properties
    prop_names = ["size", "shape", "ndim"]

    v = random_normal((3, 4))
    for pn in prop_names:
        assert getattr(v, pn) == getattr(v.mean(), pn)

    # nlat
    assert v.nlat == len(v.lat)
    assert v.nlat == len(v.a)

    # delta property
    tol = 1e-10
    vd = v.delta
    assert isinstance(vd, Normal)
    assert vd.lat == v.lat
    assert np.max(np.abs(vd.mean())) < tol
    assert np.max(np.abs(vd.var() - v.var())) < tol 

    # Checks the complex flag.

    v = normal(1, 2.)
    assert v.iscomplex == False

    v = normal(1 + 2j, 2.)
    assert v.iscomplex == True

    v = normal(1, 2.) + 1j * normal(size=(2, 2))
    assert v.iscomplex == True

    # either a or b need to be complex, not both
    v.b = v.b.real
    assert v.iscomplex == True

    v.a = v.a.real
    assert v.iscomplex == False


def test_repr():
    for v in [normal(), normal(size=(2, 3))]:
        assert type(repr(v)) == type("")
    
    assert "\n" not in repr(5 + normal(size=(3,)))
    assert "\n" in repr(normal(size=(2, 3)))


def test_logp():
    # The validation of the log likelihood calculation for real normal arrays.

    tol = 1e-15

    nc = np.log(1/np.sqrt(2 * np.pi))  # Normalization constant.
    
    # Scalar variables
    xi = normal()
    assert xi.logp(0).shape == tuple()
    assert xi.logp(0) == nc
    assert xi.logp(1.1) == -1.1**2/2 + nc

    assert xi.logp([0]).shape == (1,)
    assert xi.logp([0]) == nc

    xi = normal(0.9, 3.3)
    assert xi.logp(2).shape == tuple()
    assert xi.logp(2) == (-(2-0.9)**2/(2 * 3.3) 
                          + np.log(1/np.sqrt(2 * np.pi * 3.3)))

    # Vector variables
    xi = normal(0.9, 3.3, size=2)
    assert xi.logp([2, 1]).shape == tuple()
    assert xi.logp([2, 1]) == (-(2-0.9)**2/(2 * 3.3)-(1-0.9)**2/(2 * 3.3) 
                               + 2 * np.log(1/np.sqrt(2 * np.pi * 3.3)))
    
    assert xi.logp(np.random.rand(3, 2)).shape == (3,)

    xi = normal(0.9, 3.3, size=2)
    with pytest.raises(ValueError):
        xi.logp(0)
    with pytest.raises(ValueError):
        xi.logp([0, 0, 0])
    with pytest.raises(ValueError):
        xi.logp([[0], [0]])
    with pytest.raises(ValueError):
        xi.logp([[[0, 2], [0, 2]]])  # Too many dimensions.

    # A higher-dimensional variable.
    sh = (3, 5)
    xi = random_normal(sh, dtype=np.float64)
    xif = xi.ravel()

    tol_ = 1e-8  # increased tolerance margin
    x = np.random.rand(*sh)
    logpref = mvn.logpdf(x.ravel(), xif.mean(), xif.cov())
    assert np.abs(xi.logp(x) - logpref) < tol_
    assert xi.logp(x).shape == x.shape[:-xi.ndim]

    x = np.random.rand(3, *sh)
    logpref = mvn.logpdf(x.reshape(-1, xif.size), xif.mean(), xif.cov())
    assert np.max(np.abs(xi.logp(x) - logpref)) < tol_
    assert xi.logp(x).shape == x.shape[:-xi.ndim]

    # Degenerate cases.
        
    # Zero scalar random variable.
    assert (0 * normal()).logp(0.) > float("-inf")
    assert (0 * normal()).logp(0.1) == float("-inf")

    assert (0 * normal()).logp(0.).shape == tuple()
    assert (0 * normal()).logp([0., 0.1]).shape == (2,)

    # Deterministic variables.
    xi = hstack([normal(), 1])
    assert xi.logp([0, 1.1]) == float("-inf")
    assert xi.logp([0, 1.]) == nc
    assert xi.logp([10.2, 1.]) == -(10.2)**2/(2) + nc

    assert xi.logp([0, 1.]).shape == tuple()
    assert xi.logp([[0, 1.1], [0, 1.]]).shape == (2,)
    
    # Degenerate covariance matrix. 
    xi1 = normal()
    xi2 = 0 * normal()
    xi12 = xi1 & xi2
    assert xi12.logp([1.2, 0]) == -(1.2)**2/(2) + nc
    assert xi12.logp([1.2, 0.1]) == float("-inf")

    # A higher-dimensional example.
    xi1 = normal([[0.1, 0.2], [0.3, 0.4]], 2)
    xi2 = normal([0.1, 0.2, 0.3, 0.4], 2)
    assert np.abs(xi1.logp(np.ones((2, 2))) - xi2.logp(np.ones((4,)))) < tol
    assert np.abs(xi1.logp([np.ones((2, 2)), 0.5 * np.ones((2, 2))]) 
                  - xi2.logp([np.ones((4,)), 0.5 * np.ones((4,))])).max() < tol
    
    assert xi1.logp(np.ones((2, 2))).shape == tuple()
    assert xi1.logp(np.ones((1, 2, 2))).shape == (1,)

    # More degenerate cases.

    tol_ = 1e-7

    # Self-consistency: doubling the dimension does not change logp
    sh = (2, 3, 1)
    xi = random_normal(sh, dtype=np.float64)
    xi_ = vstack([xi, xi]) / np.sqrt(2)
    x = xi.sample()
    x_ = np.vstack([x, x]) / np.sqrt(2)
    assert np.abs(xi.logp(x) - xi_.logp(x_)) < tol_

    for sh in [tuple(), (5,), (3, 3), (3, 20, 4)]:
        xi = random_normal(sh, dtype=np.float64)
        xi = vstack([xi, xi, xi, xi])
        xif = xi.ravel()

        with pytest.raises(LinAlgError):  # Asserts the degeneracy.
            np.linalg.cholesky(xif.cov())

        # Single possible sample.
        x = xi.sample()
        xf = x.ravel()
        logpref = mvn.logpdf(xf, xif.mean(), xif.cov(), 
                             allow_singular=True)
        assert np.abs(xi.logp(x) - logpref) < tol_
        assert xi.logp(x).shape == x.shape[:-xi.ndim]

        # Single impossible sample.
        x = np.random.rand(*xi.shape)
        assert xi.logp(x) == float("-inf")

        # Multiple possible samples.
        x = xi.sample(3)
        xf = x.reshape(-1, xi.size)
        logpref = mvn.logpdf(xf, xif.mean(), xif.cov(), 
                             allow_singular=True)
        assert np.max(np.abs(xi.logp(x) - logpref)) < tol_
        assert xi.logp(x).shape == x.shape[:-xi.ndim]

        # One impossible sample among several possible.
        x[0] = np.random.rand(*xi.shape)
        xf = x.reshape(-1, xi.size)
        logpref = mvn.logpdf(xf, xif.mean(), xif.cov(), 
                             allow_singular=True)
        assert np.max(np.abs(xi.logp(x)[1:] - logpref[1:])) < tol_
        assert xi.logp(x)[0] == float("-inf")
        assert xi.logp(x).shape == x.shape[:-xi.ndim]


def test_complex_logp():
    # Logp for complex arrays.

    def complex_pdf(x, m, cov, rel):
        dx = x - m
        rmat = rel.T.conj() @ np.linalg.inv(cov)
        pmat = cov.conj() - rmat @ rel
        pci = np.linalg.inv(pmat).conj()

        _, ld1 = np.linalg.slogdet(cov)
        _, ld2 = np.linalg.slogdet(pmat)

        norm = len(x) * np.log(np.pi) + 0.5 * (ld1 + ld2)
        return -dx.conj() @ pci @ dx + np.real(dx @ rmat.T @ pci @ dx) - norm

    tol = 1e-8

    xi = normal() + 1j * normal()
    xi2 = normal(size=2)
    assert np.abs(xi.logp(0.1 - 1j * 0.2) - xi2.logp([0.1, -0.2])) < tol
    assert xi.logp(0.1 - 1j * 0.2).shape == tuple()
    assert xi.logp([0.1 - 1j * 0.2]).shape == (1,)

    with pytest.warns(ComplexWarning):
        normal().logp(1+0.j)
    with pytest.warns(ComplexWarning):
        normal().logp([1+0.j])

    xi = normal(size=2) + 1j * normal(size=2)
    xi2 = normal(size=4)
    sa = [0.1 - 1j * 0.2, 0.3 + 1j * 0.6]
    assert np.abs(xi.logp(sa) - xi2.logp([0.1, -0.2, 0.3, 0.6])) < tol
    assert xi.logp(sa).shape == tuple()
    assert xi.logp([sa]).shape == (1,)

    sh = (6,)
    xi = random_normal(sh, dtype=np.complex128)

    a = xi.a
    m = xi.mean()
    cov = a.T @ a.conj()
    rel = a.T @ a

    x = np.random.rand(*sh) + 1j * np.random.rand(*sh)
    logpref = complex_pdf(x, m, cov, rel)
    assert np.abs(xi.logp(x) - logpref) < tol
    assert xi.logp(x).shape == x.shape[:-xi.ndim]

    x = np.random.rand(3, *sh)
    logpref = np.array([complex_pdf(x_, m, cov, rel) for x_ in x])
    assert np.max(np.abs(xi.logp(x) - logpref)) < tol
    assert xi.logp(x).shape == x.shape[:-xi.ndim]

    # A higher-dimensional array.
    sh = (3, 2)
    xi = random_normal(sh, dtype=np.complex128)
    
    xif = xi.flatten()
    a = xif.a
    m = xif.mean()
    cov = a.T @ a.conj()
    rel = a.T @ a

    x = np.random.rand(*sh) + 1j * np.random.rand(*sh)
    xf = x.flatten()
    logpref = complex_pdf(xf, m, cov, rel)
    assert np.abs(xi.logp(x) - logpref) < tol
    assert xi.logp(x).shape == x.shape[:-xi.ndim]

    x = np.random.rand(3, *sh)
    xf = x.reshape(3, -1)
    logpref = np.array([complex_pdf(x_, m, cov, rel) for x_ in xf])
    assert np.max(np.abs(xi.logp(x) - logpref)) < tol
    assert xi.logp(x).shape == x.shape[:-xi.ndim]

    # A degenerate case.
    sh = (3,)
    xi = random_normal(sh, dtype=np.complex128)
    xi = hstack([xi, xi])
    x = xi.sample()

    m = xi.mean()
    a = xi.a
    x2 = np.hstack([x.real, x.imag])
    m2 = np.hstack([m.real, m.imag])
    a2 = np.hstack([a.real, a.imag])

    with pytest.raises(LinAlgError):  # Asserts the degeneracy.
        np.linalg.cholesky(a2.T @ a2)

    logpref = mvn.logpdf(x2, m2, a2.T @ a2, allow_singular=True)
    assert np.abs(xi.logp(x) - logpref) < tol

    # An impossible sample.
    x = np.random.rand(*xi.shape) + 1j * np.random.rand(*xi.shape)
    assert xi.logp(x) == float("-Inf")


def test_len():
    xi = normal()
    with pytest.raises(TypeError):
        len(xi)

    xi = normal(size=(1,))
    assert len(xi) == 1

    xi = normal(0, 2, size=2)
    assert len(xi) == 2

    xi = normal(0, 2, size=(2, 3))
    assert len(xi) == 2

    xi = hstack([normal(), normal(), normal(), normal()])
    assert len(xi) == 4


def test_sample():
    
    # Checks the formats returned by sample()
    
    v = normal(0, 1)
    s = v.sample()
    assert np.array(s).ndim == 0

    s = v.sample(3)
    assert s.shape == (3,)

    v = normal(0, 1, size=5)
    s = v.sample()
    assert s.shape == (5,)

    s = v.sample(3)
    assert s.shape == (3, 5)


def test_stack():

    v1 = normal()
    v2 = normal()
    v3 = normal()

    vm = hstack([v1, v2, v3])  # list input
    assert isinstance(vm, Normal) and len(vm) == 3

    vm = hstack((v1, v2, v3))  # tuple input
    assert isinstance(vm, Normal) and len(vm) == 3


def test_latent_ordering():
    # Requires python >= 3.6

    def isordered(v: Normal):
        ed = v.lat
        return all(ed[k] == i for i, k in enumerate(ed))
    
    # Scalars

    v1 = normal(0.1, 100)
    v2 = normal()
    v3 = normal()

    v4 = v1 + 0.8 * v3 + 200.
    assert isordered(v4)

    v5 = 0.3 * v4 + 0.4 * v2
    assert isordered(v5)

    v5 = 0.02 + 13. * v5 + 20 * v1
    assert isordered(v5)

    v5 = 0.02 + 13. * v5 + 0.4 * normal() + 20 * v1
    assert isordered(v5)

    nrv = 20

    vl1 = [normal() for _ in range(nrv)]
    assert isordered(hstack(vl1))

    vl2 = [normal() for _ in range(nrv)]
    assert isordered(hstack(vl2))

    vl3 = [normal() for _ in range(nrv // 2)]  # A shorter list.
    assert isordered(hstack(vl3))

    assert isordered(hstack(vl1 + vl2))
    assert isordered(hstack(vl3 + vl2))
    assert isordered(hstack(vl2 + vl3))
    assert isordered(hstack(vl2 + vl3 + vl2))
    assert isordered(hstack(vl2 + vl3 + vl2 + vl2 + vl1 + vl2))

    assert isordered(hstack(vl2) + hstack(vl1))
    assert isordered(hstack(vl2) - 1)

    # Shuffled lists of scalars

    idx1 = np.random.randint(0, len(vl1), size=len(vl1))
    idx12 = np.random.randint(0, len(vl1), size=len(vl1))
    idx2 = np.random.randint(0, len(vl2), size=len(vl2))
    vl1_s = [vl1[i] for i in idx1]
    vl12_s = [vl1[i] for i in idx12]
    vl2_s = [vl2[i] for i in idx2]
    
    assert isordered(hstack(vl1_s + vl1_s))
    assert isordered(hstack(vl12_s + vl1_s))
    assert isordered(hstack(vl2_s + vl1_s))
    assert isordered(hstack(vl1_s + vl2_s))

    totvl1 = 0.
    for v in vl1_s:
        totvl1 += v

    totvl12 = 0.
    for v in vl12_s:
        totvl12 -= v

    totvl2 = 0.
    for v in vl2_s:
        totvl2 += v

    assert isordered(totvl1)
    assert isordered(totvl12)
    assert isordered(totvl2)
    assert isordered(totvl1 - totvl12)
    assert isordered(totvl1 + totvl2)

    # Vectors

    vl1 = [normal(0.3, 12., size=3) for _ in range(nrv)]
    assert isordered(hstack(vl1))

    vl2 = [normal(size=3) for _ in range(nrv)]
    assert isordered(hstack(vl2))

    vl3 = [normal(-3, 2., size=3) for _ in range(nrv // 2)]  # A shorter list.
    assert isordered(hstack(vl3))

    assert isordered(hstack(vl1 + vl2))
    assert isordered(hstack(vl3 + vl2))
    assert isordered(hstack(vl2 + vl3))
    assert isordered(hstack(vl2 + vl3 + vl2))
    assert isordered(hstack(vl2 + vl3 + vl2 + vl2 + vl1 + vl2))

    assert isordered(hstack(vl2) + hstack(vl1))
    assert isordered(hstack(vl2) - 1)

    # Shuffled lists of vectors

    idx1 = np.random.randint(0, len(vl1), size=len(vl1))
    idx12 = np.random.randint(0, len(vl1), size=len(vl1))
    idx2 = np.random.randint(0, len(vl2), size=len(vl2))
    vl1_s = [vl1[i] for i in idx1]
    vl12_s = [vl1[i] for i in idx12]
    vl2_s = [vl2[i] for i in idx2]
    
    assert isordered(hstack(vl1_s + vl1_s))
    assert isordered(hstack(vl12_s + vl1_s))
    assert isordered(hstack(vl2_s + vl1_s))
    assert isordered(hstack(vl1_s + vl2_s))

    totvl1 = 0.
    for v in vl1_s:
        totvl1 += v

    totvl12 = 0.
    for v in vl12_s:
        totvl12 -= v

    totvl2 = 0.
    for v in vl2_s:
        totvl2 += v

    assert isordered(totvl1)
    assert isordered(totvl12)
    assert isordered(totvl2)
    assert isordered(totvl1 - totvl12)
    assert isordered(totvl1 + totvl2)


def test_operations():
    # arithmetic operations between normal variables and other types

    def isclose(v1, v2, tol=1e-14):
        return ((np.abs(v1.a - v2.a) < tol).all() 
                and (np.abs(v1.b - v2.b) < tol).all())
    
    # 0d-1d
    v = normal(8, 1)
    x_li = [2, 3]
    x_tu = tuple(x_li)
    x_ar = np.array(x_li)

    assert np.all((x_li + v).b == [10, 11])
    assert np.all((x_li * v).b == [16, 24])
    
    assert isclose(x_li + v, x_ar + v)
    assert isclose(x_tu + v, x_ar + v)
    assert isclose(v + x_li, x_ar + v)

    assert isclose(x_li - v, x_ar - v)
    assert isclose(x_tu - v, x_ar - v)
    assert isclose((-1) * v + x_li, x_ar - v)
    assert isclose((-1) * (x_li - v), v - x_ar)

    assert isclose(x_li * v, x_ar * v)
    assert isclose(x_tu * v, x_ar * v)
    assert isclose(v * x_li, x_ar * v)

    assert isclose(v / x_li,  v / x_ar)
    assert isclose(v / x_tu, v / x_ar)

    # 1d-1d
    v = normal(0, 1, size=2)
    x_li = [2, 3]
    x_tu = tuple(x_li)
    x_ar = np.array(x_li) 

    assert isclose(x_li + v, x_ar + v)
    assert isclose(x_tu + v, x_ar + v)
    assert isclose(v + x_li, x_ar + v)

    assert isclose(x_li - v, x_ar - v)
    assert isclose(x_tu - v, x_ar - v)
    assert isclose((-1) * v + x_li, x_ar - v)
    assert isclose((-1) * (x_li - v), v - x_ar)

    assert isclose(x_li * v, x_ar * v)
    assert isclose(x_tu * v, x_ar * v)
    assert isclose(v * x_li, x_ar * v)

    assert isclose(v / x_li,  v / x_ar)
    assert isclose(v / x_tu, v / x_ar)

    v1 = -normal(0, 1, size=2) + np.array([2, 3])
    v2 = np.array([2, 3]) - normal(0, 1, size=2)
    assert isclose(v1, v2)

    v1 = [2, 3] * normal(0, 1, size=2)
    v2 = np.array([2, 3]) * normal(0, 1, size=2)
    assert isclose(v1, v2)

    v1 = normal(0, 1, size=2) * np.array([2, 3])
    v2 = np.array([2, 3]) * normal(0, 1, size=2)
    assert isclose(v1, v2)

    v1 = normal(0, 1, size=2) * np.sqrt(2)
    v2 = np.sqrt(2) * normal(0, 1, size=2)
    assert isclose(v1, v2)

    # Operations with an object that is neither convertible to a numeric array, 
    # nor a normal variable itself.

    tol = 1e-8

    # Addition.
    v1 = normal() + [normal(), 2, normal()]
    v2 = normal() + hstack([normal(), 2, normal()])
    assert isinstance(v1, Normal)
    assert v1.shape == (3,)
    assert np.max(np.abs(v1.mean() - v2.mean())) < tol 
    assert np.max(np.abs(v1.cov() - v2.cov())) < tol 

    v1 = [normal(), 2, normal()] + normal()
    v2 = hstack([normal(), 2, normal()]) + normal()
    assert isinstance(v1, Normal)
    assert v1.shape == (3,)
    assert np.max(np.abs(v1.mean() - v2.mean())) < tol 
    assert np.max(np.abs(v1.cov() - v2.cov())) < tol 

    # Subtraction.
    v1 = normal() - [normal(), 2, normal()]
    v2 = normal() - hstack([normal(), 2, normal()])
    assert isinstance(v1, Normal)
    assert v1.shape == (3,)
    assert np.max(np.abs(v1.mean() - v2.mean())) < tol 
    assert np.max(np.abs(v1.cov() - v2.cov())) < tol 

    v1 = [normal(), 2, normal()] - normal()
    v2 = hstack([normal(), 2, normal()]) - normal()
    assert isinstance(v1, Normal)
    assert v1.shape == (3,)
    assert np.max(np.abs(v1.mean() - v2.mean())) < tol 
    assert np.max(np.abs(v1.cov() - v2.cov())) < tol

    # Multiplication.
    v1 = normal(0.5, 2) * [normal(0.1, 1), 2, normal(0.1, 1)]
    v2 = normal(0.5, 2) * hstack([normal(0.1, 1), 2, normal(0.1, 1)])
    assert isinstance(v1, Normal)
    assert v1.shape == (3,)
    assert np.max(np.abs(v1.mean() - v2.mean())) < tol 
    assert np.max(np.abs(v1.cov() - v2.cov())) < tol 

    v1 = [normal(0.1, 1), 2, normal(0.1, 1)] * normal(0.5, 2)
    v2 = hstack([normal(0.1, 1), 2, normal(0.1, 1)]) * normal(0.5, 2)
    assert isinstance(v1, Normal)
    assert v1.shape == (3,)
    assert np.max(np.abs(v1.mean() - v2.mean())) < tol 
    assert np.max(np.abs(v1.cov() - v2.cov())) < tol 

    # Division.
    v1 = normal(0.5, 2) / [normal(0.1, 1), 2, normal(0.1, 1)]
    v2 = normal(0.5, 2) / hstack([normal(0.1, 1), 2, normal(0.1, 1)])
    assert isinstance(v1, Normal)
    assert v1.shape == (3,)
    assert np.max(np.abs(v1.mean() - v2.mean())) < tol 
    assert np.max(np.abs(v1.cov() - v2.cov())) < tol 

    v1 = [normal(0.1, 1), 2, normal(0.1, 1)] / normal(0.5, 2)
    v2 = hstack([normal(0.1, 1), 2, normal(0.1, 1)]) / normal(0.5, 2)
    assert isinstance(v1, Normal)
    assert v1.shape == (3,)
    assert np.max(np.abs(v1.mean() - v2.mean())) < tol 
    assert np.max(np.abs(v1.cov() - v2.cov())) < tol 

    # Matrix multiplication.
    v = random_normal((3, 3))
    v1 = v @ [normal(0.1, 1), 2, normal(0.1, 1)]
    v2 = v @ hstack([normal(0.1, 1), 2, normal(0.1, 1)])
    assert isinstance(v1, Normal)
    assert v1.shape == (3,)
    assert np.max(np.abs(v1.mean() - v2.mean())) < tol 
    assert np.max(np.abs(v1.cov() - v2.cov())) < tol 

    v1 = [normal(0.1, 1), 2, normal(0.1, 1)] @ v
    v2 = hstack([normal(0.1, 1), 2, normal(0.1, 1)]) @ v
    assert isinstance(v1, Normal)
    assert v1.shape == (3,)
    assert np.max(np.abs(v1.mean() - v2.mean())) < tol 
    assert np.max(np.abs(v1.cov() - v2.cov())) < tol 

    # Power.
    v = 2 + random_normal((2, 3))
    v1 = v ** [normal(0.1, 1), 2, normal(0.1, 1)]
    v2 = v ** hstack([normal(0.1, 1), 2, normal(0.1, 1)])
    assert isinstance(v1, Normal)
    assert v1.shape == (2, 3)
    assert np.max(np.abs(v1.mean() - v2.mean())) < tol 
    assert np.max(np.abs(v1.cov() - v2.cov())) < tol 

    v1 = [normal(0.1, 1), 2, normal(0.1, 1)] ** v
    v2 = hstack([normal(0.1, 1), 2, normal(0.1, 1)]) ** v
    assert isinstance(v1, Normal)
    assert v1.shape == (2, 3)
    assert np.max(np.abs(v1.mean() - v2.mean())) < tol 
    assert np.max(np.abs(v1.cov() - v2.cov())) < tol


def test_normal_operations():
    # normal-normal operations

    tol = 1e-14

    f = lambda x: 0.1 + x**0. - 2.5 * x**2 + x*x*x * 4 + 1/(4. + x)**3 - 0.5**x
    df = lambda x: -5 * x + 12. * x**2 - 3 * (4.+x)**(-4) - 0.5**x * np.log(0.5)

    x = np.linspace(-3, 3, 100) 
    v = normal(0, 0.3, size=100) + x
    v_ = f(v)
    assert np.abs((v_.mean() - f(x))/f(x)).max() < tol
    assert np.abs((v_.var() - df(x)**2 * v.var())/v_.var()).max() < tol


def test_broadcasting():
    # In this test, normal arrays interacts with a higher-dimensional constant
    # or normal arrays that broadcast them to new shapes.

    s = (5, 3)
    assert broadcast_to(normal(size=(1, 3)), s).shape == s

    s = (5, 2, 2)
    assert broadcast_to(normal(size=(5, 1, 2)), s).shape == s

    # Numeric operand smaller than the random variable
    xi = normal(size=(1, 2, 3))
    y = np.array([2, 3, 4])
    assert (xi + y).shape == xi.shape
    assert (y + xi).shape == xi.shape
    assert (xi - y).shape == xi.shape
    assert (y - xi).shape == xi.shape
    assert (xi * y).shape == xi.shape
    assert (y * xi).shape == xi.shape
    assert (xi / y).shape == xi.shape
    assert (xi ** y).shape == xi.shape
    assert broadcast_to(xi, xi.shape).shape == xi.shape

    # Numeric operand larger than the random variable
    xi = normal(size=(1, 2, 3))
    y = np.ones((4, 2, 2, 3))
    assert (xi + y).shape == y.shape
    assert (y + xi).shape == y.shape
    assert (xi - y).shape == y.shape
    assert (y - xi).shape == y.shape
    assert (xi * y).shape == y.shape
    assert (y * xi).shape == y.shape
    assert (xi / y).shape == y.shape
    assert (xi ** y).shape == y.shape
    assert broadcast_to(xi, y.shape).shape == y.shape

    tol = 1e-15

    xi1 = normal(mu=0.1)
    xi2 = xi1 * (-3, -4)
    assert xi2.shape == (2,)
    assert np.abs(xi2.b - (-0.3, -0.4)).max() < tol
    assert np.abs(xi2.a[0] - (-3, -4)).max() < tol

    xi = normal(1, 1)**[2, 0]
    assert xi.shape == (2,)
    assert np.abs(xi.a - np.array([[2., 0.]])).max() < tol

    m = np.array([[1, 0], [0, 1], [2, 2]])
    xi1 = Normal(np.array([[1, 0.5], [0, -1]]), np.array([0.3, -0.3]))
    xi2 = xi1 * m
    assert xi2.shape == (3, 2)
    assert np.abs(xi2.b - [[0.3, 0], [0, -0.3], [0.6, -0.6]]).max() < tol
    for r1, r2 in zip(xi1.a, xi2.a):
        assert np.abs(r2 - r1 * m).max() < tol

    xi2 = m * xi1
    assert xi2.shape == (3, 2)
    assert np.abs(xi2.b - [[0.3, 0], [0, -0.3], [0.6, -0.6]]).max() < tol
    for r1, r2 in zip(xi1.a, xi2.a):
        assert np.abs(r2 - r1 * m).max() < tol

    # random nd shapes
    xi1 = Normal(np.array([[1, 0.5], [0, -1], [8., 9.]]), 
                 np.array([0.3, -0.3]))
    
    # addition
    sh = tuple(np.random.randint(1, 4, 8))
    m = np.random.rand(*(sh + (2,)))

    xi2 = m + xi1
    assert xi2.shape == sh + (2,)
    assert xi2.a.shape == (3,) + sh + (2,)
    rng = int(np.prod(sh))
    a2_fl = np.reshape(xi2.a, (3, rng, 2))
    for i in range(rng):
        assert np.abs(a2_fl[:, i, :] - xi1.a).max() < tol

    # multiplication
    sh = tuple(np.random.randint(1, 4, 9))
    m = np.random.rand(*(sh + (2,)))

    xi2 = m * xi1
    assert xi2.shape == sh + (2,)
    assert xi2.a.shape == (3,) + sh + (2,)
    rng = int(np.prod(sh))
    m_fl = np.reshape(m, (rng, 2))
    a2_fl = np.reshape(xi2.a, (3, rng, 2))
    for i in range(rng):
        assert np.abs(a2_fl[:, i, :] - m_fl[i] * xi1.a).max() < tol

    # division
    sh = tuple(np.random.randint(1, 4, 7))
    m = np.random.rand(*(sh + (2,))) + 0.1

    xi2 =  xi1 / m
    assert xi2.shape == sh + (2,)
    assert xi2.a.shape == (3,) + sh + (2,)
    rng = int(np.prod(sh))
    m_fl = np.reshape(m, (rng, 2))
    a2_fl = np.reshape(xi2.a, (3, rng, 2))
    for i in range(rng):
        assert np.abs(a2_fl[:, i, :] -  xi1.a / m_fl[i]).max() < tol

    # normal-normal operations

    sz1 = (20, 5, 5)
    sz2 = (5,)

    xi1, xi2 = normal(size=sz1), normal(size=sz2)
    assert (xi1 + xi2).shape == sz1
    assert (xi1 - xi2).shape == sz1
    assert (xi1 * xi2).shape == sz1
    assert (xi1 / (xi2 + 2.)).shape == sz1

    # inverting the order by changing the length of the elementaries
    sz1 = (10, 5, 5)
    sz2 = (5,)

    xi1, xi2 = normal(size=sz1), Normal(np.ones((500, 5)), np.zeros((5,)))
    assert (xi1 + xi2).shape == sz1
    assert (xi1 - xi2).shape == sz1
    assert (xi1 * xi2).shape == sz1
    assert (xi1 / (xi2 + 2.)).shape == sz1

    # lifted constants - there can be separate optimization paths for them
    assert (normal(size=(2, 3)) + asnormal(1)).shape == (2, 3)
    assert (normal(size=(2, 3)) - asnormal(1)).shape == (2, 3)
    assert (normal(size=(2, 3)) * asnormal(1)).shape == (2, 3)
    assert (normal(size=(2, 3)) / asnormal(1)).shape == (2, 3)

    assert (normal(size=(1, 3)) + asnormal(np.ones((2, 3)))).shape == (2, 3)
    assert (normal(size=(1, 3)) - asnormal(np.ones((2, 3)))).shape == (2, 3)
    assert (normal(size=(1, 3)) * asnormal(np.ones((2, 3)))).shape == (2, 3)
    assert (normal(size=(1, 3)) / asnormal(np.ones((2, 3)))).shape == (2, 3)

    assert (normal(size=(3,)) + asnormal(np.ones((2, 3)))).shape == (2, 3)
    assert (normal(size=(3,)) - asnormal(np.ones((2, 3)))).shape == (2, 3)
    assert (normal(size=(3,)) * asnormal(np.ones((2, 3)))).shape == (2, 3)
    assert (normal(size=(3,)) / asnormal(np.ones((2, 3)))).shape == (2, 3)

    assert (normal(size=(2, 3)) + asnormal(np.ones((2, 1)))).shape == (2, 3)
    assert (normal(size=(2, 3)) - asnormal(np.ones((2, 1)))).shape == (2, 3)
    assert (normal(size=(2, 3)) * asnormal(np.ones((2, 1)))).shape == (2, 3)
    assert (normal(size=(2, 3)) / asnormal(np.ones((2, 1)))).shape == (2, 3)


def test_broadcast_to():
    v = broadcast_to(1, (2, 3))
    assert isinstance(v, Normal)
    assert v.shape == (2, 3)

    v = broadcast_to(normal(), (2, 3))
    assert isinstance(v, Normal)
    assert v.shape == (2, 3)

    v = broadcast_to(normal(size=(2, 1)), (4, 2, 3))
    assert isinstance(v, Normal)
    assert v.shape == (4, 2, 3)

    with pytest.raises(ValueError):
        broadcast_to(v, (2,))

    with pytest.raises(ValueError):
        broadcast_to(v, (3, 1))


def test_getitem():
    v = normal(size=(2, 3, 4))
    assert (v[:, 0].mean() == np.zeros((2, 4))).all()
    assert (v[:, 0].var() == np.ones((2, 4))).all()

    assert (v[:, :, 2].mean() == np.zeros((2, 3))).all()
    assert (v[:, :, 2].var() == np.ones((2, 3))).all()

    assert (v[0, :, :].mean() == np.zeros((3, 4))).all()
    
    assert (v[..., 2].var() == np.ones((2, 3))).all()
    assert (v[...].var() == np.ones((2, 3, 4))).all()
    assert (v[None].var() == np.ones((1, 2, 3, 4))).all()
    assert (v[:, None].var() == np.ones((2, 1, 3, 4))).all()
    assert (v[:, None, :, None].var() == np.ones((2, 1, 3, 1, 4))).all()
    assert (v[:, ..., None].var() == np.ones((2, 3, 4, 1))).all()


def test_setitem():
    rdtype_list = [np.float32, np.float64]
    cdtype_list = [np.complex64, np.complex128]

    for dt in rdtype_list:
        tol = 100 * np.finfo(dt).eps

        sh = (2, 3, 4)
        v = random_normal(sh, dtype=dt)

        # A deterministic value
        c = 0.5

        v[0, 0, 0] = c
        assert np.abs(v[0, 0, 0].mean() - c) < tol
        assert np.abs(v[0, 0, 0].cov()) < tol
        assert v.a.dtype == dt

        v[0, :, 0] = c
        assert np.max(np.abs(v[0, :, 0].mean() - c)) < tol
        assert np.max(np.abs(v[0, :, 0].cov())) < tol
        assert v.a.dtype == dt

        v[1, ...] = c
        assert np.max(np.abs(v[1, ...].mean() - c)) < tol
        assert np.max(np.abs(v[1, ...].cov())) < tol
        assert v.a.dtype == dt

    for dt in cdtype_list:
        tol = 100 * np.finfo(dt).eps

        sh = (2, 3, 4)
        v = random_normal(sh, dtype=dt)

        # A deterministic value
        c = 0.5 - 3j

        v[0, 0, 0] = c
        assert np.abs(v[0, 0, 0].mean() - c) < tol
        assert np.abs(v[0, 0, 0].cov()) < tol
        assert v.a.dtype == dt

        v[0, :, 0] = c
        assert np.max(np.abs(v[0, :, 0].mean() - c)) < tol
        assert np.max(np.abs(v[0, :, 0].cov())) < tol
        assert v.a.dtype == dt

        v[1, ...] = c
        assert np.max(np.abs(v[1, ...].mean() - c)) < tol
        assert np.max(np.abs(v[1, ...].cov())) < tol
        assert v.a.dtype == dt

    for dt in rdtype_list + cdtype_list:
        tol = 100 * np.finfo(dt).eps

        # A sub-array
        v = random_normal(sh, dtype=dt)
        cov_ref = v[:, :, [0, 1, 3]].cov()
        mean_ref = v[:, :, [0, 1, 3]].mean()

        x = v[0, 0, 1]
        v[0, 0, 2] = x
        assert np.max(np.abs(v[:, :, [0, 1, 3]].cov() - cov_ref)) < tol
        assert np.max(np.abs(v[:, :, [0, 1, 3]].mean() - mean_ref)) < tol
        assert np.max(np.abs(v[0, 0, 2].mean() - x.mean())) < tol
        assert np.max(np.abs(v[0, 0, 2].cov())) - x.cov() < tol

        # An independent variable
        v = random_normal(sh, dtype=dt)
        cov_ref = v[:, :, [0, 3]].cov()
        mean_ref = v[:, :, [0, 3]].mean()

        x = random_normal((2,), dtype=dt)
        v[0, 0, 1:3] = x
        assert np.max(np.abs(v[:, :, [0, 3]].cov() - cov_ref)) < tol
        assert np.max(np.abs(v[:, :, [0, 3]].mean() - mean_ref)) < tol
        assert np.max(np.abs(v[0, 0, 1:3].mean() - x.mean())) < tol
        assert np.max(np.abs(v[0, 0, 1:3].cov() - x.cov())) < tol

        # With data type conversion
        tol_ = 100 * np.finfo(np.float32).eps
        x = random_normal((2,), dtype=np.float32)
        v[0, 0, 1:3] = x
        assert np.max(np.abs(v[:, :, [0, 3]].cov() - cov_ref)) < tol_
        assert np.max(np.abs(v[:, :, [0, 3]].mean() - mean_ref)) < tol_
        assert np.max(np.abs(v[0, 0, 1:3].mean() - x.mean())) < tol_
        assert np.max(np.abs(v[0, 0, 1:3].cov() - x.cov())) < tol_

        # With data type conversion 2
        x = random_normal((2,), dtype=np.float64)
        v[0, 0, 1:3] = x
        assert np.max(np.abs(v[:, :, [0, 3]].cov() - cov_ref)) < tol
        assert np.max(np.abs(v[:, :, [0, 3]].mean() - mean_ref)) < tol
        assert np.max(np.abs(v[0, 0, 1:3].mean() - x.mean())) < tol
        assert np.max(np.abs(v[0, 0, 1:3].cov() - x.cov())) < tol

        # Checks if correlations are preserved
        sz = 50
        szv = 10
        v1, v2, v3 = random_correlate([random_normal((sz,), dtype=dt),
                                       random_normal((sz,), dtype=dt),
                                       random_normal((szv,), dtype=dt)])
        
        # Asserts dependence.
        assert np.max(np.abs(cov(v1, v2))) > 0.1
        assert np.max(np.abs(cov(v1, v3))) > 0.1
        assert np.max(np.abs(cov(v2, v3))) > 0.1

        cov_ref = cov(v1[szv:], v2)
        assert np.max(np.abs(cov_ref)) > 0.1

        v1[:szv] = v3
        assert np.max(np.abs(cov(v1[szv:], v2) - cov_ref)) < tol
        assert np.max(np.abs(v1[:szv].mean() - v3.mean())) < tol
        assert np.max(np.abs(v1[:szv].cov() - v3.cov())) < tol


def test_asnormal():
    tol = 1e-8

    for sh in [tuple(), (3,), (2, 3)]:
        x = np.random.rand(*sh)
        v = asnormal(x)
        assert isinstance(v, Normal)
        assert np.max(np.abs(v.mean() - x)) < tol
        assert np.max(v.var()) < tol

    v = asnormal([normal(), 1, normal(1.2, 2.2)])
    assert isinstance(v, Normal)
    assert v.shape == (3,)
    assert np.max(np.abs(v.mean() - [0, 1, 1.2])) < tol
    assert np.max(np.abs(v.var() - [1, 0, 2.2])) < tol

    v = asnormal([normal(size=(2,)), [1, 1], normal(1.2, 2.2, size=2)])
    assert isinstance(v, Normal)
    assert v.shape == (3, 2)
    assert np.max(np.abs(v.mean() - [[0, 0], [1, 1], [1.2, 1.2]])) < tol
    assert np.max(np.abs(v.var() - [[1, 1], [0, 0], [2.2, 2.2]])) < tol

    v = asnormal([normal(size=(3,)), normal(1.2, 2.2, size=3)])
    assert isinstance(v, Normal)
    assert v.shape == (2, 3)
    assert np.max(np.abs(v.mean() - [[0, 0, 0], [1.2, 1.2, 1.2]])) < tol
    assert np.max(np.abs(v.var() - [[1, 1, 1], [2.2, 2.2, 2.2]])) < tol

    with pytest.raises(TypeError):
        asnormal(["s", NotImplemented])

    with pytest.raises(TypeError):
        asnormal([normal(), "s"])

    with pytest.raises(TypeError) as te1:
        asnormal("s")
    
    with pytest.raises(TypeError) as te2:
        asnormal(iid_repeat(normal(), 3))

    # Checks that the messages thrown in the two cases are different.
    assert te1.value.args[0] != te2.value.args[0]


def test_cov_func():
    tol = 1e-10

    sz = 10

    for dt in [np.float64, np.complex128]:

        # A scalar.
        v1 = random_normal(tuple(), dtype=dt)
        assert np.max(np.abs(cov(v1) - v1.var())) < tol
        assert np.max(np.abs(cov(v1) - v1.cov())) < tol
        
        # 1D, same dimensions.
        v1 = random_normal((sz,), dtype=dt)
        v2 = random_normal((sz,), dtype=dt)
        v1, v2 = random_correlate([v1, v2])

        c11 = cov(v1, v1)
        c12 = cov(v1, v2)
        assert np.abs(c12).max() > 0.1  # Ensures correlation.

        c = hstack([v1, v2]).cov()
        assert np.max(np.abs(c[:sz, :sz] - c11)) < tol
        assert np.max(np.abs(c[:sz, sz:] - c12)) < tol

        # 2D, different dimensions.
        v1 = random_normal((sz, 2), dtype=dt)
        v2 = random_normal((sz, 3), dtype=dt)
        v1, v2 = random_correlate([v1, v2])

        c11 = cov(v1, v1)
        c12 = cov(v1, v2)
        assert np.abs(c12).max() > 0.1  # Ensures correlation.

        c = hstack([v1, v2]).cov()
        assert np.max(np.abs(c[:, :2, :, :2] - c11)) < tol
        assert np.max(np.abs(c[:, :2, :, 2:] - c12)) < tol

        # Too few input arguments.
        with pytest.raises(TypeError):
            cov()

        # Too many input arguments.
        with pytest.raises(TypeError):
            cov(normal(), normal(), normal())


def test_iid_copy():
    tol = 1e-8

    v = random_normal((3, 4))
    v_ = v.iid_copy()

    assert v.shape == v_.shape
    assert v.iscomplex == v_.iscomplex
    assert np.max(np.abs(cov(v, v_))) < tol
    assert np.max(np.abs(v.mean() - v_.mean())) < tol
    assert np.max(np.abs(v.cov() - v_.cov())) < tol

    v_ = iid_copy(v)

    assert v.shape == v_.shape
    assert v.iscomplex == v_.iscomplex
    assert np.max(np.abs(cov(v, v_))) < tol
    assert np.max(np.abs(v.mean() - v_.mean())) < tol
    assert np.max(np.abs(v.cov() - v_.cov())) < tol

    v = random_normal((4, 3), dtype=np.complex64)
    v_ = v.iid_copy()

    assert v.shape == v_.shape
    assert v.iscomplex == v_.iscomplex
    assert np.max(np.abs(cov(v, v_))) < tol
    assert np.max(np.abs(v.mean() - v_.mean())) < tol
    assert np.max(np.abs(v.cov() - v_.cov())) < tol


def test_and():
    # Tests the combining operation & .
    
    tol = 1e-10

    v1 = normal() + 0.1
    v2 = normal() - v1
    v = (v2 & v1) & (v1 & v2)

    assert v.shape == (2, 2)
    assert np.max(np.abs(v.mean() - [[-0.1, 0.1], [0.1, -0.1]])) < tol

    vm = [[v2, v1], [v1, v2]]
    r2 = range(2)
    cov_ref = [[[[cov(vm[i][j], vm[k][l]) 
                  for i in r2] for j in r2] for k in r2] for l in r2]

    assert np.max(np.abs(v.cov() - cov_ref)) < tol

    with pytest.raises(TypeError):
        normal() & "s"

    with pytest.raises(TypeError):
        "s" & normal()