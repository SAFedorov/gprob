import pytest
import numpy as np
from normal import N, join, Normal
from utils import logp_batch


def test_N():
    xi = N()
    assert (xi.a == np.array([[1.]])).all()
    assert (xi.b == np.array([0.])).all()

    xi = N(1.3, 4)
    assert (xi.a == np.array([[2.]])).all()
    assert (xi.b == np.array([1.3])).all()

    xi = N(0, 4, dim=3)
    assert (xi.a == 2 * np.eye(3)).all()
    assert (xi.b == np.zeros(3)).all()

    with pytest.raises(ValueError):
        N(0, -4)
    
    xi = N(0, 0)
    assert (xi.a == np.array([[0.]])).all()
    assert (xi.b == np.array([0.])).all()

    mu = [1.3, 2.5]
    cov = [[1., 0.], [0., 0.]]
    xi = N(mu, cov)
    assert (xi.a @ xi.a.T == np.array(cov)).all()
    assert (xi.b == mu).all()

    cov = [[2.1, 0.5], [0.5, 1.3]]
    xi = N(0, cov)
    tol = 1e-14
    assert (np.abs(xi.a @ xi.a.T - np.array(cov)) < tol).all()
    assert (xi.b == 0).all()

    # Covariance matrices with negative eigenvalues are not allowed
    with pytest.raises(ValueError):
        N(0, [[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        N(0, [[0, 1], [1, 0]], lu=False)
    with pytest.raises(np.linalg.LinAlgError):
        N(0, [[0, 1], [1, 0]], lu=True)

    # But covariance matrices with zero eigenvalues are.
    cov = [[1., 0.], [0., 0.]]
    xi = N(0, cov)
    assert (xi.a @ xi.a.T == np.array(cov)).all()

    # Unless lu decomposition is explicitly requested.
    with pytest.raises(np.linalg.LinAlgError):
        N(0, [[0, 1], [1, 0]], lu=True)


def test_logp():
    # Validation of the log likelihood calculation

    nc = np.log(1/np.sqrt(2 * np.pi))  # Normalization constant.
    
    # Scalar variables
    xi = N()
    assert xi.logp(0) == nc
    assert xi.logp(1.1) == -1.1**2/2 + nc

    with pytest.raises(ValueError):
        xi.logp([0, 1])

    xi = N(0.9, 3.3)
    assert xi.logp(2) == (-(2-0.9)**2/(2 * 3.3) 
                          + np.log(1/np.sqrt(2 * np.pi * 3.3)))

    # Vector variables
    xi = N(0.9, 3.3, dim=2)
    assert xi.logp([2, 1]) == (-(2-0.9)**2/(2 * 3.3)-(1-0.9)**2/(2 * 3.3) 
                               + 2 * np.log(1/np.sqrt(2 * np.pi * 3.3)))

    xi = N(0.9, 3.3, dim=2)
    with pytest.raises(ValueError):
        xi.logp(0)
    with pytest.raises(ValueError):
        xi.logp([0, 0, 0])
    with pytest.raises(ValueError):
        xi.logp([[0], [0]])

    # Degenerate cases.
        
    # Zero scalar random variable.
    assert (0 * N()).logp(0.) > float("-inf")
    assert (0 * N()).logp(0.1) == float("-inf")

    # Deterministic variables.
    xi = join(N(), 1)
    assert xi.logp([0, 1.1]) == float("-inf")
    assert xi.logp([0, 1.]) == nc
    assert xi.logp([10.2, 1.]) == -(10.2)**2/(2) + nc
    
    # Degenerate covariance matrix. 
    xi1 = N()
    xi2 = 0 * N()
    xi12 = xi1 & xi2
    assert xi12.logp([1.2, 0]) == -(1.2)**2/(2) + nc
    assert xi12.logp([1.2, 0.1]) == float("-inf")


def test_logp_batch():
    # Validation of the log likelihood calculation, batch version.

    nc = np.log(1/np.sqrt(2 * np.pi))  # Normalization constant.
    
    # Scalar variables
    xi = N()
    assert logp_batch(xi, 0) == nc
    assert (logp_batch(xi, [0]) == np.array([nc])).all()
    assert (logp_batch(xi, [0, 0]) == np.array([nc, nc])).all()
    assert (logp_batch(xi, [[0], [0]]) == np.array([nc, nc])).all()
    assert logp_batch(xi, 2) == -4/2 + nc
    assert (logp_batch(xi, [0, 1.1]) == [nc, -1.1**2/2 + nc]).all()

    with pytest.raises(ValueError):
        logp_batch(xi, [[0, 1]])

    xi = N(0.9, 3.3)
    assert logp_batch(xi, 2) == (-(2-0.9)**2/(2 * 3.3) 
                          + np.log(1/np.sqrt(2 * np.pi * 3.3)))

    # Vector variables
    xi = N(0.9, 3.3, dim=2)
    assert logp_batch(xi, [2, 1]) == (-(2-0.9)**2/(2 * 3.3)-(1-0.9)**2/(2 * 3.3) 
                               + 2 * np.log(1/np.sqrt(2 * np.pi * 3.3)))
    
    res = [-(3.2-0.9)**2/(2 * 3.3)-(1.2-0.9)**2/(2 * 3.3) 
           + 2 * np.log(1/np.sqrt(2 * np.pi * 3.3)), 
           -(-1-0.9)**2/(2 * 3.3)-(-2.2-0.9)**2/(2 * 3.3) 
           + 2 * np.log(1/np.sqrt(2 * np.pi * 3.3))]
    
    assert (logp_batch(xi, [[3.2, 1.2], [-1., -2.2]]) == np.array(res)).all()

    xi = N(0.9, 3.3, dim=2)
    with pytest.raises(ValueError):
        logp_batch(xi, 0)
    with pytest.raises(ValueError):
        logp_batch(xi, [0, 0, 0])
    with pytest.raises(ValueError):
        logp_batch(xi, [[0], [0]])
    with pytest.raises(ValueError):
        logp_batch(xi, [[0, 0, 0]])

    # Degenerate cases.
        
    # Zero scalar random variable.
    assert (0 * N()).logp(0.) > float("-inf")
    assert (0 * N()).logp(0.1) == float("-inf")

    # Deterministic variables.
    xi = join(N(), 1)
    assert logp_batch(xi, [0, 1.1]) == float("-inf")
    assert (logp_batch(xi, [[0, 1.1]]) == np.array([float("-inf")])).all()
    assert logp_batch(xi, [0, 1.]) == nc
    assert (logp_batch(xi, [[0, 1], [1.1, 1], [0, 2]]) == [nc, -(1.1)**2/(2) + nc, 
                                                    float("-inf")]).all()
    
    # Degenerate covariance matrix. 
    xi1 = N()
    xi2 = 0 * N()
    xi12 = xi1 & xi2
    assert logp_batch(xi12, [1.2, 0]) == -(1.2)**2/(2) + nc
    assert logp_batch(xi12, [1.2, 0.1]) == float("-inf")
    assert (logp_batch(xi12, [[1, 0.1]]) == np.array([float("-inf")])).all()
    assert (logp_batch(xi12, [[1, 0.1], [1.2, 0]]) == [float("-inf"), 
                                                -(1.2)**2/(2) + nc]).all()

    # TODO: add higher-dimensional examples
    
    # Integrals of the probability density
    xi = N(0, 3.3)
    npt = 200000
    ls = np.linspace(-10, 10, npt)
    err = np.abs(1 - np.sum(np.exp(logp_batch(xi, ls))) * (20)/ npt)
    assert err < 6e-6  # should be 5.03694e-06

    xi = N(0, [[2.1, 0.5], [0.5, 1.3]])
    npt = 1000
    ls = np.linspace(-7, 7, npt)
    points = [(x, y) for x in ls for y in ls]
    err = np.abs(1 - np.sum(np.exp(logp_batch(xi, points))) * ((14)/ npt)**2)
    assert err < 2.5e-3  # should be 0.00200


def test_len():
    xi = N(0, 2, dim=2)

    with pytest.raises(NotImplementedError):
        len(xi)


def test_sample():
    
    # Checks the formats returned by sample()
    
    v = N(0, 1)
    s = v.sample()
    assert np.array(s).ndim == 0

    s = v.sample(3)
    assert s.shape == (3,)

    v = N(0, 1, dim=5)
    s = v.sample()
    assert s.shape == (5,)

    s = v.sample(3)
    assert s.shape == (3, 5)


def test_join():

    x = join(1)
    assert isinstance(x, Normal) and x.dim == 1

    v1 = N()
    v2 = N()
    v3 = N()

    vm = join(v1)  # single argument input
    assert vm == v1

    # single argument sequence
    vm = join([v1])
    assert vm.a == v1.a and vm.b == v1.b and vm.iids == v1.iids

    vm = join(v1, v2, v3)  # sequence input
    assert isinstance(vm, Normal) and vm.dim == 3

    vm = join([v1, v2, v3])  # list input
    assert isinstance(vm, Normal) and vm.dim == 3

    vm = join((v1, v2, v3))  # tuple input
    assert isinstance(vm, Normal) and vm.dim == 3

    v4 = N(0, 2, dim=2)
    vm = join(v1, v2, v4)
    assert isinstance(vm, Normal) and vm.dim == 4 and len(vm.iids) == 4


def test_iids_ordering():
    # Requires python >= 3.6

    def isordered(v):
        return all([v.iids[k] == i for i, k in enumerate(v.iids)])
    
    # Scalars

    v1 = N(0.1, 100)
    v2 = N()
    v3 = N()

    v4 = v1 + 0.8 * v3 + 200.
    assert isordered(v4)

    v5 = 0.3 * v4 + 0.4 * v2
    assert isordered(v5)

    v5 = 0.02 + 13. * v5 + 20 * v1
    assert isordered(v5)

    v5 = 0.02 + 13. * v5 + 0.4 * N() + 20 * v1
    assert isordered(v5)

    nrv = 20

    vl1 = [N() for _ in range(nrv)]
    assert isordered(join(vl1))

    vl2 = [N() for _ in range(nrv)]
    assert isordered(join(vl2))

    vl3 = [N() for _ in range(nrv // 2)]  # A shorter list.
    assert isordered(join(vl3))

    assert isordered(join(vl1 + vl2))
    assert isordered(join(vl3 + vl2))
    assert isordered(join(vl2 + vl3))
    assert isordered(join(vl2 + vl3 + vl2))
    assert isordered(join(vl2 + vl3 + vl2 + vl2 + vl1 + vl2))

    assert isordered(join(vl2) + join(vl1))
    assert isordered(join(vl2) - 1)

    # Shuffled lists of scalars

    idx1 = np.random.randint(0, len(vl1), size=len(vl1))
    idx12 = np.random.randint(0, len(vl1), size=len(vl1))
    idx2 = np.random.randint(0, len(vl2), size=len(vl2))
    vl1_s = [vl1[i] for i in idx1]
    vl12_s = [vl1[i] for i in idx12]
    vl2_s = [vl2[i] for i in idx2]
    
    assert isordered(join(vl1_s + vl1_s))
    assert isordered(join(vl12_s + vl1_s))
    assert isordered(join(vl2_s + vl1_s))
    assert isordered(join(vl1_s + vl2_s))

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

    vl1 = [N(0.3, 12., dim=3) for _ in range(nrv)]
    assert isordered(join(vl1))

    vl2 = [N(dim=3) for _ in range(nrv)]
    assert isordered(join(vl2))

    vl3 = [N(-3, 2., dim=3) for _ in range(nrv // 2)]  # A shorter list.
    assert isordered(join(vl3))

    assert isordered(join(vl1 + vl2))
    assert isordered(join(vl3 + vl2))
    assert isordered(join(vl2 + vl3))
    assert isordered(join(vl2 + vl3 + vl2))
    assert isordered(join(vl2 + vl3 + vl2 + vl2 + vl1 + vl2))

    assert isordered(join(vl2) + join(vl1))
    assert isordered(join(vl2) - 1)

    # Shuffled lists of vectors

    idx1 = np.random.randint(0, len(vl1), size=len(vl1))
    idx12 = np.random.randint(0, len(vl1), size=len(vl1))
    idx2 = np.random.randint(0, len(vl2), size=len(vl2))
    vl1_s = [vl1[i] for i in idx1]
    vl12_s = [vl1[i] for i in idx12]
    vl2_s = [vl2[i] for i in idx2]
    
    assert isordered(join(vl1_s + vl1_s))
    assert isordered(join(vl12_s + vl1_s))
    assert isordered(join(vl2_s + vl1_s))
    assert isordered(join(vl1_s + vl2_s))

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
    def isclose(v1, v2, tol=1e-14):
        return ((np.abs(v1.a - v2.a) < tol).all() 
                and (np.abs(v1.b - v2.b) < tol).all())

    v1 = [2, 3] - N(0, 1, dim=2)
    v2 = np.array([2, 3]) - N(0, 1, dim=2)
    assert isclose(v1, v2)

    v1 = -N(0, 1, dim=2) + np.array([2, 3])
    v2 = np.array([2, 3]) - N(0, 1, dim=2)
    assert isclose(v1, v2)

    v1 = [2, 3] * N(0, 1, dim=2)
    v2 = np.array([2, 3]) * N(0, 1, dim=2)
    assert isclose(v1, v2)

    v1 = N(0, 1, dim=2) * np.array([2, 3])
    v2 = np.array([2, 3]) * N(0, 1, dim=2)
    assert isclose(v1, v2)

    v1 = N(0, 1, dim=2) * np.sqrt(2)
    v2 = np.sqrt(2) * N(0, 1, dim=2)
    assert isclose(v1, v2)

