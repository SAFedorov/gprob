import numpy as np
from functools import reduce
from gprob.normal_ import Normal, asnormal


def random_normal(shape, dtype=np.float64):
    """Generates a normal variable with random mean and elementary map."""

    sz = reduce(lambda x, y: x * y, shape, 1)

    if np.issubdtype(dtype, np.complexfloating):
        rdtype = dtype(0).real.dtype

        rmu = (2. * np.random.rand(sz) - 1.).astype(rdtype)
        ra = (2 * np.random.rand(2 * sz, sz) - 1.).astype(rdtype)
        imu = (2. * np.random.rand(sz) - 1.).astype(rdtype)
        ia = (2 * np.random.rand(2 * sz, sz) - 1.).astype(rdtype)

        mu = rmu + 1j * imu
        a = ra + 1j * ia
    else:
        mu = (2. * np.random.rand(sz) - 1.).astype(dtype)
        a = (2 * np.random.rand(sz, sz) - 1.).astype(dtype)

    assert mu.dtype == dtype
    assert a.dtype == dtype

    return Normal(a, mu).reshape(shape)


def random_det_normal(shape, dtype=np.float64):
    """Generates a random deterministic array lifted to the rank of a normal 
    variable with zero fluctuations."""

    sz = reduce(lambda x, y: x * y, shape, 1)

    if np.issubdtype(dtype, np.complexfloating):
        rdtype = dtype(0).real.dtype

        rmu = (2. * np.random.rand(sz) - 1.).astype(rdtype)
        imu = (2. * np.random.rand(sz) - 1.).astype(rdtype)

        mu = rmu + 1j * imu
    else:
        mu = (2. * np.random.rand(sz) - 1.).astype(dtype)

    v = asnormal(mu)
    assert v.b.dtype == dtype
    assert v.a.dtype == dtype

    return v.reshape(shape)


def random_correlate(vs):
    # Correlates the input variables by randomly mixing their elementary keys.
    union_elems = set().union(*list(v.emap.elem.keys() for v in vs))

    rng = np.random.default_rng()

    for v in vs:
        new_ind  = rng.choice(list(union_elems), size=len(v.emap.elem), 
                              replace=False)
        v.emap.elem = {i: v.emap.elem[k] for i, k in zip(new_ind, v.emap.elem)}
    return vs