import numpy as np
from functools import reduce
from gprob import maps, normal_, sparse


def asnormal(x):
    return maps.lift(normal_.Normal, x)


def assparsenormal(x):
    return sparse.lift(sparse.SparseNormal, x)


def random_normal(rng, shape, dtype=np.float64):
    """Generates a normal variable with random mean and latent map. 
    The mean and the map coefficients are uniformly distributed over [-1, 1] 
    for real data types, or made of the real and imaginary parts 
    that are uniformly distributed over [-1, 1] for complex data types."""

    sz = reduce(lambda x, y: x * y, shape, 1)

    if np.issubdtype(dtype, np.complexfloating):
        rdtype = dtype(0).real.dtype

        rmu = rng.uniform(-1, 1, sz).astype(rdtype)
        ra = rng.uniform(-1, 1, (2 * sz, sz)).astype(rdtype)
        imu = rng.uniform(-1, 1, sz).astype(rdtype)
        ia = rng.uniform(-1, 1, (2 * sz, sz)).astype(rdtype)

        mu = rmu + 1j * imu
        a = ra + 1j * ia
    else:
        mu = rng.uniform(-1, 1, sz).astype(dtype)
        a = rng.uniform(-1, 1, (sz, sz)).astype(dtype)

    assert mu.dtype == dtype
    assert a.dtype == dtype

    return normal_.Normal(a, mu).reshape(shape)


def random_det_normal(rng, shape, dtype=np.float64):
    """Generates a random deterministic array lifted to the rank of a normal 
    variable with zero fluctuations."""

    sz = reduce(lambda x, y: x * y, shape, 1)

    if np.issubdtype(dtype, np.complexfloating):
        rdtype = dtype(0).real.dtype

        rmu = rng.uniform(-1, 1, sz).astype(rdtype)
        imu = rng.uniform(-1, 1, sz).astype(rdtype)

        mu = rmu + 1j * imu
    else:
        mu = rng.uniform(-1, 1, sz).astype(dtype)

    v = asnormal(mu)
    assert v.b.dtype == dtype
    assert v.a.dtype == dtype

    return v.reshape(shape)


def random_correlate(rng, vs):
    # Correlates the input variables by randomly mixing their latent keys.
    union_elems = set().union(*list(v.lat.keys() for v in vs))

    for v in vs:
        new_ind  = rng.choice(list(union_elems), size=len(v.lat), 
                              replace=False)
        v.lat = {i: v.lat[k] for i, k in zip(new_ind, v.lat)}
    return vs


def get_message(e):
    """Extracts the message from the error object captured in 
    `pytest.raises(SomeException) as e`
    """
    return e.value.args[0]