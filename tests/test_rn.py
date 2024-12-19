import numpy as np
from numpy.random import Generator, SFC64

import gprob as gp


def _get_samples():
    """Produces a few samples from variable of different shapes and types."""

    v1 = gp.normal()
    v2 = gp.normal(size=(2, 3))
    v3 = gp.iid(v1, 3)
    v4 = gp.iid(gp.iid(gp.iid(v2, 2), 3), 4, axis=-1)

    return (v1.sample(), v2.sample(), v3.sample(), v4.sample())


def test_setgen():
    assert isinstance(gp.rn.gen, Generator)
    _get_samples()

    gp.rn.setgen(SFC64())
    assert isinstance(gp.rn.gen, Generator)
    assert isinstance(gp.rn.gen.bit_generator, SFC64)
    _get_samples()

    gp.rn.setgen(0)
    assert isinstance(gp.rn.gen, Generator)
    sl1 = _get_samples()

    gp.rn.setgen(0)
    assert isinstance(gp.rn.gen, Generator)
    sl2 = _get_samples()

    for x, y in zip(sl1, sl2):
        assert np.max(np.abs(1 - x / y)) < 1e-15