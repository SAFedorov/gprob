# Tests for the high-level oprations, type dispatch, and error handling.

import pytest
import numpy as np
import gprob as gp


def test_cov():
    x = gp.cov([1, 1])
    assert x.shape == (2, 2)
    assert np.max(np.abs(x)) < 1e-10

    x = gp.cov([1], [1])
    assert x.shape == (1, 1)
    assert np.max(np.abs(x)) < 1e-10

    # Wrong numbers of inputs.

    with pytest.raises(TypeError):
        gp.cov()

    with pytest.raises(TypeError):
        gp.cov(gp.normal(), gp.normal(), gp.normal())

    # Type dispatch (the resulting shape depends on the type).

    x = gp.cov(gp.normal(size=(2, 3)), gp.normal(size=(2, 3)))
    assert x.shape == (2, 3, 2, 3)

    x = gp.cov(gp.normal(size=(3, 4)))
    assert x.shape == (3, 4, 3, 4)

    x = gp.cov(gp.normal(size=(2, 3)), np.ones((2, 3)))
    assert x.shape == (2, 3, 2, 3)

    x = gp.cov(gp.iid_repeat(gp.normal(), 4), gp.iid_repeat(gp.normal(), 4))
    assert x.shape == (4,)

    v1 = gp.iid_repeat(gp.normal(size=3), 2)  # shape (2, 3)
    v2 = gp.iid_repeat(gp.normal(size=(3, 4)), 2)  # shape (2, 3, 4)

    x = gp.cov(v1, v2)
    assert x.shape == (3, 3, 4, 2)

    x = gp.cov(v1, np.ones((2, 3)))
    assert x.shape == (3, 3, 2)
    assert np.max(np.abs(x)) < 1e-10


