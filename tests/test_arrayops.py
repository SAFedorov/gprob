# Tests for the high-level oprations, type dispatch, and error handling.

import pytest
import inspect
import numpy as np
import gprob as gp

from gprob.normal_ import Normal


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


def test_fallback_to_normal():

    # Gets the names of all decorated functions following the recipe 
    # from https://stackoverflow.com/questions/5910703/how-to-get-all-methods-of-a-python-class-with-given-decorator

    src_lines, _ = inspect.getsourcelines(gp.arrayops)
    func_names = []
    for i, line in enumerate(src_lines):
        if line.strip() == "@fallback_to_normal":
            next_line = src_lines[i+1]
            func_names.append(next_line.split("def")[1].split("(")[0].strip())

    assert len(func_names) >= 15  # May change in the future.

    excl_names = {"reshape", "moveaxis",  # require extra arguments
                  "split", "hsplit", "vsplit", "dsplit", 
                  "mean", "var", "cov"  # always produce numerical outputs
                  }

    for fn in func_names:
        if fn in excl_names:
            continue
        
        f = getattr(gp, fn)
        assert isinstance(f([[1, 2], [2, 3]]), Normal)

    # The separate tests for the cases that require extra input arguments.

    v = gp.reshape([1, 2, 3, 4], (2, 2))
    assert isinstance(v, Normal)

    v = gp.moveaxis([[1], [2], [3], [4]], 0, 1)
    assert isinstance(v, Normal)
    assert v.shape == (1, 4)

    v1, v2 = gp.split([1, 2, 3, 4], 2)
    assert isinstance(v1, Normal)
    assert isinstance(v2, Normal)

    v1, v2 = gp.hsplit([1, 2, 3, 4], 2)
    assert isinstance(v1, Normal)
    assert isinstance(v2, Normal)

    v1, v2 = gp.vsplit([[1, 2, 3], [4, 5, 6]], 2)
    assert isinstance(v1, Normal)
    assert isinstance(v2, Normal)

    v1, v2 = gp.dsplit([[[1, 2, 3, 4], [4, 5, 6, 4]]], 2)
    assert isinstance(v1, Normal)
    assert isinstance(v2, Normal)

    tol = 1e-8

    assert np.abs(gp.mean(0)) < tol
    assert np.abs(gp.var(0)) < tol
    assert np.max(np.abs(gp.cov(0))) < tol