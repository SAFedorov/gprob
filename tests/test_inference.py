import sys
sys.path.append('..')  # Until there is a package structure.

import pytest
import numpy as np
from external.Infer import Infer
from plaingaussian.normal import N, join


def test_conditioning():
    
    tol = 1e-15  # Tolerance for float

    # Single constraint

    # This package
    v1 = N(0.93, 1)
    v2 = N(0, 1)
    v3 = N(0, 1)

    vm = join(v1, v2, v3)
    vc = vm | {v1 + 0.2*v2 + 0.4*v3: 1.4}


    # GaussianInfer
    g = Infer()

    r1 = g.N(0.93, 1)
    r2 = g.N(0,1)
    r3 = g.N(0,1)

    g.condition(r1 + 0.2*r2 + 0.4*r3, 1.4)
    m = g.marginals(r1, r2, r3)

    assert (np.abs(vc.cov() - m.Sigma) < tol).all()
    assert (np.abs(vc.var() - np.diag(m.Sigma)) < tol).all()
    assert (np.abs(vc.mean() - m.b[:, 0]) < tol).all()

    # Adding second constraint

    vc2 = vm | {v1 + 0.2*v2 + 0.4*v3: 1.4, v2 + 0.8* v3: -1}
    g.condition(r2 + 0.8* r3, -1)
    m2 = g.marginals(r1, r2, r3)

    assert (np.abs(vc2.cov() - m2.Sigma) < tol).all()
    assert (np.abs(vc2.var() - np.diag(m2.Sigma)) < tol).all()
    assert (np.abs(vc2.mean() - m2.b[:, 0]) < tol).all()

    # Incompatible conditions
    with pytest.raises(RuntimeError):
        join(v1, v2, v3) | {v2: 0, v3: 1, v1:0, v1+v2:1}


def test_linear_regression():
    # Using the linear regression example from GaussianInfer

    tol = 1e-10
    
    # GaussianInfer example

    g = Infer()

    xs = [1.0, 2.0, 2.25, 5.0, 10.0]
    ys = [-3.5, -6.4, -4.0, -8.1, -11.0]
    mn = [g.N(0, 0.1) for _ in range(len(xs))]

    a = g.N(0, 10)
    b = g.N(0, 10)

    f = lambda x: a*x + b

    for (x, y, n) in zip(xs, ys, mn):
        g.condition(f(x), y + n)

    mab = g.marginals(a, b)
    mfull = g.marginals(a, b, *mn)

    # Comparison to a non-vectorized calculation using this package

    mn = [N(0, 0.1) for _ in range(len(xs))]

    a = N(0, 10)
    b = N(0, 10)

    cond = {f(x): y + n for (x, y, n) in zip(xs, ys, mn)}

    ab = (a & b) | cond
    jointd = join(a, b, *mn) | cond

    assert (np.abs(mfull.Sigma - jointd.cov()) < tol).all()
    assert (np.abs(mfull.b[:, 0] - jointd.mean()) < tol).all()
    assert (np.abs(mab.Sigma - ab.cov()) < tol).all()
    assert (np.abs(mab.b[:, 0] - ab.mean()) < tol).all()

    # Comparison to a vectorized calculation using this package

    fv = a * xs + b
    mnv = N(0, 0.1, size=len(xs))

    ab2 = (a & b) | {fv: ys + mnv}
    jointd2 = join(a, b, mnv) | {fv: ys + mnv}

    assert (np.abs(mfull.Sigma - jointd2.cov()) < tol).all()
    assert (np.abs(mfull.b[:, 0] - jointd2.mean()) < tol).all()
    assert (np.abs(mab.Sigma - ab2.cov()) < tol).all()
    assert (np.abs(mab.b[:, 0] - ab2.mean()) < tol).all()
