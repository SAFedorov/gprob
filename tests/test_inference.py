import pytest
import numpy as np
from external.Infer import Infer
from plaingaussian.normal import normal, hstack, stack, covariance
from plaingaussian.func import ConditionError
from utils import random_normal, random_correlate

np.random.seed(0)


def test_conditioning():
    tol = 1e-15  # Tolerance for float

    # Single constraint

    # This package
    v1 = normal(0.93, 1)
    v2 = normal(0, 1)
    v3 = normal(0, 1)

    vm = hstack([v1, v2, v3])
    vc = vm | {v1 + 0.2*v2 + 0.4*v3: 1.4}  # Conditioning on a dict
    vc_ = vm | v1 + 0.2*v2 + 0.4*v3 - 1.4  # Conditioning on a single variable

    # GaussianInfer
    g = Infer()

    r1 = g.N(0.93, 1)
    r2 = g.N(0,1)
    r3 = g.N(0,1)

    g.condition(r1 + 0.2*r2 + 0.4*r3, 1.4)
    m = g.marginals(r1, r2, r3)

    # Validation
    assert (np.abs(vc.cov() - m.Sigma) < tol).all()
    assert (np.abs(vc.var() - np.diag(m.Sigma)) < tol).all()
    assert (np.abs(vc.mean() - m.b[:, 0]) < tol).all()

    assert (np.abs(vc_.cov() - m.Sigma) < tol).all()
    assert (np.abs(vc_.var() - np.diag(m.Sigma)) < tol).all()
    assert (np.abs(vc_.mean() - m.b[:, 0]) < tol).all()

    # Adding a second constraint
    vc2 = vm | {v1 + 0.2*v2 + 0.4*v3: 1.4, v2 + 0.8*v3: -1}
    
    g.condition(r2 + 0.8* r3, -1)
    m2 = g.marginals(r1, r2, r3)

    assert (np.abs(vc2.cov() - m2.Sigma) < tol).all()
    assert (np.abs(vc2.var() - np.diag(m2.Sigma)) < tol).all()
    assert (np.abs(vc2.mean() - m2.b[:, 0]) < tol).all()

    # Incompatible conditions
    with pytest.raises(ConditionError):
        hstack([v1, v2, v3]) | {v2: 0, v3: 1, v1:0, v1+v2:1}

    with pytest.raises(ConditionError):
        normal() | {0:1e-8}
    
    with pytest.raises(ConditionError):
        normal() | 1e-8

    # But compatible degenerate conditions should work.
    assert (normal() | {1:1.}).mean() == 0
    assert (normal() | 0).mean() == 0
    assert (normal() | {1:1.}).var() == 1
    assert (normal() | 0).var() == 1


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

    mn = [normal(0, 0.1) for _ in range(len(xs))]

    a = normal(0, 10)
    b = normal(0, 10)

    cond = {f(x): y + n for (x, y, n) in zip(xs, ys, mn)}

    ab = (a & b) | cond
    jointd = hstack([a, b, *mn]) | cond

    assert (np.abs(mfull.Sigma - jointd.cov()) < tol).all()
    assert (np.abs(mfull.b[:, 0] - jointd.mean()) < tol).all()
    assert (np.abs(mab.Sigma - ab.cov()) < tol).all()
    assert (np.abs(mab.b[:, 0] - ab.mean()) < tol).all()

    # Comparison to a vectorized calculation using this package

    fv = a * xs + b
    mnv = normal(0, 0.1, size=len(xs))

    ab2 = (a & b) | {fv: ys + mnv}
    jointd2 = hstack([a, b, mnv]) | {fv: ys + mnv}

    assert (np.abs(mfull.Sigma - jointd2.cov()) < tol).all()
    assert (np.abs(mfull.b[:, 0] - jointd2.mean()) < tol).all()
    assert (np.abs(mab.Sigma - ab2.cov()) < tol).all()
    assert (np.abs(mab.b[:, 0] - ab2.mean()) < tol).all()


def test_conditioning_commutativity():
    # Sequential conditioning on independent variables should be commutative.

    tol = 1e-10

    sh = (5, 2)
    v1, v2, v3, v4 = [random_normal(sh, dtype=np.float64) for _ in range(4)]

    v = 3.2*v1 + 4.1*v2 + 0.7*v3 + v4
    
    vc1 = v | {v1:0, v2:0, v3:0}
    vc2 = v | v1 | v2 | v3
    vc3 = v | v3 | v2 | v1
    vc4 = v | v2 | v3 | v1

    assert (np.abs(vc2.mean() - vc1.mean()) < tol).all()
    assert (np.abs(vc2.cov() - vc1.cov()) < tol).all()

    assert (np.abs(vc3.mean() - vc1.mean()) < tol).all()
    assert (np.abs(vc3.cov() - vc1.cov()) < tol).all()

    assert (np.abs(vc4.mean() - vc1.mean()) < tol).all()
    assert (np.abs(vc4.cov() - vc1.cov()) < tol).all()

    # Conditioning second time on the same condition does not do anything.
    vc1o = v | v1
    vc11o = v | v1 | v1
    assert (np.abs(vc1o.mean() - vc11o.mean()) < tol).all()
    assert (np.abs(vc1o.cov() - vc11o.cov()) < tol).all()


def test_complex_conditioning():
    tol = 1e-9

    sh = (5, 2)
    shc = (4, 1)

    v = random_normal(sh, dtype=np.complex128)
    vc = random_normal(shc, dtype=np.complex128)
    v, vc = random_correlate([v, vc])

    assert np.abs(covariance(v, vc)).max() > 0.1  # Asserts correlation.

    vr = hstack([v.real, v.imag])
    vcr = hstack([vc.real, vc.imag])

    # Complex-complex
    vcond = v | {vc: 0}
    vrcond = vr | {vcr: 0}
    vrcond2 = hstack([vcond.real, vcond.imag])
    assert np.max(np.abs(vrcond.mean() - vrcond2.mean())) < tol
    assert np.max(np.abs(vrcond.covariance() - vrcond2.covariance())) < tol

    # Complex-real
    vcond = v | vc.real
    vrcond = vr | vc.real
    vrcond2 = hstack([vcond.real, vcond.imag])
    assert np.max(np.abs(vrcond.mean() - vrcond2.mean())) < tol
    assert np.max(np.abs(vrcond.covariance() - vrcond2.covariance())) < tol

    # Real-complex
    vcond = v.real | vc
    vrcond = v.real | vcr
    assert np.max(np.abs(vrcond.mean() - vcond.mean())) < tol
    assert np.max(np.abs(vrcond.covariance() - vcond.covariance())) < tol

    # Complex mean but real map
    v.emap.a = v.emap.a.real
    vr = hstack([v.real, v.imag])

    vcond = v | vc
    vrcond = vr | vcr
    vrcond2 = hstack([vcond.real, vcond.imag])
    assert np.max(np.abs(vrcond.mean() - vrcond2.mean())) < tol
    assert np.max(np.abs(vrcond.covariance() - vrcond2.covariance())) < tol


def test_masked_conditioning():
    tol = 1e-10

    test_sets = [(
        (5,),  # sh
        (4,),  # shc
        [1, 2, 2, 4, 4],  # idx
    ), (
        (1,),  # sh
        (1, 1),  # shc
        [1],  # idx
    ), (
        (7,),  # sh
        (5, 2),  # shc
        [1, 1, 2, 3, 4, 5, 5],  # idx
    ), (
        (7, 2),  # sh
        (5,),  # shc
        [1, 1, 2, 3, 4, 5, 5],  # idx
    ), (
        (7, 3, 2),  # sh
        (6, 2),  # shc
        [1, 1, 2, 3, 4, 5, 5],  # idx
    )]

    for dt in [np.float64, np.complex128]:
        for sh, shc, idx in test_sets:
            mask = np.array([range(shc[0])] * sh[0]).T < idx

            v = random_normal(sh, dtype=dt)
            vc = random_normal(shc, dtype=dt)
            v, vc = random_correlate([v, vc])
            
            # Ensures correlation. 
            try:
                assert np.abs(covariance(v, vc)).max() > 1e-3
            except Exception:
                # random_correlate may produce no correlations 
                # with a probability non-negligible for very small arrays
                # because it uses random shuffling.
                vc = vc + v

            mc_cond = v.condition({vc: 0}, mask=mask)  # Causal mask

            ref = stack([v[i] | {vc[:idx[i]]: 0} for i in range(len(v))])
            assert np.max(np.abs(ref.mean() - mc_cond.mean())) < tol
            assert np.max(np.abs(ref.covariance() - mc_cond.covariance())) < tol

            ma_cond = v.condition({vc: 0}, mask=~mask)  # Anti-causal mask

            ref = stack([(v[i] | vc[idx[i]:]) if len(vc[idx[i]:]) > 0 else v[i] 
                        for i in range(len(v))])

            assert np.max(np.abs(ref.mean() - ma_cond.mean())) < tol
            assert np.max(np.abs(ref.covariance() - ma_cond.covariance())) < tol

            # Redundant checks of the variances.
            for i in range(len(v)):
                xi1 = v[i] | {vc[:idx[i]]: 0}
                xi2 = mc_cond[i]
                assert np.max(np.abs(xi1.mean() - xi2.mean())) < tol
                assert np.max(np.abs(xi1.variance() - xi2.variance())) < tol

            for i in range(len(v)):
                if len(vc[idx[i]:]) == 0:
                    break

                xi1 = v[i] | {vc[idx[i]:]: 0}
                xi2 = ma_cond[i]
                assert np.max(np.abs(xi1.mean() - xi2.mean())) < tol
                assert np.max(np.abs(xi1.variance() - xi2.variance())) < tol

        # A test with more than one condition variable
        sh = (5,)
        shc1 = (4,)
        shc2 = (4, 2)
        idx = [1, 1, 3, 3, 4]

        mask = np.array([range(shc1[0])] * sh[0]).T < idx

        for dt in [np.float64, np.complex128]:
            v = random_normal(sh, dtype=dt)
            vc1 = random_normal(shc1, dtype=dt)
            vc2 = -2.1 * vc1.reshape((4, 1)) + random_normal(shc2, dtype=dt)
            v, vc2 = random_correlate([v, vc2])
            
            # Ensures correlation.
            assert np.abs(covariance(v, vc1)).max() > 0.1
            assert np.abs(covariance(v, vc2)).max() > 0.1

            # Causal masking.
            mc_cond = v.condition({vc1: 0, vc2: 0}, mask=mask)
            
            ref = stack([(v[i] | {vc1[:idx[i]]: 0, vc2[:idx[i]]: 0}) 
                         for i in range(len(v))])
            assert np.max(np.abs(ref.mean() - mc_cond.mean())) < tol
            assert np.max(np.abs(ref.covariance() - mc_cond.covariance())) < tol

            # Anti-causal masking.
            ma_cond = v.condition({vc1: 0, vc2: 0}, mask=~mask)

            ref = stack([(v[i] | {vc1[idx[i]:]:0, vc2[idx[i]:]:0})
                         if len(vc1[idx[i]:]) > 0 else v[i]
                         for i in range(len(v))])

            assert np.max(np.abs(ref.mean() - ma_cond.mean())) < tol
            assert np.max(np.abs(ref.covariance() - ma_cond.covariance())) < tol