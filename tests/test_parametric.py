import pytest
import numpy as np
from numpy.exceptions import ComplexWarning
import gprob as gp
from gprob import normal, iid
from utils import random_normal

np.random.seed(0)


try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    from parametric import pnormal

    have_jax = True
except ImportError:
    have_jax = False


@pytest.mark.skipif(not have_jax, reason="jax is not installed")
def test_parametric_methods():
    # Tests logp, dlogp, fisher, natdlogp

    tol = 1e-8

    def call_methods(v, p, x):
        llk1 = v.logp(p, x)
        llk2 = v(p).logp(x)
        assert llk1.shape == llk2.shape
        assert np.abs(llk1 - llk2) < tol

        # Simply checks that the methods don't fail when called
        v.dlogp(p, x)
        v.natdlogp(p, x)
        v.fisher(p)
        return

    dt_list = [np.float64, np.complex128]
    for dt in dt_list:

        # Single scalar input
        sh = tuple()
        vin = random_normal(sh, dtype=dt)
        vp = pnormal(lambda p, v: p[1] * v + p[0], vin)
        p0 = [1., 2.]
        x = 0.1

        call_methods(vp, p0, x)

        # Two scalar inputs
        sh = tuple()
        vin1, vin2 = random_normal(sh, dtype=dt), random_normal(sh, dtype=dt)
        vp = pnormal(lambda p, v: p[0] * v[0] + p[1] * v[1], [vin1, vin2])
        p0 = [1., 2.]
        x = 0.1

        call_methods(vp, p0, x)

        # Single multi-dimensional input
        sh = (3, 2, 4)
        vin = random_normal(sh, dtype=dt)
        vp = pnormal(lambda p, v: v @ p, vin, jit=False)
        p0 = np.array([1., 2., 0.5, 0.1])
        x = np.random.rand(3, 2)

        call_methods(vp, p0, x)

        # Several multi-dimensional inputs with different sizes
        sh = (3, 1, 2)
        vin1 = random_normal(sh, dtype=dt)
        vin2 = random_normal((1,), dtype=dt)
        vp = pnormal(lambda p, v: v[0] @ p + v[1] * p[0], [vin1, vin2], jit=False)
        p0 = np.array([1., 2.])
        x = np.random.rand(3, 1)

        call_methods(vp, p0, x)

    vp = pnormal(lambda p, v: p[0] * v + p[1], normal())
    p0 = [1., 2.]
    with pytest.warns(ComplexWarning):
        vp.logp(p0, 1+0.j)


def test_unary_definitions():
    # Checks that all the unary functions are defined in 
    # the top-level name space
    
    global FN_LIST
    FN_LIST = [gp.exp, gp.exp2, gp.log, gp.log2, gp.log10, gp.sqrt, 
               gp.sin, gp.cos, gp.tan, gp.arcsin, gp.arccos, gp.arctan, 
               gp.sinh, gp.cosh, gp.tanh, gp.arcsinh, gp.arctanh, 
               gp.conjugate, gp.conj, gp.absolute, gp.abs]


def test_linearized_unaries_1():
    # Tests some basic properties of the linearized unary functions 
    # without using jax.

    assert gp.conjugate is gp.conj
    assert gp.absolute is gp.abs
    
    tol = 1e-8
    sh_list = [tuple(), (2,), (2, 3)]
    dt_list = [np.float64, np.complex128]

    for dt in dt_list:
        for sh in sh_list:
            vin = 0.99 * np.pi * random_normal(sh, dtype=dt)
            
            vout1 = gp.abs(vin) ** 2
            vout2 = vin * gp.conj(vin)

            assert np.allclose(vout1.b, vout2.b, rtol=tol, atol=tol)
            assert np.allclose(vout1.a, vout2.a, rtol=tol, atol=tol)

            vout1 = gp.exp(2 * vin)
            vout2 = gp.exp(vin) * gp.exp(vin)

            assert np.allclose(vout1.b, vout2.b, rtol=tol, atol=tol)
            assert np.allclose(vout1.a, vout2.a, rtol=tol, atol=tol)

            vout1 = gp.exp(np.log(2.) * vin)
            vout2 = gp.exp2(vin)

            assert np.allclose(vout1.b, vout2.b, rtol=tol, atol=tol)
            assert np.allclose(vout1.a, vout2.a, rtol=tol, atol=tol)

            vout1 = gp.sin(2 * vin)
            vout2 = 2 * gp.sin(vin) * gp.cos(vin)

            assert np.allclose(vout1.b, vout2.b, rtol=tol, atol=tol)
            assert np.allclose(vout1.a, vout2.a, rtol=tol, atol=tol)

            vout1 = gp.tan(vin)
            vout2 = gp.sin(vin) / gp.cos(vin)

            assert np.allclose(vout1.b, vout2.b, rtol=tol, atol=tol)
            assert np.allclose(vout1.a, vout2.a, rtol=tol, atol=tol)

            vout1 = gp.tanh(vin)
            vout2 = gp.sinh(vin) / gp.cosh(vin)

            assert np.allclose(vout1.b, vout2.b, rtol=tol, atol=tol)
            assert np.allclose(vout1.a, vout2.a, rtol=tol, atol=tol)

            vin = 0.49 * np.pi * random_normal(sh, dtype=dt)
            vout1 = vin
            vout2 = gp.arcsin(gp.sin(vin))

            assert np.allclose(vout1.b, vout2.b, rtol=tol, atol=tol)
            assert np.allclose(vout1.a, vout2.a, rtol=tol, atol=tol)

            vout1 = vin
            vout2 = gp.arctan(gp.tan(vin))

            assert np.allclose(vout1.b, vout2.b, rtol=tol, atol=tol)
            assert np.allclose(vout1.a, vout2.a, rtol=tol, atol=tol)

            vin = 0.49 * np.pi * (1 + random_normal(sh, dtype=dt))
            vout1 = vin
            vout2 = gp.arccos(gp.cos(vin))

            assert np.allclose(vout1.b, vout2.b, rtol=tol, atol=tol)
            assert np.allclose(vout1.a, vout2.a, rtol=tol, atol=tol)

            vin = random_normal(sh, dtype=dt)
            vout1 = vin
            vout2 = gp.arcsinh(gp.sinh(vin))

            assert np.allclose(vout1.b, vout2.b, rtol=tol, atol=tol)
            assert np.allclose(vout1.a, vout2.a, rtol=tol, atol=tol)

            vout1 = vin
            vout2 = gp.arctanh(gp.tanh(vin))

            assert np.allclose(vout1.b, vout2.b, rtol=tol, atol=tol)
            assert np.allclose(vout1.a, vout2.a, rtol=tol, atol=tol)

            vin = 1 + random_normal(sh, dtype=dt)
            vout1 = vin
            vout2 = gp.arccosh(gp.cosh(vin))

            assert np.allclose(vout1.b, vout2.b, rtol=tol, atol=tol)
            assert np.allclose(vout1.a, vout2.a, rtol=tol, atol=tol)


@pytest.mark.skipif(not have_jax, reason="jax is not installed")
def test_linearized_unaries_2():
    tol = 1e-8
    sh_list = [tuple(), (2,), (2, 3)]
    dt_list = [np.float64, np.complex128]

    for dt in dt_list:
        for sh in sh_list:
            for fn in FN_LIST:
                jfn = getattr(jnp, fn.__name__)

                if dt is np.complex128 and fn in (gp.cbrt,):
                    # Cubic root in numpy is not supported for complex types.
                    continue

                vin = 0.5 + random_normal(sh, dtype=dt) / 4  # scales to [0, 1]
                vout = fn(vin)
                vout_p = pnormal(lambda p, v: jfn(v), vin)(0.)

                assert np.allclose(vout.b, vout_p.b, rtol=tol, atol=tol)
                assert np.allclose(vout.a, vout_p.a, rtol=tol, atol=tol)
            
            fn = gp.arccosh  
            jfn = getattr(jnp, fn.__name__)
            # This function is special because it needs the inputs 
            # to be greater than 1.
            
            vin = 1.5 + random_normal(sh) / 4
            vout = fn(vin)
            vout_p = pnormal(lambda p, v: jfn(v), vin)(0.)

            assert np.allclose(vout.b, vout_p.b, rtol=tol, atol=tol)
            assert np.allclose(vout.a, vout_p.a, rtol=tol, atol=tol)


def test_sparse_linearized_unaries():
    tol = 1e-8

    ro = np.random.rand(3, 2)  # should be in [0, 1] 
    rs = np.random.rand(3, 2) - 0.5

    vin = ro + rs * normal(size=(3, 2))
    svin = ro + rs * iid(normal(size=2), 3)

    for fn in FN_LIST + [gp.cbrt]:
        vout = fn(vin)
        svout = fn(svin)

        assert np.max(np.abs(vout.mean() - svout.mean())) < tol
        assert np.max(np.abs(vout.var() - svout.var())) < tol
        assert svout.iaxes == svin.iaxes

    # Complex numbers.
    ro = np.random.rand(3, 2) + 1j * np.random.rand(3, 2)
    rs = (np.random.rand(3, 2) - 0.5) + 1j * (np.random.rand(3, 2) - 0.5)

    vin = ro + rs * normal(size=(3, 2))
    svin = ro + rs * iid(normal(size=2), 3)

    for fn in FN_LIST:
        vout = fn(vin)
        svout = fn(svin)

        assert np.max(np.abs(vout.mean() - svout.mean())) < tol
        assert np.max(np.abs(vout.var() - svout.var())) < tol
        assert svout.iaxes == svin.iaxes

    # arccosh - a special case because it needs the input mean to be > 1.
    ro = 1 + np.random.rand(3, 2)
    rs = np.random.rand(3, 2) - 0.5

    vin = ro + rs * normal(size=(3, 2))
    svin = ro + rs * iid(normal(size=2), 3)

    vout = gp.arccosh(vin)
    svout = gp.arccosh(svin)

    assert np.max(np.abs(vout.mean() - svout.mean())) < tol
    assert np.max(np.abs(vout.var() - svout.var())) < tol
    assert svout.iaxes == svin.iaxes

    ro = 1 + np.random.rand(3, 2) + 1j * np.random.rand(3, 2)
    rs = (np.random.rand(3, 2) - 0.5) + 1j * (np.random.rand(3, 2) - 0.5)

    vin = ro + rs * normal(size=(3, 2))
    svin = ro + rs * iid(normal(size=2), 3)

    vout = gp.arccosh(vin)
    svout = gp.arccosh(svin)

    assert np.max(np.abs(vout.mean() - svout.mean())) < tol
    assert np.max(np.abs(vout.var() - svout.var())) < tol
    assert svout.iaxes == svin.iaxes