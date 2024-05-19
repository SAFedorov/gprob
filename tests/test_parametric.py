import pytest
import numpy as np

np.random.seed(0)

from plaingaussian import (exp, exp2, log, log2, log10, sqrt, cbrt, sin, cos, 
                           tan, arcsin, arccos, arctan, sinh, cosh, tanh, 
                           arcsinh, arccosh, arctanh, conjugate, conj)
from plaingaussian import pnormal

from utils import random_normal

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

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


@pytest.mark.skipif(not have_jax, reason="jax is not installed")
def test_linearized_unaries():
    tol = 1e-8

    fn_list = [exp, exp2, log, log2, log10, sqrt, cbrt, sin, cos, tan, arcsin, 
               arccos, arctan, sinh, cosh, tanh, arcsinh, arctanh, 
               conjugate, conj]
    
    sh_list = [tuple(), (2,), (2, 3)]
    dt_list = [np.float64, np.complex128]

    for dt in dt_list:
        for sh in sh_list:
            for fn in fn_list:
                jfn = getattr(jnp, fn.__name__)

                if dt is np.complex128 and fn in (cbrt,):
                    # Cubic root in numpy is not supported for complex types.
                    continue

                vin = 0.5 + random_normal(sh, dtype=dt) / 4  # scales to [0, 1]
                vout = fn(vin)
                vout_p = pnormal(lambda p, v: jfn(v), vin)(0.)

                assert np.allclose(vout.b, vout_p.b, rtol=tol, atol=tol)
                assert np.allclose(vout.a, vout_p.a, rtol=tol, atol=tol)
            
            fn = arccosh  
            jfn = getattr(jnp, fn.__name__)
            # This function is special because it needs the inputs 
            # to be greater than 1.
            
            vin = 1.5 + random_normal(sh) / 4
            vout = fn(vin)
            vout_p = pnormal(lambda p, v: jfn(v), vin)(0.)

            assert np.allclose(vout.b, vout_p.b, rtol=tol, atol=tol)
            assert np.allclose(vout.a, vout_p.a, rtol=tol, atol=tol)