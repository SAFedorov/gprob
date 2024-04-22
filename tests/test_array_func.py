import sys
sys.path.append('..')  # Until there is a package structure.


from functools import reduce
import numpy as np
from numpy.random import default_rng

import plaingaussian.emaps as emaps
from plaingaussian.normal import (normal,
                                  stack, hstack, vstack, dstack, concatenate,
                                  split, hsplit, vsplit, dsplit,
                                  sum, cumsum, trace, diagonal, reshape, moveaxis, ravel, transpose,
                                  einsum, dot, matmul, inner, outer, kron, tensordot)

from plaingaussian.fft import (fft, fft2, fftn, 
                               ifft, ifft2, ifftn,
                               rfft, rfft2, rfftn,
                               irfft, irfft2, irfftn,
                               hfft, ihfft)

def _gts(ndim = None):
    """Get test shapes"""
    def flatten(list_of_lists):
        return (elem for list in list_of_lists for elem in list)

    test_shapes = [[tuple()], 
                   [(1,), (3,), (10,)], 
                   [(3, 1), (3, 5), (2, 4), (2, 10), (10, 2)], 
                   [(2, 3, 4), (3, 5, 1), (3, 8, 2), (3, 2, 10)], 
                   [(3, 2, 5, 4), (2, 1, 3, 4), (2, 3, 3, 2)],
                   [(2, 3, 1, 5, 1)], 
                   [(3, 2, 1, 2, 1, 4)]]
    
    if ndim is None:
        return flatten(test_shapes)
    
    if isinstance(ndim, int):
        return test_shapes[ndim]
    
    if isinstance(ndim, str):
        return flatten(test_shapes[int(ndim[0]):])
    
    raise ValueError("ndim must be None, an integer or a string")


def _random_normal(shape):
    sz = reduce(lambda x, y: x * y, shape, 1)
    mu = 2. * np.random.rand(sz) - 1.
    a = 2 * np.random.rand(sz, sz) - 1.
    cov = a.T @ a
    return normal(mu, cov).reshape(shape)


def _random_correlate(vs):
    # Correlates the input variables by randomly mixing their elementary keys.
    union_elems = set().union(*list(v.emap.elem.keys() for v in vs))

    rng = default_rng()

    for v in vs:
        new_ind  = rng.choice(list(union_elems), size=len(v.emap.elem), 
                              replace=False)
        v.emap.elem = {i: v.emap.elem[k] for i, k in zip(new_ind, v.emap.elem)}
    return vs


def _test_array_func(f, *args, test_shapes=None, **kwargs):
    
    # single input and single output functions
    npf = getattr(np, f.__name__)

    if test_shapes is None or isinstance(test_shapes, (str, int)):
        test_shapes = _gts(test_shapes)
        
    for sh in test_shapes:
        vin = _random_normal(sh)
        vout = f(vin, *args, **kwargs)

        assert vin.emap.a.shape[1:] == vin.b.shape
        assert vout.emap.a.shape[1:] == vout.b.shape

        refmean = npf(vin.b, *args, **kwargs)
        assert vout.b.shape == refmean.shape
        assert vout.b.dtype == refmean.dtype
        assert np.all(vout.b == refmean)

        dt = refmean.dtype
        tol = 10 * np.finfo(dt).eps

        assert vout.emap.a.dtype == dt

        for arin, arout in zip(vin.emap.a, vout.emap.a):
            aref = npf(arin, *args, **kwargs)
            assert arout.shape == aref.shape
            assert np.allclose(arout, aref, rtol=tol, atol=tol * np.max(arin))


def test_sum():
    _test_array_func(sum)
    _test_array_func(sum, axis=0)
    _test_array_func(sum, axis=0, keepdims=True)
    _test_array_func(sum, axis=0, dtype=np.float16)
    _test_array_func(sum, axis=-1, keepdims=True)
    _test_array_func(sum, axis=-2, keepdims=False, test_shapes="2dmin")
    _test_array_func(sum, axis=-2, keepdims=True, test_shapes="2dmin")
    _test_array_func(sum, axis=1, keepdims=True, dtype=np.float16, test_shapes="2dmin")
    _test_array_func(sum, axis=2, test_shapes="3dmin")
    _test_array_func(sum, axis=-2, test_shapes="3dmin")
    _test_array_func(sum, axis=(0, 1), test_shapes="2dmin")
    _test_array_func(sum, axis=(0, 2), test_shapes="3dmin")
    _test_array_func(sum, axis=(1, -1), test_shapes="3dmin")
    _test_array_func(sum, axis=(-1, -2), keepdims=True, test_shapes="3dmin")


def test_cumsum():
    _test_array_func(cumsum)
    _test_array_func(cumsum, axis=0)
    _test_array_func(cumsum, axis=0, dtype=np.float16)
    _test_array_func(cumsum, axis=-1)
    _test_array_func(cumsum, axis=-2, test_shapes="2dmin")
    _test_array_func(cumsum, axis=1, dtype=np.float16, test_shapes="2dmin")
    _test_array_func(cumsum, axis=2, test_shapes="3dmin")
    _test_array_func(cumsum, axis=-2, test_shapes="3dmin")


def test_trace():
    _test_array_func(trace, test_shapes="2dmin")
    _test_array_func(trace, dtype=np.float16, test_shapes="2dmin")
    _test_array_func(trace, offset=1, test_shapes="2dmin")
    _test_array_func(trace, offset=-2, test_shapes="2dmin")
    _test_array_func(trace, offset=1, axis1=1, axis2=0, test_shapes="3dmin")
    _test_array_func(trace, offset=1, axis1=1, axis2=-1, test_shapes="4dmin")
    _test_array_func(trace, offset=1, axis1=-3, axis2=-2, test_shapes="4dmin")


def test_diagonal():
    _test_array_func(diagonal, test_shapes="2dmin")
    _test_array_func(diagonal, offset=1, test_shapes="2dmin")
    _test_array_func(diagonal, offset=-2, test_shapes="2dmin")
    _test_array_func(diagonal, offset=1, axis1=1, axis2=0, test_shapes="3dmin")
    _test_array_func(diagonal, offset=1, axis1=1, axis2=-1, test_shapes="4dmin")
    _test_array_func(diagonal, offset=0, axis1=-3, axis2=-2, test_shapes="4dmin")
    _test_array_func(diagonal, offset=-3, axis1=-3, axis2=-2, test_shapes="4dmin")


def test_ravel():
    _test_array_func(ravel)


def test_transpose():
    _test_array_func(transpose)
    _test_array_func(transpose, axes=(1, 0, 2), test_shapes=3)
    _test_array_func(transpose, axes=(-2, 0, -1), test_shapes=3)
    _test_array_func(transpose, axes=(2, 1, 3, 0), test_shapes=4)
    _test_array_func(transpose, axes=(-3, -1, -2, 0), test_shapes=4)
    _test_array_func(transpose, axes=(2, 1, 3, 5, 4, 0), test_shapes=6)


def test_moveaxis():
    _test_array_func(moveaxis, 0, 0, test_shapes="1dmin")
    _test_array_func(moveaxis, 0, 1, test_shapes="2dmin")
    _test_array_func(moveaxis, -1, 0, test_shapes="2dmin")
    _test_array_func(moveaxis, -1, -2, test_shapes="3dmin")
    _test_array_func(moveaxis, 0, 2, test_shapes="3dmin")


def test_reshape():
    def prime_factors(n):
        # https://stackoverflow.com/questions/15347174/python-finding-prime-factors
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors

    shapes = _gts()

    for sh in shapes:
        if len(sh) == 0:
            new_shapes = [tuple(), (1,), (1, 1, 1)]
        elif len(sh) == 1:
            new_shapes = [(1, sh[0]), (1, sh[0], 1)]
        elif len(sh) == 2:
            new_shapes = [(sh[0] * sh[1],), (sh[1], 1, sh[0])]
        elif len(sh) == 3:
            new_shapes = [(sh[0] * sh[1] * sh[2],), (sh[1], sh[0] * sh[2]),
                          (sh[0] * sh[1], sh[2])]
        else:
            sz = reduce(lambda x, y: x * y, sh)
            new_shapes = [(sz,), prime_factors(sz)]

        for new_sh in new_shapes:
            _test_array_func(reshape, new_sh, test_shapes=[sh])


def _test_array_func2(f, op1_shape=None, op2_shape=None, **kwargs):
    if op1_shape is None:
        shapes = [[tuple(), tuple()], [tuple(), (8,)], [tuple(), (2, 5)], 
                  [(3,), (7,)], [(11,), (1,)], [(5,), (3, 2)], [(5,), (3, 2, 4)],
                  [(3, 2), (4, 5)], [(2, 3), (5, 1, 4)], 
                  [(2, 3, 4), (2, 5, 7)], [(2, 3, 4), (2, 5, 6, 7)]]
        
        for sh1, sh2 in shapes:
            _test_array_func2(f, sh1, sh2, **kwargs)

        return
        
    npf = getattr(np, f.__name__)

    # Random variable first

    vin = _random_normal(op1_shape)
    op2 = (2. * np.random.rand(*op2_shape) - 1)

    vout = f(vin, op2, **kwargs)

    refmean = npf(vin.b, op2, **kwargs)
    assert vout.b.shape == refmean.shape
    assert vout.b.dtype == refmean.dtype
    assert np.all(vout.b == refmean)

    dt = refmean.dtype
    tol = 10 * np.finfo(dt).eps

    assert vout.emap.a.dtype == dt

    for arin, arout in zip(vin.emap.a, vout.emap.a):
        aref = npf(arin, op2, **kwargs)
        assert arout.shape == aref.shape

        atol = tol * max(1, np.max(np.abs(aref)))
        assert np.allclose(arout, aref, rtol=tol, atol=atol)

    # Random variable second
    
    op1 = np.random.uniform(-1., 1., size=op1_shape)
    vin = _random_normal(op2_shape)

    vout = f(op1, vin, **kwargs)

    refmean = npf(op1, vin.b, **kwargs)
    assert vout.b.shape == refmean.shape
    assert vout.b.dtype == refmean.dtype
    assert np.all(vout.b == refmean)

    assert vout.emap.a.dtype == refmean.dtype

    for arin, arout in zip(vin.emap.a, vout.emap.a):
        aref = npf(op1, arin, **kwargs)
        assert arout.shape == aref.shape

        atol = tol * max(1, np.max(np.abs(aref)))
        assert np.allclose(arout, aref, rtol=tol, atol=atol)  
    

def test_matmul():
    for sh in _gts(1):
        _test_array_func2(matmul, sh, sh)
        _test_array_func2(matmul, sh + sh, sh)
        _test_array_func2(matmul, (2, 3) + sh, sh)
        _test_array_func2(matmul, sh, sh + sh)
        _test_array_func2(matmul, sh, sh + (3,))
        _test_array_func2(matmul, sh, (2, 4) + sh + (7,))
        _test_array_func2(matmul, (8,) + sh, (2, 4) + sh + (7,))
        _test_array_func2(matmul, (5, 3) + sh, (5,) + sh + (7,))
        _test_array_func2(matmul, (3,) + sh, (5, 3) + sh + (7,))
        _test_array_func2(matmul, (5, 3) + sh, (5, 1) + sh + (7,))
        _test_array_func2(matmul, (5, 1, 3, 2) + sh, (5, 3, 1) + sh + (2,))


def test_dot():
    for sh in _gts(1):
        _test_array_func2(dot, sh, sh)
        _test_array_func2(dot, sh + sh, sh)
        _test_array_func2(dot, (2, 3) + sh, sh)
        _test_array_func2(dot, sh, sh + sh)
        _test_array_func2(dot, sh, sh + (3,))
        _test_array_func2(dot, sh, (2, 4) + sh + (7,))
        _test_array_func2(dot, (8,) + sh, (2, 4) + sh + (7,))
        _test_array_func2(dot, (5, 3) + sh, (5,) + sh + (7,))
        _test_array_func2(dot, (3,) + sh, (5, 3) + sh + (7,))
        _test_array_func2(dot, (5, 3) + sh, (5, 1) + sh + (7,))
        _test_array_func2(dot, (5, 1, 3, 2) + sh, (5, 3, 1) + sh + (2,))


def test_inner():
    for sh in _gts(1):
        _test_array_func2(inner, sh, sh)
        _test_array_func2(inner, sh + sh, sh)
        _test_array_func2(inner, (2, 3) + sh, sh)
        _test_array_func2(inner, sh, sh + sh)
        _test_array_func2(inner, sh, (2, 3,) + sh)
        _test_array_func2(inner, (2, 3,) + sh, sh)
        _test_array_func2(inner, (4,) + sh, (2, 4) + sh)
        _test_array_func2(inner, (2, 1) + sh, (5,) + sh)
        _test_array_func2(inner, (3,) + sh, (5, 3) + sh)
        _test_array_func2(inner, (5, 3) + sh, (3,) + sh)
        _test_array_func2(inner, (5, 1, 2) + sh, (5, 3, 1) + sh)


def test_tensordot():
    _test_array_func2(tensordot, axes=0)

    for sh in _gts(1):
        _test_array_func2(tensordot, sh, sh, axes=1)
        _test_array_func2(tensordot, sh, sh, axes=((0,), (0,)))
        _test_array_func2(tensordot, sh, sh, axes=[[0], [-1]])
        _test_array_func2(tensordot, sh, sh, axes=[[-1], [-1]])
        _test_array_func2(tensordot, sh, sh + (3,), axes=1)
        _test_array_func2(tensordot, sh, sh + (3,), axes=[[0], [0]])
        _test_array_func2(tensordot, sh, (3,) + sh, axes=[[0], [1]])
        _test_array_func2(tensordot, sh, (3,) + sh, axes=[[0], [-1]])
        _test_array_func2(tensordot, (2,) + sh, sh, axes=1)
        _test_array_func2(tensordot, (2,) + sh, sh, axes=[[-1], [0]])
        _test_array_func2(tensordot, sh + (2,), sh, axes=[[0], [0]])
        _test_array_func2(tensordot, (2,) + sh, sh + (3,), axes=1)
        _test_array_func2(tensordot, (2,) + sh, sh + (3,), axes=[[-1], [0]])
        _test_array_func2(tensordot, sh + (2,), sh + (3,), axes=[[0], [0]])
        _test_array_func2(tensordot, (2,) + sh, (3,) + sh, axes=[[1], [1]])
        _test_array_func2(tensordot, (2, 4) + sh, sh + (3, 7), axes=1)
        _test_array_func2(tensordot, (2, 4) + sh, (3, 7) + sh, axes=[[-1], [-1]])
        _test_array_func2(tensordot, (2,) + sh + (4,), (3, 7) + sh, axes=[[1], [2]])

    for sh in _gts(2):
        _test_array_func2(tensordot, sh, sh, axes=2)
        _test_array_func2(tensordot, sh, sh, axes=((0, 1), (0, 1)))
        _test_array_func2(tensordot, sh, sh[::-1], axes=((0, 1), (1, 0)))
        _test_array_func2(tensordot, sh, sh[::-1], axes=((1, 0), (0, 1)))
        _test_array_func2(tensordot, sh, sh, axes=[[-1, 0], [-1, 0]])
        _test_array_func2(tensordot, sh, sh + (3,), axes=2)
        _test_array_func2(tensordot, sh, sh + (3,), axes=[[0, 1], [0, -2]])
        _test_array_func2(tensordot, sh, sh[::-1] + (3,), axes=[[0, 1], [1, 0]])
        _test_array_func2(tensordot, sh, (3,) + sh, axes=[[1, 0], [2, 1]])
        _test_array_func2(tensordot, sh, (3,) + sh[::-1], axes=[[0, 1], [-1, -2]])
        _test_array_func2(tensordot, (2,) + sh, sh, axes=2)
        _test_array_func2(tensordot, (2,) + sh, sh, axes=[[-2, -1], [0, 1]])
        _test_array_func2(tensordot, sh + (2,), sh, axes=[[0, 1], [0, 1]])
        _test_array_func2(tensordot, (2,) + sh, sh + (3,), axes=2)
        _test_array_func2(tensordot, (2,) + sh, sh + (3,), axes=[[1, 2], [0, 1]])
        _test_array_func2(tensordot, sh + (2,), sh + (3,), axes=[[-3, -2], [0, 1]])
        _test_array_func2(tensordot, (2,) + sh, (3,) + sh, axes=[[-2, -1], [-2, -1]])
        _test_array_func2(tensordot, (2, 4) + sh, sh + (3, 7), axes=2)
        _test_array_func2(tensordot, (2, 4) + sh, (3, 7) + sh, axes=[[2, 3], [2, 3]])
        _test_array_func2(tensordot, (2,) + sh + (4,), (3, 7) + sh, axes=[[1, 2], [2, 3]])
        _test_array_func2(tensordot, (2,) + sh + (4,), sh[::-1] + (3, 7), axes=[[-2, -3], [0, 1]])

    for sh in _gts(3):
        _test_array_func2(tensordot, sh, sh, axes=3)
        _test_array_func2(tensordot, sh, sh + (3,), axes=3)
        _test_array_func2(tensordot, (3,) + sh, sh, axes=3)
        _test_array_func2(tensordot, (2, 3) + sh, sh + (1, 5), axes=3)
        _test_array_func2(tensordot, (sh[1], sh[0], 3, sh[2]), 
                          (sh[2], 1, sh[1], sh[0]), axes=[[1, 0, -1], [3, 2, 0]])

    for sh in _gts(4):
        _test_array_func2(tensordot, sh, sh, axes=4)
        _test_array_func2(tensordot, sh, sh + (3,), axes=4)
        _test_array_func2(tensordot, (3,) + sh, sh, axes=4)
        _test_array_func2(tensordot, (2, 3) + sh, sh + (1, 5), axes=4)
        _test_array_func2(tensordot, (sh[3], sh[1], sh[0], 3, sh[2]), 
                          (sh[2], 1, sh[1], sh[0], sh[3]), 
                          axes=[[2, 1, -1, 0], [3, 2, 0, -1]])

    for sh in _gts(5):
        _test_array_func2(tensordot, sh, sh, axes=5)


def test_outer():
    _test_array_func2(outer)


def test_kron():
    _test_array_func2(kron)
    

def _test_array_method(name, *args, test_shapes=None, **kwargs):
    if test_shapes is None or isinstance(test_shapes, (str, int)):
        test_shapes = _gts(test_shapes)

    for sh in test_shapes:
        vin = _random_normal(sh)
        vout = getattr(vin, name)(*args, **kwargs)

        assert vin.emap.a.shape[1:] == vin.b.shape
        assert vout.emap.a.shape[1:] == vout.b.shape

        refmean = getattr(vin.b, name)(*args, **kwargs)
        assert vout.b.shape == refmean.shape
        assert vout.b.dtype == refmean.dtype
        assert np.all(vout.b == refmean)

        dt = vout.b.dtype
        tol = 10 * np.finfo(dt).eps

        assert vout.emap.a.dtype == dt

        for arin, arout in zip(vin.emap.a, vout.emap.a):
            aref = getattr(arin, name)(*args, **kwargs)
            assert arout.shape == aref.shape

            atol = tol * max(1, np.max(np.abs(aref)))
            assert np.allclose(arout, aref, rtol=tol, atol=atol) 
            

def test_flatten():
    _test_array_method("flatten")


def _test_concat_func(f, *args, test_shapes=None, vins_list=None, **kwargs):
    npf = getattr(np, f.__name__)

    if test_shapes is None or isinstance(test_shapes, (str, int)):
        test_shapes = _gts(test_shapes)

    ns = [1, 2, 3, 11]  # Numbers of input arrays.

    if vins_list is None:
        vins_list = []
        for sh in test_shapes:
            vins_max = _random_correlate([_random_normal(sh) 
                                          for _ in range(ns[-1])])
            vins_list += [vins_max[:n] for n in ns]
            
            # Adds special cases of two inputs with different numbers of 
            # the elementary variables, because there are separate evaluation
            # branches for the optimization of those.
            vins2 = _random_correlate([_random_normal(sh) for _ in range(2)])
            vins2[0] = vins2[0] + np.random.rand(*sh) * normal(0.1, 0.9)

            vins_list += [[vins2[0], vins2[1]], [vins2[1], vins2[0]]]

    for vins in vins_list:
        vout = f(vins, *args, **kwargs)

        assert all(vin.emap.a.shape[1:] == vin.b.shape for vin in vins)
        assert vout.emap.a.shape[1:] == vout.b.shape

        assert np.all(vout.b == npf([vin.b for vin in vins], *args, **kwargs))
        assert vout.emap.a.dtype == vout.b.dtype

        vins_ext = emaps.complete([vin.emap for vin in vins])
        # Now the testing silently relies on the fact that emaps.complete
        # produces in the same order of the elementary random variables 
        # as emaps.concatenate or emaps.stack.

        for i in range(len(vout.emap.a)):
            arins = [vin.a[i] for vin in vins_ext]
            arout = vout.emap.a[i]
            aref = npf(arins, *args, **kwargs)
            assert arout.shape == aref.shape
            assert np.all(arout == aref)


def test_stack():
    _test_concat_func(stack)
    _test_concat_func(stack, axis=0, test_shapes="2dmin")
    _test_concat_func(stack, axis=1, test_shapes="2dmin")
    _test_concat_func(stack, axis=-1, test_shapes="1dmin")
    _test_concat_func(stack, axis=-2, test_shapes="3dmin")
    _test_concat_func(stack, dtype=np.float16) #TODO: add data type argument to emap functions


def test_vstack():
    _test_concat_func(vstack)


def test_hstack():
    _test_concat_func(hstack)


def test_dstack():
    _test_concat_func(dstack)


def test_concatenate():
    _test_concat_func(concatenate, test_shapes="1dmin")
    _test_concat_func(concatenate, axis=0, test_shapes="2dmin")
    _test_concat_func(concatenate, axis=1, test_shapes="2dmin")
    _test_concat_func(concatenate, axis=-1, test_shapes="1dmin")
    _test_concat_func(concatenate, axis=-2, test_shapes="3dmin")

    # Different array sizes along the concatenation axis.
    ts = _gts("2dmin")
    ax = 2
    for s in ts:
        vins = _random_correlate([_random_normal((*s[:ax], i, *s[ax:]))
                                  for i in range(1, 4)])
        _test_concat_func(concatenate, axis=ax, vins_list=[vins])

    ts = _gts("2dmin")
    ax = -2
    for s in ts:
        print(s)
        vins = _random_correlate([_random_normal((*s[:ax+1], i, *s[ax+1:]))
                                  for i in range(1, 4)])
        _test_concat_func(concatenate, axis=ax, vins_list=[vins])


def _test_split_func(f, test_shapes="1dmin", test_axis=None, **kwargs):
    npf = getattr(np, f.__name__)

    if test_axis is None:
        test_axis = kwargs.get("axis", 0)
    
    # single input and multiple output functions

    if test_shapes is None or isinstance(test_shapes, (str, int)):
        test_shapes = _gts(test_shapes)
        
    for sh in test_shapes:
        sz = sh[test_axis]
        args_lists = [[1], [sz], [[0]], [[sz]], [[sz//2]]]
        if sz > 3:
            args_lists += [[[sz//3, 2*sz//3]]]

        for args in args_lists:
            vin = _random_normal(sh)
            vouts = f(vin, *args, **kwargs)

            refmeans = npf(vin.b, *args, **kwargs)

            assert len(vouts) == len(refmeans)
            for vout, refmean in zip(vouts, refmeans):
                assert vout.b.dtype == refmean.dtype
                assert vout.b.shape == refmean.shape
                assert np.all(vout.b == refmean)

                assert vout.emap.a.dtype == refmean.dtype

            for i in range(len(vin.emap.a)):
                arin = vin.emap.a[i]
                arouts = [vout.emap.a[i] for vout in vouts]
                arefs = npf(arin, *args, **kwargs)

                for arout, aref in zip(arouts, arefs):
                    assert arout.shape == aref.shape
                    assert np.all(arout == aref)


def test_split():
    _test_split_func(split)
    _test_split_func(split, axis=1, test_shapes="2dmin")
    _test_split_func(split, axis=-2, test_shapes="3dmin")


def test_hsplit():
    _test_split_func(hsplit, test_axis=0, test_shapes=1)
    _test_split_func(hsplit, test_axis=1, test_shapes="2dmin")


def test_vsplit():
    _test_split_func(vsplit, test_axis=0, test_shapes="2dmin")


def test_dsplit():
    _test_split_func(dsplit, test_axis=2, test_shapes="3dmin")