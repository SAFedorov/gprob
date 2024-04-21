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


TEST_SHAPES = [[tuple()], 
               [(1,), (4,), (3,)], 
               [(3, 1), (3, 5), (2, 4)], 
               [(2, 3, 4), (3, 5, 2), (3, 2, 1)], 
               [(3, 2, 5, 4), (2, 1, 3, 4), (2, 3, 3, 2)],
               [(2, 3, 1, 5, 1)], 
               [(3, 2, 1, 2, 1, 4)]]


def _flatten(list_of_lists):
    return (elem for list in list_of_lists for elem in list)


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

    if test_shapes is None:
        test_shapes = _flatten(TEST_SHAPES)
    elif test_shapes == "1dmin":
        test_shapes = _flatten(TEST_SHAPES[1:])
    elif test_shapes == "2dmin":
        test_shapes = _flatten(TEST_SHAPES[2:])
    elif test_shapes == "3dmin":
        test_shapes = _flatten(TEST_SHAPES[3:])
    elif test_shapes == "4dmin":
        test_shapes = _flatten(TEST_SHAPES[4:])
        
    for sh in test_shapes:
        vin = _random_normal(sh)
        vout = f(vin, *args, **kwargs)
        npf = getattr(np, f.__name__)

        assert vin.emap.a.shape[1:] == vin.b.shape
        assert vout.emap.a.shape[1:] == vout.b.shape

        refmean = npf(vin.b, *args, **kwargs)
        assert np.all(vout.b == refmean)

        dt = refmean.dtype
        tol = 6 * np.finfo(dt).eps

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
    _test_array_func(transpose, axes=(1, 0, 2), test_shapes=TEST_SHAPES[3])
    _test_array_func(transpose, axes=(-2, 0, -1), test_shapes=TEST_SHAPES[3])
    _test_array_func(transpose, axes=(2, 1, 3, 0), test_shapes=TEST_SHAPES[4])
    _test_array_func(transpose, axes=(-3, -1, -2, 0), test_shapes=TEST_SHAPES[4])
    _test_array_func(transpose, axes=(2, 1, 3, 5, 4, 0), test_shapes=TEST_SHAPES[6])


def test_moveaxis():
    _test_array_func(moveaxis, 0, 0, test_shapes="1dmin")
    _test_array_func(moveaxis, 0, 1, test_shapes="2dmin")
    _test_array_func(moveaxis, -1, 0, test_shapes="2dmin")
    _test_array_func(moveaxis, -1, -2, test_shapes="3dmin")
    _test_array_func(moveaxis, 0, 2, test_shapes="3dmin")


def _test_array_method(name, *args, **kwargs):
    test_shapes=_flatten(TEST_SHAPES)

    for sh in test_shapes:
        vin = _random_normal(sh)
        vout = getattr(vin, name)(*args, **kwargs)

        assert vin.emap.a.shape[1:] == vin.b.shape
        assert vout.emap.a.shape[1:] == vout.b.shape

        refmean = getattr(vin.b, name)(*args, **kwargs)
        assert np.all(vout.b == refmean)

        dt = vout.b.dtype
        tol = 6 * np.finfo(dt).eps

        assert vout.emap.a.dtype == dt

        for arin, arout in zip(vin.emap.a, vout.emap.a):
            aref = getattr(arin, name)(*args, **kwargs)
            assert arout.shape == aref.shape
            assert np.allclose(arout, aref, rtol=tol, atol=tol * np.max(aref)) 
            

def test_flatten():
    _test_array_method("flatten")


def _test_concat_func(f, *args, test_shapes=None, **kwargs):
    ns = [1, 2, 3, 11]  # Numbers of input arrays.

    if test_shapes is None:
        test_shapes = _flatten(TEST_SHAPES)
    elif test_shapes == "1dmin":
        test_shapes = _flatten(TEST_SHAPES[1:])
    elif test_shapes == "2dmin":
        test_shapes = _flatten(TEST_SHAPES[2:])
    elif test_shapes == "3dmin":
        test_shapes = _flatten(TEST_SHAPES[3:])
    elif test_shapes == "4dmin":
        test_shapes = _flatten(TEST_SHAPES[4:])

    for sh in test_shapes:
        vins_max = _random_correlate([_random_normal(sh) for _ in range(ns[-1])])
        vins_list = [vins_max[:n] for n in ns]
        
        # Adds special cases of two inputs with different numbers of 
        # the elementary variables, because there are separate evaluation
        # branches for the optimization of those.
        vins2 = _random_correlate([_random_normal(sh) for _ in range(2)])
        vins2[0] = vins2[0] + np.random.rand(*sh) * normal(0.1, 0.9)

        vins_list += [[vins2[0], vins2[1]], [vins2[1], vins2[0]]]

        for vins in vins_list:

            vout = f(vins, *args, **kwargs)
            npf = getattr(np, f.__name__)

            assert all(vin.emap.a.shape[1:] == vin.b.shape for vin in vins)
            assert vout.emap.a.shape[1:] == vout.b.shape

            #print(f"\nsh = {sh}")  #TODO: delete
            #print("\n")
            #print(npf([vin.b for vin in vins], *args, **kwargs))
            #print(vout.b)
            #print("inputs=")
            #print([vin.b for vin in vins])
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
    #_test_concat_func(stack, dtype=np.float16) #TODO: add data type argument to emap functions


def test_concatenate():
    pass