from functools import reduce
import copy
import numpy as np
import pytest

np.random.seed(0)

from gprob import maps
from gprob import (normal,
                   stack, hstack, vstack, dstack, concatenate,
                   split, hsplit, vsplit, dsplit,
                   sum, cumsum, trace, diagonal, reshape, moveaxis, ravel, 
                   transpose, add, subtract, multiply, divide, power, 
                   einsum, dot, matmul, inner, outer, kron, tensordot)

from gprob.fft import (fft, fft2, fftn, 
                       ifft, ifft2, ifftn,
                       rfft, rfft2, rfftn,
                       irfft, irfft2, irfftn,
                       hfft, ihfft)

from utils import random_normal, random_det_normal, random_correlate


def _gts(ndim = None):
    """Get test shapes"""
    def flatten(list_of_lists):
        return (elem for list in list_of_lists for elem in list)

    test_shapes = [[tuple()], 
                   [(1,), (3,), (10,)], 
                   [(3, 1), (2, 4), (2, 10), (10, 2)], 
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


def _test_array_func(f, args=tuple(), pargs=tuple(), test_shapes=None, 
                     test_dtype=np.float64, mod="", **kwargs):
    """Tests a numpy-like function taking a single random variable as an input
    and producing a single random variable at the output.
    
    Args:
        f (callable): 
            The random function to be tested.
        args (sequence): 
            The positional arguments to be supplied to the tested function 
            after the random variable.
        pargs (sequence):
            The positional arguments to be supplied to the tested function 
            before the random variable.
        test_shapes (None, int, str or sequence):
            The shapes of the random variables to be generated for testing.
            if not a sequence, a default set of shapes is used.
        test_dtype (numpy dtype):
            The data type of the test variables.
        mod (str):
            The name of the module where the function is located.
        **kwargs:
            The keyword arguments to be passed to the tested function.

    Returns:
        None
    """

    if mod != "":
        mod = getattr(np, mod)
        npf = getattr(mod, f.__name__)
    else:
        npf = getattr(np, f.__name__)

    if test_shapes is None or isinstance(test_shapes, (str, int)):
        test_shapes = _gts(test_shapes)
        
    for sh in test_shapes:
        # The operation on random variables.
        vin = random_normal(sh, test_dtype)
        vout = f(*pargs, vin, *args, **kwargs)

        assert vin.a.shape[1:] == vin.b.shape
        assert vout.a.shape[1:] == vout.b.shape

        refmean = npf(*pargs, vin.b, *args, **kwargs)
        assert vout.b.shape == refmean.shape
        assert vout.b.dtype == refmean.dtype
        assert np.all(vout.b == refmean)

        dt = refmean.dtype
        tol = 10 * np.finfo(dt).eps

        assert vout.a.dtype == dt

        for arin, arout in zip(vin.a, vout.a):
            aref = npf(*pargs, arin, *args, **kwargs)
            assert arout.shape == aref.shape
            assert np.allclose(arout, aref, rtol=tol, atol=tol * np.max(np.abs(arin)))

        # The operation on deterministic variables.
        vin = random_det_normal(sh, test_dtype)
        vout = f(*pargs, vin, *args, **kwargs)

        assert vin.a.size == 0
        assert vin.a.shape[1:] == vin.b.shape
        assert vout.a.shape[1:] == vout.b.shape

        refmean = npf(*pargs, vin.b, *args, **kwargs)
        assert vout.b.shape == refmean.shape
        assert vout.b.dtype == refmean.dtype
        assert np.all(vout.b == refmean)


def test_sum():
    _test_array_func(sum)
    _test_array_func(sum, axis=0)
    _test_array_func(sum, axis=0, keepdims=True)
    _test_array_func(sum, axis=-1, keepdims=True)
    _test_array_func(sum, axis=-2, keepdims=False, test_shapes="2dmin")
    _test_array_func(sum, axis=-2, keepdims=True, test_shapes="2dmin")
    _test_array_func(sum, axis=1, keepdims=True, test_shapes="2dmin")
    _test_array_func(sum, axis=2, test_shapes="3dmin")
    _test_array_func(sum, axis=-2, test_shapes="3dmin")
    _test_array_func(sum, axis=(0, 1), test_shapes="2dmin")
    _test_array_func(sum, axis=(0, 2), test_shapes="3dmin")
    _test_array_func(sum, axis=(1, -1), test_shapes="3dmin")
    _test_array_func(sum, axis=(-1, -2), keepdims=True, test_shapes="3dmin")


def test_cumsum():
    _test_array_func(cumsum)
    _test_array_func(cumsum, axis=0)
    _test_array_func(cumsum, axis=-1)
    _test_array_func(cumsum, axis=-2, test_shapes="2dmin")
    _test_array_func(cumsum, axis=1, test_shapes="2dmin")
    _test_array_func(cumsum, axis=2, test_shapes="3dmin")
    _test_array_func(cumsum, axis=-2, test_shapes="3dmin")


def test_trace():
    _test_array_func(trace, test_shapes="2dmin")
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
    _test_array_func(ravel, order="F")


def test_transpose():
    _test_array_func(transpose)
    _test_array_func(transpose, axes=(1, 0, 2), test_shapes=3)
    _test_array_func(transpose, axes=(-2, 0, -1), test_shapes=3)
    _test_array_func(transpose, axes=(2, 1, 3, 0), test_shapes=4)
    _test_array_func(transpose, axes=(-3, -1, -2, 0), test_shapes=4)
    _test_array_func(transpose, axes=(2, 1, 3, 5, 4, 0), test_shapes=6)


def test_moveaxis():
    _test_array_func(moveaxis, [0, 0], test_shapes="1dmin")
    _test_array_func(moveaxis, [0, 1], test_shapes="2dmin")
    _test_array_func(moveaxis, [-1, 0], test_shapes="2dmin")
    _test_array_func(moveaxis, [-1, -2], test_shapes="3dmin")
    _test_array_func(moveaxis, [0, 2], test_shapes="3dmin")


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
            new_shapes = [1, tuple(), (1,), (1, 1, 1)]
        elif len(sh) == 1:
            new_shapes = [sh[0], np.array(sh[0]), (1, sh[0]), 
                          (1, sh[0], 1), (-1,)]
        elif len(sh) == 2:
            new_shapes = [sh[0] * sh[1], (sh[0] * sh[1],), (sh[1], 1, sh[0]),
                          (-1,), (sh[1], -1)]
        elif len(sh) == 3:
            new_shapes = [(sh[0] * sh[1] * sh[2],), (sh[1], sh[0] * sh[2]),
                          (-1, sh[2]), (sh[1], -1), (sh[2], -1), 
                          (sh[0] * sh[1], sh[2]), (-1,)]
        else:
            sz = reduce(lambda x, y: x * y, sh)
            new_shapes = [(sz,), prime_factors(sz), (-1,), 
                          (*sh, 1), (sh[0], 1, 1, *sh[1:]), (1, *sh)]

        for new_sh in new_shapes:
            _test_array_func(reshape, [new_sh], test_shapes=[sh])
            _test_array_func(reshape, [new_sh], test_shapes=[sh], order="F")
            _test_array_func(reshape, [new_sh], test_shapes=[sh], order="A")


def test_fft():
    cfft_funcs = [fft, ifft, hfft]
    rfft_funcs = [rfft, irfft, ihfft]

    ts = [(2,), (4,), (16,), (128,), (3, 32), (3, 2, 7)]

    for f in cfft_funcs + rfft_funcs:
        _test_array_func(f, test_shapes=ts, mod="fft")
        _test_array_func(f, axis=-2, 
                         test_shapes=[s + (2,) for s in ts], mod="fft")
        _test_array_func(f, axis=1, 
                         test_shapes=[(1, 10) + s for s in ts], mod="fft")
        _test_array_func(f, axis=1, n=20,
                         test_shapes=[(2, 63, 2)], mod="fft")
        _test_array_func(f, axis=0, n=129,
                         test_shapes=[(128, 2)], mod="fft")
        
    for f in cfft_funcs:
        _test_array_func(f, test_shapes=ts, 
                         test_dtype=np.complex128, mod="fft")
        _test_array_func(f, axis=1, n=90, test_shapes=[(2, 63, 2)],
                         test_dtype=np.complex128, mod="fft")
        
    ts = [(128, 2), (89, 1)]

    for f in cfft_funcs + rfft_funcs:
        _test_array_func(f, axis=0, n=129, norm="ortho",
                         test_shapes=ts, mod="fft")
        _test_array_func(f, axis=0, norm="backward",
                         test_shapes=ts, mod="fft")
        _test_array_func(f, axis=0, n=64, norm="forward",
                         test_shapes=ts, mod="fft")
    
    for f in cfft_funcs:
        _test_array_func(f, axis=0, n=129, norm="ortho",
                         test_dtype=np.complex128, test_shapes=ts, mod="fft")
        _test_array_func(f, axis=0, norm="backward",
                         test_dtype=np.complex128, test_shapes=ts, mod="fft")
        _test_array_func(f, axis=0, n=64, norm="forward",
                         test_dtype=np.complex128, test_shapes=ts, mod="fft")


def test_fft2():
    cfft_funcs = [fft2, ifft2]
    rfft_funcs = [rfft2, irfft2]

    ts = [(2, 2), (4, 2), (5, 3), (3, 17), (2, 8, 4)]

    for f in cfft_funcs + rfft_funcs:
        _test_array_func(f, test_shapes=ts, mod="fft")
        _test_array_func(f, axes=[0, 1], 
                         test_shapes=[s + (2,) for s in ts], mod="fft")
        _test_array_func(f, axes=[-3, 1], 
                         test_shapes=[(1, 10) + s for s in ts], mod="fft")
        _test_array_func(f, axes=(0, 1), s=(5, 20),
                         test_shapes=[(2, 63, 2)], mod="fft")
        _test_array_func(f, s=(64, 3),
                         test_shapes=[(128, 2)], mod="fft")
        
    for f in cfft_funcs:
        _test_array_func(f, test_shapes=ts, test_dtype=np.complex128, mod="fft")
        _test_array_func(f, axes=(0, 1), s=(5, 20), test_dtype=np.complex128,
                         test_shapes=[(2, 63, 2)], mod="fft")
  
    ts = [(33, 3, 2), (8, 7, 2)]

    for f in cfft_funcs + rfft_funcs:
        _test_array_func(f, axes=(0, 1), norm="ortho",
                         test_shapes=ts, mod="fft")
        _test_array_func(f, axes=(0, 1), norm="backward",
                         test_shapes=ts, mod="fft")
        _test_array_func(f, axes=(0, 1), s=(32, 5), norm="forward",
                         test_shapes=ts, mod="fft")
    
    for f in cfft_funcs:
        _test_array_func(f, axes=(0, 1), s=(29, 5), norm="ortho",
                         test_dtype=np.complex128, test_shapes=ts, mod="fft")
        _test_array_func(f, axes=(0, 1), norm="backward",
                         test_dtype=np.complex128, test_shapes=ts, mod="fft")
        _test_array_func(f, axes=(0, 1), s=(32, 5), norm="forward",
                         test_dtype=np.complex128, test_shapes=ts, mod="fft")
        

def test_fftn():
    # tests for 3 fft dimensions max

    cfft_funcs = [fftn, ifftn]
    rfft_funcs = [rfftn, irfftn]

    ts = [(16, 3, 2), (2, 4, 7, 2)]

    for f in cfft_funcs + rfft_funcs:
        _test_array_func(f, test_shapes=ts, mod="fft")
        _test_array_func(f, axes=[-2, -3, -1], test_shapes=ts, mod="fft")
        _test_array_func(f, axes=[-2, -3, -1], s=[3, 5, 7], test_shapes=ts, mod="fft")
        _test_array_func(f, axes=None, s=[5, 7], test_shapes=ts, mod="fft")
        _test_array_func(f, axes=[0, 2, 1], test_shapes=ts, mod="fft")
        _test_array_func(f, axes=[0, 2, 1], norm="ortho", test_shapes=ts, mod="fft")
        _test_array_func(f, axes=[0, 2, 1], norm="forward", test_shapes=ts, mod="fft")
        _test_array_func(f, axes=[2, 1, 0], norm="backward", test_shapes=ts, mod="fft")

    for f in cfft_funcs:
        _test_array_func(f, axes=[0, 2, 1], norm="ortho", s=[3, 5, 7],
                         test_dtype=np.complex128, test_shapes=ts, mod="fft")
        _test_array_func(f, axes=[0, 2, 1], norm="forward", 
                         test_dtype=np.complex128, test_shapes=ts, mod="fft")
        _test_array_func(f, axes=[2, 1, 0], 
                         test_dtype=np.complex128, test_shapes=ts, mod="fft")


def _test_array_func2(f, op1_shape=None, op2_shape=None, *args, **kwargs):
    # *args so far are only used for the subscripts of einsum, 
    # which is why they go into the functions before the operands.

    if op1_shape is None:
        shapes = [[tuple(), tuple()], [tuple(), (8,)], [tuple(), (2, 5)], 
                  [(3,), (7,)], [(11,), (1,)], [(5,), (3, 2)], [(5,), (3, 2, 4)],
                  [(3, 2), (4, 5)], [(2, 3), (5, 1, 4)], 
                  [(2, 3, 4), (2, 5, 7)], [(2, 3, 4), (2, 5, 6, 7)]]
        
        for sh1, sh2 in shapes:
            _test_array_func2(f, sh1, sh2, **kwargs)

        return
        
    npf = getattr(np, f.__name__)

    # Normal variable first.

    vin = random_normal(op1_shape)
    op2 = (2. * np.random.rand(*op2_shape) - 1)
    vout = f(*args, vin, op2, **kwargs)
    refmean = npf(*args, vin.b, op2, **kwargs)

    assert vout.b.shape == refmean.shape
    assert vout.b.dtype == refmean.dtype
    assert np.all(vout.b == refmean)

    dt = refmean.dtype
    tol = 10 * np.finfo(dt).eps

    assert vout.a.dtype == dt

    for arin, arout in zip(vin.a, vout.a):
        aref = npf(*args, arin, op2, **kwargs)
        assert arout.shape == aref.shape

        atol = 2 * tol * max(1, np.max(np.abs(aref)))
        assert np.allclose(arout, aref, rtol=tol, atol=atol)

    # Normal variable second.
    
    op1 = (2. * np.random.rand(*op1_shape) - 1)
    vin = random_normal(op2_shape)
    vout = f(*args, op1, vin, **kwargs)
    refmean = npf(*args, op1, vin.b, **kwargs)

    assert vout.b.shape == refmean.shape
    assert vout.b.dtype == refmean.dtype
    assert np.all(vout.b == refmean)

    assert vout.a.dtype == refmean.dtype

    for arin, arout in zip(vin.a, vout.a):
        aref = npf(*args, op1, arin, **kwargs)
        assert arout.shape == aref.shape

        atol = 2 * tol * max(1, np.max(np.abs(aref)))
        assert np.allclose(arout, aref, rtol=tol, atol=atol)

    # Both variables are normal. 

    vin1 = random_normal(op1_shape)
    vin2 = random_normal(op2_shape)
    vout = f(*args, vin1, vin2, **kwargs)
    refmean = npf(*args, vin1.b, vin2.b, **kwargs)

    assert vout.b.shape == refmean.shape
    assert vout.b.dtype == refmean.dtype
    assert np.max(vout.b - refmean) < atol

    assert vout.a.dtype == refmean.dtype

    vref = (f(*args, vin1, vin2.b, **kwargs) 
            + f(*args, vin1.b, (vin2 - vin2.b), **kwargs))

    atol = 2 * tol * max(1, np.max(np.abs(vref.a)))

    assert vout.lat == vref.lat
    assert np.max(vout.a - vref.a) < atol

    # Both variables are deterministic promoted to Normal.

    vin1 = random_det_normal(op1_shape)
    vin2 = random_det_normal(op2_shape)
    vout = f(*args, vin1, vin2, **kwargs)
    refmean = npf(*args, vin1.b, vin2.b, **kwargs)

    atol = 2 * tol * max(1, np.max(np.abs(refmean)))

    assert vout.b.shape == refmean.shape
    assert vout.b.dtype == refmean.dtype
    assert np.max(vout.b - refmean) < atol

    assert vout.a.size == 0
    assert vout.a.dtype == refmean.dtype


def test_multiply():
    # Only tests the multiplication of normals by constants.
    for sh in _gts(1):
        _test_array_func2(multiply, sh, tuple())
        _test_array_func2(multiply, sh, sh)
        _test_array_func2(multiply, sh + sh, sh)
        _test_array_func2(multiply, (2, 3) + sh, sh)
        _test_array_func2(multiply, sh, sh + sh)
        _test_array_func2(multiply, sh, (2, 3,) + sh)
        _test_array_func2(multiply, (4,) + sh, (2, 4) + sh)
        _test_array_func2(multiply, (2, 1) + sh, (5,) + sh)
        _test_array_func2(multiply, (3,) + sh, (5, 3) + sh)
        _test_array_func2(multiply, (5, 3) + sh, (3,) + sh)
        _test_array_func2(multiply, (5, 1, 2) + sh, (5, 3, 1) + sh)


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

    # No matrix multiplication by scalars.

    x = normal(size=(2, 3))
    y = np.array(2.)

    with pytest.raises(ValueError):
        x @ y
    with pytest.raises(ValueError):
        y @ x
    with pytest.raises(ValueError):
        matmul(x, y)
    with pytest.raises(ValueError):
        matmul(y, x)


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


def test_einsum():
    for sh in _gts(1):
        _test_array_func2(einsum, sh, sh, "i, i -> ")  # inner
        _test_array_func2(einsum, sh, sh, "i, i")  # inner implicit
        _test_array_func2(einsum, sh, sh, "i, j -> ij")  # outer
        _test_array_func2(einsum, sh, sh, "i, j -> ji")
        _test_array_func2(einsum, sh, sh, "i, j")  # outer implicit
        _test_array_func2(einsum, sh, sh + (3,), "i, i... -> ...")  # ellipsis
        _test_array_func2(einsum, sh, sh + (3,), "i, j... -> j...i")
        _test_array_func2(einsum, sh, sh + (3,), "i, j...")
        _test_array_func2(einsum, (3,) + sh, sh + (3,), "ki, jk -> ijk")
        _test_array_func2(einsum, (1,) + sh, sh + (3,), "ki, jk -> ijk") # broadcasting

    for sh in _gts(2):
        _test_array_func2(einsum, sh, sh, "ij, ij")
        _test_array_func2(einsum, sh, sh, "ij, kj")
        _test_array_func2(einsum, sh, sh[::-1], "ij, jk")
        _test_array_func2(einsum, sh, sh, "ij, kj -> ki")
        _test_array_func2(einsum, sh, sh, "ij, kl")
        _test_array_func2(einsum, sh, sh, "ij, kl -> kilj")
        _test_array_func2(einsum, (3, 2) + sh, sh, "...ij, kj -> ki...")
        _test_array_func2(einsum, (3, 2) + sh, sh + (1, 2), "...ij, kj... -> ki...")
        _test_array_func2(einsum, sh, (5, 3) + sh, "ij, ...ij -> ...")

    for sh in _gts(3):
        _test_array_func2(einsum, sh, sh, "..., ...")
        _test_array_func2(einsum, sh, sh, "i..., i...-> ...")
        _test_array_func2(einsum, sh, sh, "ijk, ilk-> lj")
        _test_array_func2(einsum, (2,) + sh, sh, "nijk, ilk-> nlj")
        _test_array_func2(einsum, (2,) + sh, sh, "nijk, ilk")
        _test_array_func2(einsum, (2, 3) + sh, sh, "...ijk, ilk-> ...lj")
        _test_array_func2(einsum, (2, 3) + sh, sh, "...ijk, ijk-> ...j")
        _test_array_func2(einsum, (2, 3) + sh, sh, "...ijk, ijk-> ...")

    for sh in _gts(4):
        _test_array_func2(einsum, sh, sh, "..., ...")
        _test_array_func2(einsum, sh, sh[::-1], "ijkl, lkjs")
        _test_array_func2(einsum, sh, sh, "...ij, ...ij-> ...ij")
        _test_array_func2(einsum, sh, sh, "ij..., ij...-> ...j")
    

def _test_array_method(name, *args, test_shapes=None, **kwargs):
    if test_shapes is None or isinstance(test_shapes, (str, int)):
        test_shapes = _gts(test_shapes)

    for sh in test_shapes:
        vin = random_normal(sh)
        vout = getattr(vin, name)(*args, **kwargs)

        assert vin.a.shape[1:] == vin.b.shape
        assert vout.a.shape[1:] == vout.b.shape

        refmean = getattr(vin.b, name)(*args, **kwargs)
        assert vout.b.shape == refmean.shape
        assert vout.b.dtype == refmean.dtype
        assert np.all(vout.b == refmean)

        dt = vout.b.dtype
        tol = 10 * np.finfo(dt).eps

        assert vout.a.dtype == dt

        for arin, arout in zip(vin.a, vout.a):
            aref = getattr(arin, name)(*args, **kwargs)
            assert arout.shape == aref.shape

            atol = tol * max(1, np.max(np.abs(aref)))
            assert np.allclose(arout, aref, rtol=tol, atol=atol) 
            

def test_flatten():
    _test_array_method("flatten")
    _test_array_method("flatten", order="F")


def _test_concat_func(f, *args, test_shapes=None, vins_list=None, **kwargs):
    npf = getattr(np, f.__name__)

    if test_shapes is None or isinstance(test_shapes, (str, int)):
        test_shapes = _gts(test_shapes)

    ns = [1, 2, 3, 11]  # Numbers of input arrays.

    # Composes lists of input variables.
    if vins_list is None:
        vins_list = []
        for sh in test_shapes:
            vins_max = random_correlate([random_normal(sh) 
                                          for _ in range(ns[-1])])
            vins_list += [vins_max[:n] for n in ns]
            
            # Adds special cases of two inputs with different numbers of 
            # the latent variables, because there are separate evaluation
            # branches for the optimization of those.
            vins2 = random_correlate([random_normal(sh) for _ in range(2)])
            vins2[0] = vins2[0] + np.random.rand(*sh) * normal(0.1, 0.9)

            vins_list += [[vins2[0], vins2[1]], [vins2[1], vins2[0]]]

            # Adds a cace of deterministic arrays promoted to Normals.
            vins_list += [[random_det_normal(sh) for _ in range(1)],
                          [random_det_normal(sh) for _ in range(2)],
                          [random_det_normal(sh) for _ in range(3)]]
            
            # Cases with one of the input arrays being complex.
            vins3 = random_correlate([random_normal(sh), 
                                      random_normal(sh, dtype=np.complex64)])
            vins3[0] = vins3[0] + np.random.rand(*sh) * normal(-0.1, 0.8)
            vins_list += [[vins3[0], vins3[1]], [vins3[1], vins3[0]]]

            vins4 = random_correlate([random_normal(sh),
                                      random_normal(sh), 
                                      random_normal(sh, dtype=np.complex64)])
            vins_list += [vins4]

            # A case to check the optimization branch for two operands, 
            # where the latent variables are the same.
            vin = random_normal(sh, dtype=np.complex64)
            assert vin.real.lat is vin.imag.lat
            vins_list += [[vin.real, vin.imag], [vin.imag, vin.imag]]

            # A strange case where the variables have identical lists 
            # of latent variables, which are not the same object.
            vin_imag_c = copy.deepcopy(vin.imag)
            assert vin.real.lat is not vin_imag_c.lat
            assert vin.real.lat == vin_imag_c.lat
            vins_list += [[vin.real, vin_imag_c]]

    # Tests the function for all input lists.
    for vins in vins_list:
        vout = f(vins, *args, **kwargs)

        assert all(vin.a.shape[1:] == vin.b.shape for vin in vins)
        assert vout.a.shape[1:] == vout.b.shape

        assert np.all(vout.b == npf([vin.b for vin in vins], *args, **kwargs))
        assert vout.a.dtype == vout.b.dtype

        lat, ains_ext = maps.complete(vins)
        # Now the testing silently relies on the fact that maps.complete
        # produces in the same order of the latent random variables 
        # as maps.concatenate or maps.stack.

        assert lat == vout.lat

        for i in range(len(vout.a)):
            arins = [a[i] for a in ains_ext]
            arout = vout.a[i]
            aref = npf(arins, *args, **kwargs)
            assert arout.shape == aref.shape
            assert np.all(arout == aref)

    with pytest.raises(ValueError):
        # Error for empty inputs.
        f([], *args, **kwargs)


def test_stack():
    _test_concat_func(stack)
    _test_concat_func(stack, axis=0, test_shapes="2dmin")
    _test_concat_func(stack, axis=1, test_shapes="2dmin")
    _test_concat_func(stack, axis=-1, test_shapes="1dmin")
    _test_concat_func(stack, axis=-2, test_shapes="3dmin")


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
        vins = random_correlate([random_normal((*s[:ax], i, *s[ax:]))
                                  for i in range(1, 4)])
        _test_concat_func(concatenate, axis=ax, vins_list=[vins])

    ts = _gts("2dmin")
    ax = -2
    for s in ts:
        print(s)
        vins = random_correlate([random_normal((*s[:ax+1], i, *s[ax+1:]))
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
            vin = random_normal(sh)
            vouts = f(vin, *args, **kwargs)

            refmeans = npf(vin.b, *args, **kwargs)

            assert len(vouts) == len(refmeans)
            for vout, refmean in zip(vouts, refmeans):
                assert vout.b.dtype == refmean.dtype
                assert vout.b.shape == refmean.shape
                assert np.all(vout.b == refmean)

                assert vout.a.dtype == refmean.dtype

            for i in range(len(vin.a)):
                arin = vin.a[i]
                arouts = [vout.a[i] for vout in vouts]
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


def test_dtype_promotion():
    # concatenate, stack, sum, complete

    sh = (2, 3)

    # Real types
    v1 = random_normal(sh)
    v1.a = v1.a.astype(np.float16)
    v1.b = v1.b.astype(np.float16)

    v2 = random_normal(sh, dtype=np.float32)
    v3 = random_normal(sh, dtype=np.float64)

    # Complex types
    v5 = random_normal(sh, dtype=np.complex64)
    v6 = random_normal(sh, dtype=np.complex128)

    funcs = [stack, concatenate, 
             lambda a: add(a[0], a[1]), lambda a: add(a[1], a[0]),
             lambda a: subtract(a[0], a[1]), lambda a: subtract(a[1], a[0])]
    
    for f in funcs:
        v12 = f([v1, v2])
        assert v12.a.dtype == np.float32
        assert v12.b.dtype == np.float32

        v21 = f([v2, v1])
        assert v21.a.dtype == np.float32
        assert v21.b.dtype == np.float32

        v12 = f([v1, v2])
        assert v12.a.dtype == np.float32
        assert v12.b.dtype == np.float32

        v55 = f([v5, v5])
        assert v55.a.dtype == np.complex64
        assert v55.b.dtype == np.complex64

        v56 = f([v5, v6])
        assert v56.a.dtype == np.complex128
        assert v56.b.dtype == np.complex128

        v15 = f([v1, v5])
        assert v15.a.dtype == np.complex64
        assert v15.b.dtype == np.complex64

        v53 = f([v5, v3])
        assert v53.a.dtype == np.complex128
        assert v53.b.dtype == np.complex128

    for f in [stack, concatenate]:
        v1_ = f([v1])
        assert v1_.a.dtype == np.float16
        assert v1_.b.dtype == np.float16

        v123 = f([v1, v2, v3])
        assert v123.a.dtype == np.float64
        assert v123.b.dtype == np.float64

        v5_ = f([v5])
        assert v5_.a.dtype == np.complex64
        assert v5_.b.dtype == np.complex64

        v564 = f([v5, v6, v3])
        assert v564.a.dtype == np.complex128
        assert v564.b.dtype == np.complex128

    map_lists = [[v1, v5], [v5, v1], 
                 [v1, v5, v3]]
    
    for ml in map_lists:
        _, as_ = maps.complete(ml)
        for a, m in zip(as_, ml):
            assert a.dtype == m.a.dtype


def _assert_normals_close(v1, v2):
    assert v1.shape == v2.shape
    assert v1.b.shape == v2.b.shape
    assert v1.b.dtype == v2.b.dtype
    assert v1.a.shape == v2.a.shape
    assert v1.a.dtype == v2.a.dtype

    rtolb = 100 * np.finfo(v1.b.dtype).eps
    assert np.allclose(v1.b, v2.b, 
                       rtol=rtolb, atol=rtolb * np.max(np.abs(v1.b)))

    rtola = 100 * np.finfo(v1.a.dtype).eps
    assert np.allclose(v1.a, v2.a, 
                       rtol=rtola, atol=rtola * np.max(np.abs(v1.a)))
    

def test_divide():
    for bsh in _gts(1):
        test_shapes = [[tuple(), bsh], [bsh, tuple()], [bsh, bsh], [bsh + bsh, bsh], 
                       [(2, 3) + bsh, bsh], [bsh, bsh + bsh], [bsh, (2, 3,) + bsh], 
                       [(4,) + bsh, (2, 4) + bsh], [(2, 1) + bsh, (5,) + bsh],
                       [(3,) + bsh, (5, 3) + bsh], [(5, 3) + bsh, (3,) + bsh],
                       [(5, 1, 2) + bsh, (5, 3, 1) + bsh]]
        
        for sh1, sh2 in test_shapes:
            v = random_normal(sh1)
            ar = np.random.rand(*sh2)

            _assert_normals_close(divide(v, ar), multiply(v, 1/ar))


def test_power():

    # Identical shapes of the operands.
    test_shapes = [tuple(), (2,), (3, 2)]
    for sh in test_shapes:

        # normal-constant
        v = 1.001 + random_normal(sh)
        x = (2 * np.random.rand(*sh) - 1)

        v_ = power(v, x)
        v_ref = power(v.mean(), x) * (1 + x * v.delta / v.mean())
        _assert_normals_close(v_, v_ref)

        v_ = power(v, 3)
        v_ref = v * v * v
        _assert_normals_close(v_, v_ref)

        v_ = power(v, -2)
        v_ref = 1 / v / v
        _assert_normals_close(v_, v_ref)

        v_ = power(v, 0)
        v_ref = normal(1., 0., size=v.shape)
        _assert_normals_close(v_, v_ref)
        
        # constant-normal
        v = random_normal(sh)
        x = np.random.rand(*sh) + 0.001

        v_ = power(x, v)
        v_ref = power(x, v.mean()) * (1 + np.log(x) * v.delta)
        _assert_normals_close(v_, v_ref)

        v_ = power(0, 1 + v)  # negative powers of 0 are ill-defined
        v_ref = normal(0., 0., size=v.shape)
        _assert_normals_close(v_, v_ref)

        v_ = power(1, v)
        v_ref = normal(1., 0., size=v.shape)
        _assert_normals_close(v_, v_ref)

        # normal-normal
        v = 1.001 + random_normal(sh)
        v2 = random_normal(sh)
        v_ = power(v, v2)

        y, z = v.mean(), v2.mean()
        v_ref = (y**z + y**z*v2.delta*np.log(y) 
                 + v.delta*(y**(-1+z)*z+y**(-1+z)*v2.delta*(1+z*np.log(y))))
        _assert_normals_close(v_, v_ref)

    # Another case for broadcasting over the second operand.
    v = random_normal(tuple())
    v_ = power(v, [2, 3, 4])
    v_ref = stack([v * v, v * v * v, v * v * v * v])
    _assert_normals_close(v_, v_ref)


def test_heterogeneous_ops():
    # Heterogeneous variables have mean and variance with different data types.

    tol = 1e-8

    x = normal(0, 0.05)  # int mean and float var
    y = normal(1, 0.05)  # int mean and float var

    z = x + y

    assert np.abs(z.mean() - 1) < tol
    assert np.abs(z.var() - 0.1) < tol

    z = stack([x, y])

    assert np.max(np.abs(z.mean() - [0, 1])) < tol
    assert np.max(np.abs(z.var() - [0.05, 0.05])) < tol

    z = stack([normal(0, 0.05), normal(1, 0.05), normal(2, 0.05)])

    assert np.max(np.abs(z.mean() - [0, 1, 2])) < tol
    assert np.max(np.abs(z.var() - [0.05, 0.05, 0.05])) < tol

    z = concatenate([normal(0, 0.05, size=(1,)), 
                     normal(1, 0.05, size=(1,))])

    assert np.max(np.abs(z.mean() - [0, 1])) < tol
    assert np.max(np.abs(z.var() - [0.05, 0.05])) < tol

    z = concatenate([normal(0, 0.05, size=(1,)), 
                     normal(1, 0.05, size=(1,)),
                     normal(2, 0.05, size=(1,))])

    assert np.max(np.abs(z.mean() - [0, 1, 2])) < tol
    assert np.max(np.abs(z.var() - [0.05, 0.05, 0.05])) < tol