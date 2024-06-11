import pytest
import numpy as np
from numpy.exceptions import AxisError

import gprob as gp
from gprob.normal_ import normal
from gprob.sparse import _parse_index_key, iid_repeat, assparsenormal

from utils import random_normal

np.random.seed(0)


def test_iid_repeat():
    # A numeric constant.
    number = 1
    assert iid_repeat(number, 5, axis=0).shape == (5,)
    assert iid_repeat(number, 5, axis=0).iaxes == (0,)

    arr = np.ones((3, 4))
    assert iid_repeat(arr, 5, axis=0).shape == (5, 3, 4)
    assert iid_repeat(arr, 5, axis=0).iaxes == (0,)

    # One independence axis.

    assert iid_repeat(normal(), 0, axis=0).shape == (0,)
    assert iid_repeat(normal(), 0, axis=0).iaxes == (0,)

    assert iid_repeat(normal(), 5, axis=0).shape == (5,)
    assert iid_repeat(normal(), 5, axis=0).iaxes == (0,)

    v = normal(size=(2, 3))

    assert iid_repeat(v, 5, axis=0).shape == (5, 2, 3)
    assert iid_repeat(v, 5, axis=0).iaxes == (0,)

    assert iid_repeat(v, 5, axis=-3).shape == (5, 2, 3)
    assert iid_repeat(v, 5, axis=-3).iaxes == (0,)

    assert iid_repeat(v, 5, axis=1).shape == (2, 5, 3)
    assert iid_repeat(v, 5, axis=1).iaxes == (1,)

    assert iid_repeat(v, 5, axis=-2).shape == (2, 5, 3)
    assert iid_repeat(v, 5, axis=-2).iaxes == (1,)

    assert iid_repeat(v, 5, axis=2).shape == (2, 3, 5)
    assert iid_repeat(v, 5, axis=2).iaxes == (2,)

    assert iid_repeat(v, 5, axis=-1).shape == (2, 3, 5)
    assert iid_repeat(v, 5, axis=-1).iaxes == (2,)

    # Two independence axes.

    v = iid_repeat(v, 5, axis=1)  # shape (2, 5, 3), iaxes (1,)

    assert iid_repeat(v, 6, axis=0).shape == (6, 2, 5, 3)
    assert iid_repeat(v, 6, axis=0).iaxes == (0, 2)

    assert iid_repeat(v, 6, axis=-4).shape == (6, 2, 5, 3)
    assert iid_repeat(v, 6, axis=-4).iaxes == (0, 2)

    assert iid_repeat(v, 6, axis=1).shape == (2, 6, 5, 3)
    assert iid_repeat(v, 6, axis=1).iaxes == (1, 2)

    assert iid_repeat(v, 6, axis=-3).shape == (2, 6, 5, 3)
    assert iid_repeat(v, 6, axis=-3).iaxes == (1, 2)

    assert iid_repeat(v, 6, axis=2).shape == (2, 5, 6, 3)
    assert iid_repeat(v, 6, axis=2).iaxes == (1, 2)

    assert iid_repeat(v, 6, axis=-2).shape == (2, 5, 6, 3)
    assert iid_repeat(v, 6, axis=-2).iaxes == (1, 2)

    assert iid_repeat(v, 6, axis=3).shape == (2, 5, 3, 6)
    assert iid_repeat(v, 6, axis=3).iaxes == (1, 3)

    assert iid_repeat(v, 6, axis=-1).shape == (2, 5, 3, 6)
    assert iid_repeat(v, 6, axis=-1).iaxes == (1, 3)

    # Three independence axes.

    v = iid_repeat(v, 6, axis=-1)  # shape (2, 5, 3, 6), iaxes (1, 3)

    assert iid_repeat(v, 4, axis=0).shape == (4, 2, 5, 3, 6)
    assert iid_repeat(v, 4, axis=0).iaxes == (0, 2, 4)

    assert iid_repeat(v, 4, axis=-5).shape == (4, 2, 5, 3, 6)
    assert iid_repeat(v, 4, axis=-5).iaxes == (0, 2, 4)

    assert iid_repeat(v, 4, axis=1).shape == (2, 4, 5, 3, 6)
    assert iid_repeat(v, 4, axis=1).iaxes == (1, 2, 4)

    assert iid_repeat(v, 4, axis=-4).shape == (2, 4, 5, 3, 6)
    assert iid_repeat(v, 4, axis=-4).iaxes == (1, 2, 4)

    assert iid_repeat(v, 4, axis=2).shape == (2, 5, 4, 3, 6)
    assert iid_repeat(v, 4, axis=2).iaxes == (1, 2, 4)

    assert iid_repeat(v, 4, axis=-3).shape == (2, 5, 4, 3, 6)
    assert iid_repeat(v, 4, axis=-3).iaxes == (1, 2, 4)

    assert iid_repeat(v, 4, axis=3).shape == (2, 5, 3, 4, 6)
    assert iid_repeat(v, 4, axis=3).iaxes == (1, 3, 4)

    assert iid_repeat(v, 4, axis=-2).shape == (2, 5, 3, 4, 6)
    assert iid_repeat(v, 4, axis=-2).iaxes == (1, 3, 4)

    assert iid_repeat(v, 4, axis=4).shape == (2, 5, 3, 6, 4)
    assert iid_repeat(v, 4, axis=4).iaxes == (1, 3, 4)

    assert iid_repeat(v, 4, axis=-1).shape == (2, 5, 3, 6, 4)
    assert iid_repeat(v, 4, axis=-1).iaxes == (1, 3, 4)

    # Axis out of bound.
    # raised exception is AxisError, which is a subtype of ValueError
    with pytest.raises(AxisError):
        iid_repeat(v, 4, axis=5)
    with pytest.raises(AxisError):
        iid_repeat(v, 4, axis=-6)
    with pytest.raises(AxisError):
        iid_repeat(v, 4, axis=24)

    with pytest.raises(AxisError):
        iid_repeat(normal(), 7, axis=1)
    with pytest.raises(AxisError):
        iid_repeat(normal(), 7, axis=-2)
    with pytest.raises(AxisError):
        iid_repeat(normal(), 7, axis=22)


def test_index_key_parsing():
    v = iid_repeat(iid_repeat(normal(size=(4, 5)), 2, axis=1), 3, axis=1)
    # v.shape is (4, 3, 2, 5), v.iaxes are (1, 2)

    assert v[2].shape == (3, 2, 5)
    assert v[2].iaxes == (0, 1)

    assert v[2, :].shape == (3, 2, 5)
    assert v[2, :].iaxes == (0, 1)

    assert v[1, :, :].shape == (3, 2, 5)
    assert v[1, :, :].iaxes == (0, 1)

    assert v[1, :, :, ...].shape == (3, 2, 5)
    assert v[1, :, :, ...].iaxes == (0, 1)

    assert v[1, :, :, 0].shape == (3, 2)
    assert v[1, :, :, 0].iaxes == (0, 1)

    assert v[::2, :, :].shape == (2, 3, 2, 5)
    assert v[::2, :, :].iaxes == (1, 2)

    assert v[1, ..., 2].shape == (3, 2)
    assert v[1, ..., 2].iaxes == (0, 1)

    assert v[1, ...].shape == (3, 2, 5)
    assert v[1, ...].iaxes == (0, 1)

    assert v[1, :, ...].shape == (3, 2, 5)
    assert v[1, :, ...].iaxes == (0, 1)

    assert v[None, None, ...].shape == (1, 1, 4, 3, 2, 5)
    assert v[None, None, ...].iaxes == (3, 4)

    assert v[None, :, :, None, ..., 2].shape == (1, 4, 3, 1, 2)
    assert v[None, :, :, None, ..., 2].iaxes == (2, 4)

    assert v[1, :, None, :, 0].shape == (3, 1, 2)
    assert v[1, :, None, :, 0].iaxes == (0, 2)

    assert v[1, None, :, :, 1, ..., None].shape == (1, 3, 2, 1)
    assert v[1, None, :, :, 1, ..., None].iaxes == (1, 2)

    # No indices except full slices are allowed for independence axes.
    with pytest.raises(IndexError):
        _parse_index_key(v, np.s_[:, 1])
    with pytest.raises(IndexError):
        _parse_index_key(v, np.s_[:, :, 1])
    with pytest.raises(IndexError):
        _parse_index_key(v, np.s_[:, :2])
    with pytest.raises(IndexError):
        _parse_index_key(v, np.s_[:, :, :2])
    with pytest.raises(IndexError):
        _parse_index_key(v, np.s_[:, ::2])
    with pytest.raises(IndexError):
        _parse_index_key(v, np.s_[:, [True, False, True]])
    with pytest.raises(IndexError):
        _parse_index_key(v, np.s_[:, True])
    with pytest.raises(IndexError):
        _parse_index_key(v, np.s_[:, [0, 1, 2]])
    
    # Generally invalid indices.
    with pytest.raises(IndexError):
        _parse_index_key(v, np.s_["s"])
    with pytest.raises(IndexError):
        _parse_index_key(v, np.s_[1, :, :, 2, 1])
    with pytest.raises(IndexError):
        _parse_index_key(v, np.s_[..., ...])
    with pytest.raises(IndexError):
        _parse_index_key(v, np.s_[..., 1, ...])


def test_broadcasting():

    # Combine variables with compatible shapes and independence axes 
    # but different numbers of dimensions.
    v1 = iid_repeat(normal(size=(2,)), 3, axis=1)
    v2 = iid_repeat(normal() + 2., 3, axis=0)

    assert (v1 + v2).shape == (2, 3)
    assert (v1 + v2).iaxes == (1,)
    assert (v1 - v2).shape == (2, 3)
    assert (v1 - v2).iaxes == (1,)
    assert (v1 * v2).shape == (2, 3)
    assert (v1 * v2).iaxes == (1,)
    assert (v1 / v2).shape == (2, 3)
    assert (v1 / v2).iaxes == (1,)
    assert (v1 ** v2).shape == (2, 3)
    assert (v1 ** v2).iaxes == (1,)


def test_cumsum():
    v = iid_repeat(iid_repeat(normal(size=(4, 5)), 2, axis=1), 3, axis=1)
    # v.shape is (4, 3, 2, 5), v.iaxes are (1, 2)

    for ax in [0, 3, -1, -4]:
        vout = v.cumsum(axis=ax)
        assert vout.shape == v.shape
        assert vout.iaxes == v.iaxes
        assert np.all(vout.mean() == np.cumsum(v.mean(), axis=ax))

        vvs = v.v.cumsum(axis=ax)
        assert np.all(vvs.emap.a == vout.v.emap.a)
        assert np.all(vvs.emap.elem == vout.v.emap.elem)

    # Independence axes are not allowed.
    with pytest.raises(ValueError):
        v.cumsum(axis=1)
    with pytest.raises(ValueError):
        v.cumsum(axis=-2)

    # Flattening is not allowed.
    with pytest.raises(ValueError):
        v.cumsum(axis=None)

    # Axes out of bound.
    with pytest.raises(AxisError):
        v.cumsum(axis=4)
    with pytest.raises(AxisError):
        v.cumsum(axis=44)
    with pytest.raises(AxisError):
        v.cumsum(axis=-5)


def test_diagonal():
    v = iid_repeat(iid_repeat(normal(size=(4, 5)), 2, axis=0), 3, axis=0)
    # v.shape is (3, 2, 4, 5), v.iaxes are (0, 1)
    assert v.diagonal(axis1=-2, axis2=-1).shape == (3, 2, 4)
    assert v.diagonal(axis1=-2, axis2=-1).iaxes == (0, 1)

    assert v.diagonal(axis1=3, axis2=2).shape == (3, 2, 4)
    assert v.diagonal(axis1=3, axis2=2).iaxes == (0, 1)

    with pytest.raises(ValueError):
        v.diagonal(axis1=0, axis2=1)

    with pytest.raises(ValueError):
        v.diagonal(axis1=1, axis2=2)

    # Repeated axes.
    with pytest.raises(ValueError):
        v.diagonal(axis1=2, axis2=2)

    # Out-of-range axes.
    with pytest.raises(AxisError):
        v.diagonal(axis1=2, axis2=4)
    with pytest.raises(AxisError):
        v.diagonal(axis1=2, axis2=-5)

    v = iid_repeat(iid_repeat(normal(size=(4, 5)), 2, axis=-1), 3, axis=-1)
    # v.shape is (4, 5, 2, 3), v.iaxes are (2, 3)
    assert v.diagonal(axis1=0, axis2=1).shape == (2, 3, 4)
    assert v.diagonal(axis1=0, axis2=1).iaxes == (0, 1)

    v = iid_repeat(iid_repeat(normal(size=(4, 5)), 2, axis=1), 3, axis=1)
    # v.shape is (4, 3, 2, 5), v.iaxes are (1, 2)

    assert v.diagonal(axis1=0, axis2=-1).shape == (3, 2, 4)
    assert v.diagonal(axis1=0, axis2=-1).iaxes == (0, 1)

    assert v[:, :, None].diagonal(axis1=0, axis2=2).shape == (3, 2, 5, 1)
    assert v[:, :, None].diagonal(axis1=0, axis2=2).iaxes == (0, 1)

    assert v[:, :, None].diagonal(axis1=2, axis2=4).shape == (4, 3, 2, 1)
    assert v[:, :, None].diagonal(axis1=2, axis2=4).iaxes == (1, 2)

    with pytest.raises(ValueError):
        v.diagonal(axis1=0, axis2=1)

    with pytest.raises(ValueError):
        v.diagonal(axis1=-1, axis2=-2)

    with pytest.raises(ValueError):
        v.diagonal(axis1=-2, axis2=1)


def test_flatten_ravel():
    names = ["flatten", "ravel"]

    for nm in names:
        v = assparsenormal([])        
        v_ = getattr(v, nm)()
        assert v_.ndim == 1
        assert v_.shape == (0,)
        assert v_.size == 0
        assert v_.iaxes == tuple()

        v = assparsenormal(1.)        
        v_ = getattr(v, nm)()
        assert v_.ndim == 1
        assert v_.iaxes == tuple()
        
        v = iid_repeat(normal(), 3)
        v_ = getattr(v, nm)()
        assert v_.ndim == 1
        assert v_.size == 3
        assert v_.iaxes == (0,)

        v = iid_repeat(normal(size=(1, 1, 1, 1)), 3, axis=2)
        assert v.shape == (1, 1, 3, 1, 1)
        assert v.iaxes == (2,)
        v_ = getattr(v, nm)()
        assert v_.ndim == 1
        assert v_.size == 3
        assert v_.iaxes == (0,)

        v = iid_repeat(normal(size=(1, 1, 1)), 3, axis=2)
        v = iid_repeat(v, 1, axis=0)
        v = iid_repeat(v, 1, axis=-1)
        assert v.shape == (1, 1, 1, 3, 1, 1)
        assert v.iaxes == (0, 3, 5)
        v_ = getattr(v, nm)()
        assert v_.ndim == 1
        assert v_.size == 3
        assert v_.iaxes == (0,)

        v = iid_repeat(normal(size=(0, 1, 0)), 3, axis=2)
        v = iid_repeat(v, 1, axis=0)
        v = iid_repeat(v, 1, axis=-1)
        assert v.shape == (1, 0, 1, 3, 0, 1)
        assert v.iaxes == (0, 3, 5)
        v_ = getattr(v, nm)()
        assert v_.ndim == 1
        assert v_.size == 0
        assert v_.iaxes == (0,)

        v = iid_repeat(normal(size=(1, 3, 1)), 1, axis=1)
        v = iid_repeat(v, 1, axis=0)
        v = iid_repeat(v, 1, axis=-1)
        assert v.shape == (1, 1, 1, 3, 1, 1)
        assert v.iaxes == (0, 2, 5)
        v_ = getattr(v, nm)()
        assert v_.ndim == 1
        assert v_.size == 3
        assert v_.iaxes == tuple()

        with pytest.raises(ValueError):
            v = iid_repeat(normal(size=(2,)), 3)
            v_ = getattr(v, nm)()

        with pytest.raises(ValueError):
            v = iid_repeat(iid_repeat(normal(), 3), 2)
            v_ = getattr(v, nm)()


def test_moveaxis():
    v = iid_repeat(iid_repeat(normal(size=(4, 6, 5)), 2, axis=1), 3, axis=1)
    # v.shape is (4, 3, 2, 6, 5), v.iaxes are (1, 2)

    v_ = v.moveaxis(3, 4)
    assert v_.shape == (4, 3, 2, 5, 6)
    assert v_.iaxes == (1, 2)

    v_ = v.moveaxis(4, 3)
    assert v_.shape == (4, 3, 2, 5, 6)
    assert v_.iaxes == (1, 2)

    v_ = v.moveaxis(0, 1)
    assert v_.shape == (3, 4, 2, 6, 5)
    assert v_.iaxes == (0, 2)

    v_ = v.moveaxis(1, 0)
    assert v_.shape == (3, 4, 2, 6, 5)
    assert v_.iaxes == (0, 2)

    v_ = v.moveaxis(0, 2)
    assert v_.shape == (3, 2, 4, 6, 5)
    assert v_.iaxes == (0, 1)

    v_ = v.moveaxis(2, 0)
    assert v_.shape == (2, 4, 3, 6, 5)
    assert v_.iaxes == (0, 2)

    v_ = v.moveaxis(3, 0)
    assert v_.shape == (6, 4, 3, 2, 5)
    assert v_.iaxes == (2, 3)

    v_ = v.moveaxis(4, 2)
    assert v_.shape == (4, 3, 5, 2, 6)
    assert v_.iaxes == (1, 3)

    v_ = v.moveaxis(-3, -1)
    assert v_.shape == (4, 3, 6, 5, 2)
    assert v_.iaxes == (1, 4)

    v_ = v.moveaxis(-3, -5)
    assert v_.shape == (2, 4, 3, 6, 5)
    assert v_.iaxes == (0, 2)

    v_ = v.moveaxis(-3, 0)
    assert v_.shape == (2, 4, 3, 6, 5)
    assert v_.iaxes == (0, 2)

    v_ = v.moveaxis(1, -4)
    v__ = v_.moveaxis(-4, 1)
    assert v.shape == v__.shape
    assert v.iaxes == v__.iaxes

    with pytest.raises(AxisError):
        v.moveaxis(3, 5)
    with pytest.raises(AxisError):
        v.moveaxis(5, -1)
    with pytest.raises(AxisError):
        v.moveaxis(-6, 1)


def test_reshape():

    # A constant array.
    v = assparsenormal(np.ones((9, 8)))
    
    sh = (9*8,)
    assert v.reshape(sh).shape == sh
    assert v.reshape(sh).iaxes == tuple()

    assert v.reshape((-1,)).shape == sh
    assert v.reshape((-1,)).iaxes == tuple()

    sh = (2, 3, 4, 3)
    assert v.reshape(sh).shape == sh
    assert v.reshape(sh).iaxes == tuple()

    assert v.reshape((2, 3, -1, 3)).shape == sh
    assert v.reshape((2, 3, -1, 3)).iaxes == tuple()

    # Zero-sized arrays.
    v = iid_repeat(iid_repeat(normal(size=(2, 3, 0)), 0), 5, axis=-2)

    sh = (1, 0, 3, 4, 0)
    assert v.reshape(sh).shape == sh
    assert v.reshape(sh).iaxes == tuple()

    v = assparsenormal([])

    sh = (1, 0, 3, 4, 0)
    assert v.reshape(sh).shape == sh
    assert v.reshape(sh).iaxes == tuple()

    # A scalar.
    v = assparsenormal(normal())
    
    sh = (1,)
    assert v.reshape(sh).shape == sh
    assert v.reshape(sh).iaxes == tuple()

    sh = (1, 1, 1, 1)
    assert v.reshape(sh).shape == sh
    assert v.reshape(sh).iaxes == tuple()

    with pytest.raises(ValueError):
        v.reshape((1, 2, 1))

    with pytest.raises(ValueError):
        v.reshape((1, 0, 1))

    # Non-trivial examples.

    v = iid_repeat(iid_repeat(normal(size=(4, 5)), 2, axis=0), 3, axis=0)
    assert v.shape == (3, 2, 4, 5)
    assert v.iaxes == (0, 1)

    sh = (3, 2, 20)
    assert v.reshape(sh).shape == sh
    assert v.reshape(sh).iaxes == (0, 1)

    assert v.reshape((3, 2, -1)).shape == sh
    assert v.reshape((3, 2, -1)).iaxes == (0, 1)

    with pytest.raises(ValueError):
        assert v.reshape((3, 2, 21))

    with pytest.raises(ValueError):
        assert v.reshape((3, 2, 19))

    sh = (3, 8, 5)
    assert np.reshape(np.ones(v.shape), sh).shape == sh
    # To check that the shape is valid for a numeric array.

    with pytest.raises(ValueError):
        assert v.reshape(sh)

    sh = (6, 4, 5)
    assert np.reshape(np.ones(v.shape), sh).shape == sh
    with pytest.raises(ValueError):
        assert v.reshape(sh)

    sh = (2, 3, 2, 2, 5)
    assert np.reshape(np.ones(v.shape), sh).shape == sh
    with pytest.raises(ValueError):
        assert v.reshape(sh)

    sh = (3, 1, 2, 4, 5)
    assert v.reshape(sh).shape == sh
    assert v.reshape(sh).iaxes == (0, 2)

    sh = (1, 3, 1, 2, 4, 5)
    v_ = v.reshape(sh)
    assert v_.shape == sh
    assert v_.iaxes == (1, 3)

    sh = (1, 3, 2, 2, 5, 2)
    assert v_.reshape(sh).shape == sh
    assert v_.reshape(sh).iaxes == (1, 2)

    v_ = iid_repeat(v, 1, axis=-1)
    assert v_.shape == (3, 2, 4, 5, 1)
    assert v_.iaxes == (0, 1, 4)

    # The preservation of a 1-sized independence axis.
    sh = (3, 2, 20, 1)
    v_ = v_.reshape(sh)
    assert v_.shape == sh
    assert v_.iaxes == (0, 1, 3)

    sh = (3, 2, 5, 2, 2, 1)
    v_ = v_.reshape(sh)
    assert v_.shape == sh
    assert v_.iaxes == (0, 1, 5)

    # The removal of iid axes, even trivial, is not allowed
    sh = (3, 2, 5, 4)
    with pytest.raises(ValueError):
        v_ = v_.reshape(sh)

    # Checking the arrangement of elements.
    tol = 1e-10

    v = iid_repeat(random_normal((8, 9)), 5)  # First axis.

    sh = (5, 3, 4, 2, 3)
    vvar = v.var()
    assert np.max(np.abs(v.reshape(sh).var() - vvar.reshape(sh))) < tol
    assert np.max(np.abs(v.reshape(sh, order="F").var() 
                         - vvar.reshape(sh, order="F"))) < tol
    
    # A sanity check that changing the order is not doing nothing.
    assert np.max(np.abs(v.reshape(sh, order="F").var() 
                         - vvar.reshape(sh, order="C"))) > tol
    
    v = iid_repeat(random_normal((8, 9)), 5, axis=-1)  # Last axis.

    sh = (3, 4, 2, 3, 5)
    vvar = v.var()
    assert np.max(np.abs(v.reshape(sh).var() - vvar.reshape(sh))) < tol
    assert np.max(np.abs(v.reshape(sh, order="F").var() 
                  - vvar.reshape(sh, order="F"))) < tol
    
    # A sanity check that changing the order is not doing nothing.
    assert np.max(np.abs(v.reshape(sh, order="F").var() 
                         - vvar.reshape(sh, order="C"))) > tol
    

    v = iid_repeat(random_normal((8, 9)), 5, axis=1)  # Middle axis.

    sh = (2, 4, 5, 3, 3)
    vvar = v.var()
    assert np.max(np.abs(v.reshape(sh).var() - vvar.reshape(sh))) < tol
    assert np.max(np.abs(v.reshape(sh, order="F").var() 
                         - vvar.reshape(sh, order="F"))) < tol
    
    # A sanity check that changing the order is not doing nothing.
    assert np.max(np.abs(v.reshape(sh, order="F").var() 
                  - vvar.reshape(sh, order="C"))) > tol
    
    # A shape that affects the iaxis.
    sh = (3, 4, 5, 2, 3)
    with pytest.raises(ValueError):
        v.reshape(sh)
    assert vvar.reshape(sh).shape == sh  # But this should work.


def test_sum():
    v = iid_repeat(iid_repeat(normal(size=(4, 5)), 2, axis=1), 3, axis=1)
    # v.shape is (4, 3, 2, 5), v.iaxes are (1, 2)

    for ax, iax in zip([0, 3, -1, -4, (0,), (0, 3), (-1, -4)],
                       [(0, 1), (1, 2), (1, 2), (0, 1), (0, 1), (0, 1), (0, 1)]):
        vout = v.sum(axis=ax)
        mean_ref = np.sum(v.mean(), axis=ax)
        assert vout.shape == mean_ref.shape
        assert vout.iaxes == iax
        assert np.all(vout.mean() == mean_ref)

        vvs = v.v.sum(axis=ax)
        assert np.all(vvs.emap.a == vout.v.emap.a)
        assert np.all(vvs.emap.elem == vout.v.emap.elem)

        vout = v.sum(axis=ax, keepdims=True)
        mean_ref = np.sum(v.mean(), axis=ax, keepdims=True)
        assert vout.shape == mean_ref.shape
        assert vout.iaxes == v.iaxes
        assert np.all(vout.mean() == mean_ref)

        vvs = v.v.sum(axis=ax, keepdims=True)
        assert np.all(vvs.emap.a == vout.v.emap.a)
        assert np.all(vvs.emap.elem == vout.v.emap.elem)

    assert v[:, :, None].sum(axis=2).shape == v.shape
    assert v[:, :, None].sum(axis=2).iaxes == v.iaxes

    # Independence axes are not allowed.
    with pytest.raises(ValueError):
        v.sum(axis=1)
    with pytest.raises(ValueError):
        v.sum(axis=2)
    with pytest.raises(ValueError):
        v.sum(axis=(0, 2))

    # None is not allowed.
    with pytest.raises(ValueError):
        v.sum(axis=None)

    # Axes out of bound.
    with pytest.raises(AxisError):
        v.sum(axis=4)
    with pytest.raises(AxisError):
        v.sum(axis=44)
    with pytest.raises(AxisError):
        v.sum(axis=-5)

    # Duplicated axis.
    with pytest.raises(ValueError):
        v.sum(axis=(0, 0))


def test_split():
    v = iid_repeat(iid_repeat(normal(size=(4, 6, 5)), 2, axis=1), 3, axis=1)
    # v.shape is (4, 3, 2, 6, 5), v.iaxes are (1, 2)

    with pytest.raises(ValueError):
        v.split(3, axis=1)
    np.split(v.var(), 3, axis=1)  # To check that a numeric array can be split.

    with pytest.raises(ValueError):
        v.split(3, axis=-4)
    np.split(v.var(), 3, axis=-4)

    with pytest.raises(ValueError):
        v.split(2, axis=2)
    np.split(v.var(), 2, axis=2)

    with pytest.raises(ValueError):
        v.split(2, axis=-3)
    np.split(v.var(), 2, axis=-3)

    # Axis out of bounds.
    with pytest.raises(AxisError):
        v.split(1, axis=5)
    with pytest.raises(AxisError):
        v.split(1, axis=-42)
    with pytest.raises(AxisError):
        v.split(1, axis=-6)

    tol = 1e-10

    v = iid_repeat(iid_repeat(random_normal((4, 6, 5)), 2, axis=1), 3, axis=1)
    # v.shape is (4, 3, 2, 6, 5), v.iaxes are (1, 2)

    assert len(v.split(3, axis=-2)) == 3
    for vs in v.split(3, axis=-2):
        assert vs.__class__ == v.__class__
    
    vvars = [vs.var() for vs in v.split(3, axis=-2)]
    vvars_ref = np.split(v.var(), 3, axis=-2)
    for vv, vvr in zip(vvars, vvars_ref):
        assert vv.shape == vvr.shape
        assert np.max(np.abs(vv - vvr)) < tol

    vvars = [vs.var() for vs in v.split([2, 3], axis=0)]
    vvars_ref = np.split(v.var(), [2, 3], axis=0)
    for vv, vvr in zip(vvars, vvars_ref):
        assert vv.shape == vvr.shape
        assert np.max(np.abs(vv - vvr)) < tol

    vvars = [vs.var() for vs in v.split([2, 4], axis=4)]
    vvars_ref = np.split(v.var(), [2, 4], axis=4)
    for vv, vvr in zip(vvars, vvars_ref):
        assert vv.shape == vvr.shape
        assert np.max(np.abs(vv - vvr)) < tol


def test_transpose():
    # Also .T property

    tol = 1e-10

    v = assparsenormal(1).transpose()
    assert v.shape == tuple()
    assert v.iaxes == tuple()

    v = assparsenormal(normal(size=(2, 3, 4))).transpose()
    assert v.shape == (4, 3, 2)
    assert v.iaxes == tuple()

    v = assparsenormal(normal(size=(2, 3, 4))).transpose((1, 0, 2))
    assert v.shape == (3, 2, 4)
    assert v.iaxes == tuple()

    # A matrix variable with one independence axis.
    v = iid_repeat(random_normal((4,)), 2, axis=1)
    # v.shape is (4, 2), v.iaxes are (1,)

    assert v.transpose().iaxes == (0,)

    vvar1 = v.transpose().var()
    vvar2 = v.var().transpose()
    assert vvar1.shape == vvar2.shape
    assert np.max(np.abs(vvar1 - vvar2)) < tol

    vvar1 = v.T.var()
    assert vvar1.shape == vvar2.shape
    assert np.max(np.abs(vvar1 - vvar2)) < tol
    
    # A multi-dimensional variable.
    v = iid_repeat(iid_repeat(random_normal((4, 6, 5)), 2, axis=1), 3, axis=1)
    # v.shape is (4, 3, 2, 6, 5), v.iaxes are (1, 2)

    assert v.T.shape == (5, 6, 2, 3, 4)
    assert v.T.iaxes == (2, 3)

    ax = (0, 1, 3, 4, 2)
    vvar1 = v.transpose(ax).var()
    vvar2 = v.var().transpose(ax)
    assert vvar1.shape == vvar2.shape
    assert np.max(np.abs(vvar1 - vvar2)) < tol
    assert v.transpose(ax).iaxes == (1, 4)

    ax = (-1, 2, 3, 0, 1)
    vvar1 = v.transpose(ax).var()
    vvar2 = v.var().transpose(ax)
    assert vvar1.shape == vvar2.shape
    assert np.max(np.abs(vvar1 - vvar2)) < tol
    assert v.transpose(ax).iaxes == (1, 4)

    ax = (-1, 3, 2, 0, 1)
    vvar1 = v.transpose(ax).var()
    vvar2 = v.var().transpose(ax)
    assert vvar1.shape == vvar2.shape
    assert np.max(np.abs(vvar1 - vvar2)) < tol
    assert v.transpose(ax).iaxes == (2, 4)

    ax = (1, -2, 2, -1, 0)
    vvar1 = v.transpose(ax).var()
    vvar2 = v.var().transpose(ax)
    assert vvar1.shape == vvar2.shape
    assert np.max(np.abs(vvar1 - vvar2)) < tol
    assert v.transpose(ax).iaxes == (0, 2)

    ax = None
    vvar1 = v.transpose(ax).var()
    vvar2 = v.var().transpose(ax)
    assert vvar1.shape == vvar2.shape
    assert np.max(np.abs(vvar1 - vvar2)) < tol
    assert v.transpose(ax).iaxes == (2, 3)

    vvar1 = v.T.var()
    assert vvar1.shape == vvar2.shape
    assert np.max(np.abs(vvar1 - vvar2)) < tol
    assert v.transpose(ax).iaxes == (2, 3)

    v = iid_repeat(iid_repeat(random_normal((4, 5)), 2, axis=1), 3, axis=3)
    # v.shape is (4, 2, 5, 3), v.iaxes are (1, 3)

    ax = (2, 3, 0, 1)
    vvar1 = v.transpose(ax).var()
    vvar2 = v.var().transpose(ax)
    assert vvar1.shape == vvar2.shape
    assert np.max(np.abs(vvar1 - vvar2)) < tol
    assert v.transpose(ax).iaxes == (1, 3)

    ax = (1, 0, -1, -2)
    vvar1 = v.transpose(ax).var()
    vvar2 = v.var().transpose(ax)
    assert vvar1.shape == vvar2.shape
    assert np.max(np.abs(vvar1 - vvar2)) < tol
    assert v.transpose(ax).iaxes == (0, 2)

    ax = (1, 0, -2, 3)
    vvar1 = v.transpose(ax).var()
    vvar2 = v.var().transpose(ax)
    assert vvar1.shape == vvar2.shape
    assert np.max(np.abs(vvar1 - vvar2)) < tol
    assert v.transpose(ax).iaxes == (0, 3)


def test_trace():
    v = iid_repeat(iid_repeat(normal(size=(4, 5)), 2, axis=0), 3, axis=0)
    # v.shape is (3, 2, 4, 5), v.iaxes are (0, 1)
    assert v.trace(axis1=-2, axis2=-1).shape == (3, 2)
    assert v.trace(axis1=-2, axis2=-1).iaxes == (0, 1)

    assert v.trace(axis1=3, axis2=2).shape == (3, 2)
    assert v.trace(axis1=3, axis2=2).iaxes == (0, 1)

    with pytest.raises(ValueError):
        v.trace(axis1=0, axis2=1)

    with pytest.raises(ValueError):
        v.trace(axis1=1, axis2=2)

    v = iid_repeat(iid_repeat(normal(size=(4, 5)), 2, axis=-1), 3, axis=-1)
    # v.shape is (4, 5, 2, 3), v.iaxes are (2, 3)
    assert v.trace(axis1=0, axis2=1).shape == (2, 3)
    assert v.trace(axis1=0, axis2=1).iaxes == (0, 1)

    v = iid_repeat(iid_repeat(normal(size=(4, 5)), 2, axis=1), 3, axis=1)
    # v.shape is (4, 3, 2, 5), v.iaxes are (1, 2)

    assert v.trace(axis1=0, axis2=-1).shape == (3, 2)
    assert v.trace(axis1=0, axis2=-1).iaxes == (0, 1)

    assert v[:, :, None].trace(axis1=0, axis2=2).shape == (3, 2, 5)
    assert v[:, :, None].trace(axis1=0, axis2=2).iaxes == (0, 1)

    assert v[:, :, None].trace(axis1=2, axis2=4).shape == (4, 3, 2)
    assert v[:, :, None].trace(axis1=2, axis2=4).iaxes == (1, 2)

    with pytest.raises(ValueError):
        v.trace(axis1=0, axis2=1)

    with pytest.raises(ValueError):
        v.trace(axis1=-1, axis2=-2)

    with pytest.raises(ValueError):
        v.trace(axis1=-2, axis2=1)


def test_concatenate():
    tol = 1e-10

    xi = random_normal((8, 2))
    v = iid_repeat(xi, 7, axis=-1)
    
    v1 = v[:3]  # (3, 2, 7)
    v2 = v[3:]  # (5, 2, 7)

    v_ = gp.concatenate([v1, v2], axis=0)
    assert v.shape == v_.shape
    assert v.iaxes == v_.iaxes
    assert np.max(np.abs(v.mean() - v_.mean())) < tol
    assert np.max(np.abs(v.var() - v_.var())) < tol

    v_ = gp.concatenate([v1, v2], axis=-3)
    assert v.shape == v_.shape
    assert v.iaxes == v_.iaxes
    assert np.max(np.abs(v.mean() - v_.mean())) < tol
    assert np.max(np.abs(v.var() - v_.var())) < tol

    v1 = v[:4]  # (4, 2, 7)
    v2 = v[4:]  # (4, 2, 7)

    v_ = gp.concatenate([v1, v2], axis=0)
    assert v.shape == v_.shape
    assert v.iaxes == v_.iaxes
    assert np.max(np.abs(v.mean() - v_.mean())) < tol
    assert np.max(np.abs(v.var() - v_.var())) < tol

    v_ = gp.concatenate([v1, v1, v1, v1], axis=1)
    assert v_.shape == (4, 8, 7)
    assert v.iaxes == v_.iaxes

    v_ = gp.concatenate([v1, v1, v1, v2, v1, v2], axis=1)
    assert v_.shape == (4, 12, 7)
    assert v.iaxes == v_.iaxes

    v_ = gp.concatenate([v1, np.ones((4, 4, 7)), np.ones((4, 1, 7))], axis=1)
    assert v_.shape == (4, 7, 7)
    assert v.iaxes == v_.iaxes

    with pytest.raises(ValueError):
        v_ = gp.concatenate([v1, v2], axis=-1)

    # Axis out of bounds.
    v1 = v[:4]  # (4, 2, 7)
    v2 = v[4:]  # (4, 2, 7)

    with pytest.raises(AxisError):
        gp.concatenate([v1, v2], axis=3)
    with pytest.raises(AxisError):
        gp.concatenate([v1, v2], axis=-4)
    with pytest.raises(AxisError):
        gp.concatenate([v1, v2], axis=42)

    vs = []
    for _ in range(100):
        xi = normal(size=(2,))
        v = iid_repeat(iid_repeat(iid_repeat(xi, 3, axis=0), 4, axis=0), 5, axis=0)
        # shape (5, 4, 3, 2), iaxes (0, 1, 2)

        vs.append(v)

    v = gp.concatenate(vs, axis=3)
    assert v.shape == (5, 4, 3, 200)
    assert v.iaxes == (0, 1, 2)

    v = iid_repeat(normal(size=(1,)), 7)

    with pytest.raises(ValueError):
        gp.concatenate([v, normal(size=(7, 1))], axis=1)
        # Concatenation of regular and sparse normal variables is never possible 
        # because regular variables do not have independence axes.

    # Concatenation with numeric arrays, however, is possible.
    v_ = gp.concatenate([v, np.ones((7, 1))], axis=1)
    assert v_.shape == (7, 2)
    assert v_.iaxes == (0,)


def test_stack():
    tol = 1e-10

    xi = random_normal((8, 2))
    v = iid_repeat(xi, 7, axis=-1)
    v = iid_repeat(v, 3, axis=1)

    v1 = v[:4]  # shape (4, 3, 2, 7), iaxes (1, 3) 
    v2 = v[4:]  # shape (4, 3, 2, 7), iaxes (1, 3)

    for ax, sh, iax in zip([0, 1, 2, -2, 4],
                           [(2, 4, 3, 2, 7), (4, 2, 3, 2, 7), (4, 3, 2, 2, 7), (4, 3, 2, 2, 7), (4, 3, 2, 7, 2)],
                           [(2, 4), (2, 4), (1, 4), (1, 4), (1, 3)]):
        
        v_ = gp.stack([v1, v2], axis=ax)
        refmean = np.stack([v1.mean(), v2.mean()], axis=ax)
        refvar = np.stack([v1.var(), v2.var()], axis=ax)
        assert v_.shape == sh
        assert v_.iaxes == iax
        assert np.max(np.abs(v_.mean() - refmean)) < tol
        assert np.max(np.abs(v_.var() - refvar)) < tol

    v_ = gp.stack([v1, v1, v1, v1], axis=-5)
    assert v_.shape == (4, 4, 3, 2, 7)
    assert v_.iaxes == (2, 4)

    with pytest.raises(AxisError):
        gp.stack([v1, v2], axis=5)
    with pytest.raises(AxisError):
        gp.stack([v1, v2], axis=-6)
    with pytest.raises(AxisError):
        gp.stack([v1, v2], axis=42)

    vs = []
    for _ in range(100):
        xi = normal(size=(2,))
        v = iid_repeat(iid_repeat(iid_repeat(xi, 3, axis=0), 4, axis=0), 5, axis=0)
        # shape (5, 4, 3, 2), iaxes (0, 1, 2)

        vs.append(v)

    v = gp.stack(vs, axis=3)
    assert v.shape == (5, 4, 3, 100, 2)
    assert v.iaxes == (0, 1, 2)

    v = iid_repeat(normal(size=(1,)), 7)

    with pytest.raises(ValueError):
        gp.stack([v, normal(size=(7, 1))], axis=1)
        # Stacking of regular and sparse normal variables is never possible 
        # because regular variables do not have independence axes.

    # Concatenation with numeric arrays, however, is possible.
    v_ = gp.stack([v, np.ones((7, 1))], axis=1)
    assert v_.shape == (7, 2, 1)
    assert v_.iaxes == (0,)


def test_matmul():
    # Some trivial cases first.

    v = assparsenormal(1)
    with pytest.raises(ValueError):
        v @ [1, 2]
    
    v = assparsenormal([1, 2, 3])
    w = v @ [1, 1, 1]
    assert w.shape == tuple()
    assert w.iaxes == tuple()

    w = [1, 1, 1] @ v
    assert w.shape == tuple()
    assert w.iaxes == tuple()

    # When at least one variable is a proper sparse normal.

    with pytest.raises(ValueError):
        iid_repeat(normal(), 3) @ [1, 1, 1]
    with pytest.raises(ValueError):
         [1, 1, 1] @ iid_repeat(normal(), 3)

    v = iid_repeat(normal(size=(3,)), 7)  # shape (7, 3), iaxes (0,)

    # Simple cases, no broadcasting.
    w = v @ np.ones((3,))
    assert w.shape == (7,)
    assert w.iaxes == (0,)

    w = v @ np.ones((3, 5))
    assert w.shape == (7, 5)
    assert w.iaxes == (0,)

    w = np.ones((3,)) @ v.T
    assert w.shape == (7,)
    assert w.iaxes == (0,)

    w = np.ones((6, 3)) @ v.T
    assert w.shape == (6, 7)
    assert w.iaxes == (1,)

    # Broadcasting when the numerical operand has larger dimension.
    w = v @ np.ones((4, 3, 5))
    assert w.shape == (4, 7, 5)
    assert w.iaxes == (1,)

    w = v @ np.ones((2, 3, 1, 5, 4, 3, 5))
    assert w.shape == (2, 3, 1, 5, 4, 7, 5)
    assert w.iaxes == (5,)

    w = np.ones((4, 6, 3)) @ v.T
    assert w.shape == (4, 6, 7)
    assert w.iaxes == (2,)

    w = np.ones((2, 3, 1, 5, 4, 6, 3)) @ v.T
    assert w.shape == (2, 3, 1, 5, 4, 6, 7)
    assert w.iaxes == (6,)

    # Contraction over independence axes is not allowed.
    with pytest.raises(ValueError):
        np.ones((5, 7)) @ v
    with pytest.raises(ValueError):
        v.T @ np.ones((7, 5))

    # Scalars are not allowed.
    with pytest.raises(ValueError):
        v @ 1
    with pytest.raises(ValueError):
        1 @ v

    # A larger-dimensional variable.
    # Including broadcasting when the random operand has larger dimension.

    v = iid_repeat(iid_repeat(normal(size=(3, 5)), 7, axis=1), 4)  
    # shape (4, 3, 7, 5), iaxes (0, 2)

    w = v @ np.ones((5,))
    assert w.shape == (4, 3, 7,)
    assert w.iaxes == (0, 2)

    w = v @ np.ones((5, 8))
    assert w.shape == (4, 3, 7, 8)
    assert w.iaxes == (0, 2)

    w = v @ np.ones((4, 3, 5, 8))
    assert w.shape == (4, 3, 7, 8)
    assert w.iaxes == (0, 2)

    w = v @ np.ones((2, 4, 3, 5, 8))
    assert w.shape == (2, 4, 3, 7, 8)
    assert w.iaxes == (1, 3)

    v = v.transpose((0, 1, 3, 2))  # shape (4, 3, 5, 7), iaxes (0, 3)
    
    w = np.ones((5,)) @ v
    assert w.shape == (4, 3, 7)
    assert w.iaxes == (0, 2)

    w = np.ones((8, 5)) @ v
    assert w.shape == (4, 3, 8, 7)
    assert w.iaxes == (0, 3)

    w = np.ones((4, 3, 8, 5)) @ v
    assert w.shape == (4, 3, 8, 7)
    assert w.iaxes == (0, 3)

    w = np.ones((2, 4, 3, 8, 5)) @ v
    assert w.shape == (2, 4, 3, 8, 7)
    assert w.iaxes == (1, 4)

    # An example with two random variables.

    tol = 1e-10
    
    v1 = iid_repeat(iid_repeat(random_normal((3, 5)), 7), 4)
    # shape (4, 7, 3, 5), iaxes (0, 1)

    v2 = iid_repeat(iid_repeat(random_normal((5, 1)), 7), 4)
    # shape (4, 7, 5, 1), iaxes (0, 1)

    v2.v += v1.v[0, 0, 0, 0]  # for establishing correlation.

    w = v1 @ v2
    assert w.shape == (4, 7, 3, 1)
    assert w.iaxes == (0, 1)

    v1m = v1.mean()
    v2m = v2.mean()
    wref = v1m @ v2m + (v1 - v1m) @ v2m + v1m @ (v2 - v2m)

    assert w.shape == wref.shape
    assert np.max(np.abs(w.mean() - wref.mean())) < tol
    assert np.max(np.abs(w.var() - wref.var())) < tol

    v3 = iid_repeat(normal(size=(7, 5, 1)), 4)
    with pytest.raises(ValueError):
        v1 @ v3


def test_einsum():
    tol = 1e-10

    def assert_equal(v1, v2):
        assert v1.shape == v2.shape
        assert v1.iaxes == v2.iaxes
        assert np.max(np.abs(v1.v.emap.a - v2.v.emap.a)) < tol
        assert v1.v.emap.elem == v2.v.emap.elem
        assert np.max(np.abs(v1.mean() - v2.mean())) < tol

    # Tests against matrix multiplication.

    xi = random_normal((3, 2))
    v = iid_repeat(xi, 5, axis=1)  # shape (3, 5, 2), iaxes (1,)
    
    x = np.random.rand(2)  # 1D

    v_ei = gp.einsum("rij, j -> ri", v, x)
    v_mm = gp.matmul(v, x)
    assert_equal(v_ei, v_mm)

    v_ei = gp.einsum("ij, j -> i", v[0], x)
    v_mm = gp.matmul(v[0], x)
    assert_equal(v_ei, v_mm)

    v_ei = gp.einsum("j, rji -> ri", x, v.transpose((0, 2, 1)))
    v_mm = gp.matmul(x, v.transpose((0, 2, 1)))
    assert_equal(v_ei, v_mm)

    v_ei = gp.einsum("j, rj... -> ...r", x.T, v.transpose((0, 2, 1)))
    v_mm = gp.matmul(x.T, v.transpose((0, 2, 1)))
    assert_equal(v_ei.T, v_mm)

    v_ei = gp.einsum("j, ...ji -> i...", x, v.transpose((0, 2, 1))[1, ...])
    v_mm = gp.matmul(x.T, v.transpose((0, 2, 1))[1, ...])
    assert_equal(v_ei.T, v_mm)

    x = np.random.rand(2, 8)  # 2D

    v_ei = gp.einsum("rij, jk -> rik", v, x)
    v_mm = gp.matmul(v, x)
    assert_equal(v_ei, v_mm)

    v_ei = gp.einsum("ij, jk -> ik", v[0], x)
    v_mm = gp.matmul(v[0], x)
    assert_equal(v_ei, v_mm)

    # Swapping the order 1.
    v_ei = gp.einsum("jk, rij -> rik", x, v)
    v_mm = gp.matmul(v, x)
    assert_equal(v_ei, v_mm)

    # Swapping the order 2.
    v_ei = gp.einsum("kj, rji -> rki", x.T, v.transpose((0, 2, 1)))
    v_mm = gp.matmul(x.T, v.transpose((0, 2, 1)))
    assert_equal(v_ei, v_mm)

    # With an ellipsis.
    v_ei = gp.einsum("...j, jk -> ...k", v, x)
    v_mm = gp.matmul(v, x)
    assert_equal(v_ei, v_mm)

    # Implicit output.
    v_ei = gp.einsum("...j, jk", v, x)
    v_mm = gp.matmul(v, x)
    assert_equal(v_ei, v_mm)

    # Attempts contracting over an independence axis.
    x = np.random.rand(5, 8)
    gp.einsum("rji, jk -> rik", v.mean(), x)  # No error because of the shapes.
    with pytest.raises(ValueError):
        gp.einsum("rji, jk -> rik", v, x)
    
    gp.einsum("j, jk -> k", v.mean()[1, :, 1], x)
    with pytest.raises(ValueError):
        gp.einsum("j, jk -> k", v[1, :, 1], x)

    xi = random_normal((3, 2))
    v = iid_repeat(xi, 5, axis=1)  # shape (3, 5, 2), iaxes (1,)

    x = np.random.rand(4, 3, 2, 8)  # 4D

    v_ei = gp.einsum("ijk, liko -> lijo", v, x)
    v_mm = gp.matmul(v, x)
    assert_equal(v_ei, v_mm)

    v_ei = gp.einsum("liko, ijk -> jilo", x, v)
    v_mm = gp.matmul(v, x).transpose((-2, 1, 0, -1))
    assert_equal(v_ei, v_mm)

    # TODO:
    # Tests agains inner.
    # Tests against outer.