import pytest
import numpy as np

from gprob.normal_ import normal
from gprob.sparse import _parse_index_key, iid_repeat, assparsenormal


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

    assert iid_repeat(v, 4, axis=2).shape == (2, 5, 4, 3, 6)
    assert iid_repeat(v, 4, axis=2).iaxes == (1, 2, 4)

    assert iid_repeat(v, 4, axis=3).shape == (2, 5, 3, 4, 6)
    assert iid_repeat(v, 4, axis=3).iaxes == (1, 3, 4)


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

    with pytest.raises(ValueError):
        v.cumsum(axis=1)

    with pytest.raises(ValueError):
        v.cumsum(axis=-2)

    # Flattening is not allowed.
    with pytest.raises(ValueError):
        v.cumsum(axis=None)


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