import pytest
import itertools
import numpy as np
from numpy.exceptions import AxisError
from numpy.exceptions import ComplexWarning

import gprob as gp
from gprob.normal_ import normal, Normal
from gprob.sparse import (_parse_index_key, iid_repeat, assparsenormal, 
                          SparseNormal, SparseConditionWarning)

from utils import random_normal, get_message

np.random.seed(0)


def dense_to_sparse_cov(cov, iaxes):
    ndim = cov.ndim // 2
    for i, ax in enumerate(iaxes):
        cov = np.diagonal(cov, axis1=ax - i, axis2=ax + ndim - 2*i)
    return cov


def test_construction():
    tol = 1e-8

    m = [[1, 2, 3], [4, 5, 6]]
    fcv = m + normal()
    v = SparseNormal(fcv, (1, None))
    assert v.iaxes == (0,)
    assert np.max(np.abs(v.var() - np.ones((2, 3)))) < tol
    assert np.max(np.abs(v.mean() - m)) < tol

    # A non-normal input as the kernel.
    with pytest.raises(TypeError):
        SparseNormal(m, (1, None))
    
    # A non-tuple iaxid.
    with pytest.raises(ValueError):
        SparseNormal(fcv, [1, None])

    # Wrong size of iaxid.
    with pytest.raises(ValueError):
        SparseNormal(fcv, (1, None, None))
    with pytest.raises(ValueError):
        SparseNormal(fcv, tuple())

    # Wrong content in iaxid.
    with pytest.raises(ValueError):
        SparseNormal(fcv, (1, 0))
    with pytest.raises(ValueError):
        SparseNormal(fcv, (True, False))


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


def test_properties():
    # size, shape, ndim, iscomplex, real, imag, iaxes, T

    tol = 1e-8

    v1 = iid_repeat(iid_repeat(normal(), 4), 8) * np.random.rand(8, 4)
    v2 = iid_repeat(iid_repeat(normal(), 4), 8)

    vc = v1 + 1j * v2

    assert v1.iscomplex == False
    assert vc.iscomplex == True

    for v in [v1, vc]:
        assert v.size == 4 * 8
        assert v.shape == (8, 4)
        assert v.ndim == 2
        assert v.iaxes == (0, 1)

        vt = v.T
        assert isinstance(vt, SparseNormal)
        assert vt.shape == (4, 8)
        assert vt.iscomplex == v.iscomplex
        assert np.max(np.abs(vt.mean() - v.mean().T)) < tol
        assert np.max(np.abs(vt.var() - v.var().T)) < tol

        vr = v.real
        assert isinstance(vr, SparseNormal)
        assert vr.shape == (8, 4)
        assert vr.iscomplex == False
        assert np.max(np.abs(vr.mean() - v.mean().real)) < tol

        vi = v.imag
        assert isinstance(vi, SparseNormal)
        assert vi.shape == (8, 4)
        assert vi.iscomplex == False
        assert np.max(np.abs(vi.mean() - v.mean().imag)) < tol

        assert np.max(np.abs(vr.var() + vi.var() - v.var())) < tol


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


def test_broadcast_to():
    # broadcast_to of the sparse variable module is a convenience function for 
    # explicit broadcasting, while broadcasting in arithmetic operations is
    # implemented differently.

    # Trivial examples.

    v = assparsenormal(normal())
    v = gp.broadcast_to(v, (3, 4))
    assert v.shape == (3, 4)
    assert v.iaxes == tuple()

    v = assparsenormal(normal(size=(3, 1)))
    v = gp.broadcast_to(v, (5, 3, 4))
    assert v.shape == (5, 3, 4)
    assert v.iaxes == tuple()

    with pytest.raises(ValueError):
        gp.broadcast_to(v, (4,))

    with pytest.raises(ValueError):
        gp.broadcast_to(v, (4, 1))

    # 1 Independence axis.

    v = iid_repeat(normal(), 4)
    v = gp.reshape(v, (1, 4, 1))
    assert v.iaxes == (1,)
    v = gp.broadcast_to(v, (5, 3, 4, 2))
    assert v.shape == (5, 3, 4, 2)
    assert v.iaxes == (2,)

    v = iid_repeat(normal(size=(2, 3)), 4)
    assert v.iaxes == (0,)
    v = gp.broadcast_to(v, (2, 6, 4, 2, 3))
    assert v.shape == (2, 6, 4, 2, 3)
    assert v.iaxes == (2,)

    with pytest.raises(ValueError):
        gp.broadcast_to(v, (4, 3, 3))

    with pytest.raises(ValueError):
        gp.broadcast_to(v, (2, 3))

    # 2 independence axes.

    v = iid_repeat(normal(size=(2, 4)), 3, axis=1)
    v = iid_repeat(v, 5, axis=-1)
    assert v.iaxes == (1, 3)
    v = gp.broadcast_to(v, (6, 2, 3, 4, 5))
    assert v.shape == (6, 2, 3, 4, 5)
    assert v.iaxes == (2, 4)

    v = iid_repeat(normal(size=(2, 1)), 3, axis=1)
    v = iid_repeat(v, 5, axis=-1)
    assert v.iaxes == (1, 3)
    v = gp.broadcast_to(v, (6, 2, 3, 4, 5))
    assert v.shape == (6, 2, 3, 4, 5)
    assert v.iaxes == (2, 4)
    

def test_cov():
    tol = 1e-10

    nv = random_normal((2, 3))
    v = iid_repeat(iid_repeat(nv, 4, axis=1), 5, axis=-1)
    # shape (2, 4, 3, 5), iaxes (1, 3)

    r1 = np.random.rand(1, 4, 1, 1)
    r2 = np.random.rand(1, 1, 1, 5)

    v1 = r1 * r2 * v

    # A dense Normal variable with the same statistics.
    v2 = gp.stack([nv.iid_copy() for _ in range(4)], axis=1)
    v2 = gp.stack([v2.iid_copy() for _ in range(5)], axis=-1)

    v2 = r1 * r2 * v2

    c1 = v1.cov()
    c2 = v2.cov()
    c2 = np.diagonal(c2, axis1=1, axis2=5)
    c2 = np.diagonal(c2, axis1=2, axis2=5)
    assert c2.shape == c1.shape
    assert np.max(np.abs(c1 - c2)) < tol
    
    # Covariance betveen two variables.

    n = 20
    v1 = iid_repeat(normal(), n)
    v2 = iid_repeat(normal(), n)
    v = v1 + v2

    r1 = 2 * np.random.rand(n) - 1
    r2 = 2 * np.random.rand(n) - 1

    c1 = 2 * gp.sparse.cov(v * r1, v1 * r2)
    c2 = (v * r1 + v1 * r2).var() - (r1 ** 2) * v.var() - (r2 ** 2) * v1.var()
    assert c1.shape == c2.shape
    assert np.max(np.abs(c1 - c2)) < tol

    v1_ = normal(size=(n,))
    v2_ = normal(size=(n,))
    v_ = v1_ + v2_

    c1_ = np.diagonal(2 * gp.normal_.cov(v_ * r1, v1_ * r2))
    assert c1.shape == c1_.shape
    assert np.max(np.abs(c1 - c1_)) < tol

    v = iid_repeat(iid_repeat(normal(), 4), 4)
    with pytest.raises(ValueError):
        gp.sparse.cov(v, v.T)
    with pytest.raises(ValueError):
        gp.sparse.cov(v, iid_repeat(normal((4,)), 4))


def test_cov_det():
    # Tests for sparse covariance with deterministic constants, covering
    # all the error messaves in different branches.

    tol = 1e-8

    with pytest.raises(ValueError):
        gp.sparse.cov(1, 1)  # Two deterministic constants are not allowed 
                             # in the sparse implementation.

    # One independence axis.

    v = iid_repeat(normal(), 4)  # shape (4,)

    c = gp.sparse.cov(v, np.ones((4,)))
    assert c.shape == (4,)
    assert np.max(np.abs(c)) < tol

    c = gp.sparse.cov(np.ones((4,)), v)
    assert c.shape == (4,)
    assert np.max(np.abs(c)) < tol

    v = iid_repeat(normal(size=(4,)), 4)  # shape (4,)

    with pytest.raises(ValueError):
        gp.sparse.cov(v, np.ones((4, 4)))

    with pytest.raises(ValueError):
        gp.sparse.cov(np.ones((4, 4)), v)

    v = iid_repeat(normal(size=(3,)), 4)  # shape (4, 3)

    c = gp.sparse.cov(v, np.ones((2, 4)))
    assert c.shape == (3, 2, 4)
    assert np.max(np.abs(c)) < tol

    c = gp.sparse.cov(np.ones((2, 4)), v)
    assert c.shape == (2, 3, 4)
    assert np.max(np.abs(c)) < tol
    
    with pytest.raises(ValueError) as e_1l:
        gp.sparse.cov(v, np.ones((2, 3)))  # Mismatching shape.

    with pytest.raises(ValueError) as e_1r:
        gp.sparse.cov(np.ones((2, 3)), v)
    
    with pytest.raises(ValueError) as e_2l:
        gp.sparse.cov(v, np.ones((4, 4)))  # Ambiguous shape.

    with pytest.raises(ValueError) as e_2r:
        gp.sparse.cov(np.ones((4, 4)), v)

    # Two independence axes.

    v = iid_repeat(iid_repeat(normal(), 4), 4)

    r = np.random.rand(4, 4)
    c = gp.sparse.cov(v, r)
    assert c.shape == (4, 4)
    assert np.max(np.abs(c)) < tol

    c = gp.sparse.cov(r, v)  # Changing the order.
    assert c.shape == (4, 4)
    assert np.max(np.abs(c)) < tol

    v = iid_repeat(iid_repeat(normal(size=(3,)), 4), 4)  # shape (4, 4, 3)

    c = gp.sparse.cov(v, np.ones((4, 2, 4)))
    assert c.shape == (3, 2, 4, 4)
    assert np.max(np.abs(c)) < tol

    c = gp.sparse.cov(np.ones((4, 2, 5, 4)), v)
    assert c.shape == (2, 5, 3, 4, 4)
    assert np.max(np.abs(c)) < tol

    with pytest.raises(ValueError) as e_3l:
        gp.sparse.cov(v, np.ones((2, 4)))  # Too few dimensions of size 4.

    with pytest.raises(ValueError) as e_3r:
        gp.sparse.cov(np.ones((2, 4)), v)

    assert get_message(e_1l) == get_message(e_1r)
    assert get_message(e_2l) == get_message(e_2r)
    assert get_message(e_3l) == get_message(e_3r)

    emsgs = [get_message(e_1l), get_message(e_2l), get_message(e_3l)]

    # All the error messages are unique.
    assert len(set(emsgs)) == 3

    # Checks that there were no other errors while generating the message.
    for m in emsgs:
        assert "sparse" in m

    # More error tests.

    with pytest.raises(ValueError):
        gp.sparse.cov(v, np.ones((4, 2, 4, 4)))

    with pytest.raises(ValueError):
        gp.sparse.cov(np.ones((4, 2, 4, 4)), v)


def test_sample():
    v = assparsenormal(1)
    assert v.sample().shape == v.shape
    assert v.sample(1).shape == (1,) + v.shape

    v1 = iid_repeat(iid_repeat(random_normal((2, 3)), 4, axis=1), 5, axis=-1)
    v2 = iid_repeat(iid_repeat(random_normal((2, 3)), 4, axis=1), 5, axis=-1)
    v = 0.5 * v1 - v2
    # shape (2, 4, 3, 5), iaxes (1, 3)

    assert v.sample().shape == v.shape
    assert v.sample(1).shape == (1,) + v.shape

    ns = 10000
    s = v.sample(ns)
    assert len(s) == ns

    m = np.sum(s, axis=0) / ns
    vv = np.sum((s - m) ** 2, axis=0) / ns

    tol = 10 / np.sqrt(ns)
    assert m.shape == v.mean().shape
    assert np.mean((m - v.mean())**2) / np.max(v.mean()**2) < tol ** 2
    assert np.mean((vv - v.var())**2) / np.max(v.var()**2) < tol ** 2


def test_condition():
    tol = 1e-8

    def check_sparse_vs_normal(snv, nv):
        assert snv.shape == nv.shape
        assert np.max(np.abs(snv.mean() - nv.mean())) < tol
        assert np.max(np.abs(snv.cov() - 
                             dense_to_sparse_cov(nv.cov(), snv.iaxes))) < tol
        
    def check_permutations(snv_list, nv_list):
        # Checks sparse vs normal for all the permutations of the axes of 
        # the variable being conditioned.

        # 4d variables minimum.
        # minimum 5 items in the lists.

        nv = sum(nv_list)
        snv = sum(snv_list)

        # Transpositions of the variable.
        for ax in itertools.permutations(tuple(range(nv.ndim))):
            # Real-real.
            nv_ = nv.transpose(ax)
            snv_ = snv.transpose(ax)
            nvc = nv_.condition(nv_list[0] - 1.2 * nv_list[3])
            snvc = snv_.condition(snv_list[0] - 1.2 * snv_list[3])
            check_sparse_vs_normal(snvc, nvc)
            del snvc, nvc

            # Real-complex.
            nv_ = nv.transpose(ax)
            snv_ = snv.transpose(ax)
            nvc = nv_.condition(nv_list[0] + 0.4j * nv_list[1] 
                                + (0.3 + 2j) * nv_list[3])
            snvc = snv_.condition(snv_list[0] + 0.4j * snv_list[1]
                                  + (0.3 + 2j) * snv_list[3])
            check_sparse_vs_normal(snvc, nvc)
            del snvc, nvc

            # Complex-complex.
            nv_ = (nv + 2j * nv_list[1]).transpose(ax)
            snv_ = (snv + 2j * snv_list[1]).transpose(ax)
            nvc = nv_.condition(nv_list[0] + 0.4j * nv_list[1] 
                                + (0.3 + 2j) * nv_list[3])
            snvc = snv_.condition(snv_list[0] + 0.4j * snv_list[1]
                                  + (0.3 + 2j) * snv_list[3])
            check_sparse_vs_normal(snvc, nvc)
            del snvc, nvc

            # Complex-complex, with a transposed condition.
            nv_ = (nv + 2j * nv_list[1]).transpose(ax)
            snv_ = (snv + 2j * snv_list[1]).transpose(ax)
            nvc = nv_.condition({(nv_list[0] + 0.4j * nv_list[1] 
                                  + (0.3 + 2j) * nv_list[3]) : 0,
                                 nv_list[4][..., 1].transpose((2, 0, 1)) : 0.1})
            snvc = snv_.condition({(snv_list[0] + 0.4j * snv_list[1]
                                    + (0.3 + 2j) * snv_list[3]) : 0,
                                   snv_list[4][..., 1].transpose((2, 0, 1)) : 0.1})
            check_sparse_vs_normal(snvc, nvc)
            del snvc, nvc

    # 0 independence axes. -----------------------------------------------------

    # A scalar variable.
    nv1 = normal(-2, 10.3)
    nv2 = normal(0.2, 3.4)
    snv1 = assparsenormal(nv1)
    snv2 = assparsenormal(nv2)

    nvc = (nv1 + nv2).condition(nv1 - 0.3 * nv2)
    snvc = (snv1 + snv2).condition(snv1 - 0.3 * snv2)
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    # Trivial condition.
    nvc = (nv1 + nv2).condition(normal())
    snvc = (snv1 + snv2).condition(normal())
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    nvc = (nv1 + nv2).condition({nv1 - 0.3 * nv2 : 0.5})
    snvc = (snv1 + snv2).condition({snv1 - 0.3 * snv2 : 0.5})
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    # Multiple conditions.
    nv1 = normal(-2, 10.3)
    nv2 = normal(0.2, 3.4)
    snv1 = assparsenormal(nv1)
    snv2 = assparsenormal(nv2)

    nvc = (nv1 + nv2).condition({nv1 - 0.3 * nv2 : 0.5, nv1 + 0.3 * nv2 : 0.1})
    snvc = (snv1 + snv2).condition({snv1 - 0.3 * snv2 : 0.5, 
                                    snv1 + 0.3 * snv2 : 0.1})
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    # A multi-dimensional variable.
    nv1 = normal(-2, 10.3, size=(2, 2, 2, 2, 2)).trace(axis1=-1, axis2=-2)
    nv2 = normal(0.2, 3.4, size=(1, 2, 2, 2, 2)).trace(axis1=-1, axis2=-2)
    snv1 = assparsenormal(nv1)
    snv2 = assparsenormal(nv2)

    nvc = (nv1 + nv2).condition(nv1 - 0.3 * nv2)
    snvc = (snv1 + snv2).condition(snv1 - 0.3 * snv2)
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    x = np.random.rand(2, 2, 2)
    nvc = (nv1 + nv2).condition({nv1 - 0.3 * nv2 : x})
    snvc = (snv1 + snv2).condition({snv1 - 0.3 * snv2 : x})
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    # 1 independence axis. -----------------------------------------------------

    # Scalar dense subspace.
    sz = 4
    nv_list = []
    snv_list = []

    for _ in range(3):
        rs = 2 * np.random.rand(sz) - 1
        ro = 2 * np.random.rand(sz) - 1
        rv = normal(-0.3, 2.3)
        nv_list.append(gp.stack([o + s * rv.iid_copy() for o, s in zip(ro, rs)]))
        snv_list.append(ro + rs * iid_repeat(rv, sz))

    nvc = sum(nv_list).condition(nv_list[0] + 0.4 * nv_list[1])
    snvc = sum(snv_list).condition(snv_list[0] + 0.4 * snv_list[1])
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    # Vector dense subspace.
    sz = 4
    nv_list = []
    snv_list = []

    for _ in range(5):
        rs = 2 * np.random.rand(sz, 2) - 1
        ro = 2 * np.random.rand(sz, 2) - 1
        rv = random_normal((2,))
        nv_list.append(gp.stack([o + s * rv.iid_copy() for o, s in zip(ro, rs)]))
        snv_list.append(ro + rs * iid_repeat(rv, sz))

    nv = sum(nv_list)
    snv = sum(snv_list)

    nvc = nv.condition(nv_list[0] + 0.4 * nv_list[1])
    snvc = snv.condition(snv_list[0] + 0.4 * snv_list[1])
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    # Two conditions.
    nvc = nv.condition({nv_list[0] + 0.2 * nv_list[-1]: 1, 
                        nv_list[2] - 0.2 * nv_list[-1]: -0.3})
    snvc = snv.condition({snv_list[0] + 0.2 * snv_list[-1]: 1, 
                          snv_list[2] - 0.2 * snv_list[-1]: -0.3})
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    # Two conditions, out of which one is transposed.
    nvc = nv.condition({nv_list[0] + 0.2 * nv_list[-1]: 1, 
                        (nv_list[2] - 0.2 * nv_list[-1]).T: -0.3})
    snvc = snv.condition({snv_list[0] + 0.2 * snv_list[-1]: 1, 
                          (snv_list[2] - 0.2 * snv_list[-1]).T: -0.3})
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    # Two conditions, out of which one is transposed, 
    # and the other is of different shape.
    nvc = nv.condition({nv_list[0][:, 0] + 0.2 * nv_list[-1][:, 0]: 1, 
                        (nv_list[2] - 0.2 * nv_list[-1]).T: -0.3})
    snvc = snv.condition({snv_list[0][:, 0] + 0.2 * snv_list[-1][:, 0]: 1, 
                          (snv_list[2] - 0.2 * snv_list[-1]).T: -0.3})
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    # Two conditions, out of which one is transposed, 
    # plus the variable itself is transposed.
    nvc = nv.T.condition({nv_list[0] + 0.2 * nv_list[-1]: 1, 
                          (nv_list[2] - 0.2 * nv_list[-1]).T: -0.3})
    snvc = snv.T.condition({snv_list[0] + 0.2 * snv_list[-1]: 1, 
                            (snv_list[2] - 0.2 * snv_list[-1]).T: -0.3})
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    # Adds one complex condition.
    nvc = nv.T.condition({nv_list[0][:, 0] + 0.2 * nv_list[4][:, 0]: 1, 
                          (nv_list[2] - 0.2 * nv_list[4]).T: -0.3,
                          (nv_list[1] - 2j * nv_list[3] 
                           + (1 + 0.5j) * nv_list[4]): 1})
    snvc = snv.T.condition({snv_list[0][:, 0] + 0.2 * snv_list[4][:, 0]: 1, 
                            (snv_list[2] - 0.2 * snv_list[4]).T: -0.3,
                            (snv_list[1] - 2j * snv_list[3] 
                             + (1 + 0.5j) * snv_list[4]): 1})
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    # 3-dimensional dense subspace.
    sz = 6
    nv_list = []
    snv_list = []

    for _ in range(5):
        rs = 2 * np.random.rand(sz, 2, 3, 4) - 1
        ro = 2 * np.random.rand(sz, 2, 3, 4) - 1
        rv = random_normal((2, 3, 4))
        nv_list.append(gp.stack([o + s * rv.iid_copy() for o, s in zip(ro, rs)]))
        snv_list.append(ro + rs * iid_repeat(rv, sz))

    nv = sum(nv_list)
    snv = sum(snv_list)

    nvc = nv.condition(nv_list[0] + 0.4 * nv_list[1])
    snvc = snv.condition(snv_list[0] + 0.4 * snv_list[1])
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    # Several conditions of different shapes.
    nvc = nv.condition({nv_list[0] + 0.2 * nv_list[-1]: 1, 
                        nv_list[2][:, 1] - 0.2 * nv_list[4][:, 1]: -0.3,
                        nv_list[2][:, 1, 0] - 2.2 * nv_list[3][:, 1, 0]: 0})
    snvc = snv.condition({snv_list[0] + 0.2 * snv_list[-1]: 1, 
                          snv_list[2][:, 1] - 0.2 * snv_list[-1][:, 1]: -0.3,
                          snv_list[2][:, 1, 0] - 2.2 * snv_list[3][:, 1, 0]: 0})
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    # The varaible is complex.
    nv = sum(nv_list) + 5j * nv_list[1]
    snv = sum(snv_list) + 5j * snv_list[1]

    nvc = nv.condition(nv_list[0] + 0.4 * nv_list[1])
    snvc = snv.condition(snv_list[0] + 0.4 * snv_list[1])
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    nvc = nv.condition(nv_list[0] + 0.4j * nv_list[1] 
                       + (0.8 - 2.3j) * nv_list[2])
    snvc = snv.condition(snv_list[0] + 0.4j * snv_list[1]
                         + (0.8 - 2.3j) * snv_list[2])
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    check_permutations(snv_list, nv_list)

    # 2 independence axes. -----------------------------------------------------

    sz1 = 5
    sz2 = 6
    nv_list = []
    snv_list = []

    for _ in range(5):
        rs = 2 * np.random.rand(sz1, sz2) - 1
        ro = 2 * np.random.rand(sz1, sz2) - 1
        nv_list.append(ro + rs * normal(size=(sz1, sz2)))
        snv_list.append(ro + rs * iid_repeat(iid_repeat(normal(), sz2), sz1))

    nv = sum(nv_list)
    snv = sum(snv_list)

    nvc = nv.condition(nv_list[0] + 0.4 * nv_list[1])
    snvc = snv.condition(snv_list[0] + 0.4 * snv_list[1])
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    # One condition transposed.
    nvc = nv.condition({(nv_list[0] + 0.4 * nv_list[1]): - 0.1,
                        (nv_list[1] - 0.4 * nv_list[2]).T: - 0.1})
    snvc = snv.condition({(snv_list[0] + 0.4 * snv_list[1]): - 0.1,
                          (snv_list[1] - 0.4 * snv_list[2]).T: - 0.1})
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    # Higher-dimensional dense subspace.
    nv_list = []
    snv_list = []

    for _ in range(5):
        rs = 2 * np.random.rand(3, 4, 5, 2) - 1
        ro = 2 * np.random.rand(3, 4, 5, 2) - 1
        rv = random_normal((5, 2))
        nv = gp.stack([rv.iid_copy() for _ in range(4)])
        nv = gp.stack([nv.iid_copy() for _ in range(3)])    
        nv_list.append(ro + rs * nv)
        snv_list.append(ro + rs * iid_repeat(iid_repeat(rv, 4), 3))

    nv = sum(nv_list)
    snv = sum(snv_list)

    nvc = nv.condition(nv_list[0] + 0.4 * nv_list[1])
    snvc = snv.condition(snv_list[0] + 0.4 * snv_list[1])
    check_sparse_vs_normal(snvc, nvc)
    del snvc, nvc

    check_permutations(snv_list, nv_list)

    # Other cases. -------------------------------------------------------------

    snv1 = iid_repeat(iid_repeat(normal(), 4), 5)
    snv2 = iid_repeat(normal(), 4)

    # Conditioning a variable on itself.
    x = np.random.rand(5, 4)
    snvc = snv1 | {snv1 : x}
    assert np.max(np.abs(snvc.mean() - x)) < tol 
    assert np.max(np.abs(snvc.var())) < tol

    # Conditioning a variable on an empty dictionary.
    snvc = snv1 | dict()
    assert snvc is snv1

    # Different numbers of axes.
    with pytest.warns(SparseConditionWarning):
        snvc = snv1 | normal()
        assert snvc is snv1

    with pytest.warns(SparseConditionWarning):
        snvc = snv2 | snv1
        assert snvc is snv2

    with pytest.warns(SparseConditionWarning):
        snvc = snv1 | snv2
        assert snvc is snv1

    # Different sizes of independence axes.
    snv1 = iid_repeat(normal(), 5)
    snv2 = iid_repeat(normal(), 4)
    with pytest.warns(SparseConditionWarning):
        snvc = snv2 | snv1
        assert snvc is snv2

    with pytest.warns(SparseConditionWarning):
        snvc = snv1 | snv2
        assert snvc is snv1

    snv1 = iid_repeat(iid_repeat(normal(), 5), 4)
    snv2 = iid_repeat(iid_repeat(normal(), 5), 3)
    with pytest.warns(SparseConditionWarning):
        snvc = snv2 | snv1
        assert snvc is snv2

    with pytest.warns(SparseConditionWarning):
        snvc = snv1 | snv2
        assert snvc is snv1

    snv3 = iid_repeat(iid_repeat(normal(), 5), 4)
    snv13 = snv1 + snv3
    with pytest.warns(SparseConditionWarning):
        nv1 = normal(size=(4, 5))
        nv3 = normal(size=(4, 5))
        nvc = (nv1 + nv3) | {nv1 - 1.3 * nv3: 0.2}
        snvc = snv13 | {snv2: 1, snv1 - 1.3 * snv3: 0.2}
        check_sparse_vs_normal(snvc, nvc)


def test_cumsum():
    v = iid_repeat(iid_repeat(normal(size=(4, 5)), 2, axis=1), 3, axis=1)
    # v.shape is (4, 3, 2, 5), v.iaxes are (1, 2)

    for ax in [0, 3, -1, -4]:
        vout = v.cumsum(axis=ax)
        assert vout.shape == v.shape
        assert vout.iaxes == v.iaxes
        assert np.all(vout.mean() == np.cumsum(v.mean(), axis=ax))

        vvs = v.fcv.cumsum(axis=ax)
        assert np.all(vvs.emap.a == vout.fcv.emap.a)
        assert np.all(vvs.emap.elem == vout.fcv.emap.elem)

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


def test_logp():
    tol = 1e-10

    def check_sparse_vs_normal(snv, nv, x):
        x = np.asanyarray(x)
        logp2 = nv.logp(x)
        tol_ = tol * nv.size

        # Checks all permutations of axes.
        for ax in itertools.permutations(tuple(range(-nv.ndim, 0))):
            snv_ = snv.transpose(ax)
            x_ = x.transpose((0,) * (x.ndim - nv.ndim) + ax)

            logp1 = snv_.logp(x_)
            assert logp1.shape == logp2.shape
            assert np.max(np.abs(1 - logp1/logp2)) < tol_

    # Trivial cases first - no independence axes.

    nv = normal(0.2, 3.4)
    snv = assparsenormal(nv)
    check_sparse_vs_normal(snv, nv, 1.9)
    check_sparse_vs_normal(snv, nv, [0.5, 1.9, -4.1])

    nv = random_normal((3,))
    snv = assparsenormal(nv)
    x = 2 * np.random.rand(2, 3) - 1
    check_sparse_vs_normal(snv, nv, x[0])
    check_sparse_vs_normal(snv, nv, x)

    nv = random_normal((3, 2))
    snv = assparsenormal(nv)
    x = 2 * np.random.rand(4, 3, 2) - 1
    check_sparse_vs_normal(snv, nv, x[0])
    check_sparse_vs_normal(snv, nv, x)

    # One independence axis.

    # Scalar dense subspace.
    sz = 4
    rs = 2 * np.random.rand(sz) - 1
    ro = 2 * np.random.rand(sz) - 1
    nv = gp.stack([o + s * normal() for o, s in zip(ro, rs)])
    snv = ro + rs * iid_repeat(normal(), sz)
    x = 2 * np.random.rand(3, sz) - 1
    check_sparse_vs_normal(snv, nv, x[0])
    check_sparse_vs_normal(snv, nv, x)

    # 1D dense subspace.
    sparse_sz = 4
    dense_sz = 3
    rs = 2 * np.random.rand(sparse_sz, dense_sz) - 1
    ro = 2 * np.random.rand(sparse_sz, dense_sz) - 1
    rv = random_normal((dense_sz,))
    nv = gp.stack([o + s * rv.iid_copy() for o, s in zip(ro, rs)])
    snv = ro + rs * iid_repeat(rv, sparse_sz)
    x = 2 * np.random.rand(2, sparse_sz, dense_sz) - 1
    check_sparse_vs_normal(snv, nv, x[0])
    check_sparse_vs_normal(snv, nv, x)

    # 1D dense subspace, swap axes.
    sparse_sz = 4
    dense_sz = 3
    nv = gp.stack([o + s * rv.iid_copy() for o, s in zip(ro, rs)], axis=1)
    snv = ro.T + rs.T * iid_repeat(rv, sparse_sz, axis=1)
    x = 2 * np.random.rand(2, dense_sz, sparse_sz) - 1
    check_sparse_vs_normal(snv, nv, x[0])
    check_sparse_vs_normal(snv, nv, x)

    # 2D dense subspace.
    sparse_sz = 4
    dense_sz = (3, 5)
    rs = 2 * np.random.rand(sparse_sz, *dense_sz) - 1
    ro = 2 * np.random.rand(sparse_sz, *dense_sz) - 1
    rv = random_normal(dense_sz)
    nv = gp.stack([o + s * rv.iid_copy() for o, s in zip(ro, rs)])
    snv = ro + rs * iid_repeat(rv, sparse_sz)
    x = 2 * np.random.rand(2, sparse_sz, *dense_sz) - 1
    check_sparse_vs_normal(snv, nv, x[0])
    check_sparse_vs_normal(snv, nv, x)

    # Two independence axes.

    # Scalar dense subspace.
    rs = 2 * np.random.rand(3, 4) - 1
    ro = 2 * np.random.rand(3, 4) - 1
    nv = ro + rs * normal(size=(3, 4))
    snv = ro + rs * iid_repeat(iid_repeat(normal(), 4), 3)
    x = 2 * np.random.rand(2, 3, 4) - 1
    check_sparse_vs_normal(snv, nv, x[0])
    check_sparse_vs_normal(snv, nv, x)

    # 1D dense subspace.
    rs = 2 * np.random.rand(3, 4, 5) - 1
    ro = 2 * np.random.rand(3, 4, 5) - 1
    rv = random_normal((5,))
    nv = gp.stack([rv.iid_copy() for _ in range(4)])
    nv = gp.stack([nv.iid_copy() for _ in range(3)])    
    nv = ro + rs * nv
    snv = ro + rs * iid_repeat(iid_repeat(rv, 4), 3)
    x = 2 * np.random.rand(2, 3, 4, 5) - 1
    check_sparse_vs_normal(snv, nv, x[0])
    check_sparse_vs_normal(snv, nv, x)

    # 2D dense subspace.
    rs = 2 * np.random.rand(3, 4, 5, 2) - 1
    ro = 2 * np.random.rand(3, 4, 5, 2) - 1
    rv = random_normal((5, 2))
    nv = gp.stack([rv.iid_copy() for _ in range(4)])
    nv = gp.stack([nv.iid_copy() for _ in range(3)])    
    nv = ro + rs * nv
    snv = ro + rs * iid_repeat(iid_repeat(rv, 4), 3)
    x = 2 * np.random.rand(2, 3, 4, 5, 2) - 1
    check_sparse_vs_normal(snv, nv, x[0])
    check_sparse_vs_normal(snv, nv, x)

    # Three independence axes.

    # Scalar dense subspace.
    rs = 2 * np.random.rand(3, 4, 5) - 1
    ro = 2 * np.random.rand(3, 4, 5) - 1
    nv = ro + rs * normal(size=(3, 4, 5))
    snv = ro + rs * iid_repeat(iid_repeat(iid_repeat(normal(), 5), 4), 3)
    x = 2 * np.random.rand(2, 3, 4, 5) - 1
    check_sparse_vs_normal(snv, nv, x[0])
    check_sparse_vs_normal(snv, nv, x)

    # 1D dense subspace.
    rs = 2 * np.random.rand(3, 4, 5, 2) - 1
    ro = 2 * np.random.rand(3, 4, 5, 2) - 1
    rv = random_normal((2,))
    nv = gp.stack([rv.iid_copy() for _ in range(5)])
    nv = gp.stack([nv.iid_copy() for _ in range(4)])
    nv = gp.stack([nv.iid_copy() for _ in range(3)])    
    nv = ro + rs * nv
    snv = ro + rs * iid_repeat(iid_repeat(iid_repeat(rv, 5), 4), 3)
    x = 2 * np.random.rand(2, 3, 4, 5, 2) - 1
    check_sparse_vs_normal(snv, nv, x[0])
    check_sparse_vs_normal(snv, nv, x)

    with pytest.warns(ComplexWarning):
        iid_repeat(normal(), 1).logp([1+0.j])

    with pytest.raises(ValueError):
        iid_repeat(normal(), 1).logp([[[1]]])  # Too high dimension of x.


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

    v = iid_repeat(normal(size=(1,)), 1)  # shape (1, 1), iaxes (0,)
    assert v.reshape((1,)).shape == (1,)
    assert v.reshape((1,)).iaxes == (0,)

    v = iid_repeat(normal(size=(1,)), 1, axis=-1)  # shape (1, 1), iaxes (1,)
    assert v.reshape((1,)).shape == (1,)
    assert v.reshape((1,)).iaxes == (0,)

    # Tests the preservation of axes's ids.

    # A trivial reshaping.
    v = iid_repeat(iid_repeat(iid_repeat(normal(), 3), 3), 3, axis=1)
    assert v._iaxid == v.reshape((3, 3, 3))._iaxid

    # A case with transposition.
    v1 = iid_repeat(normal(size=(2, 3)), 5, axis=-1) 
    v1 = iid_repeat(v1, 5)
    v1 = iid_repeat(v1, 5)
    # shape (5, 5, 2, 3, 5), iaxes (0, 1, 4)

    v2 = iid_repeat(normal(size=(2, 3)), 5, axis=-1) 
    v2 = iid_repeat(v2, 5)
    v2 = iid_repeat(v2, 5, axis=1)

    with pytest.raises(ValueError):
        v1 + v2  # iaxids are incompatible.

    # But after the transposition the addition works.
    assert (v1 + v2.transpose((1, 0, 2, 3, 4))).iaxes == (0, 1, 4)

    # Now check the sampe operations after reshaping.
    v1_ = v1.reshape((5, 5, 6, 5))
    v2_ = v2.reshape((5, 5, 6, 5))

    with pytest.raises(ValueError):
        v1_ + v2_

    assert (v1_ + v2_.transpose((1, 0, 2, 3))).iaxes == (0, 1, 3)


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

        vvs = v.fcv.sum(axis=ax)
        assert np.all(vvs.emap.a == vout.fcv.emap.a)
        assert np.all(vvs.emap.elem == vout.fcv.emap.elem)

        vout = v.sum(axis=ax, keepdims=True)
        mean_ref = np.sum(v.mean(), axis=ax, keepdims=True)
        assert vout.shape == mean_ref.shape
        assert vout.iaxes == v.iaxes
        assert np.all(vout.mean() == mean_ref)

        vvs = v.fcv.sum(axis=ax, keepdims=True)
        assert np.all(vvs.emap.a == vout.fcv.emap.a)
        assert np.all(vvs.emap.elem == vout.fcv.emap.elem)

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


def test_iaxes_compatibility():

    # The numbers of independence axes are different.
    v1 = iid_repeat(iid_repeat(normal(), 3), 3)
    v2 = iid_repeat(normal(), 3)
    with pytest.raises(ValueError) as num_e:
        v1 - v2

    # The locations of the independence axes are different.
    v1 = iid_repeat(normal(size=(3,)), 3)
    v2 = iid_repeat(normal(size=(3,)), 3, axis=1)
    with pytest.raises(ValueError) as loc_e:
        v1 - v2
    
    # After the transposition, however, it should work.
    assert (v1 - v2.T).shape == (3, 3)
    assert (v1 - v2.T).iaxes == (0,)

    # Exchanging the order of the independence axes, even of the same size, 
    # should make the transposed and the original variables incompatible 
    # in arithmetic operations.

    v1 = iid_repeat(iid_repeat(normal(), 3), 3)
    v2 = iid_repeat(iid_repeat(normal(), 3), 3)

    assert (v1 - v2).shape == (3, 3)
    assert (v1 - v2).iaxes == (0, 1)
    
    v1.mean() + v1.mean().T  # Does not raise an error.
    with pytest.raises(ValueError):
        v1 + v1.T

    v2 = iid_repeat(iid_repeat(normal(), 3), 3)
    assert (v1 + v2).shape == (3, 3)
    assert (v1.mean() + v2.mean().T).shape == (3, 3)
    with pytest.raises(ValueError) as ord_e:
        v1 + v2.T

    # Also, if both are transposed, there should not be any error.
    assert (v1.T - v2.T).shape == (3, 3)
    assert (v1.T - v2.T).iaxes == (0, 1)

    v = iid_repeat(iid_repeat(random_normal((4, 5)), 3, axis=1), 3, axis=3)
    # v.shape is (4, 3, 5, 3), v.iaxes are (1, 3)

    v.mean() + v.mean().transpose((0, -1, 2, 1))  # Does not raise an error.
    with pytest.raises(ValueError):
        v + v.transpose((0, -1, 2, 1))

    v.mean() + v.moveaxis(-1, 1).moveaxis(2, 3).mean()
    with pytest.raises(ValueError):
        v + v.moveaxis(-1, 1).moveaxis(2, 3)

    # Finally, check that the three types of iaxes incompatibility,
    # number, location, and order, produce different error messages.
    assert num_e.value.args[0] != loc_e.value.args[0]
    assert loc_e.value.args[0] != ord_e.value.args[0]
    assert ord_e.value.args[0] != num_e.value.args[0]


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

    v2.fcv += v1.fcv[0, 0, 0, 0]  # for establishing correlation.

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

        if len(v1.fcv.emap.a) > 0:
            assert np.max(np.abs(v1.fcv.emap.a - v2.fcv.emap.a)) < tol
        else:
            assert len(v2.fcv.emap.a) == 0
        
        assert v1.fcv.emap.elem == v2.fcv.emap.elem
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

    # Tests agains inner.

    xi = random_normal((3, 2))
    v = iid_repeat(xi, 5, axis=1)  # shape (3, 5, 2), iaxes (1,)

    x = np.random.rand(4, 3, 8, 2)  # 4D

    v_ei = gp.einsum("ij, j -> i", v[1], x[0, 0, 0])
    v_in = gp.inner(v[1], x[0, 0, 0])
    assert_equal(v_ei, v_in)

    v_ei = gp.einsum("kij, lj -> kil", v, x[0, 0])
    v_in = gp.inner(v, x[0, 0])
    assert_equal(v_ei, v_in)

    v_ei = gp.einsum("ij, klmj -> iklm", v[1], x)
    v_in = gp.inner(v[1], x)
    assert_equal(v_ei, v_in)

    v_ei = gp.einsum("klmj, ij -> klmi", x, v[1])
    v_in = gp.inner(x, v[1])
    assert_equal(v_ei, v_in)

    v_ei = gp.einsum("...j, ij", x, v[1])
    v_in = gp.inner(x, v[1])
    assert_equal(v_ei, v_in)

    # Tests against outer.

    v = iid_repeat(normal(0.5, 0.1), 5)
    x = np.random.rand(4)

    v_ei = gp.einsum("i, j -> ij", x, v)
    v_ou = gp.outer(x, v)
    assert_equal(v_ei, v_ou)

    v_ei = gp.einsum("i, j", x, v)  # Implicit output indices.
    v_ou = gp.outer(x, v)
    assert_equal(v_ei, v_ou)

    v_ei = gp.einsum("i, j -> ij", v, x)
    v_ou = gp.outer(v, x)
    assert_equal(v_ei, v_ou)

    xi = random_normal((1,))
    v = iid_repeat(xi, 5, axis=1)  # shape (1, 5), iaxes (1,)
    x = np.random.rand(1, 3)  # 4D

    v_ei = gp.einsum("ij, kl -> jl", v, x)
    v_ou = gp.outer(v, x)
    assert_equal(v_ei, v_ou)

    v_ei = gp.einsum("kl, ij -> lj", x, v)
    v_ou = gp.outer(x, v)
    assert_equal(v_ei, v_ou)

    # A purely deterministic input.

    v = assparsenormal(2 * np.random.rand(2, 4, 3, 2) - 1)
    x = 2 * np.random.rand(4, 3, 2, 5) - 1

    v_ei = gp.einsum("ijkl, jklm", v, x)
    v_td = gp.tensordot(v, x, axes=3)
    assert_equal(v_ei, v_td)

    v_ei = gp.einsum("ijkl, jklm", x.T, v.T)
    v_td = gp.tensordot(x.T, v.T, axes=3)
    assert_equal(v_ei, v_td)


def _check_w_scalar_operand(sv, func_name="dot"):
        # Checks the operation of a bilinear dunction one scalar argument.
        # `sv` is a sparse normal variable.
        tol = 1e-8

        sv_ = getattr(gp, func_name)(sv, 2.)
        mean_ref = getattr(np, func_name)(sv.mean(), 2.)
        assert sv_.shape == mean_ref.shape
        assert np.max(np.abs(sv_.fcv.emap.a - 2 * sv.fcv.emap.a)) < tol
        assert np.max(np.abs(sv_.mean() - 2 * sv.mean())) < tol

        sv_ = getattr(gp, func_name)(2., sv)
        mean_ref = getattr(np, func_name)(2., sv.mean())
        assert sv_.shape == mean_ref.shape
        assert np.max(np.abs(sv_.fcv.emap.a - 2 * sv.fcv.emap.a)) < tol
        assert np.max(np.abs(sv_.mean() - 2 * sv.mean())) < tol


def test_dot():
    tol = 1e-8

    # Degenerate cases.

    v = assparsenormal(1)
    v_ = gp.dot(v, 2.)
    assert v_.shape == tuple()
    assert v_.iaxes == tuple()
    assert np.max(np.abs(v_.mean() - 2 * v.mean())) < tol
    assert np.max(v_.var()) < tol

    v_ = gp.dot(2., v)
    assert v_.shape == tuple()
    assert v_.iaxes == tuple()
    assert np.max(np.abs(v_.mean() - 2 * v.mean())) < tol
    assert np.max(v_.var()) < tol

    v = assparsenormal([1, 2, 3])
    v_ = gp.dot(v, [1, 2, 3])
    assert v_.shape == tuple()
    assert v_.iaxes == tuple()
    assert v_.var() < tol
    assert np.abs(v_.mean() - 14) < tol

    v_ = gp.dot([1, 2, 3], v)
    assert v_.shape == tuple()
    assert v_.iaxes == tuple()
    assert v_.var() < tol
    assert np.abs(v_.mean() - 14) < tol

    v = assparsenormal(2 * np.random.rand(2, 3, 4, 5) - 1)
    x = 2 * np.random.rand(6, 7, 5, 4) - 1

    v_ = gp.dot(v, x)
    assert v_.shape == (2, 3, 4, 6, 7, 4)
    assert v_.iaxes == tuple()
    assert np.max(v_.var()) < tol
    assert np.max(np.abs(v_.mean() - np.dot(v.mean(), x))) < tol

    v_ = gp.dot(x, v)
    assert v_.shape == (6, 7, 5, 2, 3, 5)
    assert v_.iaxes == tuple()
    assert np.max(v_.var()) < tol
    assert np.max(np.abs(v_.mean() - np.dot(x, v.mean()))) < tol

    # 1 independence axis.

    # - 1-d sparse variable.
    sz = 5
    v = iid_repeat(normal(), sz)
    rs = 2 * np.random.rand(sz) - 1
    ro = 2 * np.random.rand(sz) - 1
    v = ro + rs * v

    # -- 0-d operand 2.
    _check_w_scalar_operand(v)

    # -- 1-d operand 2.
    x = np.arange(5)
    assert np.dot(v.mean(), x).shape == tuple()
    with pytest.raises(ValueError):
        gp.dot(v, x)
    assert np.dot(x, v.mean()).shape == tuple()
    with pytest.raises(ValueError):
        gp.dot(x, v)

    # - Adds one dense dimension.
    v = iid_repeat(normal(size=3), sz)  # shape (5, 3), iaxes (0,)
    rs = 2 * np.random.rand(sz, 1) - 1
    ro = 2 * np.random.rand(sz, 1) - 1
    v = ro + rs * v

    # -- 0-d operand 2.
    _check_w_scalar_operand(v)

    # -- 1-d operand 2.
    x3 = np.arange(3)
    x5 = np.arange(5)
    
    v_ = gp.dot(v, x3)
    mean_ref = np.dot(v.mean(), x3)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (0,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.dot(v.var(), x3**2))) < tol

    v_ = gp.dot(x3, v.T)
    mean_ref = np.dot(x3, v.T.mean())
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (0,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.dot(x3**2, v.T.var()))) < tol

    assert np.dot(v.T.mean(), x5).shape == (3,)
    with pytest.raises(ValueError):
        gp.dot(v.T, x5)
    assert np.dot(x5, v.mean()).shape == (3,)
    with pytest.raises(ValueError):
        gp.dot(x5, v)

    # -- 2-d operand 2.
    x35 = np.arange(3 * 5).reshape((3, 5))

    v_ = gp.dot(v, x35)
    mean_ref = np.dot(v.mean(), x35)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (0,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.dot(v.var(), x35**2))) < tol

    v_ = gp.dot(x35.T, v.T)
    mean_ref = np.dot(x35.T, v.T.mean())
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (1,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.dot(x35.T**2, v.T.var()))) < tol

    assert np.dot(v.T.mean(), x35.T).shape == (3, 3)
    with pytest.raises(ValueError):
        gp.dot(v.T, x35.T)
    assert np.dot(x35, v.mean()).shape == (3, 3)
    with pytest.raises(ValueError):
        gp.dot(x35, v)

    # - Two dense dimensions.
    nv = normal(size=(2, 3))
    v = iid_repeat(nv, sz, axis=1)  # shape (2, 5, 3), iaxes (1,)
    rs = 2 * np.random.rand(sz, 1) - 1
    ro = 2 * np.random.rand(sz, 1) - 1
    v = ro + rs * v

    # -- 0-d operand 2.
    _check_w_scalar_operand(v)

    # -- 1-d operand 2.
    x3 = np.arange(3)
    x5 = np.arange(5)

    v_ = gp.dot(v, x3)
    mean_ref = np.dot(v.mean(), x3)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (1,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.dot(v.var(), x3**2))) < tol

    vt = v.transpose((0, 2, 1))  # shape (2, 3, 5), iaxes (2,)
    v_ = gp.dot(x3, vt)
    mean_ref = np.dot(x3, vt.mean())
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (1,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.dot(x3**2, vt.var()))) < tol

    assert np.dot(vt.mean(), x5).shape == (2, 3)
    with pytest.raises(ValueError):
        gp.dot(vt, x5)
    assert np.dot(x5, v.mean()).shape == (2, 3)
    with pytest.raises(ValueError):
        gp.dot(x5, v)

    # -- 2-d operand 2.
    # v: shape (2, 5, 3), iaxes (1,) 
    # vt: shape (2, 3, 5), iaxes (2,)
    x35 = np.arange(3 * 5).reshape((3, 5))

    v_ = gp.dot(v, x35)
    mean_ref = np.dot(v.mean(), x35)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (1,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.dot(v.var(), x35**2))) < tol

    v_ = gp.dot(x35.T, vt)
    mean_ref = np.dot(x35.T, vt.mean())
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (2,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.dot(x35.T**2, vt.var()))) < tol

    assert np.dot(vt.mean(), x35.T).shape == (2, 3, 3)
    with pytest.raises(ValueError):
        gp.dot(vt, x35.T)
    assert np.dot(x35, v.mean()).shape == (3, 2, 3)
    with pytest.raises(ValueError):
        gp.dot(x35, v)

    # -- 3-d operand 2.
    # v: shape (2, 5, 3), iaxes (1,) 
    # vt: shape (2, 3, 5), iaxes (2,)
    x435 = np.arange(3 * 4 * 5).reshape((4, 3, 5))
    x453 = x435.transpose((0, 2, 1))

    v_ = gp.dot(v, x435)
    mean_ref = np.dot(v.mean(), x435)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (1,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.dot(v.var(), x435**2))) < tol

    v_ = gp.dot(x453, vt)
    mean_ref = np.dot(x453, vt.mean())
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (3,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.dot(x453**2, vt.var()))) < tol

    assert np.dot(vt.mean(), x453).shape == (2, 3, 4, 3)
    with pytest.raises(ValueError):
        gp.dot(vt, x453)
    assert np.dot(x435, v.mean()).shape == (4, 3, 2, 3)
    with pytest.raises(ValueError):
        gp.dot(x435, v)

    # 2 independence axis.

    # - 2-d sparse variable.
    sz1 = 5
    sz2 = 6
    v = iid_repeat(iid_repeat(normal(), sz1), sz2)  # shape (6, 5)
    rs = 2 * np.random.rand(sz2, sz1) - 1
    ro = 2 * np.random.rand(sz2, sz1) - 1
    v = ro + rs * v

    # -- 0-d operand 2.
    _check_w_scalar_operand(v)

    # -- 1-d operand 2.
    x5 = np.arange(5)

    assert np.dot(v.mean(), x5).shape == (6,)
    with pytest.raises(ValueError):
        gp.dot(v, x5)
    assert np.dot(x5, v.T.mean()).shape == (6,)
    with pytest.raises(ValueError):
        gp.dot(x5, v.T)

    # -- 2-d operand 2.
    x35 = np.arange(3 * 5).reshape((3, 5))

    assert np.dot(v.mean(), x35.T).shape == (6, 3)
    with pytest.raises(ValueError):
        gp.dot(v, x35.T)
    assert np.dot(x35, v.T.mean()).shape == (3, 6)
    with pytest.raises(ValueError):
        gp.dot(x35, v.T)

    # - Adds one dense dimension.
    sz1 = 5
    sz2 = 6
    dsz = 7
    nv = normal(size=dsz)
    v = iid_repeat(iid_repeat(nv, sz1), sz2)  # shape (6, 5, 7)
    rs = 2 * np.random.rand(sz2, sz1, dsz) - 1
    ro = 2 * np.random.rand(sz2, sz1, dsz) - 1
    v = ro + rs * v

    # -- 0-d operand 2.
    _check_w_scalar_operand(v)

    # -- 1-d operand 2.
    x5 = np.arange(5)
    x7 = np.arange(7)
    vt = v.transpose((0, 2, 1))  # shape (6, 7, 5), iaxes (0, 2)
    
    v_ = gp.dot(v, x7)
    mean_ref = np.dot(v.mean(), x7)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (0, 1)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.dot(v.var(), x7**2))) < tol

    v_ = gp.dot(x7, vt)
    mean_ref = np.dot(x7, vt.mean())
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (0, 1)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.dot(x7**2, vt.var()))) < tol

    assert np.dot(vt.mean(), x5).shape == (6, 7)
    with pytest.raises(ValueError):
        gp.dot(vt, x5)
    assert np.dot(x5, v.mean()).shape == (6, 7)
    with pytest.raises(ValueError):
        gp.dot(x5, v)

    # -- 2-d operand 2: skip.

    # -- 3-d operand 2.
    # v: shape (6, 5, 7), iaxes (0, 1) 
    # vt: shape (6, 7, 5), iaxes (0, 2)
    x = np.arange(3 * 7 * 5).reshape((3, 7, 5))
    xt = x.transpose((0, 2, 1))

    v_ = gp.dot(v, x)
    mean_ref = np.dot(v.mean(), x)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (0, 1)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.dot(v.var(), x**2))) < tol

    v_ = gp.dot(xt, vt)
    mean_ref = np.dot(xt, vt.mean())
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (2, 3)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.dot(xt**2, vt.var()))) < tol

    assert np.dot(vt.mean(), xt).shape == (6, 7, 3, 7)
    with pytest.raises(ValueError):
        gp.dot(vt, xt)
    assert np.dot(x, v.mean()).shape == (3, 7, 6, 7)
    with pytest.raises(ValueError):
        gp.dot(x, v)

    # A more complex example with correlations between elements.
    nv = normal(size=(3, 2))
    v1 = iid_repeat(iid_repeat(nv, 4, axis=-1), 5)  # shape (5, 3, 2, 4)
    rs = 2 * np.random.rand(5, 3, 2, 4) - 1
    ro = 2 * np.random.rand(5, 3, 2, 4) - 1
    v1 = ro + rs * v1
    v1 = v1.sum(axis=2)

    nv1_ref = normal(size=(5, 3, 2, 4))
    nv1_ref = ro + rs * nv1_ref
    nv1_ref = nv1_ref.sum(axis=2)

    nv = normal(size=(3, 2))
    v2 = iid_repeat(iid_repeat(nv, 4, axis=-1), 5)  # shape (5, 3, 2, 4)
    rs = 2 * np.random.rand(5, 3, 2, 4) - 1
    ro = 2 * np.random.rand(5, 3, 2, 4) - 1
    v2 = ro + rs * v2
    v2 = v2.sum(axis=2)

    nv2_ref = normal(size=(5, 3, 2, 4))
    nv2_ref = ro + rs * nv2_ref
    nv2_ref = nv2_ref.sum(axis=2)

    v = v1 - v2
    nv_ref = nv1_ref - nv2_ref

    x = 2 * np.random.rand(7, 4, 3) - 1
    v_ = gp.dot(x, v)
    nv_ref_ = gp.dot(x, nv_ref)

    assert v_.shape == nv_ref_.shape
    assert v_.iaxes == (2, 3)
    assert np.max(np.abs(v_.mean() - nv_ref_.mean())) < tol
    assert np.max(np.abs(v_.cov() - 
                         dense_to_sparse_cov(nv_ref_.cov(), (2, 3)))) < tol
    
    vt = v.transpose((0, 2, 1))
    nv_ref_t = nv_ref.transpose((0, 2, 1))
    xt = x.transpose((0, 2, 1))

    v_ = gp.dot(vt, xt)
    nv_ref_ = gp.dot(nv_ref_t, xt)

    assert v_.shape == nv_ref_.shape
    assert v_.iaxes == (0, 1)
    assert np.max(np.abs(v_.mean() - nv_ref_.mean())) < tol
    assert np.max(np.abs(v_.cov() - 
                         dense_to_sparse_cov(nv_ref_.cov(), (0, 1)))) < tol


def test_inner():
    tol = 1e-8

    # Degenerate cases.

    v = assparsenormal(1)
    v_ = gp.inner(v, 2.)
    assert v_.shape == tuple()
    assert v_.iaxes == tuple()
    assert np.max(np.abs(v_.mean() - 2 * v.mean())) < tol
    assert np.max(v_.var()) < tol

    v_ = gp.inner(2., v)
    assert v_.shape == tuple()
    assert v_.iaxes == tuple()
    assert np.max(np.abs(v_.mean() - 2 * v.mean())) < tol
    assert np.max(v_.var()) < tol

    v = assparsenormal([1, 2, 3])
    v_ = gp.inner(v, [1, 2, 3])
    assert v_.shape == tuple()
    assert v_.iaxes == tuple()
    assert v_.var() < tol
    assert np.abs(v_.mean() - 14) < tol

    v_ = gp.inner([1, 2, 3], v)
    assert v_.shape == tuple()
    assert v_.iaxes == tuple()
    assert v_.var() < tol
    assert np.abs(v_.mean() - 14) < tol

    v = assparsenormal(2 * np.random.rand(2, 3, 4, 5) - 1)
    x = 2 * np.random.rand(6, 7, 4, 5) - 1

    v_ = gp.inner(v, x)
    assert v_.shape == (2, 3, 4, 6, 7, 4)
    assert v_.iaxes == tuple()
    assert np.max(v_.var()) < tol
    assert np.max(np.abs(v_.mean() - np.inner(v.mean(), x))) < tol

    v_ = gp.inner(x, v)
    assert v_.shape == (6, 7, 4, 2, 3, 4)
    assert v_.iaxes == tuple()
    assert np.max(v_.var()) < tol
    assert np.max(np.abs(v_.mean() - np.inner(x, v.mean()))) < tol

    # 1 independence axis.

    # - 1-d sparse variable.
    sz = 5
    v = iid_repeat(normal(), sz)
    rs = 2 * np.random.rand(sz) - 1
    ro = 2 * np.random.rand(sz) - 1
    v = ro + rs * v

    # -- 0-d operand 2.
    _check_w_scalar_operand(v, func_name="inner")

    # -- 1-d operand 2.
    x = np.arange(5)

    assert np.inner(x, v.mean()).shape == tuple()
    with pytest.raises(ValueError):
        gp.inner(x, v)
    with pytest.raises(ValueError):
        gp.inner(v, x)

    # - 2-d variable with one sparse axis.
    v = iid_repeat(normal(size=3), 5)  # shape (5, 3), iaxes (0,)
    rs = 2 * np.random.rand(5, 3) - 1
    ro = 2 * np.random.rand(5, 3) - 1
    v = ro + rs * v

    # -- 0-d operand 2.
    _check_w_scalar_operand(v, func_name="inner")

    # -- 1-d operand 2.
    x3 = np.arange(3)
    x5 = np.arange(5)
    
    v_ = gp.inner(v, x3)
    mean_ref = np.inner(v.mean(), x3)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (0,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.inner(v.var(), x3**2))) < tol

    v_ = gp.inner(x3, v)
    mean_ref = np.inner(x3, v.mean())
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (0,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.inner(x3**2, v.var()))) < tol

    assert np.inner(v.T.mean(), x5).shape == (3,)
    with pytest.raises(ValueError):
        gp.inner(v.T, x5)
    assert np.inner(x5, v.T.mean()).shape == (3,)
    with pytest.raises(ValueError):
        gp.inner(x5, v.T)

    # -- 2-d operand 2.
    x53 = np.arange(3 * 5).reshape((5, 3))

    v_ = gp.inner(v, x53)
    mean_ref = np.inner(v.mean(), x53)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (0,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.inner(v.var(), x53**2))) < tol

    v_ = gp.inner(x53, v)
    mean_ref = np.inner(x53, v.mean())
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (1,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.inner(x53**2, v.var()))) < tol

    assert np.inner(v.T.mean(), x53.T).shape == (3, 3)
    with pytest.raises(ValueError):
        gp.inner(v.T, x53.T)
    assert np.inner(x53.T, v.T.mean()).shape == (3, 3)
    with pytest.raises(ValueError):
        gp.inner(x53.T, v.T)

    # - Two dense and one sparse dimensions.
    nv = normal(size=(2, 3))
    v = iid_repeat(nv, 5, axis=1)  # shape (2, 5, 3), iaxes (1,)
    rs = 2 * np.random.rand(2, 5, 3) - 1
    ro = 2 * np.random.rand(2, 5, 3) - 1
    v = ro + rs * v

    # -- 0-d operand 2.
    _check_w_scalar_operand(v, func_name="inner")

    # -- 1-d operand 2.
    x3 = np.arange(3)
    x5 = np.arange(5)

    v_ = gp.inner(v, x3)
    mean_ref = np.inner(v.mean(), x3)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (1,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.inner(v.var(), x3**2))) < tol

    v_ = gp.inner(x3, v)
    mean_ref = np.inner(x3, v.mean())
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (1,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.inner(x3**2, v.var()))) < tol

    vt = v.transpose((0, 2, 1))
    assert np.inner(vt.mean(), x5).shape == (2, 3)
    with pytest.raises(ValueError):
        gp.inner(vt, x5)
    assert np.inner(x5, vt.mean()).shape == (2, 3)
    with pytest.raises(ValueError):
        gp.inner(x5, vt)

    # -- 2-d operand 2.
    # v: shape (2, 5, 3), iaxes (1,) 
    # vt: shape (2, 3, 5), iaxes (2,)
    x53 = np.arange(3 * 5).reshape((5, 3))

    v_ = gp.inner(v, x53)
    mean_ref = np.inner(v.mean(), x53)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (1,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.inner(v.var(), x53**2))) < tol

    v_ = gp.inner(x53, v)
    mean_ref = np.inner(x53, v.mean())
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (2,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.inner(x53**2, v.var()))) < tol

    assert np.inner(vt.mean(), x53.T).shape == (2, 3, 3)
    with pytest.raises(ValueError):
        gp.inner(vt, x53.T)
    assert np.inner(x53.T, vt.mean()).shape == (3, 2, 3)
    with pytest.raises(ValueError):
        gp.inner(x53.T, vt)

    # -- 3-d operand 2.
    # v: shape (2, 5, 3), iaxes (1,) 
    # vt: shape (2, 3, 5), iaxes (2,)
    x = np.arange(3 * 4 * 5).reshape((5, 4, 3))

    v_ = gp.inner(v, x)
    mean_ref = np.inner(v.mean(), x)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (1,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.inner(v.var(), x**2))) < tol

    v_ = gp.inner(x, v)
    mean_ref = np.inner(x, v.mean())
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (3,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.inner(x**2, v.var()))) < tol

    assert np.inner(vt.mean(), x.T).shape == (2, 3, 3, 4)
    with pytest.raises(ValueError):
        gp.inner(vt, x.T)
    assert np.inner(x.T, vt.mean()).shape == (3, 4, 2, 3)
    with pytest.raises(ValueError):
        gp.inner(x.T, vt)

    # 2 independence axis.

    # - 2-d sparse variable.
    v = iid_repeat(iid_repeat(normal(), 5), 6)  # shape (6, 5)
    rs = 2 * np.random.rand(6, 5) - 1
    ro = 2 * np.random.rand(6, 5) - 1
    v = ro + rs * v

    # -- 0-d operand 2.
    _check_w_scalar_operand(v, func_name="inner")

    # -- 1-d operand 2.
    x5 = np.arange(5)

    assert np.inner(v.mean(), x5).shape == (6,)
    with pytest.raises(ValueError):
        gp.inner(v, x5)
    assert np.inner(x5, v.mean()).shape == (6,)
    with pytest.raises(ValueError):
        gp.inner(x5, v)

    # -- 2-d operand 2.
    x35 = np.arange(3 * 5).reshape((3, 5))

    assert np.inner(v.mean(), x35).shape == (6, 3)
    with pytest.raises(ValueError):
        gp.inner(v, x35)
    assert np.inner(x35, v.mean()).shape == (3, 6)
    with pytest.raises(ValueError):
        gp.inner(x35, v)

    # - Adds one dense dimension.
    nv = normal(size=7)
    v = iid_repeat(iid_repeat(nv, 5), 6)  # shape (6, 5, 7)
    rs = 2 * np.random.rand(6, 5, 7) - 1
    ro = 2 * np.random.rand(6, 5, 7) - 1
    v = ro + rs * v

    # -- 0-d operand 2.
    _check_w_scalar_operand(v, func_name="inner")

    # -- 1-d operand 2.
    x5 = np.arange(5)
    x7 = np.arange(7)
    
    v_ = gp.inner(v, x7)
    mean_ref = np.inner(v.mean(), x7)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (0, 1)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.inner(v.var(), x7**2))) < tol

    v_ = gp.inner(x7, v)
    mean_ref = np.inner(x7, v.mean())
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (0, 1)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.inner(x7**2, v.var()))) < tol

    vt = v.transpose((0, 2, 1))  # shape (6, 7, 5), iaxes (0, 2)
    assert np.inner(vt.mean(), x5).shape == (6, 7)
    with pytest.raises(ValueError):
        gp.inner(vt, x5)
    assert np.inner(x5, vt.mean()).shape == (6, 7)
    with pytest.raises(ValueError):
        gp.inner(x5, vt)

    # -- 2-d operand 2: skip.

    # -- 3-d operand 2.
    # v: shape (6, 5, 7), iaxes (0, 1) 
    # vt: shape (6, 7, 5), iaxes (0, 2)
    x = np.arange(3 * 7 * 5).reshape((3, 5, 7))
    xt = x.transpose((0, 2, 1))  # shape (3, 7, 5)

    v_ = gp.inner(v, x)
    mean_ref = np.inner(v.mean(), x)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (0, 1)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.inner(v.var(), x**2))) < tol

    v_ = gp.inner(x, v)
    mean_ref = np.inner(x, v.mean())
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (2, 3)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - np.inner(x**2, v.var()))) < tol

    assert np.inner(vt.mean(), xt).shape == (6, 7, 3, 7)
    with pytest.raises(ValueError):
        gp.inner(vt, xt)
    assert np.inner(xt, vt.mean()).shape == (3, 7, 6, 7)
    with pytest.raises(ValueError):
        gp.inner(xt, vt)


def test_outer():
    tol = 1e-8

    # Degenerate cases.

    v = assparsenormal(1)
    v_ = gp.outer(v, 2.)
    assert v_.shape == (1, 1)
    assert v_.iaxes == tuple()
    assert np.max(np.abs(v_.mean() - 2 * v.mean())) < tol
    assert np.max(v_.var()) < tol

    v_ = gp.outer(2., v)
    assert v_.shape == (1, 1)
    assert v_.iaxes == tuple()
    assert np.max(np.abs(v_.mean() - 2 * v.mean())) < tol
    assert np.max(v_.var()) < tol

    v = assparsenormal([1, 2, 3])
    v_ = gp.outer(v, [2, 3, 4])
    assert v_.shape == (3, 3)
    assert v_.iaxes == tuple()
    assert np.max(v_.var()) < tol
    assert np.max(np.abs(v_.mean() - np.outer([1, 2, 3], [2, 3, 4]))) < tol

    v_ = gp.outer([2, 3, 4], v)
    assert v_.shape == (3, 3)
    assert v_.iaxes == tuple()
    assert np.max(v_.var()) < tol
    assert np.max(np.abs(v_.mean() - np.outer([2, 3, 4], [1, 2, 3]))) < tol

    # Flattening higher-dimensional inputs.
    v = assparsenormal([-1, 1, 2, 3]).reshape((2, 2))
    v_ = gp.outer(v, [2, 3, 4])
    assert v_.shape == (4, 3)
    assert v_.iaxes == tuple()
    assert np.max(v_.var()) < tol
    assert np.max(np.abs(v_.mean() - np.outer([-1, 1, 2, 3], [2, 3, 4]))) < tol

    v_ = gp.outer([2, 3, 4], v)
    assert v_.shape == (3, 4)
    assert v_.iaxes == tuple()
    assert v_._iaxid == (None, None)
    assert np.max(v_.var()) < tol
    assert np.max(np.abs(v_.mean() - np.outer([2, 3, 4], [-1, 1, 2, 3]))) < tol

    v = assparsenormal(normal() + normal())
    _check_w_scalar_operand(v, func_name="outer")

    # 0 independence axes.

    nv = random_normal((4, 2))
    v = assparsenormal(nv)

    x = 2 * np.random.rand(3) - 1

    nv_ = gp.outer(nv, x)
    v_ = gp.outer(v, x)
    assert v_.shape == nv_.shape
    assert v_.iaxes == tuple()
    assert np.max(np.abs(v_.mean() - nv_.mean())) < tol
    assert np.max(np.abs(v_.cov() - nv_.cov())) < tol

    nv_ = gp.outer(x, nv)
    v_ = gp.outer(x, v)
    assert isinstance(nv_, Normal)
    assert isinstance(v_, SparseNormal)
    assert v_.shape == nv_.shape
    assert v_.iaxes == tuple()
    assert np.max(np.abs(v_.mean() - nv_.mean())) < tol
    assert np.max(np.abs(v_.cov() - nv_.cov())) < tol

    # 1 independence axis.
    v = iid_repeat(normal(), 5)
    rs = 2 * np.random.rand(5) - 1
    ro = 2 * np.random.rand(5) - 1
    v = ro + rs * v

    nv = ro + rs * normal(size=5)

    nv_ = gp.outer(nv, x)
    v_ = gp.outer(v, x)

    assert isinstance(nv_, Normal)
    assert isinstance(v_, SparseNormal)
    assert v_.shape == nv_.shape
    assert v_.iaxes == (0,)
    assert v_._iaxid == (1, None)
    assert np.max(np.abs(v_.mean() - nv_.mean())) < tol
    assert np.max(np.abs(v_.cov() - 
                         dense_to_sparse_cov(nv_.cov(), (0,)))) < tol
    
    nv_ = gp.outer(x, nv)
    v_ = gp.outer(x, v)

    assert isinstance(nv_, Normal)
    assert isinstance(v_, SparseNormal)
    assert v_.shape == nv_.shape
    assert v_.iaxes == (1,)
    assert v_._iaxid == (None, 1)
    assert np.max(np.abs(v_.mean() - nv_.mean())) < tol
    assert np.max(np.abs(v_.cov() - 
                         dense_to_sparse_cov(nv_.cov(), (1,)))) < tol

    v = iid_repeat(normal(size=4), 5)

    with pytest.raises(ValueError):
        gp.outer(v, [1, 2, 3])
        
    with pytest.raises(ValueError):
        gp.outer([1, 2, 3], v)

    # 2 independence axes.
    v = iid_repeat(iid_repeat(normal(), 4), 5)

    with pytest.raises(ValueError):
        gp.outer(v, [1, 2, 3])

    with pytest.raises(ValueError):
        gp.outer([1, 2, 3], v)


def test_kron():
    tol = 1e-8

    # Trivial cases.

    v = assparsenormal(1)
    v_ = gp.kron(v, 2.)
    assert v_.shape == tuple()
    assert v_.iaxes == tuple()
    assert np.max(np.abs(v_.mean() - 2 * v.mean())) < tol
    assert np.max(v_.var()) < tol

    v_ = gp.kron(2., v)
    assert v_.shape == tuple()
    assert v_.iaxes == tuple()
    assert np.max(np.abs(v_.mean() - 2 * v.mean())) < tol
    assert np.max(v_.var()) < tol

    v = assparsenormal([1, 2, 3])
    v_ = gp.kron(v, [2, 3, 4])
    assert v_.shape == (3 * 3,)
    assert v_.iaxes == tuple()
    assert np.max(v_.var()) < tol
    assert np.max(np.abs(v_.mean() - np.kron([1, 2, 3], [2, 3, 4]))) < tol

    v_ = gp.kron([2, 3, 4], v)
    assert v_.shape == (3 * 3,)
    assert v_.iaxes == tuple()
    assert np.max(v_.var()) < tol
    assert np.max(np.abs(v_.mean() - np.kron([2, 3, 4], [1, 2, 3]))) < tol

    # 0 independence axes.
    nv = random_normal((4, 2))
    v = assparsenormal(nv)

    x = 2 * np.random.rand(3, 5) - 1

    nv_ = gp.kron(nv, x)
    v_ = gp.kron(v, x)
    assert v_.shape == nv_.shape
    assert v_.iaxes == tuple()
    assert np.max(np.abs(v_.mean() - nv_.mean())) < tol
    assert np.max(np.abs(v_.cov() - nv_.cov())) < tol

    nv_ = gp.kron(x, nv)
    v_ = gp.kron(x, v)
    assert isinstance(nv_, Normal)
    assert isinstance(v_, SparseNormal)
    assert v_.shape == nv_.shape
    assert v_.iaxes == tuple()
    assert np.max(np.abs(v_.mean() - nv_.mean())) < tol
    assert np.max(np.abs(v_.cov() - nv_.cov())) < tol

    # 1 independence axis.

    v = iid_repeat(normal(size=2), 5, axis=-1)
    rs = 2 * np.random.rand(2, 5) - 1
    ro = 2 * np.random.rand(2, 5) - 1
    v = ro + rs * v

    x = 2 * np.random.rand(3, 5) - 1

    nv = ro + rs * normal(size=(2, 5))

    nv_ = gp.kron(nv, x)
    v_ = gp.kron(v, x)

    assert isinstance(nv_, Normal)
    assert isinstance(v_, SparseNormal)
    assert v_.shape == nv_.shape
    assert v_.iaxes == (1,)
    assert v_._iaxid == (None, 1)
    assert np.max(np.abs(v_.mean() - nv_.mean())) < tol
    assert np.max(np.abs(v_.cov() - 
                         dense_to_sparse_cov(nv_.cov(), (1,)))) < tol
    
    nv_ = gp.kron(x, nv)
    v_ = gp.kron(x, v)

    assert isinstance(nv_, Normal)
    assert isinstance(v_, SparseNormal)
    assert v_.shape == nv_.shape
    assert v_.iaxes == (1,)
    assert v_._iaxid == (None, 1)
    assert np.max(np.abs(v_.mean() - nv_.mean())) < tol
    assert np.max(np.abs(v_.cov() - 
                         dense_to_sparse_cov(nv_.cov(), (1,)))) < tol

    # A high-dimensional case, with broadcasting.
    
    v = iid_repeat(iid_repeat(normal(size=(2, 3)), 5, axis=1), 6, axis=-1)
    # shape (2, 5, 3, 6), iaxes (1, 3)

    rs = 2 * np.random.rand(2, 5, 3, 6) - 1
    ro = 2 * np.random.rand(2, 5, 3, 6) - 1
    v = ro + rs * v

    nv = ro + rs * normal(size=(2, 5, 3, 6))

    x = 2 * np.random.rand(4, 1, 3) - 1  # x.ndim < v.ndim

    nv_ = gp.kron(nv, x)
    v_ = gp.kron(v, x)

    assert isinstance(nv_, Normal)
    assert isinstance(v_, SparseNormal)
    assert v_.shape == nv_.shape
    assert v_.iaxes == (1, 3)
    assert np.max(np.abs(v_.mean() - nv_.mean())) < tol
    assert np.max(np.abs(v_.cov() - 
                         dense_to_sparse_cov(nv_.cov(), (1, 3)))) < tol
    
    nv_ = gp.kron(x, nv)
    v_ = gp.kron(x, v)

    assert isinstance(nv_, Normal)
    assert isinstance(v_, SparseNormal)
    assert v_.shape == nv_.shape
    assert v_.iaxes == (1, 3)
    assert np.max(np.abs(v_.mean() - nv_.mean())) < tol
    assert np.max(np.abs(v_.cov() - 
                         dense_to_sparse_cov(nv_.cov(), (1, 3)))) < tol
    
    x = 2 * np.random.rand(2, 1, 3, 1, 2) - 1  # x.ndim > v.ndim

    nv_ = gp.kron(nv, x)
    v_ = gp.kron(v, x)

    assert isinstance(nv_, Normal)
    assert isinstance(v_, SparseNormal)
    assert v_.shape == nv_.shape
    assert v_.iaxes == (2, 4)
    assert np.max(np.abs(v_.mean() - nv_.mean())) < tol
    assert np.max(np.abs(v_.cov() - 
                         dense_to_sparse_cov(nv_.cov(), (2, 4)))) < tol
    
    nv_ = gp.kron(x, nv)
    v_ = gp.kron(x, v)

    assert isinstance(nv_, Normal)
    assert isinstance(v_, SparseNormal)
    assert v_.shape == nv_.shape
    assert v_.iaxes == (2, 4)
    assert np.max(np.abs(v_.mean() - nv_.mean())) < tol
    assert np.max(np.abs(v_.cov() - 
                         dense_to_sparse_cov(nv_.cov(), (2, 4)))) < tol


def test_tensordot():
    tol = 1e-8

    # Degenerate cases.

    v = assparsenormal(1)
    v_ = gp.tensordot(v, 2., axes=0)
    assert v_.shape == tuple()
    assert v_.iaxes == tuple()
    assert np.max(np.abs(v_.mean() - 2 * v.mean())) < tol
    assert np.max(v_.var()) < tol

    v_ = gp.tensordot(2., v, axes=0)
    assert v_.shape == tuple()
    assert v_.iaxes == tuple()
    assert np.max(np.abs(v_.mean() - 2 * v.mean())) < tol
    assert np.max(v_.var()) < tol

    v = assparsenormal([1, 2, 3])
    v_ = gp.tensordot(v, [1, 2, 3], axes=1)
    assert v_.shape == tuple()
    assert v_.iaxes == tuple()
    assert v_.var() < tol
    assert np.abs(v_.mean() - 14) < tol

    v_ = gp.tensordot([1, 2, 3], v, axes=1)
    assert v_.shape == tuple()
    assert v_.iaxes == tuple()
    assert v_.var() < tol
    assert np.abs(v_.mean() - 14) < tol

    v = assparsenormal(2 * np.random.rand(2, 3, 4, 5) - 1)
    x = 2 * np.random.rand(3, 4, 5, 7) - 1

    v_ = gp.tensordot(v, x, axes=3)
    assert v_.shape == (2, 7)
    assert v_.iaxes == tuple()
    assert np.max(v_.var()) < tol
    assert np.max(np.abs(v_.mean() - np.tensordot(v.mean(), x, axes=3))) < tol

    vt = v.transpose((1, 2, 3, 0))
    xt = x.transpose((3, 0, 1, 2))
    v_ = gp.tensordot(xt, vt, axes=3)
    assert v_.shape == (7, 2)
    assert v_.iaxes == tuple()
    assert np.max(v_.var()) < tol
    assert np.max(np.abs(v_.mean() - 
                         np.tensordot(xt, vt.mean(), axes=3))) < tol
    
    # 1 independence axis.

    v = iid_repeat(normal(size=(2, 3)), 5, 1)  # shape (2, 5, 3), iaxes (1,)

    rs = 2 * np.random.rand(2, 5, 3) - 1
    ro = 2 * np.random.rand(2, 5, 3) - 1
    v = ro + rs * v

    x = 2 * np.random.rand(3, 2, 6) - 1

    # - Contraction over 0 axes.

    v_ = gp.tensordot(v, x, axes=0)
    mean_ref = np.tensordot(v.mean(), x, axes=0)
    var_ref = np.tensordot(v.var(), x**2, axes=0)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (1,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - var_ref)) < tol

    v_ = gp.tensordot(x, v, axes=0)
    mean_ref = np.tensordot(x, v.mean(), axes=0)
    var_ref = np.tensordot(x**2, v.var(), axes=0)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (4,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - var_ref)) < tol

    # - Contraction over 1 axis.

    v_ = gp.tensordot(v, x, axes=1)
    mean_ref = np.tensordot(v.mean(), x, axes=1)
    var_ref = np.tensordot(v.var(), x**2, axes=1)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (1,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - var_ref)) < tol

    v_ = gp.tensordot(x.T, v.T, axes=1)
    mean_ref = np.tensordot(x.T, v.T.mean(), axes=1)
    var_ref = np.tensordot(x.T**2, v.T.var(), axes=1)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (2,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - var_ref)) < tol

    x_ = [1, 2, 3, 4, 5]
    assert np.tensordot(v.mean(), x_, axes=((1,), (0,))).shape == (2, 3)
    with pytest.raises(ValueError):
        gp.tensordot(v, x_, axes=((1,), (0,)))

    assert np.tensordot(x_, v.mean(), axes=((0,), (1,))).shape == (2, 3)
    with pytest.raises(ValueError):
        gp.tensordot(x_, v, axes=((0,), (1,)))

    # - Contraction over 2 axes.

    vt = v.transpose((1, 2, 0))  # shape (5, 3, 2), iaxes (0,)

    v_ = gp.tensordot(vt, x, axes=2)
    mean_ref = np.tensordot(vt.mean(), x, axes=2)
    var_ref = np.tensordot(vt.var(), x**2, axes=2)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (0,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - var_ref)) < tol

    vt = v.transpose((2, 0, 1))  # shape (3, 2, 5), iaxes (2,)
    xt = x.transpose((2, 0, 1))  # shape (6, 3, 2)

    v_ = gp.tensordot(xt, vt, axes=2)
    mean_ref = np.tensordot(xt, vt.mean(), axes=2)
    var_ref = np.tensordot(xt**2, vt.var(), axes=2)
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (1,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - var_ref)) < tol

    # Contraction over 2 axes with explicit designation of axes.

    v_ = gp.tensordot(v, x, axes=((0, 2), (1, 0)))
    mean_ref = np.tensordot(v.mean(), x, axes=((0, 2), (1, 0)))
    var_ref = np.tensordot(v.var(), x**2, axes=((0, 2), (1, 0)))
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (0,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - var_ref)) < tol

    v_ = gp.tensordot(x, v, axes=((1, 0), (0, 2)))
    mean_ref = np.tensordot(x, v.mean(), axes=((1, 0), (0, 2)))
    var_ref = np.tensordot(x**2, v.var(), axes=((1, 0), (0, 2)))
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (1,)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - var_ref)) < tol

    # 2 independence axis.

    v = iid_repeat(iid_repeat(normal(size=(2, 3)), 5, axis=1), 6, axis=-1)
    # shape (2, 5, 3, 6), iaxes (1, 3)

    rs = 2 * np.random.rand(2, 5, 3, 6) - 1
    ro = 2 * np.random.rand(2, 5, 3, 6) - 1
    v = ro + rs * v

    x = 2 * np.random.rand(3, 2, 4, 6) - 1

    v_ = gp.tensordot(v, x, axes=((2,), (0,)))
    mean_ref = np.tensordot(v.mean(), x, axes=((2,), (0,)))
    var_ref = np.tensordot(v.var(), x**2, axes=((2,), (0,)))
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (1, 2)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - var_ref)) < tol

    v_ = gp.tensordot(x, v, axes=((0,), (2,)))
    mean_ref = np.tensordot(x, v.mean(), axes=((0,), (2,)))
    var_ref = np.tensordot(x**2, v.var(), axes=((0,), (2,)))
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (4, 5)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - var_ref)) < tol

    v_ = gp.tensordot(v, x, axes=((2, 0), (0, 1)))
    mean_ref = np.tensordot(v.mean(), x, axes=((2, 0), (0, 1)))
    var_ref = np.tensordot(v.var(), x**2, axes=((2, 0), (0, 1)))
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (0, 1)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - var_ref)) < tol

    # Check addition after tensordot to see that iaxid are correct.

    v = iid_repeat(iid_repeat(normal(size=(2, 3)), 5, axis=1), 5, axis=-1)
    # shape (2, 5, 3, 5), iaxes (1, 3)

    rs = 2 * np.random.rand(2, 5, 3, 5) - 1
    ro = 2 * np.random.rand(2, 5, 3, 5) - 1
    v = ro + rs * v

    w = iid_repeat(iid_repeat(normal(), 5), 5, axis=-1)

    x = 2 * np.random.rand(3, 2) - 1

    v_ = gp.tensordot(x, v, axes=((1, 0), (0, 2)))
    mean_ref = np.tensordot(x, v.mean(), axes=((1, 0), (0, 2)))
    var_ref = np.tensordot(x**2, v.var(), axes=((1, 0), (0, 2)))
    assert v_.shape == mean_ref.shape
    assert v_.iaxes == (0, 1)
    assert np.max(np.abs(v_.mean() - mean_ref)) < tol
    assert np.max(np.abs(v_.var() - var_ref)) < tol

    assert (w + v_).shape == (5, 5)
    assert (w.T + v_.T).shape == (5, 5)
    with pytest.raises(ValueError):
        assert (w.T + v_).shape == (5, 5)


def test_iid_copy():
    tol = 1e-8

    v = iid_repeat(random_normal((3, 4)), 5, axis=1)
    v = iid_repeat(v, 6, axis=-1)
    v_ = v.iid_copy()

    assert v.shape == v_.shape
    assert v.iscomplex == v_.iscomplex
    assert np.max(np.abs(gp.sparse.cov(v, v_))) < tol
    assert np.max(np.abs(v.mean() - v_.mean())) < tol
    assert np.max(np.abs(v.cov() - v_.cov())) < tol

    v = iid_repeat(random_normal((4, 3), dtype=np.complex64), 5, axis=1)
    v = iid_repeat(v, 6, axis=-2)
    v_ = v.iid_copy()

    assert v.shape == v_.shape
    assert v.iscomplex == v_.iscomplex
    assert np.max(np.abs(gp.sparse.cov(v, v_))) < tol
    assert np.max(np.abs(v.mean() - v_.mean())) < tol
    assert np.max(np.abs(v.cov() - v_.cov())) < tol


def test_doc_string_sync():
    methods = ["iid_copy", "mean", "var", "sample", "logp", "__or__"]

    for m in methods:
        assert getattr(Normal, m).__doc__
        assert getattr(Normal, m).__doc__ == getattr(SparseNormal, m).__doc__


def test_array_conversion():
    # The conversion of sparse normal variables to numpy arrays is 
    # not supported. The sparse axes cannot be indexed, and it makes 
    # the default conversion method of numpy.array() silently yield 
    # an empty array.

    v = iid_repeat(normal(), 3)

    with pytest.raises(TypeError):
        np.array(v)
    with pytest.raises(TypeError):
        np.asarray(v)
    with pytest.raises(TypeError):
        np.multiply(v, 3)
