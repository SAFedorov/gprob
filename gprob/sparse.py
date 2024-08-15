from importlib import import_module
from functools import reduce
from operator import mul
from warnings import warn

import numpy as np
from numpy.linalg import LinAlgError
from numpy.exceptions import AxisError

from . import normal_
from .normal_ import (Normal, complete, match_, complete_tensordot_axes,
                      validate_logp_samples, print_)
from .external import einsubs


def iid_repeat(x, nrep=1, axis=0):
    """Creates a sparse array of `nrep` independent identically distributed 
    copies of `x` stacked together along `axis`.
    
    Args:
        x: Normal or sparse normal random variable.
        nrep: Integer number of repetitions.
        axis: Integer index of the axis along which the repetitions are stacked.

    Returns:
        A sparse normal random variable with `axis` being an independence axis.
    """

    x = lift(SparseNormal, x).iid_copy()
    
    axis = _normalize_axis(axis, x.ndim + 1)

    iaxid = x._iaxid[:axis] + (x._niax + 1,) + x._iaxid[axis:]

    # Index that adds a new axis.
    idx = [slice(None, None, None)] * x.ndim
    idx.insert(axis, None)
    idx = tuple(idx)

    new_shape = x.shape[:axis] + (nrep,) + x.shape[axis:]
    return _as_sparse(x[idx].broadcast_to(new_shape), iaxid)


class SparseNormal(Normal):
    """Array of block-independent normal random variables."""

    __slots__ = ("_iaxid",)  
    # iaxid is a tuple of the length ndim indicating the axes along which 
    # the random variables at different indices are independent of each other. 
    # It contains integer ids at positions corresponding to the independence 
    # axes, and `None`s at the positions corresponding to regular axes.

    _mod = import_module(__name__)
    
    @property
    def delta(self):
        return _as_sparse(super().delta, self._iaxid)
    
    @property
    def real(self):
        return _as_sparse(super().real, self._iaxid)
    
    @property
    def imag(self):
        return _as_sparse(super().imag, self._iaxid)
    
    @property
    def iaxes(self):
        """Ordered sequence of axes along which the array elements 
        are independent from each other."""
        return tuple([i for i, b in enumerate(self._iaxid) if b])
    
    @property
    def _niax(self):
        return len(self._iaxid) - self._iaxid.count(None)
    
    def __array__(self):
        # By default, the application of numpy.array() to a sparse variable 
        # can silently return an empty array, because such variables cannot 
        # be iterated over along their independence axes. As such a behavior 
        # can be confusing, conversion to numpy arrays is disallowed.

        raise TypeError(f"{self.__class__.__name__} variables cannot "
                        "be converted to numpy arrays.")
    
    def __repr__(self):
        return print_(self, extra_attrs=("iaxes",))
    
    def __neg__(self):
        return _as_sparse(super().__neg__(), self._iaxid)

    def __add__(self, other):
        x = super().__add__(other)
        iaxid = _validate_iaxid([self, other])
        return _as_sparse(x, iaxid)

    def __mul__(self, other):
        x = super().__mul__(other)
        iaxid = _validate_iaxid([self, other])
        return _as_sparse(x, iaxid)

    def __truediv__(self, other):
        x = super().__truediv__(other)
        iaxid = _validate_iaxid([self, other])
        return _as_sparse(x, iaxid)

    def __rtruediv__(self, other):
        x = super().__rtruediv__(other)
        iaxid = _validate_iaxid([self, other])
        return _as_sparse(x, iaxid)

    def __matmul__(self, other):
        other, isnumeric = match_(self.__class__, other)

        if not isnumeric:
            raise ValueError("Bilinear operations between two sparse normal "
                             "variables are not supported.")
        
        x = super().__matmul__(other)
        
        op_axis = self.ndim - 1
        if self._iaxid[op_axis]:
            raise ValueError("Matrix multiplication contracting over "
                             "independence axes is not supported. "
                             f"Axis {op_axis} of operand 1 is contracted.")

        if other.ndim == 1:
            iaxid = self._iaxid[:-1]
        elif other.ndim <= self.ndim:
            iaxid = self._iaxid
        else:
            # There are dimensions added by broadcasting.
            iaxid = (None,) * (other.ndim - self.ndim) + self._iaxid

        return _as_sparse(x, iaxid)

    def __rmatmul__(self, other):
        other, isnumeric = match_(self.__class__, other)

        if not isnumeric:
            raise ValueError("Bilinear operations between two sparse normal "
                             "variables are not supported.")
        
        x = super().__rmatmul__(other)
        
        if self.ndim == 1:
            op_axis = 0
        else:
            op_axis = self.ndim - 2

        if self._iaxid[op_axis]:
            raise ValueError("Matrix multiplication contracting over "
                             "independence axes is not supported. "
                             f"Axis {op_axis} of operand 2 is contracted.")

        if other.ndim == 1:
            iaxid = tuple([b for i, b in enumerate(self._iaxid) 
                           if i != op_axis])
        elif other.ndim >= 2 and other.ndim <= self.ndim - 1:
            iaxid = self._iaxid
        else:
            d = other.ndim - self.ndim
            iaxid = (None,) * d + self._iaxid

        return _as_sparse(x, iaxid)
    
    def __pow__(self, other):
        x = super().__pow__(other)
        iaxid = _validate_iaxid([self, other])
        return _as_sparse(x, iaxid)

    def __rpow__(self, other):
        x = super().__rpow__(other)
        iaxid = _validate_iaxid([self, other])
        return _as_sparse(x, iaxid)
    
    def __getitem__(self, key):
        x = super().__getitem__(key)
        iaxid = _item_iaxid(self, key)
        return _as_sparse(x, iaxid)
    
    def __setitem__(self, key, value):
        value, isnumeric = match_(self.__class__, value)

        if not isnumeric:
            iaxid = _item_iaxid(self, key)
            val_iaxid = (None,) * (len(iaxid) - value.ndim) + value._iaxid
            
            if val_iaxid != iaxid:
                raise ValueError("The independence axes of the indexing result "
                                 "do not match those of the assigned value.")
            
            # Deterministic values can skip this check, 
            # as their axes are always compatible.
        
        super().__setitem__(key, value)
        
    # ---------- array methods ----------

    def conjugate(self):
        return _as_sparse(super().conjugate(), self._iaxid)
    
    def cumsum(self, axis):
        if axis is None:
            # For regular arrays, None is the default value. For sparse arrays, 
            # it is not supported because these arrays usually 
            # cannot be flattened.
            raise ValueError("None value for the axis is not supported.")
        
        axis = _normalize_axis(axis, self.ndim)
        if self._iaxid[axis]:
            raise ValueError("The computation of cumulative sums along "
                             "independence axes is not supported.")

        return _as_sparse(super().cumsum(axis), self._iaxid)
    
    def diagonal(self, offset=0, axis1=0, axis2=1):
        [axis1, axis2] = _normalize_axes([axis1, axis2], self.ndim)

        if self._iaxid[axis1] or self._iaxid[axis2]:
            raise ValueError("Taking diagonals along independence axes "
                             "is not supported.")
        s = {axis1, axis2}
        iaxid = (tuple([idx for i, idx in enumerate(self._iaxid) if i not in s]) 
                 + (None,))
        return _as_sparse(super().diagonal(offset, axis1, axis2), iaxid)

    def flatten(self, order="C"):
        if not any(self._iaxid):  # This covers the case of self.ndim == 0.
            return _as_sparse(super().flatten(order=order), (None,))
        
        if self.ndim == 1:
            return _as_sparse(super().flatten(order=order), self._iaxid)
        
        # Checks if all dimensions except maybe one are <= 1.
        max_sz = max(self.shape)
        max_dim = self.shape.index(max_sz)
        if all(x <= 1 for i, x in enumerate(self.shape) if i != max_dim):
            iaxid = (self._iaxid[max_dim],)
            return _as_sparse(super().flatten(order=order), iaxid)
        
        raise ValueError("Sparse normal variables can only be flattened if "
                         "all their dimensions except maybe one have size <=1, "
                         "or they have no independence axes.")
    
    def moveaxis(self, source, destination):
        source = _normalize_axis(source, self.ndim)
        destination = _normalize_axis(destination, self.ndim)
        
        iaxid = list(self._iaxid)
        iaxid.insert(destination, iaxid.pop(source))
        iaxid = tuple(iaxid)

        return _as_sparse(super().moveaxis(source, destination), iaxid)
    
    def ravel(self, order="C"):
        if not any(self._iaxid):  # This covers the case of self.ndim == 0.
            return _as_sparse(super().ravel(order=order), (None,))
        
        if self.ndim == 1:
            return _as_sparse(super().ravel(order=order), self._iaxid)
        
        # Checks if all dimensions except maybe one are <= 1.
        max_sz = max(self.shape)
        max_dim = self.shape.index(max_sz)
        if all(x <= 1 for i, x in enumerate(self.shape) if i != max_dim):
            iaxid = (self._iaxid[max_dim],)
            return _as_sparse(super().ravel(order=order), iaxid)
        
        raise ValueError("Sparse normal variables can only be flattened if "
                         "all their dimensions except maybe one have size <=1, "
                         "or they have no independence axes.")

    def reshape(self, newshape, order="C"):
        x = super().reshape(newshape, order)  
        # Reshapes the map before the axes check to yield a meaningful 
        # error message if the old and the new shapes are inconsistent.

        newshape = x.shape  # Replaces '-1' if it was in newshape.

        if x.size == 0:
            # The transformation of independence axes for zero-size variables 
            # cannot be determined unambiguously, so we always assume that 
            # the transformed variable has no independence axes.
            return SparseNormal(x, (None,) * x.ndim)

        new_dim = 0

        new_cnt = 1
        old_cnt = 1

        iaxid = [None] * len(newshape)
        for i, n in enumerate(self.shape):
            if self._iaxid[i]:
                if n != 1:
                    # Skips all trivial dimensions first.
                    while new_dim < len(newshape) and newshape[new_dim] == 1:
                        new_dim += 1

                if (new_dim < len(newshape) and newshape[new_dim] == n 
                    and new_cnt == old_cnt):
                    
                    iaxid[new_dim] = self._iaxid[i]
                else:
                    raise ValueError("Reshaping that affects independence axes "
                                     f"is not supported. Axis {i} is affected "
                                     "by the requested shape transformation "
                                     f"{self.shape} -> {newshape}.")
                old_cnt *= n
                new_cnt *= newshape[new_dim]
                new_dim += 1
            else:
                old_cnt *= n

                while new_cnt < old_cnt:
                    new_cnt *= newshape[new_dim]
                    new_dim += 1
        
        iaxid = tuple(iaxid)
        return _as_sparse(x, iaxid)

    def sum(self, axis, keepdims=False):
        if not isinstance(axis, int) and not isinstance(axis, tuple):
            raise ValueError("`axis` must be an integer or "
                             "a tuple of integers.")
            # None, the default value for non-sparse arrays, is not supported,
            # because in the typical case the variable has at least one 
            # independence axis that cannot be contracted.

        sum_axes = _normalize_axes(axis, self.ndim)

        if any(self._iaxid[ax] for ax in sum_axes):
            raise ValueError("The computation of sums along "
                             "independence axes is not supported.")
        if keepdims:
            iaxid = self._iaxid
        else:
            iaxid = tuple([b for i, b in enumerate(self._iaxid) 
                           if i not in sum_axes])

        return _as_sparse(super().sum(axis, keepdims), iaxid)

    def transpose(self, axes=None):
        if axes is None:
            iaxid = self._iaxid[::-1]
        else:
            iaxid = tuple([self._iaxid[ax] for ax in axes])

        return _as_sparse(super().transpose(axes), iaxid)
    
    def trace(self, offset=0, axis1=0, axis2=1):
        [axis1, axis2] = _normalize_axes([axis1, axis2], self.ndim)

        if self._iaxid[axis1] or self._iaxid[axis2]:
            raise ValueError("Traces along independence axes "
                             "are not supported.")
        s = {axis1, axis2}
        iaxid = tuple([b for i, b in enumerate(self._iaxid) if i not in s])
        return _as_sparse(super().trace(offset, axis1, axis2), iaxid)
    
    def split(self, indices_or_sections, axis=0):
        axis = _normalize_axis(axis, self.ndim)

        if self._iaxid[axis]:
            raise ValueError("Splitting along independence axes "
                             "is not supported.")
        
        xs = super().split(indices_or_sections, axis)
        return [_as_sparse(x, self._iaxid) for x in xs]
    
    def broadcast_to(self, shape):
        iaxid = (None,) * (len(shape) - self.ndim) + self._iaxid
        return _as_sparse(super().broadcast_to(shape), iaxid)

    # ---------- probability-related methods ----------

    def iid_copy(self):
        """Creates an independent identically distributed copy 
        of the varaible."""
        return _as_sparse(super().iid_copy(), self._iaxid)

    def condition(self, observations):
        """Conditioning operation. Applicable between variables having the same 
        numbers and sizes of independence axes.
        
        Args:
            observations (SparseNormal or dict):
                A single sparse normal variable or a dictionary
                of observations of the format 
                {`variable`: `value`, ...}, where `variable`s are sparse normal 
                variables, and `value`s can be numerical constants or 
                sparse normal variables. Specifying a single normal `variable` 
                is equavalent to specifying {`variable`: `0`}.
        
        Returns:
            Conditional sparse normal variable.
        """

        if isinstance(observations, dict):
            obs = [lift(SparseNormal, k-v) for k, v in observations.items()]
        else:
            obs = [lift(SparseNormal, observations)]

        niax = self._niax  # A shorthand.

        # Moves the sparse axes first, reordering them in increasing order,
        # and flattens the dense subspaces.

        s_sparse_ax = [i for i, b in enumerate(self._iaxid) if b]
        s_dense_ax = [i for i, b in enumerate(self._iaxid) if not b]
        dense_sz = reduce(mul, [self.shape[i] for i in s_dense_ax], 1)

        self_fl = self.transpose(tuple(s_sparse_ax + s_dense_ax))
        self_fl = self_fl.reshape(self_fl.shape[:niax] + (dense_sz,))

        mismatch_w_msg = ("Conditions with different numbers or sizes of "
                          "independence axes compared to `self` are ignored. "
                          "The consistency of such conditions is not checked.")
        obs_flat = []
        iax_ord = [i for i in self._iaxid if i]
        for c in obs:
            if c._niax != niax:
                warn(mismatch_w_msg, SparseConditionWarning)
                continue

            sparse_ax = [c._iaxid.index(i) for i in iax_ord]
            dense_ax = [i for i, b in enumerate(c._iaxid) if not b]
            dense_sz = reduce(mul, [c.shape[i] for i in dense_ax], 1)
            
            c = c.transpose(tuple(sparse_ax + dense_ax))
            c = c.reshape(c.shape[:niax] + (dense_sz,))

            if c.shape[:niax] != self_fl.shape[:niax]:
                warn(mismatch_w_msg, SparseConditionWarning)
                continue

            if c.iscomplex:
                obs_flat.extend([c.real, c.imag])
            else:
                obs_flat.append(c)

        if not obs_flat:
            return self

        # Combines the observations in one and completes them w.r.t. self.
        cond = concatenate(SparseNormal, obs_flat, axis=-1)
        lat, [av, ac] = complete([self_fl, cond])

        t_ax = tuple(range(1, niax+1)) + (0, -1)

        av = av.transpose(t_ax)
        mv = self_fl.mean()
        if self.iscomplex:
            av = np.concatenate([av.real, av.imag], axis=-1)
            mv = np.concatenate([mv.real, mv.imag], axis=-1)

        ac = ac.transpose(t_ax)
        mc = cond.mean()

        # The calculation of the conditional map and mean.

        q, r = np.linalg.qr(ac, mode="reduced")

        # If there are zero diagonal elements in the triangular matrix, 
        # some colums of `ac` are linearly dependent.
        dia_r = np.abs(np.diagonal(r, axis1=-1, axis2=-2))
        tol = np.finfo(r.dtype).eps
        if (dia_r < (tol * np.max(dia_r))).any():
            raise LinAlgError("Degenerate conditions.")

        t_ax = tuple(range(niax)) + (-1, -2)
        es = np.linalg.solve(r.transpose(t_ax), -mc)
        aproj = (q.transpose(t_ax) @ av)

        cond_a = av - q @ aproj
        cond_m = mv + np.einsum("...i, ...ij -> ...j", es, aproj)

        # Transposing and shaping back.

        t_ax = (-2,) + tuple(range(niax)) + (-1,)
        cond_a = cond_a.transpose(t_ax)

        if self.iscomplex:
            # Converting back to complex.
            n = cond_m.shape[-1] // 2
            cond_a = cond_a[..., :n] + 1j * cond_a[..., n:]
            cond_m = cond_m[..., :n] + 1j * cond_m[..., n:]

        fcv = Normal(cond_a, cond_m, lat)  
        # A proxy variable to perform transposition.

        dense_sh = tuple([n for n, i in zip(self.shape, self._iaxid) if not i])
        fcv = fcv.reshape(fcv.shape[:niax] + dense_sh)
        t_ax = tuple([i[0] for i in sorted(enumerate(s_sparse_ax + s_dense_ax), 
                                           key=lambda x:x[1])])
        fcv = fcv.transpose(t_ax)

        return _as_sparse(SparseNormal(fcv.a, fcv.b, fcv.lat), self._iaxid)  # TODO: Is there a way of doing it without a proxy?

    def cov(self):
        """Covariance, generalizing `<outer((x-<x>), (x-<x>)^H)>`, 
        where `H` denotes conjugate transposition, and `<...>` is 
        the expectation value of `...`.

        Returns:
            An array with the dimension number equal to the doubled 
            number of the regular dimensions of the variable, plus 
            the undoubled number of its sparse (independence) dimensions. 
            In the returned array, the regular dimensions 
            go first in the order they appear in the variable shape, 
            and the independence dimensions are appended at the end.
            The resulting structure is the same as the structure produced 
            by repeated applications of `np.diagonal` over all the 
            independence dimensions of the full-sized covariance matrix `c`,
            `c[ijk... lmn...] = <(x[ijk..] - <x>)(x[lmn..] - <x>)*>`, 
            where `ijk...` and `lmn...` are indices that run over 
            the elements of the variable (here `x`), 
            and `*` denotes complex conjugation.

        Examples:
            >>> v = iid_repeat(normal(size=(3,)), 4)
            >>> v.shape
            (4, 3)
            >>> v.cov().shape
            (3, 3, 4)

            >>> v = iid_repeat(normal(size=(3, 2)), 4)
            >>> v = iid_repeat(v, 5)
            >>> v.shape
            (5, 4, 3, 2)
            >>> v.cov().shape
            (3, 2, 3, 2, 5, 4)
        """

        symb = [einsubs.get_symbol(i) for i in range(2 * self.ndim + 1)]
        elem_symb = symb[0]
        out_symb = symb[1:]

        in_symb1 = out_symb[:self.ndim]
        in_symb2 = out_symb[self.ndim:]

        for i in self.iaxes:
            out_symb.remove(in_symb2[i])
            out_symb.remove(in_symb1[i])
            out_symb.append(in_symb1[i])

            in_symb2[i] = in_symb1[i]
        
        # Adds the symbol for the summation over the latent variables.
        in_symb1.insert(0, elem_symb)
        in_symb2.insert(0, elem_symb)

        subs = f"{"".join(in_symb1)},{"".join(in_symb2)}->{"".join(out_symb)}"
        return np.einsum(subs, self.a, self.a.conj())

    
    def sample(self, n=None):
        """Samples the random variable `n` times.
        
        Args:
            n: An integer number of samples or None.
        
        Returns:
            A single sample with the same shape as the varaible if `n` is None, 
            or an array of samples of the lenght `n` if `n` is an integer,
            in which case the total shape of the array is the shape of 
            the varaible plus (n,) prepended as the first dimension.

        Examples:
            >>> v.shape
            (2, 3)
            >>> v.sample()
            array([[-0.33993954, -0.26758247, -0.33927517],
                   [-0.36414751,  0.76830802,  0.0997399 ]])
            >>> v.sample(2)
            array([[[-1.78808198,  1.08481027,  0.40414722],
                    [ 0.95298205, -0.42652839,  0.62417706]],

                   [[-0.81295799,  1.76126207, -0.36532098],
                    [-0.22637276, -0.67915003, -1.55995937]]])
            >>> v.sample(5).shape
            (5, 2, 3)
        """
        
        if n is None:
            nsh = tuple()
        else:
            nsh = (n,)

        iaxsh = [m for m, b in zip(self.shape, self._iaxid) if b]
        r = np.random.normal(size=(*nsh, *iaxsh, self.a.shape[0]))

        symb = [einsubs.get_symbol(i) for i in range(self.ndim + 1 + len(nsh))]
        
        elem_symb = symb[0]
        out_symb = symb[1:]

        in_symb1 = out_symb[:len(nsh)]
        in_symb2 = out_symb[len(nsh):]

        in_symb1.extend(in_symb2[i] for i in self.iaxes)
        in_symb1.append(elem_symb)

        in_symb2.insert(0, elem_symb)

        subs = f"{"".join(in_symb1)},{"".join(in_symb2)}->{"".join(out_symb)}"
        return np.einsum(subs, r, self.a) + self.mean()
        

    def logp(self, x):
        """Log likelihood of a sample.
    
        Args:
            x: Sample value or a sequence of sample values.

        Returns:
            Natural logarithm of the probability density at the sample value - 
            a single number for single sample inputs, and an array for sequence 
            inputs.
        """

        delta_x = x - self.mean()
        validate_logp_samples(self, delta_x)

        if self.iscomplex:
            delta_x = np.hstack([delta_x.real, delta_x.imag])
            self = SparseNormal._stack([self.real, self.imag])
        elif np.iscomplexobj(delta_x):
            # Casts to real with a warning.
            delta_x = delta_x.astype(delta_x.real.dtype)

        niax = self._niax
        
        if self.ndim == niax:
            # Scalar dense subspace.

            sigmasq = np.einsum("i..., i... -> ...", self.fcv.a, self.fcv.a)
            llk = -0.5 * (delta_x**2 / sigmasq + np.log(2 * np.pi * sigmasq))
            return np.sum(llk, axis=tuple(range(-self.ndim, 0)))
        
        # In the remaining, the dimensionality of the dense subspace is >= 1.

        batch_ndim = x.ndim - self.ndim

        # Moves the sparse axes to the beginning and the batch axis to the end.
        sparse_ax = [i + batch_ndim for i, b in enumerate(self._iaxid) if b]
        dense_ax = [i + batch_ndim for i, b in enumerate(self._iaxid) if not b]

        if batch_ndim:
            dense_ax.append(0)

        delta_x = delta_x.transpose(tuple(sparse_ax + dense_ax))

        # Covariance with the sparse axes first.
        cov = self.cov()
        t = tuple(range(cov.ndim))        
        cov = cov.transpose(t[cov.ndim - niax:] + t[:cov.ndim - niax])

        # Flattens the dense subspace.
        dense_sh = [n for n, i in zip(self.shape, self._iaxid) if not i]
        dense_sz = reduce(mul, dense_sh, 1)

        cov = cov.reshape(cov.shape[:niax] + (dense_sz, dense_sz))

        new_x_sh = delta_x.shape[:niax] + (dense_sz,) + (len(x),) * batch_ndim
        delta_x = delta_x.reshape(new_x_sh)

        ltr = np.linalg.cholesky(cov)
        z = np.linalg.solve(ltr, delta_x)
        
        sparse_sz = self.size // dense_sz
        rank = cov.shape[-1] * sparse_sz  # The rank is full.
        log_sqrt_det = np.sum(np.log(np.diagonal(ltr, axis1=-1, axis2=-2)))
        norm = 0.5 * np.log(2 * np.pi) * rank + log_sqrt_det

        idx = "".join([einsubs.get_symbol(i) for i in range(niax + 1)])
        return -0.5 * np.einsum(f"{idx}..., {idx}... -> ...", z, z) - norm


def _as_sparse(x, iaxid):
    """Assigns `iaxid` to `x`, thereby initializing it as a sparse variable.
    Returns `x`."""

    if not isinstance(iaxid, tuple):
            raise ValueError("iaxid must be a tuple, while now it is "
                             f"of type {type(iaxid)}.")
        
    if len(iaxid) != x.ndim:
        raise ValueError(f"The size of iaxid ({len(iaxid)}) does not "
                         "match the number of the dimensions "
                         f"of the variable ({x.ndim}).")
    
    if not all([i is None or i for i in iaxid]):
        raise ValueError("iaxid can contain only Nones and integers "
                         f"greater than zero, while now it is {iaxid}.")

    x._iaxid = iaxid
    return x


def _normalize_axis(axis, ndim):
    """Ensures that the axis index is positive and within the array dimension.

    Returns:
        Normalized axis index.
    
    Raises:
        AxisError: 
            If the axis is out of range.
    """

    axis = axis.__index__()

    if axis < -ndim or axis > ndim - 1:
        raise AxisError(f"Axis {axis} is out of bounds "
                        f"for an array of dimension {ndim}.")
    if axis < 0:
        axis = ndim + axis

    return axis


def _normalize_axes(axes, ndim):
    """Ensures that the axes indices are positive, within the array dimension,
    and without duplicates.

    Returns:
        Tuple of normalized axes indices.
    
    Raises:
        AxisError:
            If one of the axes is out of range.
        ValueError:
            If there are duplicates among the axes.
    """

    # Essentially repeats the code of numpy's normalize_axis_tuple,
    # which does not seem to be part of the public API.

    if type(axes) not in (tuple, list):
        try:
            axes = [axes.__index__()]
        except TypeError:
            pass

    axes = tuple([_normalize_axis(ax, ndim) for ax in axes])

    if len(set(axes)) != len(axes):
        raise ValueError('Axes cannot repeat.')
    
    return axes


def _validate_iaxid(seq):
    """Checks that the independence axes of the sparse normal arrays in `seq`
    are compatible.
    
    Returns:
        `iaxid` of the final shape for the broadcasted arrays.
    """

    ndim = max(np.ndim(x) for x in seq)
    seq = [x for x in seq if hasattr(x, "_iaxid")]
    iaxids = set((None,) * (ndim - x.ndim) + x._iaxid for x in seq)

    if len(iaxids) == 0:
        return (None,) * ndim
    if len(iaxids) == 1:
        return iaxids.pop()
    
    # len > 1, which means that not all independence axes are identical.
    # This is not permitted. To raise an error, the following code determines 
    # if the problem is the number, location, or order of the axes.

    msg = ("Combining sparse normal variables requires them to have "
           "the same numbers of independence axes at the same "
           "positions in the shape and in the same order.")
    max_disp = 10  # Maximum number of values to display.

    ns = set((len(ids) - ids.count(None)) for ids in iaxids)
    if len(ns) > 1:
        if len(ns) < max_disp:
            valstr = (": " + ", ".join(str(n) for n in ns))
        else:
            valstr = ""

        raise ValueError("Mismatching numbers of independence axes "
                         f"in the operands{valstr}. {msg}")

    get_iax_numbers = lambda ids: tuple([i for i, b in enumerate(ids) if b])
    locs = set(get_iax_numbers(ids) for ids in iaxids)
    if len(locs) > 1:
        if len(ns) < max_disp:
            valstr = (": " + ", ".join(str(loc) for loc in locs))
        else:
            valstr = ""
        
        raise ValueError("Incompatible locations of the independence axes "
                         f"of the operands{valstr}. {msg}")

    orders = set(tuple([ax for ax in ids if ax is not None]) for ids in iaxids)
    assert len(orders) > 1

    if len(ns) < max_disp:
        valstr = (": " + ", ".join(str(order) for order in orders))
    else:
        valstr = ""

    raise ValueError("Incompatible orders of the independence axes "
                     f"of the operands{valstr}. {msg}")
    

def _item_iaxid(x, key):
    """Validates the key and calculates iaxid for the result of the indexing 
    of `x` with `key`.
    
    Args:
        x: 
            Sparse normal variable.
        key: 
            Numpy-syntax array indexing key. Independence axes can only 
            be indexed by full slices `:` (explicit or implicit via ellipsis). 
        
    Returns:
        A tuple of iaxids for the indexing result.
    """

    if not isinstance(key, tuple):
        key = (key,)
    
    out_axs = []  # out_axs[i] is the number of the i-th output axis 
                  # in the input shape.
    idx = []  # idx[i] is the index used for the i-th input axis.

    has_ellipsis = False
    used_dim = 0

    for k in key:
        if isinstance(k, int):
            used_dim += 1
            continue

        if isinstance(k, slice):
            idx.append(k)
            out_axs.append(used_dim)
            used_dim += 1
            continue
        
        if k is Ellipsis:
            if has_ellipsis:
                raise IndexError("An index can only have a single "
                                 "ellipsis ('...').")
            
            has_ellipsis = True
            ep_in = used_dim       # ellipsis position in the input
            ep_out = len(out_axs)  # ellipsis position in the output
            continue

        if k is np.newaxis:
            idx.append(None)
            out_axs.append(None)
            continue

        # Advanced indices (int and bool arrays) are not implemented. 

        raise IndexError("Only integers, slices (':'), ellipsis ('...'), "
                         "numpy.newaxis ('None') are valid indices.")
    
    # Checks if any input dimensions remain unconsumed.
    delta = x.ndim - used_dim
    if delta > 0:
        if has_ellipsis:
            for i in range(ep_out, len(out_axs)):
                if out_axs[i] is not None:
                    out_axs[i] += delta

            out_axs[ep_out: ep_out] = range(ep_in, ep_in + delta)
            idx[ep_out: ep_out] = (slice(None, None, None) for _ in range(delta))
        else:
            out_axs.extend(range(used_dim, used_dim + delta))
            idx.extend(slice(None, None, None) for _ in range(delta))
    elif delta < 0:
        raise IndexError(f"Too many indices: the array is {x.ndim}-dimensional,"
                         f" but {used_dim} indices were given.")
    
    for i, b in enumerate(x._iaxid):
        if i not in out_axs:
            oi = -1
        else:
            oi = out_axs.index(i)
        
        if b and (oi == -1 or idx[oi] != slice(None, None, None)):
            raise IndexError("Only full slices (':') and ellipses ('...') "
                             "are valid indices for independence axes. "
                             f"Axis {i} is indexed with an invalid key.")
    
    # Returns iaxid for the indexing result.
    return tuple([None if ax is None else x._iaxid[ax] for ax in out_axs])
    

def cov(*args):
    """The sparse implementation of the covariance. The function 
    expects `args` to have strictly one or two elements."""

    def find_det_dense_shape(v, d):
        # Tries resolving what the independence axes of the deterministic 
        # variable `d` could be based on the shape and the independence axes 
        # of the sparse variable `v`. If the resolution is ambiguous or fails, 
        # the function throws an error.

        sparse_shape_v = [s for b, s in zip(v._iaxid, v.shape) if b]
        dense_shape_d = list(d.shape)

        for s in set(sparse_shape_v):
            cnt_v = sparse_shape_v.count(s)
            cnt_d = dense_shape_d.count(s)

            if cnt_d == cnt_v:
                for _ in range(cnt_d):
                    dense_shape_d.remove(s)
            elif cnt_d == 0:
                iax = v.shape.index(s)
                raise ValueError("The shape of the deterministic variable "
                                 f"{d.shape} does not contain a dimension "
                                 f"of size {s} that can correspond to the "
                                 f"independence axis {iax} of the sparse "
                                 f"normal variable with the shape {v.shape}.")
            elif cnt_d < cnt_v:
                raise ValueError("The shape of the deterministic variable "
                                 f"{d.shape} does not contain {cnt_v} "
                                 f"dimensions of size {s} to match the "
                                 "independence axes of the sparse "
                                 f"normal variable with the shape {v.shape}.")
            else:
                # cnt_d > cnt_v
                raise ValueError("The shape of the deterministic variable "
                                 f"{d.shape} contains too many dimensions "
                                 f"of size {s}, which makes their "
                                 "correspondence with the independence axes "
                                 "of the sparse normal variable ambiguous.")
        return dense_shape_d

    args = [lift(SparseNormal, arg) for arg in args]

    if len(args) == 1:
        return args[0].cov()

    # The remaining case is len(args) == 2.
    x, y = args
    
    # If either x or y is deterministic, a zero result is returned.

    if x.nlat == 0:
        dense_shape_y = [s for b, s in zip(y._iaxid, y.shape) if not b]
        sparse_shape = [s for b, s in zip(y._iaxid, y.shape) if b]
        dense_shape_x = find_det_dense_shape(y, x)

        res_shape = tuple(dense_shape_x + dense_shape_y + sparse_shape)
        dt = np.result_type(x.a, y.a)
        return np.zeros(res_shape, dtype=dt)

    if y.nlat == 0:
        dense_shape_x = [s for b, s in zip(x._iaxid, x.shape) if not b]
        sparse_shape = [s for b, s in zip(x._iaxid, x.shape) if b]
        dense_shape_y = find_det_dense_shape(x, y)

        res_shape = tuple(dense_shape_x + dense_shape_y + sparse_shape)
        dt = np.result_type(x.a, y.a)
        return np.zeros(res_shape, dtype=dt)
    
    # The default case, when both variables are non-deterministic.

    iax_ord_x = [i for i in x._iaxid if i is not None]
    iax_ord_y = [i for i in y._iaxid if i is not None]

    if len(iax_ord_x) != len(iax_ord_y):
        raise ValueError("The numbers of the independence axes "
                         "must coincide for the two operands. "
                         f"Now operand 1 has {len(iax_ord_x)} axes, "
                         f"while operand 2 has {len(iax_ord_y)}.")
    
    if iax_ord_x != iax_ord_y:
        raise ValueError("The orders of the independence axes "
                         "must coincide for the two operands. "
                         f"Now operand 1 has the order {iax_ord_x}, "
                         f"while operand 2 has the order {iax_ord_y}.")

    symb = [einsubs.get_symbol(i) for i in range(x.ndim + y.ndim + 1)]
    elem_symb = symb[0]
    out_symb = symb[1:]

    in_symb1 = out_symb[:x.ndim]
    in_symb2 = out_symb[x.ndim:]

    for i, j in zip(x.iaxes, y.iaxes):
        out_symb.remove(in_symb2[j])
        out_symb.remove(in_symb1[i])
        out_symb.append(in_symb1[i])

        in_symb2[j] = in_symb1[i]
    
    # Adds the symbol for the summation over the latent variables.
    in_symb1.insert(0, elem_symb)
    in_symb2.insert(0, elem_symb)
    
    subs = f"{"".join(in_symb1)},{"".join(in_symb2)}->{"".join(out_symb)}"
    _, [ax, ay] = complete([x, y])
    return np.einsum(subs, ax, ay.conj())


class SparseConditionWarning(RuntimeWarning):
    """The warning issued when a condition is skipped during 
    the conditioning of a sparse variable."""
    pass


def lift(cls, x):
    if x.__class__ is cls:
        return x
    
    if x.__class__ is Normal:
        x = SparseNormal(x.a, x.b)
    else:
        x = normal_.lift(cls, x)
    return _as_sparse(x, (None,) * x.ndim)


def concatenate(cls, arrays, axis=0):
    iaxid = _validate_iaxid(arrays)
    axis = _normalize_axis(axis, len(iaxid))
    
    if iaxid[axis]:
        raise ValueError("Concatenation along independence axes "
                         "is not allowed.")
    
    return _as_sparse(normal_.concatenate(cls, arrays, axis), iaxid)


def stack(cls, arrays, axis=0):
    iaxid = _validate_iaxid(arrays)
    iaxid = iaxid[:axis] + (None,) + iaxid[axis:]
    return _as_sparse(normal_.stack(cls, arrays, axis), iaxid)


def call_linearized(cls, x, f, jmpf):
    return _as_sparse(normal_.call_linearized(cls, x, f, jmpf), x._iaxid)


def fftfunc(cls, name, x, n, axis, norm):
    raise NotImplementedError


def fftfunc_n(cls, name, x, s, axes, norm):
    raise NotImplementedError


def _check_independence(x, op_axes, n):
    """Checks that the operation axes are not independence axes of `x`.
    `n` is the operand number, normally 1 or 2, as this function is 
    a helper for bilinear functions."""
    
    if any([x._iaxid[ax] for ax in op_axes]):
        raise ValueError("Bilinear operations contracting over "
                         "independence axes are not supported. "
                         f"Axes {op_axes} of operand {n} are contracted.")
    

def bilinearfunc(cls, name, x, y, args=tuple(), pargs=tuple()):
    x, x_is_numeric = match_(cls, x)
    y, y_is_numeric = match_(cls, y)

    if not x_is_numeric and not y_is_numeric:
        raise ValueError("Bilinear operations between two sparse normal "
                         "variables are not supported.")
    
    return normal_.bilinearfunc(cls, name, x, y, args, pargs)


def _einsum_out_iaxid(x, insubs, outsubs):
    """Calculates the indices of the independence axes for
    the output operand."""
    
    for i, c in enumerate(insubs):
        if x._iaxid[i] and c not in outsubs:
            raise ValueError(f"Contraction over an independence axis ({i}).")
        
    iaxid = x._iaxid + (None,)  # Augments with a default.  
    return tuple([iaxid[insubs.find(c)] for c in outsubs])


def einsum01(cls, subs, x, y):
    # Converts the subscripts to an explicit form.
    (insu1, insu2), outsu = einsubs.parse(subs, (x.shape, y.shape))
    subs = f"{insu1},{insu2}->{outsu}"

    iaxid = _einsum_out_iaxid(x, insu1, outsu)
    return _as_sparse(normal_.einsum01(cls, subs, x, y), iaxid)


def einsum10(cls, subs, x, y):
    # Converts the subscripts to an explicit form.
    (insu1, insu2), outsu = einsubs.parse(subs, (x.shape, y.shape))
    subs = f"{insu1},{insu2}->{outsu}"

    iaxid = _einsum_out_iaxid(y, insu2, outsu)
    return _as_sparse(normal_.einsum10(cls, subs, x, y), iaxid)


def inner01(cls, x, y):
    if x.ndim == 0 or y.ndim == 0:
        return x * y
    
    op_axes = (-1,)
    _check_independence(x, op_axes, 1)

    iaxid = x._iaxid[:-1] + (None,) * (y.ndim - 1)
    return _as_sparse(normal_.inner01(cls, x, y), iaxid)
    

def inner10(cls, x, y):
    if x.ndim == 0 or y.ndim == 0:
        return x * y
    
    op_axes = (-1,)
    _check_independence(x, op_axes, 2)

    iaxid = (None,) * (x.ndim - 1) + y._iaxid[:-1]
    return _as_sparse(normal_.inner10(cls, x, y), iaxid)


def dot01(cls, x, y):
    if x.ndim == 0 or y.ndim == 0:
        return x * y
    
    op_axes = (-1,)
    _check_independence(x, op_axes, 1)

    iaxid = x._iaxid[:-1] + (None,) * (y.ndim - 1)
    return _as_sparse(normal_.dot01(cls, x, y), iaxid)


def dot10(cls, x, y):
    if x.ndim == 0 or y.ndim == 0:
        return x * y
    
    op_axes = (0,) if y.ndim == 1 else (-2,)
    _check_independence(y, op_axes, 2)

    iaxid = list(y._iaxid)
    iaxid.pop(op_axes[0])
    iaxid = (None,) * (x.ndim - 1) + tuple(iaxid)
    return _as_sparse(normal_.dot01(cls, x, y), iaxid)


def outer01(cls, x, y):
    return _as_sparse(normal_.outer01(cls, x, y), x._iaxid + (None,))


def outer10(cls, x, y):
    return _as_sparse(normal_.outer10(cls, x, y), (None,) + y._iaxid)


def kron01(cls, x, y):
    ndim = max(x.ndim, y.ndim)  # ndim of the result.
    iaxid = (None,) * (ndim - x.ndim) + x._iaxid
    return _as_sparse(normal_.kron01(cls, x, y), iaxid)


def kron10(cls, x, y):
    ndim = max(x.ndim, y.ndim)  # ndim of the result.
    iaxid = (None,) * (ndim - y.ndim) + y._iaxid
    return _as_sparse(normal_.kron10(cls, x, y), iaxid)


def tensordot01(cls, x, y, axes):
    axes1, axes2 = complete_tensordot_axes(axes)
    _check_independence(x, axes1, 1)

    iaxid = tuple([b for i, b in enumerate(x._iaxid) if i not in axes1])
    iaxid = iaxid + (None,) * (y.ndim - len(axes2))
    return _as_sparse(normal_.tensordot01(cls, x, y, (axes1, axes2)), iaxid)


def tensordot10(cls, x, y, axes):
    axes1, axes2 = complete_tensordot_axes(axes)
    _check_independence(y, axes2, 2)

    iaxid = tuple([b for i, b in enumerate(y._iaxid) if i not in axes2])
    iaxid = (None,) * (x.ndim - len(axes1)) + iaxid
    return _as_sparse(normal_.tensordot10(cls, x, y, (axes1, axes2)), iaxid)