from functools import reduce
from operator import mul
from warnings import warn

import numpy as np
from numpy.linalg import LinAlgError
from numpy.exceptions import AxisError

from . import normal_
from .normal_ import Normal, asnormal, print_normal
from .external import einsubs
from . import emaps
from .emaps import ElementaryMap


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

    x = assparsenormal(x)
    x = x.iid_copy()
    
    axis = _normalize_axis(axis, x.ndim + 1)

    n = x._niax + 1
    iaxid = x._iaxid[:axis] + (n,) + x._iaxid[axis:]
    fcv = x.fcv

    # Index that adds a new axis.
    idx = [slice(None, None, None)] * fcv.ndim
    idx.insert(axis, None)
    idx = tuple(idx)

    new_shape = fcv.shape[:axis] + (nrep,) + fcv.shape[axis:]
    fcv = normal_.broadcast_to(fcv[idx], new_shape)

    return SparseNormal(fcv, iaxid)


def assparsenormal(x):
    if isinstance(x, SparseNormal):
        return x
    
    if (hasattr(x, "_normal_priority_") 
        and x._normal_priority_ > SparseNormal._normal_priority_):

        raise TypeError(f"The variable {x} cannot be converted to "
                        "a sparse normal variable because it is already "
                        f"of higher priority ({x._normal_priority_} "
                        f"> {SparseNormal._normal_priority_}).")
    
    fcv = asnormal(x)
    return SparseNormal(fcv, (None,) * fcv.ndim)


class SparseNormal:
    """Array of block-independent normal random variables."""

    __slots__ = ("fcv", "_iaxid")
    __array_ufunc__ = None
    _normal_priority_ = 10
    
    def __init__(self, fcv, iaxid):
        if not isinstance(fcv, Normal):
            raise TypeError("fcv must be a normal variable.")
        
        self.fcv = fcv  # A fully-correlated normal variable with the same 
                        # mean and variance as the sparse varaible, serving 
                        # as the basis for most calculations.
        
        if not isinstance(iaxid, tuple):
            raise ValueError("iaxid must be a tuple, while now it is "
                             f"of type {type(iaxid)}.")
        
        if len(iaxid) != fcv.ndim:
            raise ValueError(f"The size of iaxid ({len(iaxid)}) does not "
                             "match the number of the dimensions "
                             f"of the variable ({fcv.ndim}).")
        
        if not all([i is None or i for i in iaxid]):
            raise ValueError("iaxid can contain only Nones and integers "
                             f"greater than zero, while now it is {iaxid}.")

        self._iaxid = iaxid  # A tuple of the length equal to ndim, containing 
                             # integer ids at positions corresponding to 
                             # independence axes, and `None`s at positions 
                             # corresponding to regular axes.

    @property
    def size(self):
        return self.fcv.size
    
    @property
    def shape(self):
        return self.fcv.shape
    
    @property
    def ndim(self):
        return self.fcv.ndim
    
    @property
    def real(self):
        return SparseNormal(self.fcv.real, self._iaxid)
    
    @property
    def imag(self):
        return SparseNormal(self.fcv.imag, self._iaxid)
    
    @property
    def T(self):
        return self.transpose()
    
    @property
    def iscomplex(self):
        return self.fcv.iscomplex
    
    @property
    def iaxes(self):
        """Ordered sequence of axes along which the array elements 
        are independent from each other."""
        return tuple([i for i, b in enumerate(self._iaxid) if b])
    
    @property
    def _niax(self):
        return len(self._iaxid) - self._iaxid.count(None)
    
    def __array__(self):
        # Sparse variables cannot be iterated over along their 
        # independence axes, which results in the default conversion 
        # method of numpy.array() silently returning an empty array. 
        # I find such behavior confusing, which is why conversion to 
        # numpy arrays is disallowed.

        raise TypeError(f"{self.__class__.__name__} variables cannot "
                        "be converted to numpy arrays.")
    
    def __repr__(self):
        return print_normal(self, extra_attrs=("iaxes",))
    
    def __len__(self):
        return len(self.fcv)
    
    def __neg__(self):
        return SparseNormal(-self.fcv, self._iaxid)

    def __add__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        fcv = self.fcv + other.fcv
        iaxid = _validate_iaxes([self, other])
        return SparseNormal(fcv, iaxid)

    def __radd__(self, other):
        other = assparsenormal(other)
        fcv = other.fcv + self.fcv
        iaxid = _validate_iaxes([self, other])
        return SparseNormal(fcv, iaxid)

    def __sub__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        fcv = self.fcv - other.fcv
        iaxid = _validate_iaxes([self, other])
        return SparseNormal(fcv, iaxid)

    def __rsub__(self, other):
        other = assparsenormal(other)
        fcv = other.fcv - self.fcv
        iaxid = _validate_iaxes([self, other])
        return SparseNormal(fcv, iaxid)

    def __mul__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        fcv = self.fcv * other.fcv
        iaxid = _validate_iaxes([self, other])
        return SparseNormal(fcv, iaxid)

    def __rmul__(self, other):
        other = assparsenormal(other)
        fcv = other.fcv * self.fcv
        iaxid = _validate_iaxes([self, other])
        return SparseNormal(fcv, iaxid)

    def __truediv__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        fcv = self.fcv / other.fcv
        iaxid = _validate_iaxes([self, other])
        return SparseNormal(fcv, iaxid)

    def __rtruediv__(self, other):
        other = assparsenormal(other)
        fcv = other.fcv / self.fcv
        iaxid = _validate_iaxes([self, other])
        return SparseNormal(fcv, iaxid)

    def __pow__(self, other):
        try:
            other = assparsenormal(other)
        except TypeError:
            return NotImplemented

        fcv = self.fcv ** other.fcv
        iaxid = _validate_iaxes([self, other])
        return SparseNormal(fcv, iaxid)

    def __rpow__(self, other):
        other = assparsenormal(other)
        fcv = other.fcv ** self.fcv
        iaxid = _validate_iaxes([self, other])
        return SparseNormal(fcv, iaxid)

    def __matmul__(self, other):
        other = assparsenormal(other)

        if self.ndim == 0:
            raise ValueError("Matrix multiplication requires at least one "
                             "dimension, while the operand 1 has 0.")
        if other.ndim == 0:
            raise ValueError("Matrix multiplication requires at least one "
                             "dimension, while the operand 2 has 0.")
        
        op_axis = self.ndim - 1

        if self._iaxid[op_axis]:
            raise ValueError("Matrix multiplication affecting independence "
                             "axes is not supported. "
                             f"Axis {op_axis} of operand 1 is affected.")

        if other.ndim == 1:
            iaxid = self._iaxid[:-1]
        elif other.ndim <= self.ndim:
            iaxid = self._iaxid
        else:
            # There are dimensions added by broadcasting.
            iaxid = (None,) * (other.ndim - self.ndim) + self._iaxid

        # Calculates the contribution from the deterministic part of `other`.
        fcv = self.fcv @ other.mean()
        v = SparseNormal(fcv, iaxid)
        
        if not _is_deterministic(other):
            # Adds the linearized contribution from the random part of `other`.
            # As `other` is a sparse normal, the consistency of
            # independence axes is ensured by the addition operation.
            v += self.mean() @ (other - other.mean())

        return v

    def __rmatmul__(self, other):
        other = assparsenormal(other)

        if other.ndim == 0:
            raise ValueError("Matrix multiplication requires at least one "
                             "dimension, while the operand 1 has 0.")
        if self.ndim == 0:
            raise ValueError("Matrix multiplication requires at least one "
                             "dimension, while the operand 2 has 0.")
        
        if self.ndim == 1:
            op_axis = 0
        else:
            op_axis = self.ndim - 2

        if self._iaxid[op_axis]:
            raise ValueError("Matrix multiplication affecting independence "
                             "axes is not supported. "
                             f"Axis {op_axis} of operand 2 is affected.")

        if other.ndim == 1:
            iaxid = tuple([b for i, b in enumerate(self._iaxid) 
                           if i != op_axis])
        elif other.ndim >= 2 and other.ndim <= self.ndim - 1:
            iaxid = self._iaxid
        else:
            d = other.ndim - self.ndim
            iaxid = (None,) * d + self._iaxid

        # Calculates the contribution from the deterministic part of `other`.
        fcv = other.mean() @ self.fcv
        w = SparseNormal(fcv, iaxid)
        
        if not _is_deterministic(other):
            # Adds the linearized contribution from the random part of `other`.
            # As `other`` is a sparse normal, the consistency of
            # independence axes is ensured by the addition operation.
            w += (other - other.mean()) @ self.mean()

        return w
    
    def __getitem__(self, key):
        out_ax = _parse_index_key(self, key)
        iaxid = tuple([self._iaxid[ax] if ax is not None else None 
                       for ax in out_ax])
        return SparseNormal(self.fcv[key], iaxid)
    
    def __setitem__(self, key, value):
        raise NotImplementedError
    
    def __or__(self, observations):
        """Conditioning operation."""
        return self.condition(observations)
        
    # ---------- array methods ----------

    def conjugate(self):
        return SparseNormal(self.fcv.conjugate(), self._iaxid)
    
    def conj(self):
        return self.conjugate()
    
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

        return SparseNormal(self.fcv.cumsum(axis), self._iaxid)
    
    def diagonal(self, offset=0, axis1=0, axis2=1):
        [axis1, axis2] = _normalize_axes([axis1, axis2], self.ndim)

        if self._iaxid[axis1] or self._iaxid[axis2]:
            raise ValueError("Taking diagonals along independence axes "
                             "is not supported.")
        s = {axis1, axis2}
        iaxid = (tuple([idx for i, idx in enumerate(self._iaxid) if i not in s]) 
                 + (None,))
        return SparseNormal(self.fcv.diagonal(offset, axis1, axis2), iaxid)

    def flatten(self, order="C"):
        if not any(self._iaxid):  # This covers the case of self.ndim == 0.
            return SparseNormal(self.fcv.flatten(order=order), (None,))
        
        if self.ndim == 1:
            return SparseNormal(self.fcv.flatten(order=order), self._iaxid)
        
        # Checks if all dimensions except maybe one are <= 1.
        max_sz = max(self.shape)
        max_dim = self.shape.index(max_sz)
        if all(x <= 1 for i, x in enumerate(self.shape) if i != max_dim):
            iaxid = (self._iaxid[max_dim],)
            return SparseNormal(self.fcv.flatten(order=order), iaxid)
        
        raise ValueError("Sparse normal variables can only be flattened if "
                         "all their dimensions except maybe one have size <=1, "
                         "or they have no independence axes.")
    
    def moveaxis(self, source, destination):
        source = _normalize_axis(source, self.ndim)
        destination = _normalize_axis(destination, self.ndim)
        
        iaxid = list(self._iaxid)
        iaxid.insert(destination, iaxid.pop(source))
        iaxid = tuple(iaxid)

        return SparseNormal(self.fcv.moveaxis(source, destination), iaxid)
    
    def ravel(self, order="C"):
        if not any(self._iaxid):  # This covers the case of self.ndim == 0.
            return SparseNormal(self.fcv.ravel(order=order), (None,))
        
        if self.ndim == 1:
            return SparseNormal(self.fcv.ravel(order=order), self._iaxid)
        
        # Checks if all dimensions except maybe one are <= 1.
        max_sz = max(self.shape)
        max_dim = self.shape.index(max_sz)
        if all(x <= 1 for i, x in enumerate(self.shape) if i != max_dim):
            iaxid = (self._iaxid[max_dim],)
            return SparseNormal(self.fcv.ravel(order=order), iaxid)
        
        raise ValueError("Sparse normal variables can only be flattened if "
                         "all their dimensions except maybe one have size <=1, "
                         "or they have no independence axes.")
    
    def reshape(self, newshape, order="C"):

        fcv = self.fcv.reshape(newshape, order)  
        # Reshaping the fully correlated variable before the axes check is
        # to yield a meaningful error message if the shapes are inconsistent.

        newshape = fcv.shape  # Replaces '-1' if it was in newshape.

        if fcv.size == 0:
            # The transformation of independence axes for zero-size variables 
            # cannot be determined unambiguously, so we always assume that 
            # the transformed variable has no independence axes.
            return SparseNormal(fcv, (None,) * fcv.ndim)

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
        return SparseNormal(fcv, iaxid)
    
    def split(self, indices_or_sections, axis=0):
        axis = _normalize_axis(axis, self.ndim)

        if self._iaxid[axis]:
            raise ValueError("Splitting along independence axes "
                             "is not supported.")
        
        vpieces = self.fcv.split(indices_or_sections, axis)
        return [SparseNormal(v, self._iaxid) for v in vpieces]
    
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

        fcv = self.fcv.sum(axis=axis, keepdims=keepdims)
        return SparseNormal(fcv, iaxid)

    def transpose(self, axes=None):
        if axes is None:
            iaxid = self._iaxid[::-1]
        else:
            iaxid = tuple([self._iaxid[ax] for ax in axes])

        return SparseNormal(self.fcv.transpose(axes), iaxid)
    
    def trace(self, offset=0, axis1=0, axis2=1):
        [axis1, axis2] = _normalize_axes([axis1, axis2], self.ndim)

        if self._iaxid[axis1] or self._iaxid[axis2]:
            raise ValueError("Traces along independence axes "
                             "are not supported.")
        s = {axis1, axis2}
        iaxid = tuple([b for i, b in enumerate(self._iaxid) if i not in s])
        return SparseNormal(self.fcv.trace(offset, axis1, axis2), iaxid)

    # ---------- probability-related methods ----------

    def iid_copy(self):
        """Creates an independent identically distributed copy 
        of the varaible."""
        return SparseNormal(self.fcv.iid_copy(), self._iaxid)

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
            obs = [assparsenormal(k-v) for k, v in observations.items()]
        else:
            obs = [assparsenormal(observations)]

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
        cond = concatenate(obs_flat, axis=-1)
        emv, emc = emaps.complete((self_fl.fcv.emap, cond.fcv.emap))

        t_ax = tuple(range(1, niax+1)) + (0, -1)

        a = emv.a.transpose(t_ax)
        m = self_fl.mean()
        if self.iscomplex:
            a = np.concatenate([a.real, a.imag], axis=-1)
            m = np.concatenate([m.real, m.imag], axis=-1)

        ac = emc.a.transpose(t_ax)
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
        aproj = (q.transpose(t_ax) @ a)

        cond_a = a - q @ aproj
        cond_m = m + np.einsum("...i, ...ij -> ...j", es, aproj)

        # Transposing and shaping back.

        t_ax = (-2,) + tuple(range(niax)) + (-1,)
        cond_a = cond_a.transpose(t_ax)

        if self.iscomplex:
            # Converting back to complex.
            n = cond_m.shape[-1] // 2
            cond_a = cond_a[..., :n] + 1j * cond_a[..., n:]
            cond_m = cond_m[..., :n] + 1j * cond_m[..., n:]

        fcv = Normal(ElementaryMap(cond_a, emv.elem), cond_m)

        dense_sh = tuple([n for n, i in zip(self.shape, self._iaxid) if not i])
        fcv = fcv.reshape(fcv.shape[:niax] + dense_sh)
        t_ax = tuple([i[0] for i in sorted(enumerate(s_sparse_ax + s_dense_ax), 
                                           key=lambda x:x[1])])
        fcv = fcv.transpose(t_ax)

        return SparseNormal(fcv, self._iaxid)
    
    def mean(self):
        """Mean.
        
        Returns:
            An array of the mean values with the same shape as 
            the random variable.
        """
        return self.fcv.b

    def var(self):
        """Variance, `<(x-<x>)(x-<x>)^*>`, where `*` denotes 
        complex conjugation, and `<...>` is the expectation value of `...`.
        
        Returns:
            An array of the varaince values with the same shape as 
            the random variable.
        """
        return self.fcv.var()

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
        a = self.fcv.emap.a
        return np.einsum(subs, a, a.conj())

    
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

        a = self.fcv.emap.a

        iaxsh = [m for m, b in zip(self.shape, self._iaxid) if b]
        r = np.random.normal(size=(*nsh, *iaxsh, a.shape[0]))

        symb = [einsubs.get_symbol(i) for i in range(self.ndim + 1 + len(nsh))]
        
        elem_symb = symb[0]
        out_symb = symb[1:]

        in_symb1 = out_symb[:len(nsh)]
        in_symb2 = out_symb[len(nsh):]

        in_symb1.extend(in_symb2[i] for i in self.iaxes)
        in_symb1.append(elem_symb)

        in_symb2.insert(0, elem_symb)

        subs = f"{"".join(in_symb1)},{"".join(in_symb2)}->{"".join(out_symb)}"
        return np.einsum(subs, r, a) + self.mean()
        

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
        normal_.validate_logp_samples(self, delta_x)

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


def _is_deterministic(x):
    """True if the sparse normal `x` is a promoted numeric constant, 
    False if it is not."""
    return len(x.fcv.emap.elem) == 0


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


def _validate_iaxes(seq):
    """Checks that the independence axes of the sparse normal arrays in `seq`
    are compatible.
    
    Returns:
        `iaxid` of the final shape for the broadcasted arrays.
    """

    ndim = max(x.ndim for x in seq)
    iaxids = set((None,) * (ndim - x.ndim) + x._iaxid 
                 for x in seq if not _is_deterministic(x))
    # Reversing is to account for broadcasting - the independence 
    # axes numbers counted from the beginning of the array may not be 
    # the same for broadcastable arrys. When counted from the end, 
    # the axes numbers must always be the same as broadcasting 
    # cannot add new independence axes. 

    if len(iaxids) == 1:
        return iaxids.pop()
    elif len(iaxids) == 0:
        return (None,) * ndim
    
    # len > 1, which means that not all independence axes are identical. 
    # Next determine if the problem is the number, location, or order
    # of the axes, and show an appropriate error.

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

        raise ValueError("Incompatible numbers of the independence axes "
                         f"of the operands{valstr}.\n{msg}")

    get_iax_numbers = lambda ids: tuple([i for i, b in enumerate(ids) if b])
    locs = set(get_iax_numbers(ids) for ids in iaxids)
    if len(locs) > 1:
        if len(ns) < max_disp:
            valstr = (": " + ", ".join(str(loc) for loc in locs))
        else:
            valstr = ""
        
        raise ValueError("Incompatible locations of the independence axes "
                         f"of the operands{valstr}.\n{msg}")

    orders = set(tuple([ax for ax in ids if ax is not None]) for ids in iaxids)
    assert len(orders) > 1

    if len(ns) < max_disp:
        valstr = (": " + ", ".join(str(order) for order in orders))
    else:
        valstr = ""

    raise ValueError("Incompatible orders of the independence axes "
                     f"of the operands{valstr}.\n{msg}")
    

def _parse_index_key(x, key):
    """Validates the key and calculates the numbers of the input axes 
    contributing to each output axis."""

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
    return out_axs
    

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

    args = [assparsenormal(arg) for arg in args]

    if len(args) == 1:
        return args[0].cov()

    # The remaining case is len(args) == 2.
    x, y = args

    if _is_deterministic(x) and _is_deterministic(y):
        # This branch should never be reached under normal circumstances, 
        # because two constants are dispatched to the dense 
        # implementation of covariance.

        # It is not allowed for both inputs to be deterministic 
        # in this function, because the independence axes cannot 
        # be determined.
        raise ValueError
    
    # If either x or y is deterministic, a zero result is returned.

    elif _is_deterministic(x): 
        dense_shape_y = [s for b, s in zip(y._iaxid, y.shape) if not b]
        sparse_shape = [s for b, s in zip(y._iaxid, y.shape) if b]
        dense_shape_x = find_det_dense_shape(y, x)

        res_shape = tuple(dense_shape_x + dense_shape_y + sparse_shape)
        dt = np.result_type(x.fcv.a, y.fcv.a)
        return np.zeros(res_shape, dtype=dt)

    elif _is_deterministic(y):
        dense_shape_x = [s for b, s in zip(x._iaxid, x.shape) if not b]
        sparse_shape = [s for b, s in zip(x._iaxid, x.shape) if b]
        dense_shape_y = find_det_dense_shape(x, y)

        res_shape = tuple(dense_shape_x + dense_shape_y + sparse_shape)
        dt = np.result_type(x.fcv.a, y.fcv.a)
        return np.zeros(res_shape, dtype=dt)
    
    # The default case, when both variables are non-trivially random.

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
    ax, ay = [em.a for em in emaps.complete([x.fcv.emap, y.fcv.emap])]
    return np.einsum(subs, ax, ay.conj())


class SparseConditionWarning(RuntimeWarning):
    """The warning issued when a condition is skipped during 
    the conditioning of a sparse variable."""
    pass


# ---------- array functions ----------


def broadcast_to(x, shape):
    """Broadcasts a sparse normal variable to a new shape."""
    x = assparsenormal(x)
    fcv = normal_.broadcast_to(x.fcv, shape)
    iaxid = (None,) * (len(shape) - x.ndim) + x._iaxid
    return SparseNormal(fcv, iaxid)


def concatenate(arrays, axis=0):
    arrays = [assparsenormal(ar) for ar in arrays]
    iaxid = _validate_iaxes(arrays)
    axis = _normalize_axis(axis, len(iaxid))
    
    if iaxid[axis]:
        raise ValueError("Concatenation along independence axes "
                            "is not allowed.")

    fcv = normal_.concatenate([x.fcv for x in arrays], axis=axis)
    return SparseNormal(fcv, iaxid)


def stack(arrays, axis=0):
    arrays = [assparsenormal(ar) for ar in arrays]
    iaxid = _validate_iaxes(arrays)
    iaxid = iaxid[:axis] + (None,) + iaxid[axis:]

    fcv = normal_.stack([x.fcv for x in arrays], axis=axis)
    return SparseNormal(fcv, iaxid)


def _validate_iaxes_binary(op1, op2, op_axes1, op_axes2):
    """Checks that the operation axes are not independence axes 
    for `op1` and `op2`."""
    
    if any([op1._iaxid[ax] for ax in op_axes1]):
        raise ValueError("Bilinear operations contracting over "
                         "independence axes are not supported. "
                         f"Axes {op_axes1} of operand 1 are affected.")
    
    if any([op2._iaxid[ax] for ax in op_axes2]):
        raise ValueError("Bilinear operations contracting over "
                         "independence axes are not supported. "
                         f"Axes {op_axes2} of operand 2 are affected.")
    

def bilinearfunc(name, op1, op2, iaxid1, iaxid2, args=tuple(), pargs=tuple()):
    """ Applies a bilinear function specified by its numpy name 
    to two SparseNormal operands, `op1` and `op2`."""

    if not _is_deterministic(op1):
        fcv = normal_.bilinearfunc(name, op1.fcv, op2.fcv.mean(), args, pargs)
        v = SparseNormal(fcv, iaxid1)

        if not _is_deterministic(op2):
            v += bilinearfunc(name, op1.mean(), (op2 - op2.mean()), iaxid1, iaxid2, args, pargs)

    elif not _is_deterministic(op2):
        fcv = normal_.bilinearfunc(name, op1.fcv.mean(), op2.fcv, args, pargs)
        v = SparseNormal(fcv, iaxid2)

    else:
        x = getattr(np, name)(*pargs, op1.fcv.mean(), op2.fcv.mean(), *args)
        v = assparsenormal(x)

    return v


def dot(op1, op2):
    op1 = assparsenormal(op1)
    op2 = assparsenormal(op2)

    if op1.ndim == 0 or op2.ndim == 0:
        # No axes are contracted over.
        op_axes1 = tuple()
        op_axes2 = tuple()

        iaxid1 = op1._iaxid
        iaxid2 = op2._iaxid
    else:
        # Contraction over one axis.
        op_axes1 = (-1,)
        op_axes2 = (0,) if op2.ndim == 1 else (-2,)

        iaxid1 = op1._iaxid[:-1] + (None,) * (op2.ndim - 1)

        iaxid2 = list(op2._iaxid)
        iaxid2.pop(op_axes2[0])
        iaxid2 = (None,) * (op1.ndim - 1) + tuple(iaxid2)

    _validate_iaxes_binary(op1, op2, op_axes1, op_axes2)
    return bilinearfunc("dot", op1, op2, iaxid1, iaxid2)


def inner(op1, op2):
    op1 = assparsenormal(op1)
    op2 = assparsenormal(op2)

    if op1.ndim == 0 or op2.ndim == 0:
        op_axes1 = tuple()
        op_axes2 = tuple()

        iaxid1 = op1._iaxid
        iaxid2 = op2._iaxid
    else:
        op_axes1 = (-1,)
        op_axes2 = (-1,)

        iaxid1 = op1._iaxid[:-1] + (None,) * (op2.ndim - 1)
        iaxid2 = (None,) * (op1.ndim - 1) + op2._iaxid[:-1]

    _validate_iaxes_binary(op1, op2, op_axes1, op_axes2)
    return bilinearfunc("inner", op1, op2, iaxid1, iaxid2)


def outer(op1, op2):
    op1 = assparsenormal(op1).ravel()
    op2 = assparsenormal(op2).ravel()

    iaxid1 = op1._iaxid + (None,)
    iaxid2 = (None,) + op2._iaxid

    # iaxes are always valid.
    return bilinearfunc("outer", op1, op2, iaxid1, iaxid2)


def kron(op1, op2):
    op1 = assparsenormal(op1)
    op2 = assparsenormal(op2)

    ndim = max(op1.ndim, op2.ndim)  # ndim of the result.
    iaxid1 = (None,) * (ndim - op1.ndim) + op1._iaxid
    iaxid2 = (None,) * (ndim - op2.ndim) + op2._iaxid

    # iaxes are always valid.
    return bilinearfunc("kron", op1, op2, iaxid1, iaxid2)


def tensordot(op1, op2, axes=2):
    op1 = assparsenormal(op1)
    op2 = assparsenormal(op2)

    try:
        iter(axes)
    except Exception:
        op_axes1 = set(range(op1.ndim - axes, op1.ndim))
        op_axes2 = set(range(0, axes))
    else:
        op_axes1, op_axes2 = axes
    # This follows how numpy.tensordot handles the axes.

    iaxid1 = tuple([b for i, b in enumerate(op1._iaxid) if i not in op_axes1])
    iaxid1 = iaxid1 + (None,) * (op2.ndim - len(op_axes2))

    iaxid2 = tuple([b for i, b in enumerate(op2._iaxid) if i not in op_axes2])
    iaxid2 = (None,) * (op1.ndim - len(op_axes1)) + iaxid2

    _validate_iaxes_binary(op1, op2, op_axes1, op_axes2)
    return bilinearfunc("tensordot", op1, op2, iaxid1, iaxid2, args=[axes])
    

def einsum(subs, op1, op2):
    def out_iaxes(op, insubs, outsubs):
        """Calculates the indices of the independence axes for
        the output operand."""
        
        for i, c in enumerate(insubs):
            if op._iaxid[i] and c not in outsubs:
                raise ValueError("Contraction over an independence"
                                 f" axis ({i}).")
            
        iaxid = op._iaxid + (None,)  # Augments with a default.  
        return tuple([iaxid[insubs.find(c)] for c in outsubs])
    
    op1 = assparsenormal(op1)
    op2 = assparsenormal(op2)

    # Converts the subscripts to an explicit form.
    (insu1, insu2), outsu = einsubs.parse(subs, (op1.shape, op2.shape))
    subs = f"{insu1},{insu2}->{outsu}"

    iaxid1 = out_iaxes(op1, insu1, outsu)
    iaxid2 = out_iaxes(op2, insu2, outsu)

    return bilinearfunc("einsum", op1, op2, iaxid1, iaxid2, pargs=[subs])


def fftfunc(name, x, n, axis, norm):
    raise NotImplementedError


def fftfunc_n(name, x, s, axes, norm):
    raise NotImplementedError


def call_linearized(x, func, jmpfunc):
    x = assparsenormal(x)
    fcv = normal_.call_linearized(x.fcv, func, jmpfunc)
    return SparseNormal(fcv, x._iaxid)