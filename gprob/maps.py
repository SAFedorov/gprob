from importlib import import_module
import numpy as np

from . import latent
from .external import einsubs


NUMERIC_ARRAY_KINDS = {"b", "i", "u", "f", "c"}


class LatentMap:
    """An affine map of latent random variables,
    
    x[...] = sum_k a[i...] xi[i] + b[...],
    
    where `xi`s are the independent identically-distributed latent Gaussian 
    random variables, `xi[i] ~ N(0, 1)` for all `i`, and `...` is an arbitrary 
    multi-dimensional index.
    """

    __slots__ = ("a", "b", "lat")
    __array_ufunc__ = None
    _mod = import_module(__name__)

    def __init__(self, a, b, lat=None):
        if a.shape[1:] != b.shape:
            raise ValueError(f"The shapes of the map ({a.shape}) and "
                             f"the mean ({b.shape}) do not agree.")

        if lat is None:
            lat = latent.create(a.shape[0])
        elif len(lat) != a.shape[0]:
            raise ValueError(f"The number of latent variables ({len(lat)}) "
                             "does not match the outer dimension of `a` "
                             f"({a.shape[0]}).")
        self.a = a
        self.b = b
        self.lat = lat  # Dictionary of latent variables {id -> k, ...}.

    @property
    def size(self):
        return self.b.size
    
    @property
    def shape(self):
        return self.b.shape
    
    @property
    def ndim(self):
        return self.b.ndim
    
    @property
    def nlat(self):
        return len(self.lat)
    
    @property
    def delta(self):  # TODO: add tests for this in test_property -------------------------------------------
        return self.__class__(self.a, np.zeros_like(self.b), self.lat)
    
    @property
    def real(self):
        return self.__class__(self.a.real, self.b.real, self.lat)  # TODO: Change to _updated_copy to avoid code duplication in sparse ------- 
    
    @property
    def imag(self):
        return self.__class__(self.a.imag, self.b.imag, self.lat)
    
    @property
    def T(self):
        return self.transpose()
    
    @property
    def iscomplex(self):
        return (np.iscomplexobj(self.a) or np.iscomplexobj(self.b))
    
    @property
    def a2d(self):
        return np.ascontiguousarray(self.a.reshape((self.nlat, self.size)))  # TODO: is this property actually needed? -------------------

    def __len__(self):
        if self.ndim == 0:
            raise TypeError("Scalar arrays have no lengh.")
        return len(self.b)

    def __neg__(self):
        return self.__class__(-self.a, -self.b, self.lat)

    def __add__(self, other):
        other, isnumeric = match_(self.__class__, other)

        if isnumeric:
            b = self.b + other
            a = self.a if self.shape == b.shape else _broadcast(self.a, b.shape)
            return self.__class__(a, b, self.lat)

        b = self.b + other.b

        if self.lat is other.lat:
            # An optimization primarily made to speed up in-place 
            # additions to array elements.

            a = _unsq(self.a, other.ndim) + _unsq(other.a, self.ndim)
            return self.__class__(a, b, self.lat)
    
        op1, op2 = self, other
        lat, swapped = latent.ounion(op1.lat, op2.lat)
        
        if swapped:
            op1, op2 = op2, op1

        a = np.zeros((len(lat),) + b.shape, 
                     dtype=np.promote_types(op1.a.dtype, op2.a.dtype))
        a[:len(op1.lat)] = _unsq(op1.a, b.ndim)
        idx = [lat[k] for k in op2.lat]
        a[idx] += _unsq(op2.a, b.ndim)

        return self.__class__(a, b, lat)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        other, _ = match_(self.__class__, other)
        return self.__add__(-other)
    
    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        other, isnumeric = match_(self.__class__, other)

        if isnumeric:
            b = self.b * other
            a = _unsq(self.a, other.ndim) * other
            return self.__class__(a, b, self.lat)

        # Linearized product  x * y = <x><y> + <y>dx + <x>dy,
        # for  x = <x> + dx  and  y = <y> + dy.

        b = self.b * other.b
        lat, [sa, oa] = complete([self, other])
        a = _unsq(sa, b.ndim) * other.b + _unsq(oa, b.ndim) * self.b
        return self.__class__(a, b, lat)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        other, isnumeric = match_(self.__class__, other)

        if isnumeric:
            b = self.b / other
            a = _unsq(self.a, other.ndim) / other
            return self.__class__(a, b, self.lat)

        # Linearized fraction  x/y = <x>/<y> + dx/<y> - dy<x>/<y>^2,
        # for  x = <x> + dx  and  y = <y> + dy.

        b = self.b / other.b
        lat, [sa, oa] = complete([self, other])
        a = (_unsq(sa, b.ndim) / other.b 
             + _unsq(oa, b.ndim) * ((-self.b) / other.b**2))
        return self.__class__(a, b, lat)
    
    def __rtruediv__(self, other):
        other, isnumeric = match_(self.__class__, other)

        if isnumeric:
            b = other / self.b
            a = _unsq(self.a, b.ndim) * ((-other) / self.b**2)
            return self.__class__(a, b, self.lat)
        
        # `other` has been converted to a map, but was not a map initially.
        return other / self

    def __matmul__(self, other):
        other, isnumeric = match_(self.__class__, other)

        if isnumeric:
            b = self.b @ other
            a = _unsq(self.a, other.ndim) @ other
            if self.ndim == 1 and other.ndim > 1:
                a = np.squeeze(a, axis=-2)
            return self.__class__(a, b, self.lat)
        
        return self @ other.b + self.b @ other.delta

    def __rmatmul__(self, other):
        other, isnumeric = match_(self.__class__, other)

        if isnumeric:
            b = other @ self.b
            if self.ndim > 1:
                a = other @ _unsq(self.a, other.ndim)
            else:
                a_ = other @ _unsq(self.a[..., None], other.ndim)
                a = np.squeeze(a_, axis=-1)
            return self.__class__(a, b, self.lat)

        return other @ self.b + other.b @ self.delta
    
    def __pow__(self, other):
        other, isnumeric = match_(self.__class__, other)

        if isnumeric:
            b = self.b ** other
            a_ = _unsq(self.a, b.ndim) 
            a = a_ * (other * (self.b ** np.where(other, other-1, 1.)))
            return self.__class__(a, b, self.lat)
        
        # x^y = <x>^<y> + dx <y> <x>^(<y>-1) + dy ln(<x>) <x>^<y>

        b = self.b ** other.b
        d1 = self.delta * (other.b * self.b ** np.where(other.b, other.b-1, 1.))
        d2 = other.delta * (np.log(np.where(self.b, self.b, 1.)) * b)
        return b + d1 + d2

    def __rpow__(self, other):
        other, isnumeric = match_(self.__class__, other)

        if isnumeric:
            # x^y = <x>^<y> + dy ln(<x>) <x>^<y>

            b = other ** self.b
            return b + self.delta * (np.log(np.where(other, other, 1.)) * b)

        return other ** self
    
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        
        key_a = (slice(None),) + key
        return self.__class__(self.a[key_a], self.b[key], self.lat)
    
    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            key = (key,)

        if not self.b.flags.writeable:
            self.b = self.b.copy()
        
        if not self.a.flags.writeable:
            self.a = self.a.copy()

        value = self._mod.lift(self.__class__, value)
        self.b[key] = value.b

        key_a = (slice(None),) + key
        if self.lat is not value.lat:
            self.lat, [self.a, av] = complete([self, value])
        else:
            av = value.a
        self.a[key_a] = _unsq(av, self.b[key].ndim)
    
    # ---------- array methods ----------
    
    def conjugate(self):
        return self.__class__(self.a.conj(), self.b.conj(), self.lat)
    
    def conj(self):
        return self.conjugate()
    
    def cumsum(self, axis=None):
        b = np.cumsum(self.b, axis=axis)

        if axis is None:
            a = self.a.reshape((self.nlat, b.size))
            axis_a = 1
        else:
            a = self.a

            if a.ndim < 2:
                a = a.reshape((a.shape[0], 1))

            axis_a, = _axes_a([axis])

        a = np.cumsum(a, axis=axis_a)
        return self.__class__(a, b, self.lat)
    
    def diagonal(self, offset=0, axis1=0, axis2=1):
        b = self.b.diagonal(offset=offset, axis1=axis1, axis2=axis2)
        axis1_a, axis2_a = _axes_a([axis1, axis2])
        a = self.a.diagonal(offset=offset, axis1=axis1_a, axis2=axis2_a)
        return self.__class__(a, b, self.lat)
    
    def flatten(self, order="C"):
        if order not in ["C", "F"]:
            raise ValueError("Only C and F orders are supported.")
        
        b = self.b.flatten(order=order)
        a = self.a.reshape((self.nlat, b.size), order=order).copy()
        return self.__class__(a, b, self.lat)
    
    def moveaxis(self, source, destination):
        b = np.moveaxis(self.b, source, destination)
        a = np.moveaxis(self.a, *_axes_a([source, destination]))
        return self.__class__(a, b, self.lat)
    
    def ravel(self, order="C"):
        b = self.b.ravel(order=order)

        if order == "C":
            return self.__class__(self.a2d, b, self.lat)
        elif order == "F":
            a = self.a.reshape((self.nlat, b.size), order="F")
            return self.__class__(np.asfortranarray(a), b, self.lat)

        raise ValueError("Only C and F orders are supported.")
    
    def reshape(self, newshape, order="C"):
        b = self.b.reshape(newshape, order=order)
        a = self.a.reshape((self.nlat,) + b.shape, order=order)
        return self.__class__(a, b, self.lat)
    
    def sum(self, axis=None, keepdims=False):
        # "where" is absent because its broadcasting is not implemented.
        # "initial" is also not implemented.
        b = self.b.sum(axis, keepdims=keepdims)

        if axis is None or self.ndim == 0:
            axis = tuple(range(self.ndim))
        elif not isinstance(axis, tuple):
            axis = (axis,)

        a = self.a.sum(tuple(_axes_a(axis)), keepdims=keepdims)
        return self.__class__(a, b, self.lat)
    
    def transpose(self, axes=None):
        b = self.b.transpose(axes)

        if axes is None:
            axes = range(self.ndim)[::-1]

        a = self.a.transpose((0, *_axes_a(axes)))
        return self.__class__(a, b, self.lat)
    
    def trace(self, offset=0, axis1=0, axis2=1):
        b = self.b.trace(offset=offset, axis1=axis1, axis2=axis2)
        axis1_a, axis2_a = _axes_a([axis1, axis2])
        a = self.a.trace(offset=offset, axis1=axis1_a, axis2=axis2_a)
        return self.__class__(a, b, self.lat)
    
    def split(self, indices_or_sections, axis=0):
        bs = np.split(self.b, indices_or_sections, axis=axis)
        as_ = np.split(self.a, indices_or_sections, axis=_axes_a([axis])[0])
        return [self.__class__(a, b, self.lat) for a, b in zip(as_, bs)]
    
    def broadcast_to(self, shape):
        if self.shape == shape:  # TODO: remove this optimization after checking that it is not critical ----
            return self
        
        b = np.broadcast_to(self.b, shape)
        a = _broadcast(self.a, shape)
        return self.__class__(a, b, self.lat)


def _axes_a(axes_b):
    return [a + 1 if a >= 0 else a for a in axes_b]


def _unsq(a, ndim):
    """Unsqueezes `a` so that it can be broadcasted to 
    the map dimension `ndim`."""

    dn = ndim - a.ndim + 1
    if dn <= 0:
        return a
    
    sh = list(a.shape)
    sh[1:1] = [1] * dn
    return a.reshape(sh)


def _broadcast(a, shape):
    return np.broadcast_to(_unsq(a, len(shape)), (a.shape[0],) + shape)


def complete(ops): # TODO : update docs, ---------------------------------------------------------
    """Extends the maps to the union of their latent variables."""

    def extend_a(x, new_lat):
        """Extends the map `x` to a new list of latent variables `new_lat` by
        adding zero entries. All the existing variables from `x.lat` must
        be present in `new_lat` (in arbitrary order), 
        otherwise the function will fail."""
        
        new_shape = (len(new_lat),) + x.a.shape[1:]
        new_a = np.zeros(new_shape, dtype=x.a.dtype)
        idx = [new_lat[k] for k in x.lat]
        new_a[idx] = x.a
        return new_a

    def pad_a(x, new_lat):
        """Extends the map `x` to a new list of latent variables `new_lat` by
        adding zero entries. This function assumes that `new_lat` contains 
        all the existing variables from `x.lat` in the same order
        in the beginning, and therefore the map should be just padded 
        with an appropriate number of zeros."""

        new_a = np.zeros((len(new_lat),) + x.a.shape[1:], dtype=x.a.dtype)
        new_a[:len(x.lat)] = x.a        
        return new_a

    if len(ops) == 1:
        return ops[0].lat, [ops[0].a]
    
    if len(ops) > 2:
        lat = latent.uunion(*[x.lat for x in ops])
        return lat, [extend_a(op, lat) for op in ops]

    # The rest is an optimization for the case of two operands.
    op1, op2 = ops

    if op1.lat is op2.lat:
        return op1.lat, [op1.a, op2.a]
    
    lat, swapped = latent.ounion(op1.lat, op2.lat)

    if swapped:
        return lat, [extend_a(op1, lat), pad_a(op2, lat)]
    
    return lat, [pad_a(op1, lat), extend_a(op2, lat)]
    
    
def lift(cls, x):
    """Converts `x` to a varaible of class `cls`. If `x` is such a variable 
    already, returns it unchanged. If the conversion cannot be done, 
    raises a `TypeError`."""

    if x.__class__ is cls:
        return x
    
    if issubclass(cls, x.__class__):
        return cls(x.a, x.b, x.lat)

    x_ = np.asanyarray(x)
    if x_.dtype.kind in NUMERIC_ARRAY_KINDS:
        a = np.zeros((0,) + x_.shape, dtype=x_.dtype)
        return cls(a, x_, dict())
    elif x_.ndim != 0:
        return cls._mod.stack(cls, [cls._mod.lift(cls, v) for v in x])
    
    raise TypeError(f"The variable of type '{x.__class__.__name__}' "
                    f"cannot be promoted to type '{cls.__name__}'.")


def match_(cls, x):
    """Converts `x` to either a numeric array or a variable of class `cls`, 
    and returns the converted variable with its type.

    Args:
        other: Object to be converted.
    
    Returns:
        Tuple `(converted_x, isnumeric)`. `isnumeric` is a bool flag.
    """
    
    if x.__class__ is cls:
        return x, False

    x_ = np.asanyarray(x)
    if x_.dtype.kind in NUMERIC_ARRAY_KINDS:
        return x_, True

    return cls._mod.lift(cls, x), False


def concatenate(cls, arrays, axis=0):
    b = np.concatenate([x.b for x in arrays], axis=axis)

    axis = axis if axis >= 0 else b.ndim + axis

    if len(arrays) == 1:
        return arrays[0]
    elif len(arrays) == 2 and arrays[0].lat is arrays[1].lat:
        # An optimization targeting uses like concatenate([x.real, x.imag]).
        op1, op2 = arrays
        a = np.concatenate([op1.a, op2.a], axis=axis+1)
        return cls(a, b, op1.lat)

    dims = [x.a.shape[axis+1] for x in arrays]
    base_jidx = (slice(None),) * axis

    if len(arrays) > 2:
        union_lat = latent.uunion(*[x.lat for x in arrays])
        dtype = np.result_type(*[x.a for x in arrays])
        a = np.zeros((len(union_lat),) + b.shape, dtype)
        n1 = 0
        for i, x in enumerate(arrays):
            idx = [union_lat[k] for k in x.lat]
            n2 = n1 + dims[i]
            a[idx, *base_jidx, n1: n2] = x.a
            n1 = n2

        return cls(a, b, union_lat)
    
    # The rest is an optimization for the general case of two operands.
    op1, op2 = arrays

    # Indices along the variable dimension.
    jidx1 = base_jidx + (slice(None, dims[0]),)
    jidx2 = base_jidx + (slice(-dims[1], None),)

    union_lat, swapped = latent.ounion(op1.lat, op2.lat)

    if swapped:
        op1, op2 = op2, op1
        jidx1, jidx2 = jidx2, jidx1
    
    a = np.zeros((len(union_lat),) + b.shape, 
                 np.promote_types(op1.a.dtype, op2.a.dtype))
    a[:len(op1.lat), *jidx1] = op1.a
    idx = [union_lat[k] for k in op2.lat]
    a[idx, *jidx2] = op2.a

    return cls(a, b, union_lat)


def stack(cls, arrays, axis=0):
    # Essentially a copy of `concatenate`, with slightly less overhead.

    b = np.stack([x.b for x in arrays], axis=axis)

    axis = axis if axis >= 0 else b.ndim + axis
    base_jidx = (slice(None),) * axis

    if len(arrays) == 1:
        return arrays[0][*base_jidx, None]
    elif len(arrays) == 2 and arrays[0].lat is arrays[1].lat:
        # An optimization targeting uses like stack([x.real, x.imag]).
        op1, op2 = arrays
        a = np.stack([op1.a, op2.a], axis=axis+1)
        return cls(a, b, op1.lat)

    if len(arrays) > 2:
        union_lat = latent.uunion(*[x.lat for x in arrays])
        dtype = np.result_type(*[x.a for x in arrays])
        a = np.zeros((len(union_lat),) + b.shape, dtype)
        for i, x in enumerate(arrays):
            idx = [union_lat[k] for k in x.lat]
            a[idx, *base_jidx, i] = x.a

        return cls(a, b, union_lat)
    
    # The rest is an optimization for the general case of two operands.
    op1, op2 = arrays
    j1, j2 = 0, 1

    union_lat, swapped = latent.ounion(op1.lat, op2.lat)

    if swapped:
        op1, op2 = op2, op1
        j1, j2 = j2, j1
    
    a = np.zeros((len(union_lat),) + b.shape, 
                 np.promote_types(op1.a.dtype, op2.a.dtype))
    a[:len(op1.lat), *base_jidx, j1] = op1.a
    idx = [union_lat[k] for k in op2.lat]
    a[idx, *base_jidx, j2] = op2.a

    return cls(a, b, union_lat)


def call_linearized(cls, x, func, jmpfunc):
    x = cls._mod.lift(cls, x)
    b = func(x.b)
    delta = jmpfunc(x.b, b, x.delta)
    return delta + b


def fftfunc(cls, name, x, n, axis, norm):
    func = getattr(np.fft, name)
    b = func(x.b, n, axis, norm)
    a = func(x.a, n, _axes_a([axis])[0], norm)
    return cls(a, b, x.lat)


def fftfunc_n(cls, name, x, s, axes, norm):
    if axes is None:
        axes = list(range(x.ndim))

    func = getattr(np.fft, name)
    b = func(x.b, s, axes, norm)
    a = func(x.a, s, _axes_a(axes), norm)
    return cls(a, b, x.lat)


# ---------- bilinear functions ----------


def bilinearfunc(cls, name, x, y, args=tuple(), pargs=tuple()):
    x, x_is_numeric = match_(cls, x)
    y, y_is_numeric = match_(cls, y)

    if not x_is_numeric and y_is_numeric:
        return getattr(cls._mod, name + "_1")(cls, *pargs, x, y, *args)
    elif x_is_numeric and not y_is_numeric:
        return getattr(cls._mod, name + "_2")(cls, *pargs, x, y, *args)
    elif x_is_numeric and y_is_numeric:
        return getattr(np, name)(*pargs, x, y, *args)

    return (getattr(cls._mod, name + "_1")(cls, *pargs, x, y.b, *args) 
            + getattr(cls._mod, name + "_2")(cls, *pargs, x.b, y.delta, *args))


def einsum_1(cls, subs, x, y):
    (xsubs, ysubs), outsubs = einsubs.parse(subs, (x.shape, y.shape))

    b = np.einsum(subs, x.b, y)
    a = np.einsum(f"...{xsubs}, {ysubs} -> ...{outsubs}", x.a, y)
    return cls(a, b, x.lat)


def einsum_2(cls, subs, x, y):
    (xsubs, ysubs), outsubs = einsubs.parse(subs, (x.shape, y.shape))

    b = np.einsum(subs, x, y.b)
    a = np.einsum(f"{xsubs}, ...{ysubs} -> ...{outsubs}", x, y.a)
    return cls(a, b, y.lat)


def inner_1(cls, x, y):
    if x.ndim == 0 or y.ndim == 0:
        return x * y
    
    b = np.inner(x.b, y)
    a = np.inner(x.a, y)
    return cls(a, b, x.lat)


def inner_2(cls, x, y):
    if x.ndim == 0 or y.ndim == 0:
        return x * y
    
    b = np.inner(x, y.b)
    a = np.moveaxis(np.inner(x, y.a), x.ndim - 1, 0)  # TODO: change to einsum? --------------------
    return cls(a, b, y.lat)


def dot_1(cls, x, y):
    if x.ndim == 0 or y.ndim == 0:
        return x * y
    
    b = np.dot(x.b, y)
    a = np.dot(x.a, y)
    return cls(a, b, x.lat)


def dot_2(cls, x, y):
    if x.ndim == 0 or y.ndim == 0:
        return x * y
    
    b = np.dot(x, y.b)
    if y.ndim == 1:
        a = np.einsum("...j, ij -> i...", x, y.a)
    else:
        a = np.moveaxis(np.dot(x, y.a), x.ndim - 1, 0)
    
    return cls(a, b, y.lat)


def outer_1(cls, x, y):
    b = np.outer(x.b, y)
    a = np.einsum("ij, k -> ijk", x.a2d, y.ravel())
    return cls(a, b, x.lat)


def outer_2(cls, x, y):
    b = np.outer(x, y.b)
    a = np.einsum("k, ij -> ikj", x.ravel(), y.a2d)
    return cls(a, b, y.lat)


def kron_1(cls, x, y):
    b = np.kron(x.b, y)
    a = np.kron(_unsq(x.a, y.ndim), y)
    return cls(a, b, x.lat)


def kron_2(cls, x, y):
    b = np.kron(x, y.b)
    a = np.kron(x, _unsq(y.a, x.ndim))
    return cls(a, b, y.lat)


def complete_tensordot_axes(axes):
    """Converts `axes` to an explicit form compatible with `numpy.tensordot` 
    function. If `axes` is a sequence, the function returns it unchanged, 
    and if `axes` is an integer `n`, it returns a tuple of lists
    `([-n, -n + 1, ..., -1], [0, 1, ..., n-1])`."""

    try:
        iter(axes)
    except Exception:
        return list(range(-axes, 0)), list(range(0, axes))
    return axes


def tensordot_1(cls, x, y, axes):
    b = np.tensordot(x.b, y, axes)
    axes1, axes2 = complete_tensordot_axes(axes)
    a = np.tensordot(x.a, y, axes=(_axes_a(axes1), axes2))
    return cls(a, b, x.lat)


def tensordot_2(cls, x, y, axes):
    b = np.tensordot(x, y.b, axes)
    axes1, axes2 = complete_tensordot_axes(axes)
    a_ = np.tensordot(x, y.a, axes=(axes1, _axes_a(axes2)))
    a = np.moveaxis(a_, -y.ndim - 1 + len(axes2), 0)
    return cls(a, b, y.lat)