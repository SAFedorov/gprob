from importlib import import_module
import numpy as np

from . import latent
from .external import einsubs


NUMERIC_ARRAY_KINDS = {"b", "i", "u", "f", "c"}


class LatentMap:
    """An affine map of latent variables."""

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
    def delta(self):
        return self.__class__(self.a, np.zeros_like(self.b), self.lat)
    
    @property
    def real(self):
        return self.__class__(self.a.real, self.b.real, self.lat)
    
    @property
    def imag(self):
        return self.__class__(self.a.imag, self.b.imag, self.lat)
    
    @property
    def T(self):
        return self.transpose()
    
    @property
    def iscomplex(self):
        return (np.iscomplexobj(self.a) or np.iscomplexobj(self.b))

    def __len__(self):
        if self.ndim == 0:
            raise TypeError("Scalar arrays have no lengh.")
        return len(self.b)

    def __neg__(self):
        return self.__class__(-self.a, -self.b, self.lat)

    def __add__(self, other):
        try:
            other, isnumeric = match_(self.__class__, other)
        except TypeError:
            return NotImplemented

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
    
        x, y = self, other
        lat, swapped = latent.ounion(x.lat, y.lat)
        
        if swapped:
            x, y = y, x

        a = np.zeros((len(lat),) + b.shape, 
                     dtype=np.promote_types(x.a.dtype, y.a.dtype))
        a[:len(x.lat)] = _unsq(x.a, b.ndim)
        idx = [lat[k] for k in y.lat]
        a[idx] += _unsq(y.a, b.ndim)

        return self.__class__(a, b, lat)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        try:
            other, _ = match_(self.__class__, other)
        except TypeError:
            return NotImplemented
        
        return self.__add__(-other)
    
    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        try:
            other, isnumeric = match_(self.__class__, other)
        except TypeError:
            return NotImplemented

        if isnumeric:
            b = self.b * other
            a = _unsq(self.a, other.ndim) * other
            return self.__class__(a, b, self.lat)

        # Linearized product  x * y = <x><y> + <y>dx + <x>dy,
        # for  x = <x> + dx  and  y = <y> + dy.
        return self * other.b + other.delta * self.b
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        try:
            other, isnumeric = match_(self.__class__, other)
        except TypeError:
            return NotImplemented

        if isnumeric:
            b = self.b / other
            a = _unsq(self.a, other.ndim) / other
            return self.__class__(a, b, self.lat)

        # Linearized fraction  x/y = <x>/<y> + dx/<y> - dy<x>/<y>^2,
        # for  x = <x> + dx  and  y = <y> + dy.
        return self / other.b + other.delta * (-self.b / other.b**2)
    
    def __rtruediv__(self, other):
        try:
            other, isnumeric = match_(self.__class__, other)
        except TypeError:
            return NotImplemented

        if isnumeric:
            b = other / self.b
            a = _unsq(self.a, b.ndim) * ((-other) / self.b**2)
            return self.__class__(a, b, self.lat)
        
        # ``other`` has been converted to a map, but was not a map initially.
        return other / self

    def __matmul__(self, other):
        try:
            other, isnumeric = match_(self.__class__, other)
        except TypeError:
            return NotImplemented

        if isnumeric:
            b = self.b @ other
            a = _unsq(self.a, other.ndim) @ other
            if self.ndim == 1 and other.ndim > 1:
                a = np.squeeze(a, axis=-2)
            return self.__class__(a, b, self.lat)
        
        return self @ other.b + self.b @ other.delta

    def __rmatmul__(self, other):
        try:
            other, isnumeric = match_(self.__class__, other)
        except TypeError:
            return NotImplemented

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
        try:
            other, isnumeric = match_(self.__class__, other)
        except TypeError:
            return NotImplemented

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
        try:
            other, isnumeric = match_(self.__class__, other)
        except TypeError:
            return NotImplemented

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

        value = self._mod.lift(self.__class__, value)
        in_place = (self.b.flags.writeable and self.a.flags.writeable 
                    and self.lat is value.lat)

        if not in_place:
            self.b = self.b.copy()

        self.b[key] = value.b

        if self.lat is not value.lat:
            self.lat, [self.a, av] = complete([self, value])
        else:
            av = value.a
            
            if not in_place:
                self.a = self.a.copy()

        key_a = (slice(None),) + key
        self.a[key_a] = _unsq(av, self.b[key].ndim)
    
    def conjugate(self):
        """Element-wise complex conjugate."""
        return self.__class__(self.a.conj(), self.b.conj(), self.lat)
    
    def conj(self):
        """Element-wise complex conjugate."""
        return self.conjugate()
    
    def cumsum(self, axis=None):
        """Computes the cumulative sum of the elements.

        Args:
            axis (int or None): 
                Axis along which the cumulative sum is computed. If ``None``, 
                the cumulative sum is computed over the flattened array.

        Returns:
            A new random variable representing the result of the cumulative 
            summation with the same dimensions as the input.
        """
        
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
        """Extracts a diagonal from the varaible.

        Args:
            offset (int): 
                The offset of the diagonal from the main diagonal.
            axis1 (int): 
                The first axis along which the diagonal should be taken.
            axis2 (int): 
                The second axis along which the diagonal should be taken.

        Returns:
            A new random variable consisting of the extracted diagonal elements.
        """
        
        b = self.b.diagonal(offset=offset, axis1=axis1, axis2=axis2)
        axis1_a, axis2_a = _axes_a([axis1, axis2])
        a = self.a.diagonal(offset=offset, axis1=axis1_a, axis2=axis2_a)
        return self.__class__(a, b, self.lat)
    
    def flatten(self, order="C"):
        """Flattens the random variable. 
        
        ``x.flatten()`` is equivalen to ``x.reshape((-1,))``.

        Args:
            order (str): 
                The order in which the input array elements are read.
                - 'C': C-style row-major order.
                - 'F': Fortran-style column-major order.

        Returns:
            A new one-dimensional random variable.
        """
        
        b = self.b.reshape((-1,), order=order)
        a = self.a.reshape((self.nlat, b.size), order=order)
        return self.__class__(a, b, self.lat)
    
    def moveaxis(self, source, destination):
        """Moves an axis to a new position.

        Args:
            source (int):
                The original position of the axis.
            destination (int):
                The destination position of the axis.

        Returns:
            A new random variable with the transformed layout.
        """

        b = np.moveaxis(self.b, source, destination)
        a = np.moveaxis(self.a, *_axes_a([source, destination]))
        return self.__class__(a, b, self.lat)
    
    def ravel(self, order="C"):
        """Flattens the random variable while ensuring that the underying map 
        is stored contiguously in the memory. The map arrays of the returned 
        variable are views of the map arrays of the original variable whenever 
        possible. 

        Args:
            order (str): 
                The order in which the input array elements are read.
                - 'C': C-style row-major order.
                - 'F': Fortran-style column-major order.

        Returns:
            A new one-dimensional random variable.
        """
        
        b = self.b.ravel(order=order)

        if order == "C":
            return self.__class__(a2d(self), b, self.lat)
        elif order == "F":
            a = self.a.reshape((self.nlat, b.size), order="F")
            return self.__class__(np.asfortranarray(a), b, self.lat)

        raise ValueError("Only C and F orders are supported.")
    
    def reshape(self, newshape, order="C"):
        """Gives the variable a new shape.

        Args:
            newshape (tuple of int): 
                The new shape. One dimension can be set to ``-1`` to infer its
                size from the total number of elements and the other dimensions.
            order (str): 
                The order in which the array elements are read.
                - 'C': C-style row-major order.
                - 'F': Fortran-style column-major order.

        Returns:
            A new random variable with the specified shape and order.
        """

        b = self.b.reshape(newshape, order=order)
        a = self.a.reshape((self.nlat,) + b.shape, order=order)
        return self.__class__(a, b, self.lat)
    
    def squeeze(self, axis=None):
        """Removes axis or axes of length one from the variable.
        
        Args:
            axis (None, int, or tuple of ints):
                The axis or axes to be removed. If ``None``, all axes of 
                length one are removed.
        
        Returns:
            A new random variable with the shape identical to that of the 
            original, except the removed axes.  
        """
        
        b = self.b.squeeze(axis=axis)
        a = self.a.reshape((self.nlat,) + b.shape)
        return self.__class__(a, b, self.lat)
    
    def sum(self, axis=None, keepdims=False):
        """Computes the sum of all elements of the variable along an axis 
        or axes.

        Args:
            axis (int, tuple of int, or None): 
                Axis or axes along which to sum the elements. If ``None``, 
                the function sums all elements.
            keepdims (bool): 
                If ``True``, the reduced axes are retained in the output 
                as dimensions with size one. This enables the result 
                to broadcast against the input array.

        Returns:
            A new random variable representing the result of the summation. 
            The output variable has the same dimensions as the input, except 
            the summation axes are removed unless ``keepdims`` is ``True``.
        """
        
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
        """Permutes the axes of the variable.

        Args:
            axes (tuple of int or None): 
                The desired axes order. If ``None``, the existing axes 
                order is reversed.

        Returns:
            A new random variable with the axes in the new order.
        """
        
        b = self.b.transpose(axes)

        if axes is None:
            axes = range(self.ndim)[::-1]

        a = self.a.transpose((0, *_axes_a(axes)))
        return self.__class__(a, b, self.lat)
    
    def trace(self, offset=0, axis1=0, axis2=1):
        """Calculates the sum of the diagonal elements of the variable.

        Args:
            offset (int): 
                The offset of the diagonal to be summed from the main diagonal.
            axis1 (int): 
                The first axis along which the diagonal should be summed.
            axis2 (int): 
                The second axis along which the diagonal should be summed.

        Returns:
            A new random variable, consisting of the sum(s) of the diagonal 
            elements with respect to the specified axes.
        """

        b = self.b.trace(offset=offset, axis1=axis1, axis2=axis2)
        axis1_a, axis2_a = _axes_a([axis1, axis2])
        a = self.a.trace(offset=offset, axis1=axis1_a, axis2=axis2_a)
        return self.__class__(a, b, self.lat)
    
    def split(self, indices_or_sections, axis=0):
        """Splits the variable along an axis.

        Args:
            indices_or_sections (int or sequence of int):
                - If an integer, n, the input variable is to be divided along
                  ``axis`` into n equal pieces.
                - If a sequence of sorted integers, its entries indicate where 
                  along ``axis`` the input variable is to be split.
            axis (int):
                The axis along which the variable is to be split.

        Returns:
            A list of new random variables into which the original is split.
        """

        if self.ndim == 0:
            raise ValueError("Scalar variables cannot be split.")
        
        bs = np.split(self.b, indices_or_sections, axis=axis)
        as_ = np.split(self.a, indices_or_sections, axis=_axes_a([axis])[0])
        return [self.__class__(a, b, self.lat) for a, b in zip(as_, bs)]
    
    def broadcast_to(self, shape):
        """Broadcasts the variable to a new shape.

        Args:
            shape (tuple of int): 
                The desired shape to broadcast the variable to.

        Returns:
            A new random variable with the specified shape consisting
            of duplicates of the input variable.
        """
        
        b = np.broadcast_to(self.b, shape)
        a = _broadcast(self.a, shape)
        return self.__class__(a, b, self.lat)


def _axes_a(axes_b):
    return [a + 1 if a >= 0 else a for a in axes_b]


def _unsq(a, ndim):
    """Unsqueezes ``a`` so that it can be broadcasted to 
    the map dimension ``ndim``."""

    dn = ndim - a.ndim + 1
    if dn <= 0:
        return a
    
    sh = list(a.shape)
    sh[1:1] = [1] * dn
    return a.reshape(sh)


def a2d(x):
    return np.ascontiguousarray(x.a.reshape((x.nlat, x.size)))


def _broadcast(a, shape):
    return np.broadcast_to(_unsq(a, len(shape)), (a.shape[0],) + shape)


def complete(seq):
    """Extends the transformation arrays of each latent map in 
    the sequence ``seq`` to the union of their latent variables.
    
    Returns:
        ``(union_lat, [a1, a2, ...])``, where ``union_lat`` is a combined 
        dictionary of latent variables, and ``[a1, a2, ...]`` is a list of 
        new map arrays for each map in ``seq`` extended to ``union_lat`` by 
        zero padding.
    """

    def extend_a(x, new_lat):
        """Extends the map ``x`` to a new list of latent variables ``new_lat``
        by adding zero entries. All the existing variables from ``x.lat`` must
        be present in ``new_lat`` (in arbitrary order)."""
        
        new_shape = (len(new_lat),) + x.a.shape[1:]
        new_a = np.zeros(new_shape, dtype=x.a.dtype)
        idx = [new_lat[k] for k in x.lat]
        new_a[idx] = x.a
        return new_a

    def pad_a(x, new_lat):
        """Extends the map ``x`` to a new list of latent variables ``new_lat``
        by padding. ``new_lat`` must start from the variables of ``x.lat``,
        arranged in the same order as they appear in ``x.lat``."""

        new_a = np.zeros((len(new_lat),) + x.a.shape[1:], dtype=x.a.dtype)
        new_a[:len(x.lat)] = x.a        
        return new_a

    if len(seq) == 1:
        return seq[0].lat, [seq[0].a]
    
    if len(seq) > 2:
        lat = latent.uunion(*[x.lat for x in seq])
        return lat, [extend_a(op, lat) for op in seq]

    # The rest is an optimization for the case of two operands.
    x, y = seq

    if x.lat is y.lat:
        return x.lat, [x.a, y.a]
    
    lat, swapped = latent.ounion(x.lat, y.lat)

    if swapped:
        return lat, [extend_a(x, lat), pad_a(y, lat)]
    
    return lat, [pad_a(x, lat), extend_a(y, lat)]
    
    
def lift(cls, x):
    """Converts ``x`` to a varaible of class ``cls``. If ``x`` is such 
    a variable already, returns it unchanged. If the conversion cannot be done, 
    raises a ``TypeError``."""

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
    """Converts ``x`` to either a numeric array or a variable of class ``cls``, 
    and returns the converted variable with its type.

    Args:
        other: Object to be converted.
    
    Returns:
        Tuple ``(converted_x, isnumeric)``. ``isnumeric`` is ``True`` if 
        ``converted_x`` is a numeric array, and ``False`` if it is a random 
        variable.
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
        # An optimization targeting uses like concatenate([v.real, v.imag]).
        x, y = arrays
        a = np.concatenate([x.a, y.a], axis=axis+1)
        return cls(a, b, x.lat)

    dims = [x.a.shape[axis+1] for x in arrays]
    jbase = (slice(None),) * axis

    if len(arrays) > 2:
        ulat = latent.uunion(*[x.lat for x in arrays])
        dtype = np.result_type(*[x.a for x in arrays])
        a = np.zeros((len(ulat),) + b.shape, dtype)
        n1 = 0
        for i, x in enumerate(arrays):
            n2 = n1 + dims[i]
            idx = ([ulat[k] for k in x.lat],) + jbase + (slice(n1, n2),)
            a[idx] = x.a
            n1 = n2

        return cls(a, b, ulat)
    
    # The rest is an optimization for the case of two operands.
    x, y = arrays

    # Indices along the variable dimension.
    jx = jbase + (slice(dims[0]),)
    jy = jbase + (slice(-dims[1], None),)

    ulat, swapped = latent.ounion(x.lat, y.lat)

    if swapped:
        x, y = y, x
        jx, jy = jy, jx
    
    a = np.zeros((len(ulat),) + b.shape, 
                 dtype=np.promote_types(x.a.dtype, y.a.dtype))
    a[(slice(len(x.lat)),) + jx] = x.a
    a[([ulat[k] for k in y.lat],) + jy] = y.a

    return cls(a, b, ulat)


def stack(cls, arrays, axis=0):
    # Essentially a copy of ``concatenate``, with slightly less overhead.

    b = np.stack([x.b for x in arrays], axis=axis)

    axis = axis if axis >= 0 else b.ndim + axis
    jbase = (slice(None),) * axis

    if len(arrays) == 1:
        return arrays[0][jbase + (None,)]
    elif len(arrays) == 2 and arrays[0].lat is arrays[1].lat:
        # An optimization targeting uses like stack([v.real, v.imag]).
        x, y = arrays
        a = np.stack([x.a, y.a], axis=axis+1)
        return cls(a, b, x.lat)

    if len(arrays) > 2:
        ulat = latent.uunion(*[x.lat for x in arrays])
        dtype = np.result_type(*[x.a for x in arrays])
        a = np.zeros((len(ulat),) + b.shape, dtype)
        for i, x in enumerate(arrays):
            idx = ([ulat[k] for k in x.lat],) + jbase + (i,)
            a[idx] = x.a

        return cls(a, b, ulat)
    
    # The rest is an optimization for the case of two operands.
    x, y = arrays
    
    jx = jbase + (0,)
    jy = jbase + (1,)

    ulat, swapped = latent.ounion(x.lat, y.lat)

    if swapped:
        x, y = y, x
        jx, jy = jy, jx
    
    a = np.zeros((len(ulat),) + b.shape, 
                 dtype=np.promote_types(x.a.dtype, y.a.dtype))
    a[(slice(x.nlat),) + jx] = x.a
    a[([ulat[k] for k in y.lat],) + jy] = y.a

    return cls(a, b, ulat)


def solve(cls, x, y):
    b = np.linalg.solve(x, y.b)

    if y.ndim != 1:
        a = np.linalg.solve(x, y.a)
    else:
        a = np.linalg.solve(x, y.a.T).T

    return cls(a, b, y.lat)


def asolve(cls, x, y):
    b = np.linalg.solve(x, y.b[..., None])
    a = np.linalg.solve(x, _unsq(y.a[..., None], b.ndim))
    return cls(a.squeeze(-1), b.squeeze(-1), y.lat)


def call_linearized(x, func, jmpfunc):
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
        ndim = x.ndim if s is None else len(s)
        axes = tuple(range(-ndim, 0))

    func = getattr(np.fft, name)
    b = func(x.b, s, axes, norm)
    a = func(x.a, s, _axes_a(axes), norm)
    return cls(a, b, x.lat)


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
    a = np.moveaxis(np.inner(x, y.a), x.ndim - 1, 0)
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
    a = np.einsum("ij, k -> ijk", a2d(x), y.ravel())
    return cls(a, b, x.lat)


def outer_2(cls, x, y):
    b = np.outer(x, y.b)
    a = np.einsum("k, ij -> ikj", x.ravel(), a2d(y))
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
    """Converts ``axes`` to an explicit form compatible with ``numpy.tensordot`` 
    function. If ``axes`` is a sequence, the function returns it unchanged, 
    and if ``axes`` is an integer ``n``, it returns a tuple of lists
    ``([-n, -n + 1, ..., -1], [0, 1, ..., n-1])``."""

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