import numpy as np

from . import normal_
from . import sparse
from .normal_ import Normal
from .sparse import SparseNormal


def resolve(seq):
    if any([x.__class__ is SparseNormal for x in seq]):
        return sparse, SparseNormal
    
    return normal_, Normal


def fallback_to_normal(func):
    def func_(x, *args, **kwargs):
        if not hasattr(x, func.__name__):
            x = normal_.lift(Normal, x)

        return func(x, *args, **kwargs)
        
    func_.__name__ = func.__name__
    return func_


@fallback_to_normal
def icopy(x):
    """Creates a statistically independent copy of ``x``."""
    return x.icopy()


@fallback_to_normal
def mean(x):
    """Expectation value, ``<x>``."""
    return x.mean()


@fallback_to_normal
def var(x):
    """Variance, ``<(x-<x>)(x-<x>)^*>``, where ``*`` denotes 
    complex conjugation, and ``<...>`` is the expectation value of ``...``."""
    return x.var()


def cov(*args):
    """Covariance, generalizing ``<outer((x-<x>), (y-<y>)^H)>``, 
    where `H` denotes conjugate transposition, and ``<...>`` is 
    the expectation value of ``...``.
    
    Args:
        One or two random variables.

    Returns:
        - For one random variable, ``x``, the function returns ``x.cov()``. 
        - For two random variables, ``x`` and ``y``, the function returns 
          their cross-covariance.
        
        The cross-covariance of two normal variables 
        is an array ``c`` with the dimension number equal 
        to the sum of the dimensions of ``x`` and ``y``, whose components are
        ``c[ijk... lmn...] = <(x[ijk..] - <x>)(y[lmn..] - <y>)*>``, 
        where the indices ``ijk...`` and ``lmn...`` run over the elements 
        of ``x`` and ``y``, respectively, and ``*`` denotes complex conjugation.

        The cross-covariance of two sparse variables is an array 
        with the dimension number equal to the sum of the dense dimensions 
        of ``x`` and ``y``, plus the number of their sparse (independence) 
        dimensions, which should be the same in ``x`` and ``y``. 
        In the returned array, the regular dimensions 
        go first in the order they appear in the variable shapes, 
        and the independence dimensions are appended at the end.
        The resulting structure is the same as the structure produced 
        by repeated applications of `np.diagonal` over all the 
        independence dimensions of the full-sized covariance matrix ``c`` 
        for ``x`` and ``y``.

    Raises:
        TypeError: 
            If the number of input arguments is not 1 or 2.

    Examples:
    
        Normal variables.

        >>> v1 = normal(size=(3,))  # shape (3,)
        >>> v2 = normal(size=(3,))  # shape (3,)
        >>> cov(v1, v1 + v2)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])

        >>> v = normal(size=(2, 3))
        >>> c = cov(v)
        >>> c.shape
        (2, 3, 2, 3)
        >>> np.all(c.reshape((v.size, v.size)) == np.eye(v.size))
        True

        >>> v1 = normal(size=2)
        >>> v2 = 0.5 * v1[0] + normal()
        >>> cov(v1, v2)
        array([0.5, 0. ])

        >>> v1 = normal(size=2)
        >>> v2 = 0.5 * v1[0] + normal(size=3)
        >>> cov(v1, v2)
        array([[0.5, 0.5, 0.5],
               [0. , 0. , 0. ]])

        >>> v1 = normal()
        >>> v2 = 1j * v1 + normal()
        >>> cov(v1, v2)
        array(0.-1.j)

        Sparse normal variables.

        >>> v1 = iid(normal(), 3)  # shape (3,)
        >>> v2 = iid(normal(), 3)  # shape (3,)
        >>> cov(v1, v1 + v2)
        array([1., 1., 1.])

        >>> v1 = iid(normal(size=3), 4)  # shape (4, 3)
        >>> v2 = iid(normal(size=2), 4)  # shape (4, 2)
        >>> cov(v1, v2).shape
        (3, 2, 4)
    """

    mod, cls = resolve(args)
    args = [mod.lift(cls, arg) for arg in args]

    if len(args) == 1:
        return args[0].cov()
    
    if len(args) == 0 or len(args) > 2:
        raise TypeError("The function can accept only one or two input "
                        f"arguments, while {len(args)} arguments are given.")
    
    return mod.cov(*args)  # len(args) == 2.


@fallback_to_normal
def diagonal(x, offset=0, axis1=0, axis2=1):
    """Extracts a diagonal from a multi-dimensional random variable.

    Args:
        x (random variable): 
            The input array random variable.
        offset (int): 
            The offset of the diagonal from the main diagonal.
        axis1 (int): 
            The first axis along which the diagonal should be taken.
        axis2 (int): 
            The second axis along which the diagonal should be taken.

    Returns:
        A new random variable consisting of the extracted diagonal elements.

    Note:
        This function is similar to `numpy.diagonal`.
    """
    return x.diagonal(offset=offset, axis1=axis1, axis2=axis2)


@fallback_to_normal
def flip(x, axis=None):
    """Reverses the order of elements in a random variable along an axis.

    Args:
        x (random variable): 
            The input array random variable.
        axis (int, tuple of int, or None):
            The axis or axes along which to reverse the order. ``None``
            means all axes.

    Returns:
        A new random variable with the same shape as the original and 
        reordered elements.

    Note:
        This function is similar to `numpy.flip`.
    """
    return x.flip(axis=axis)


@fallback_to_normal
def sum(x, axis=None, keepdims=False):
    """Computes the sum of all elements of a random variable along an axis 
    or axes.

    Args:
        x (random variable): 
            The input array random variable whose elements are to be summed.
        axis (int, tuple of int, or None): 
            Axis or axes along which to sum the elements. If ``None``, 
            the function sums all elements.
        keepdims (bool): 
            If ``True``, the reduced axes are retained in the output 
            as dimensions with size one. This enables the result to 
            broadcast against the input array.

    Returns:
        A new random variable representing the result of the summation. 
        The output variable has the same dimensions as the input, except 
        the summation axes are removed unless ``keepdims`` is ``True``.

    Note:
        This function is similar to `numpy.sum`.
    """
    return x.sum(axis=axis, keepdims=keepdims)


@fallback_to_normal
def cumsum(x, axis=None):
    """Computes the cumulative sum of the elements of a random variable.

    Args:
        x (random variable):
            The input array random variable.
        axis (int or None):
            Axis along which the cumulative sum is computed. If ``None``, 
            the cumulative sum is computed over the flattened array.

    Returns:
        A new random variable representing the result of the cumulative 
        summation with the same dimensions as the input.

    Note:
        This function is similar to `numpy.cumsum`.
    """
    return x.cumsum(axis=axis)


@fallback_to_normal
def moveaxis(x, source, destination):
    """Moves an axis of a random variable to a new position.

    Args:
        x (random variable):
            The input array random variable whose axis is to be moved.
        source (int):
            The original position of the axis.
        destination (int):
            The destination position of the axis.

    Returns:
        A new random variable with the transformed layout.

    Note:
        This function is similar to `numpy.moveaxis`,
        except for the lack of support of multiple source and destination axes.
    """
    return x.moveaxis(source, destination)


@fallback_to_normal
def ravel(x, order="C"):
    """Flattens a random variable while ensuring that the underying latent map 
    is stored contiguously in the memory. The map arrays of the returned 
    variable are views of the map arrays of the input variable whenever 
    possible. 

    Args:
        x (random variable): 
            The input array random variable to be flattened.
        order (str): 
            The order in which the input array elements are read.
            - 'C': C-style row-major order.
            - 'F': Fortran-style column-major order.

    Returns:
        A new one-dimensional random variable containing all the elements of 
        the input variable in the specified order.

    Note:
        This function is similar to `numpy.ravel`.
    """
    return x.ravel(order=order)


@fallback_to_normal
def reshape(x, newshape, order="C"):
    """Gives a new shape to a random variable.

    Args:
        x (random variable): 
            The input array random variable to be reshaped.
        newshape (tuple of int): 
            The new shape. One dimension can be set to `-1` to infer its
            size from the total number of elements and the other dimensions.
        order (str): 
            The order in which the array elements are read.
            - 'C': C-style row-major order.
            - 'F': Fortran-style column-major order.

    Returns:
        A new random variable with the specified shape and order.

    Note:
        This function is similar to `numpy.reshape`.
    """
    return x.reshape(newshape, order=order)


@fallback_to_normal
def squeeze(x, axis=None):
    """Removes axis or axes of length one from the variable.
    
    Args:
        x (random variable): 
            The input array random variable to be squeezed.
        axis (None, int, or tuple of ints):
            The axis or axes to be removed. If ``None``, all axes of 
            length one are removed.
    
    Returns:
        A new random variable with the shape identical to that of the 
        input, except the removed axes.  
    """
    return x.squeeze(axis=axis)


@fallback_to_normal
def transpose(x, axes=None):
    """Permutes the axes of a random variable.

    Args:
        x (random variable): 
            The input array random variable to be transposed.
        axes (tuple of int or None): 
            The desired axes order. If ``None``, the existing axes 
            order is reversed.

    Returns:
        A new random variable with the axes in the new order.

    Note:
        This function is similar to `numpy.transpose`.
    """
    return x.transpose(axes=axes)


@fallback_to_normal
def trace(x, offset=0, axis1=0, axis2=1):
    """Calculates the sum of the diagonal elements of a random variable.

    Args:
        x (random variable): 
            The input array random variable.
        offset (int): 
            The offset of the diagonal to be summed from the main diagonal.
        axis1 (int): 
            The first axis along which the diagonal should be summed.
        axis2 (int): 
            The second axis along which the diagonal should be summed.

    Returns:
        A new random variable, consisting of the sum(s) of the diagonal 
        elements with respect to the specified axes.

    Note:
        This function is similar to `numpy.trace`.
    """
    return x.trace(offset=offset, axis1=axis1, axis2=axis2)


@fallback_to_normal
def broadcast_to(x, shape):
    """Broadcasts a random variable to a new shape.

    Args:
        x (random variable): 
            The input array random variable to be broadcast.
        shape (tuple of int): 
            The desired shape to broadcast the input variable to.

    Returns:
        A new random variable with the specified shape consisting of duplicates
        of the input variable.

    Note:
        This function is similar to `numpy.broadcast_to`.
    """
    return x.broadcast_to(shape)


def concatenate(arrays, axis=0):
    """Joins a sequence of array random variables along an existing axis.
    
    Args:
        arrays (sequence of random variables):
            The input variables to be joined. The shapes of the variables
            must agree except for the size along the concatenation axis.
        axis (int):
            The axis along which the variables are to be joined.

    Returns:
        A new random variable of the type corresponding to the highest type 
        present in `arrays`.

    Note:
        This function is similar to `numpy.concatenate`.
    """

    if len(arrays) == 0:
        raise ValueError("Need at least one array to concatenate.")
    mod, cls = resolve(arrays)
    arrays = [mod.lift(cls, x) for x in arrays]
    return mod.concatenate(cls, arrays, axis)


def stack(arrays, axis=0):
    """Joins a sequence of array random variables along a new axis.
    
    Args:
        arrays (sequence of random variables):
            The input variables to be joined. All variables must have 
            the same shape.
        axis (int):
            The axis number in the output array along which the variables 
            are to be stacked.

    Returns:
        A new random variable of the type corresponding to the highest type 
        present in `arrays`.

    Note:
        This function is similar to `numpy.stack`.
    """

    if len(arrays) == 0:
        raise ValueError("Need at least one array to stack.")
    mod, cls = resolve(arrays)
    arrays = [mod.lift(cls, x) for x in arrays]
    return mod.stack(cls, arrays, axis)


def hstack(arrays):
    """Joins a sequence of array random variables horizontally.

    Args:
        arrays (sequence of random variables):
            The input variables to be joined.

    Returns:
        A new random variable of the type corresponding to the highest type 
        present in ``arrays``.
    
    Note:
        This function is similar to `numpy.hstack`.
    """
    
    if len(arrays) == 0:
        raise ValueError("Need at least one array to stack.")
    
    mod, cls = resolve(arrays)
    arrays = [mod.lift(cls, x) for x in arrays]
    
    arrays = [x.ravel() if x.ndim == 0 else x for x in arrays]
    if arrays[0].ndim == 1:
        return mod.concatenate(cls, arrays, axis=0)
    
    return mod.concatenate(cls, arrays, axis=1)
    

def vstack(arrays):
    """Joins a sequence of array random variables vertically.

    Args:
        arrays (sequence of random variables):
            The input variables to be joined.

    Returns:
        A new random variable of the type corresponding to the highest type 
        present in ``arrays``.
    
    Note:
        This function is similar to `numpy.vstack`.
    """

    if len(arrays) == 0:
        raise ValueError("Need at least one array to stack.")
    
    mod, cls = resolve(arrays)
    arrays = [mod.lift(cls, x) for x in arrays]

    if arrays[0].ndim <= 1:
        arrays = [x.reshape((1, -1)) for x in arrays]
    
    return mod.concatenate(cls, arrays, axis=0)


def dstack(arrays):
    """Joins a sequence of array random variables along their depth 
    (third axis).

    Args:
        arrays (sequence of random variables):
            The input variables to be joined.

    Returns:
        A new random variable of the type corresponding to the highest type 
        present in ``arrays``.
    
    Note:
        This function is similar to `numpy.dstack`.
    """

    if len(arrays) == 0:
        raise ValueError("Need at least one array to stack.")
    
    mod, cls = resolve(arrays)
    arrays = [mod.lift(cls, x) for x in arrays]

    if arrays[0].ndim <= 1:
        arrays = [x.reshape((1, -1, 1)) for x in arrays]
    elif arrays[0].ndim == 2:
        arrays = [x.reshape(x.shape + (1,)) for x in arrays]
    
    return mod.concatenate(cls, arrays, axis=2)


@fallback_to_normal
def split(x, indices_or_sections, axis=0): 
    """Splits a random variable along an axis.

    Args:
        x (random variable): 
            The input variable to be split.
        indices_or_sections (int or sequence of int):
            - If an integer, n, the input variable is to be divided along 
              ``axis`` into n equal pieces.
            - If a sequence of sorted integers, its entries indicate where 
              along ``axis`` the input variable is to be split.
        axis (int):
            The axis along which the variable is to be split.

    Returns:
        A list of new random variables into which the input variable is split.
    
    Note:
        This function is similar to `numpy.split`.
    """
    return x.split(indices_or_sections=indices_or_sections, axis=axis)


@fallback_to_normal
def hsplit(x, indices_or_sections):
    """Splits a random variable horizontally.

    Args:
        x (random variable): 
            The input variable to be split.
        indices_or_sections (int or sequence of int):
            - If an integer, n, the input variable is to be divided 
              into n equal pieces.
            - If a sequence of sorted integers, its entries indicate where 
              the input variable is to be split. 

    Returns:
        A list of new random variables into which the input variable is split.
    
    Note:
        This function is similar to `numpy.hsplit`.
    """

    if x.ndim < 1:
        raise ValueError("hsplit only works on arrays of 1 or more dimensions.")
    if x.ndim == 1:
        return split(x, indices_or_sections, axis=0)
    return split(x, indices_or_sections, axis=1)


@fallback_to_normal
def vsplit(x, indices_or_sections):
    """Splits a random variable vertically.

    Args:
        x (random variable): 
            The input variable to be split.
        indices_or_sections (int or sequence of int):
            - If an integer, n, the input variable is to be divided 
              into n equal pieces.
            - If a sequence of sorted integers, its entries indicate where 
              the input variable is to be split.  

    Returns:
        A list of new random variables into which the input variable is split.
    
    Note:
        This function is similar to `numpy.vsplit`.
    """

    if x.ndim < 2:
        raise ValueError("vsplit only works on arrays of 2 or more dimensions.")
    return split(x, indices_or_sections, axis=0)


@fallback_to_normal
def dsplit(x, indices_or_sections):
    """Splits a random variable along the depth (third axis).

    Args:
        x (random variable): 
            The input variable to be split.
        indices_or_sections (int or sequence of int):
            - If an integer, n, the input variable is to be divided 
              into n equal pieces.
            - If a sequence of sorted integers, its entries indicate where 
              the input variable is to be split.  

    Returns:
        A list of new random variables into which the input variable is split.
    
    Note:
        This function is similar to `numpy.dsplit`.
    """

    if x.ndim < 3:
        raise ValueError("dsplit only works on arrays of 3 or more dimensions.")
    return split(x, indices_or_sections, axis=2)


def add(x, y):
    return x + y


def subtract(x, y):
    return x - y


def multiply(x, y):
    return x * y


def divide(x, y):
    return x / y


def power(x, y):
    return x ** y


def matmul(x, y):
    """Matrix product between a random variable and a numeric array, 
    or linearized matrix multiplication between two random variables. 
    The dimensions of both operands must be greater than 0.

    Args:
        x (numeric array or random variable): 
            The first operand.
        y (numeric array or random variable): 
            The second operand.

    Returns:
        A new random variable - the matrix product of ``x`` and ``y``.

    Note:
        This function is similar to `numpy.matmul`, and follows the same rules 
        regarding the shapes of the input and output variables.
    """
    return x @ y


def dot(x, y):
    """Dot product between a random variable and a numeric array, 
    or linearized dot product between two random variables.

    Args:
        x (numeric array or random variable): 
            The first operand.
        y (numeric array or random variable): 
            The second operand.

    Returns:
        A new random variable - the dot product of ``x`` and ``y``.

    Note:
        This function is similar to `numpy.dot`, and follows the same rules 
        regarding the shapes of the input and output variables.
    """

    mod, cls = resolve([x, y])
    return mod.bilinearfunc(cls, "dot", x, y)


def inner(x, y):
    """Inner product between a random variable and a numeric array, 
    or linearized inner product between two random variables.

    Args:
        x (numeric array or random variable): 
            The first operand.
        y (numeric array or random variable): 
            The second operand.

    Returns:
        A new random variable - the inner product of ``x`` and ``y``.

    Note:
        This function is similar to `numpy.inner`, and follows the same rules 
        regarding the shapes of the input and output variables.
    """

    mod, cls = resolve([x, y])
    return mod.bilinearfunc(cls, "inner", x, y)


def outer(x, y):
    """Outer product between a random variable and a numeric array, 
    or linearized outer product between two random variables.

    Args:
        x (numeric array or random variable): 
            The first operand.
        y (numeric array or random variable): 
            The second operand.

    Returns:
        A new random variable - the outer product of ``x`` and ``y``.

    Note:
        This function is similar to `numpy.outer`, and follows the same rules 
        regarding the shapes of the input and output variables.
    """

    mod, cls = resolve([x, y])
    return mod.bilinearfunc(cls, "outer", x, y)


def kron(x, y):
    """Kronecker product between a random variable and a numeric array, 
    or linearized Kronecker product between two random variables.

    Args:
        x (numeric array or random variable): 
            The first operand.
        y (numeric array or random variable): 
            The second operand.

    Returns:
        A new random variable - the Kronecker product of ``x`` and ``y``.

    Note:
        This function is similar to `numpy.kron`, and follows the same rules 
        regarding the shapes of the variables and the arrangement of elements.
    """

    mod, cls = resolve([x, y])
    return mod.bilinearfunc(cls, "kron", x, y)


def tensordot(x, y, axes=2):
    """Tensor dot product between a random variable and a numeric array, 
    or linearized tensor dot product between two random variables.
    
    The tensor dot product is the sum of the elements of ``x`` and ``y`` 
    along the selected axes.

    Args:
        x (numeric array or random variable): 
            The first operand.
        y (numeric array or random variable): 
            The second operand.
        axes (int or tuple of two sequences of int):
            The axes to be summed over.
            - If an integer, n, the sum is to be taken over the last n axes
              of ``x`` and the first n axes of ``y``.
            - If two sequences of int of the same length, the indices of axes 
              to be summed are picked from those sequences.
            In all cases, the axes in the pairs to be summed over must have 
            the same lengths.

    Returns:
        A new random variable - the tensor dot product of ``x`` and ``y``.

    Note:
        This function is similar to `numpy.tensordot`, and follows the same 
        rules regarding the shapes of the input and output variables.
    """

    mod, cls = resolve([x, y])
    return mod.bilinearfunc(cls, "tensordot", x, y, [axes])


def einsum(subs, x, y):
    """Einstein summation between a random variable and a numeric array, 
    or linearized Einstein summation between two random variables.

    Args:
        subs (str):
            The subscripts that define the axes of ``x`` and ``y`` to be summed 
            over and the axes arrangement of the output. See `numpy.einsum`
            for more details.
        x (random variable or numeric array): 
            The first input operand.
        y (random variable or numeric array): 
            The second input operand.

    Returns:
        A new random variable - the result of the Einstein summation 
        of ``x`` and ``y`` according to the subscripts.

    Note:
        This function is similar to `numpy.einsum` and follows the same rules 
        regarding the subscripts and the variable shapes, but does not support 
        more than two operands.
    """

    mod, cls = resolve([x, y])
    return mod.bilinearfunc(cls, "einsum", x, y, pargs=[subs])


def linearized_unary(jmpf):
    if not jmpf.__name__.endswith("_jmp"):
        raise ValueError()
    
    f_name = jmpf.__name__[:-4]
    f = getattr(np, f_name)

    def flin(x):
        mod, cls = resolve([x])
        x_ = mod.lift(cls, x)
        return mod.call_linearized(x_, f, jmpf)
    
    flin.__name__ = f_name
    return flin


# Elementwise Jacobian-matrix products.
def exp_jmp(x, ans, a):     return a * ans
def exp2_jmp(x, ans, a):    return a * (ans * np.log(2.))
def log_jmp(x, ans, a):     return a / x
def log2_jmp(x, ans, a):    return a / (x * np.log(2.))
def log10_jmp(x, ans, a):   return a / (x * np.log(10.))
def sqrt_jmp(x, ans, a):    return a / (2. * ans)
def cbrt_jmp(x, ans, a):    return a / (3. * ans**2)
def sin_jmp(x, ans, a):     return a * np.cos(x)
def cos_jmp(x, ans, a):     return a * (-np.sin(x))
def tan_jmp(x, ans, a):     return a / np.cos(x)**2
def arcsin_jmp(x, ans, a):  return a / np.sqrt(1 - x**2)
def arccos_jmp(x, ans, a):  return a / (-np.sqrt(1 - x**2))
def arctan_jmp(x, ans, a):  return a / (1 + x**2)
def sinh_jmp(x, ans, a):    return a * np.cosh(x)
def cosh_jmp(x, ans, a):    return a * np.sinh(x)
def tanh_jmp(x, ans, a):    return a / np.cosh(x)**2
def arcsinh_jmp(x, ans, a): return a / np.sqrt(x**2 + 1)
def arccosh_jmp(x, ans, a): return a / np.sqrt(x**2 - 1)
def arctanh_jmp(x, ans, a): return a / (1 - x**2)
def conjugate_jmp(x, ans, a): return a.conj()


def absolute_jmp(x, ans, a):
    if np.iscomplexobj(x):
        return (np.real(x) * np.real(a) + np.imag(x) * np.imag(a)) / ans
    return np.sign(x) * a


exp = linearized_unary(exp_jmp)
exp2 = linearized_unary(exp2_jmp)
log = linearized_unary(log_jmp)
log2 = linearized_unary(log2_jmp)
log10 = linearized_unary(log10_jmp)
sqrt = linearized_unary(sqrt_jmp)
cbrt = linearized_unary(cbrt_jmp)
sin = linearized_unary(sin_jmp)
cos = linearized_unary(cos_jmp)
tan = linearized_unary(tan_jmp)
arcsin = linearized_unary(arcsin_jmp)
arccos = linearized_unary(arccos_jmp)
arctan = linearized_unary(arctan_jmp)
sinh = linearized_unary(sinh_jmp)
cosh = linearized_unary(cosh_jmp)
tanh = linearized_unary(tanh_jmp)
arcsinh = linearized_unary(arcsinh_jmp)
arccosh = linearized_unary(arccosh_jmp)
arctanh = linearized_unary(arctanh_jmp)
conjugate = conj = linearized_unary(conjugate_jmp)
absolute = abs = linearized_unary(absolute_jmp)