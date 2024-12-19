from . import fft
from . import linalg
from . import func
from . import maps
from . import normal_
from . import sparse
from . import rn

from .normal_ import Normal, normal
from .sparse import iid

from .arrayops import (icopy, mean, var, std, cov, dkl, entropy, logp, sample,
    broadcast_to, stack, hstack, vstack, dstack, concatenate,
    split, hsplit, vsplit, dsplit, sum, cumsum, trace, diagonal, reshape, flip,
    squeeze, moveaxis, ravel, transpose, add, subtract, multiply, divide, power, 
    einsum, dot, matmul, inner, outer, kron, tensordot, 
    exp, exp2, log, log2, log10, sqrt, cbrt, sin, cos, tan, arcsin, arccos, 
    arctan, sinh, cosh, tanh, arcsinh, arccosh, arctanh, conjugate, conj,
    absolute, abs)