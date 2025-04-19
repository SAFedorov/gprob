"""Container for the random number generator."""

import numpy as np


gen = np.random.default_rng()


def seed(s):
    """Updates the generator used for all random sampling in `gprob`.
    
    Args:
        s (None, int, Generator, ...): 
            The seed, whose type can be anything that `numpy.random.default_rng`
            can accept as the seed. It can be, e.g., 
            an integer, for initializing the generator in a reproducible state, 
            ``None``, for drawing a random seed from the OS, 
            or a pre-configured ``numpy.random.Generator`` of any kind.
    """

    global gen
    gen = np.random.default_rng(s)