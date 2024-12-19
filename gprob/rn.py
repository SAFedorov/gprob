"""Container for the random number generator."""

import numpy as np


gen = np.random.default_rng()


def setgen(seed):
    """Updates the random number generator to a new instance produced by 
    `numpy.random.default_rng(seed)`.
    
    Args:
        s (None, int, Generator, ...): 
            The seed, whose type can be anything that `numpy.random.default_rng`
            can accept as the seed. It can be, e.g., ``None``, meaning that a 
            random seed is drawn from the OS, an integer, setting the generator
            to a reproducible state, or a numpy ``Generator``, which 
            allows using a generator different from the default type.
    """

    global gen
    gen = np.random.default_rng(seed)