"""Container for the random number generator."""

import numpy as np


gen = np.random.default_rng()


def setgen(seed):
    """Updates the random number generator used for sampling, `gprob.rn.gen`, 
    to a new instance of `numpy.random.Generator`.
    
    Args:
        seed (None, int, Generator, ...): 
            The seed, whose type can be anything that `numpy.random.default_rng`
            can accept as the seed. It can be, e.g., ``None``, meaning that a 
            random seed is drawn from the OS, an integer, setting the generator
            to a reproducible state, or a numpy ``Generator``, which 
            allows using a generator of a type different from the default.
    """

    global gen
    gen = np.random.default_rng(seed)