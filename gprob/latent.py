import itertools


id_counter = itertools.count()


def create(n: int):
    return {next(id_counter): i for i in range(n)}


def ounion(lat1: dict, lat2: dict):
    """Ordered union of two dictionaries of latent variables, where the 
    longer dictionary goes untoched into the beginning of the resulting 
    dictionary."""

    swapped = False

    if len(lat2) > len(lat1):
        swapped = True
        lat1, lat2 = lat2, lat1

    diff = set(lat2) - set(lat1)   
    offs = len(lat1)

    union_lat = lat1.copy()
    union_lat.update({xi: (offs + i) for i, xi in enumerate(diff)}) 

    return union_lat, swapped


def uunion(*args):
    """Unordered union of multiple dictionaries of latent variables."""
    s = set().union(*args)
    return {k: i for i, k in enumerate(s)}