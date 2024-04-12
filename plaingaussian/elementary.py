import itertools


id_counter = itertools.count()


def create(n: int):
    return {next(id_counter): i for i in range(n)}


def ounion(elem1: dict, elem2: dict):
    """Ordered union of two dictionaries of elementary variables, where the 
    longer dictionary goes untoched into the beginning of the combined one."""

    swapped = False

    if len(elem2) > len(elem1):
        swapped = True
        elem1, elem2 = elem2, elem1

    diff = set(elem2) - set(elem1)   
    offs = len(elem1)

    union_elem = elem1.copy()
    union_elem.update({xi: (offs + i) for i, xi in enumerate(diff)}) 

    return union_elem, swapped


def uunion(*args):
    """Unordered union of multiple dictionaries of elementary variables."""
    s = set().union(*args)
    return {k: i for i, k in enumerate(s)}