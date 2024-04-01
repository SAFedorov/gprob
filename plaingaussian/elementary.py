import itertools
import numpy as np


class Elementary:

    id_counter = itertools.count()

    @staticmethod
    def create(n: int):
        return {next(Elementary.id_counter): i for i in range(n)}
    
    @staticmethod
    def union(iids1: dict, iids2: dict):
        """Ordered union of two dictionaries of elementary variables."""

        diff = set(iids2) - set(iids1)   
        offs = len(iids1)

        union_iids = iids1.copy()
        union_iids.update({xi: (offs + i) for i, xi in enumerate(diff)}) 

        return union_iids
    
    @staticmethod
    def uunion(*args):
        """Unordered union of multiple dictionaries of elementary variables."""
        s = set().union(*args)
        return {k: i for i, k in enumerate(s)}


# --- Operations on elementary maps ---

def longer_first(op1, op2):
    (_, iids1), (_, iids2) = op1, op2

    if len(iids1) >= len(iids2):
        return op1, op2, False
    
    return op2, op1, True


def complete_maps(op1, op2):

    (a1, iids1), (a2, iids2) = op1, op2

    if iids1 is iids2:
        return a1, a2, iids1

    (a1, iids1), (a2, iids2), swapped = longer_first(op1, op2)
        
    union_iids = Elementary.union(iids1, iids2)
    a1_ = pad_map(a1, len(union_iids))
    a2_ = extend_map(a2, iids2, union_iids)

    if swapped:
        a1_, a2_ = a2_, a1_

    return a1_, a2_, union_iids


def add_maps(op1, op2):

    (a1, iids1), (a2, iids2) = op1, op2

    if iids1 is iids2:
        return a1 + a2, iids1

    (a1, iids1), (a2, iids2), _ = longer_first(op1, op2)
        
    union_iids = Elementary.union(iids1, iids2)
    sum_a = pad_map(a1, len(union_iids))

    idx = [union_iids[k] for k in iids2]
    sum_a[idx] += a2

    return sum_a, union_iids


def pad_map(a, new_len):

    len_ = a.shape[0]
    new_shape = (new_len, *a.shape[1:])
    new_a = np.zeros(new_shape)
    new_a[:len_] = a

    return new_a


def extend_map(a, iids: dict, new_iids: dict):

    new_shape = (len(new_iids), *a.shape[1:])
    new_a = np.zeros(new_shape)
    idx = [new_iids[k] for k in iids]
    new_a[idx] = a

    return new_a


def join_maps(op1, op2):

    # Works for strictly two-dimensional matrices. Preserves the iids order.

    (a1, iids1), (a2, iids2) = op1, op2

    if iids1 is iids2:
        return a1 + a2, iids1

    (a1, iids1), (a2, iids2), swapped = longer_first(op1, op2)
        
    union_iids = Elementary.union(iids1, iids2)
    l1, l2 = a1.shape[1], a2.shape[1]
    cat_a = np.zeros((len(union_iids), l1 + l2))

    idx = [union_iids[k] for k in iids2]

    if swapped:
        cat_a[:len(iids1), l2:] = a1
        cat_a[idx, :l2] = a2
    else:
        cat_a[:len(iids1), :l1] = a1
        cat_a[idx, l1:] = a2

    return cat_a, union_iids


def u_join_maps(ops):
    # ops is a sequence ((a1, iids1), (a2, iids2), ...), where `a`s are 
    # strictly two-dimensional matrices.  

    union_iids = Elementary.uunion(*[iids for _, iids in ops])

    dims = [a.shape[1] for a, _ in ops]
    cat_a = np.zeros((len(union_iids), sum(dims)))
    n1 = 0
    for i, (a, iids) in enumerate(ops):
        n2 = n1 + dims[i]
        idx = [union_iids[k] for k in iids]
        cat_a[idx, n1: n2] = a
        n1 = n2

    return cat_a, union_iids