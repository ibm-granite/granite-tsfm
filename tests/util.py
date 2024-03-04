# Copyright contributors to the TSFM project
#

"""Utilities for testing"""

from itertools import chain, repeat


def nreps(iterable, n):
    "Returns each element in the sequence repeated n times."
    return chain.from_iterable((repeat(i, n) for i in iterable))
