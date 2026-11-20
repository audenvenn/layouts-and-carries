"""Uniform access to the two Python implementations of CuTe.

Both ship a package called `pycute`, so only one can be loaded per process:

    Impl("cutlass", "<cutlass checkout>/python")     # CUTLASS python/pycute
    Impl("nvlabs",  "<NVlabs/CuTe checkout>")        # PyCuTe reference implementation

Layouts cross this boundary as plain (shape, stride) pairs of nested tuples of ints, so the
theory code never touches an implementation's classes.
"""
from __future__ import annotations

import sys


def flatten(t):
    if isinstance(t, tuple):
        return tuple(x for a in t for x in flatten(a))
    return (t,)


def tsize(t) -> int:
    n = 1
    for x in flatten(t):
        n *= x
    return n


def coarsens(P, S) -> bool:
    """P <= S in CuTe's compatibility order: P is an integer equal to |S|, or both are tuples
    of the same length with P_i <= S_i."""
    if not isinstance(P, tuple):
        return P == tsize(S)
    return isinstance(S, tuple) and len(P) == len(S) and all(coarsens(p, s) for p, s in zip(P, S))


class Impl:
    def __init__(self, label: str, path: str):
        self.label = label
        sys.path.insert(0, path)
        import pycute  # noqa: WPS433

        self.m = pycute
        self.is_nvlabs = hasattr(pycute, "coalesce_z")

    # -- conversion --------------------------------------------------------------------
    def mk(self, shape, stride):
        return self.m.Layout(shape, stride)

    @staticmethod
    def _plain(t):
        if isinstance(t, (tuple, list)):
            return tuple(Impl._plain(x) for x in t)
        return int(t)

    def tup(self, L):
        return (self._plain(L.shape), self._plain(L.stride))

    def values(self, L, n=None):
        n = self.size(L) if n is None else n
        return [int(L(i)) for i in range(n)]

    def size(self, L) -> int:
        return int(self.m.size(L))

    # -- operations: each returns ("ok", layout) or ("err", message) -------------------------
    def _try(self, fn, *args):
        try:
            return ("ok", fn(*args))
        except Exception as e:  # noqa: BLE001 - implementations raise assorted exception types
            return ("err", f"{type(e).__name__}: {e}"[:160])

    def coalesce(self, L):
        return self._try(self.m.coalesce, L)

    def composition(self, A, B):
        return self._try(self.m.composition, A, B)

    def complement(self, A, M: int):
        return self._try(self.m.complement, A, M)

    def logical_divide(self, A, B):
        return self._try(self.m.logical_divide, A, B)

    def logical_product(self, A, B):
        return self._try(self.m.logical_product, A, B)

    def right_inverse(self, A):
        return self._try(self.m.right_inverse, A)

    def left_inverse(self, A):
        return self._try(self.m.left_inverse, A)
