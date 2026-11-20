"""Reference implementation of the carry-calculus view of CuTe layouts.

Everything here is for *flat* layouts (nested layouts have the same 1-D function as
their flattening; see Proposition 2.4 of the paper).  A flat layout is a pair (shape, stride) of
equal-length tuples; shape entries are integers >= 1; stride entries live in an abelian
group -- in this file either Python ints or equal-length tuples of ints (the group Z^k).

Conventions follow CuTe: colexicographic (mixed-radix, least-significant digit first)
coordinates, and the *last* mode is unbounded when a layout is evaluated outside its
domain ("extended" evaluation).

This module is deliberately independent of every CuTe implementation: it is the thing
the implementations are compared against.
"""
from __future__ import annotations

from itertools import product
from math import gcd, prod
from typing import Iterable, Sequence

# --------------------------------------------------------------------------------------
# abelian-group helpers (ints or tuples of ints)
# --------------------------------------------------------------------------------------

def g_zero_like(d):
    return tuple(0 for _ in d) if isinstance(d, tuple) else 0


def g_add(a, b):
    if isinstance(a, tuple):
        return tuple(x + y for x, y in zip(a, b))
    return a + b


def g_sub(a, b):
    if isinstance(a, tuple):
        return tuple(x - y for x, y in zip(a, b))
    return a - b


def g_smul(n: int, a):
    if isinstance(a, tuple):
        return tuple(n * x for x in a)
    return n * a


def g_is_zero(a) -> bool:
    return all(x == 0 for x in a) if isinstance(a, tuple) else a == 0


def g_sum(items: Iterable, zero):
    acc = zero
    for it in items:
        acc = g_add(acc, it)
    return acc


# --------------------------------------------------------------------------------------
# mixed radix
# --------------------------------------------------------------------------------------

def prefix_products(shape: Sequence[int]) -> list[int]:
    """[P_0, ..., P_m] with P_0 = 1 and P_i = s_0 * ... * s_{i-1}."""
    out = [1]
    for s in shape:
        out.append(out[-1] * s)
    return out


def digits(x: int, shape: Sequence[int], extended: bool = True) -> list[int]:
    """Mixed-radix digits of x, least significant first.

    With extended=True the last digit is unbounded (CuTe's out-of-domain convention);
    otherwise x must lie in [0, prod(shape)).
    """
    out = []
    for s in shape[:-1]:
        out.append(x % s)
        x //= s
    if extended:
        out.append(x)
    else:
        assert 0 <= x < shape[-1], "coordinate out of bounds"
        out.append(x)
    return out


def undigits(c: Sequence[int], shape: Sequence[int]) -> int:
    P = prefix_products(shape)
    return sum(ci * Pi for ci, Pi in zip(c, P))


# --------------------------------------------------------------------------------------
# layouts
# --------------------------------------------------------------------------------------

class Layout:
    """A flat layout shape:stride."""

    __slots__ = ("shape", "stride")

    def __init__(self, shape, stride):
        if isinstance(shape, int):
            shape, stride = (shape,), (stride,)
        self.shape = tuple(shape)
        self.stride = tuple(stride)
        assert len(self.shape) == len(self.stride) >= 1
        assert all(isinstance(s, int) and s >= 1 for s in self.shape)

    # -- basic attributes ------------------------------------------------------------
    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def zero(self):
        return g_zero_like(self.stride[0])

    def __call__(self, x: int):
        """Extended evaluation: last mode unbounded."""
        return g_sum((g_smul(c, d) for c, d in zip(digits(x, self.shape), self.stride)), self.zero)

    def values(self) -> list:
        return [self(x) for x in range(self.size)]

    def at(self, coord: Sequence[int]):
        return g_sum((g_smul(c, d) for c, d in zip(coord, self.stride)), self.zero)

    def __eq__(self, other):
        return isinstance(other, Layout) and self.shape == other.shape and self.stride == other.stride

    def __hash__(self):
        return hash((self.shape, self.stride))

    def __repr__(self):
        if self.rank == 1:
            return f"{self.shape[0]}:{self.stride[0]}"
        return f"{self.shape}:{self.stride}".replace(" ", "")

    # -- the jump (floor-sum) parametrisation ------------------------------------------
    def slope_and_jumps(self):
        """Return (a, [(P_i, gamma_i) for i = 1..m-1]) with
        f(x) = x*a + sum_i floor(x / P_i) * gamma_i   for all x >= 0 (extended evaluation),
        where gamma_i = d_i - s_{i-1} * d_{i-1}."""
        P = prefix_products(self.shape)
        a = self.stride[0]
        jumps = []
        for i in range(1, self.rank):
            jumps.append((P[i], g_sub(self.stride[i], g_smul(self.shape[i - 1], self.stride[i - 1]))))
        return a, jumps

    def eval_floor_sum(self, x: int):
        a, jumps = self.slope_and_jumps()
        acc = g_smul(x, a)
        for P, gam in jumps:
            acc = g_add(acc, g_smul(x // P, gam))
        return acc


def concat(*layouts: Layout) -> Layout:
    return Layout(sum((L.shape for L in layouts), ()), sum((L.stride for L in layouts), ()))


def carries(xs: Sequence[int], P: int) -> int:
    """Total carry out of the block of digits below scale P when the numbers xs are added."""
    return sum(xs) // P - sum(x // P for x in xs)


# --------------------------------------------------------------------------------------
# normal form: coalesce, floor spectrum, recognition
# --------------------------------------------------------------------------------------

def coalesce(L: Layout) -> Layout:
    """The unique minimal-rank flat layout with the same function on [0, size).

    Removes size-1 modes and merges adjacent modes whose jump vanishes.  A layout of
    size 1 coalesces to 1:0.
    """
    shape, stride = [], []
    for s, d in zip(L.shape, L.stride):
        if s == 1:
            continue
        if shape and g_is_zero(g_sub(d, g_smul(shape[-1], stride[-1]))):
            shape[-1] *= s
        else:
            shape.append(s)
            stride.append(d)
    if not shape:
        return Layout((1,), (L.zero,))
    return Layout(tuple(shape), tuple(stride))


def floor_spectrum(values: Sequence, zero=0) -> dict[int, object]:
    """The unique coefficients gamma_Q (1 <= Q < N) with
    g(x) = sum_Q floor(x / Q) * gamma_Q on [0, N), for a function g with g(0) = 0.
    Returned as {Q: gamma_Q} with zero coefficients omitted.
    (Equivalently gamma = Moebius transform of the first differences of g.)"""
    N = len(values)
    assert g_is_zero(values[0]), "a layout function vanishes at 0"
    gam: dict[int, object] = {}
    for x in range(1, N):
        acc = values[x]
        for Q, c in gam.items():
            acc = g_sub(acc, g_smul(x // Q, c))
        if not g_is_zero(acc):
            gam[x] = acc
    return gam


def recognize(values: Sequence, zero=0) -> Layout | None:
    """Decide whether a function on [0, N) (given by its value list) is a layout function.
    Returns its coalesced layout, or None.

    Criterion (Theorem 4.3(a) of the paper): g is a layout function iff g(0) = 0 and
    the scales Q >= 2 in its floor spectrum form a divisibility chain all of whose
    members divide N.
    """
    N = len(values)
    if not g_is_zero(values[0]):
        return None
    spec = floor_spectrum(values, zero)
    a = spec.pop(1, zero)
    scales = sorted(spec)
    prev = 1
    for Q in scales:
        if Q % prev != 0:
            return None
        prev = Q
    if scales and N % scales[-1] != 0:
        return None
    if N == 1:
        return Layout((1,), (zero,))
    bounds = scales + [N]
    shape = [bounds[0]] + [bounds[i + 1] // bounds[i] for i in range(len(scales))]
    stride, d, last_s = [], a, None
    for i, s in enumerate(shape):
        if i == 0:
            d = a
        else:
            d = g_add(spec[scales[i - 1]], g_smul(last_s, d))
        stride.append(d)
        last_s = s
    return Layout(tuple(shape), tuple(stride))


def recognize_greedy(values: Sequence, zero=0) -> Layout | None:
    """Independent implementation of `recognize` straight from the definition
    (digit-additivity), used to cross-check the spectral criterion."""
    N = len(values)
    if not g_is_zero(values[0]):
        return None
    if N == 1:
        return Layout((1,), (zero,))
    P, shape, stride = 1, [], []
    while P < N:
        d = values[P]
        s = 1
        while (s + 1) * P <= N and all(
            g_is_zero(g_sub(values[s * P + y], g_add(g_smul(s, d), values[y]))) for y in range(P)
        ):
            s += 1
        if s == 1 or N % (P * s) != 0:
            return None
        shape.append(s)
        stride.append(d)
        P *= s
    return Layout(tuple(shape), tuple(stride))


def is_additive_over(values: Sequence, shape: Sequence[int]) -> bool:
    """Is the function (value list on [0, prod(shape))) digit-additive for this shape?"""
    assert len(values) == prod(shape)
    P = prefix_products(shape)
    zero = g_zero_like(values[0]) if isinstance(values[0], tuple) else 0
    for x in range(len(values)):
        want = g_sum((g_smul(c, values[Pi]) for c, Pi, s in zip(digits(x, shape, False), P, shape) if s > 1), zero)
        if not g_is_zero(g_sub(values[x], want)):
            return False
    return True


# --------------------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------------------

def compose_function(A: Layout, B: Layout) -> list:
    """The honest function i |-> A(B(i)) on [0, size(B)); B must be N-valued."""
    return [A(b) for b in B.values()]


def carry_defect(A: Layout, B: Layout, coord: Sequence[int]):
    """sum_i gamma_i^A * C_i  where C_i is the total carry, in A's radix at scale P_i,
    produced by the schoolbook evaluation  sum_j coord_j * r_j  (Theorem 5.3)."""
    _, jumps = A.slope_and_jumps()
    y = sum(c * r for c, r in zip(coord, B.stride))
    acc = A.zero
    for P, gam in jumps:
        C = y // P - sum(c * (r // P) for c, r in zip(coord, B.stride))
        acc = g_add(acc, g_smul(C, gam))
    return acc


def naive_composite(A: Layout, B: Layout) -> Layout:
    """B's shape with strides A(r_j): what 'left distributivity + linearity' predicts."""
    return Layout(B.shape, tuple(A(r) for r in B.stride))


def compose(A: Layout, B: Layout) -> list[Layout] | None:
    """Exact composition (Theorem 5.5).  Returns the list [R_0, ..., R_{n-1}] of coalesced
    layouts, one per mode of B (so the result is the layout whose j-th top-level mode is
    R_j, and B's shape is refined by it), or None if no layout R with B <= R and
    R(c) = A(B(c)) exists."""
    per_mode = []
    for t, r in zip(B.shape, B.stride):
        Rj = recognize([A(r * x) for x in range(t)], A.zero)
        if Rj is None:
            return None
        per_mode.append(Rj)
    # cross-mode non-interference: A(sum_j y_j) = sum_j A(y_j)
    for coord in product(*[range(t) for t in B.shape]):
        ys = [c * r for c, r in zip(coord, B.stride)]
        lhs = A(sum(ys))
        rhs = g_sum((A(y) for y in ys), A.zero)
        if not g_is_zero(g_sub(lhs, rhs)):
            return None
    return per_mode


def carry_free(shapeA: Sequence[int], B: Layout, scales: Iterable[int] | None = None) -> bool:
    """The carry-free condition: for every scale P (default: all prefix products P_1..P_{m-1}
    of shapeA),  sum_j (t_j - 1) * (r_j mod P) < P."""
    if scales is None:
        scales = prefix_products(shapeA)[1:-1]
    return all(sum((t - 1) * (r % P) for t, r in zip(B.shape, B.stride)) < P for P in scales)


# --------------------------------------------------------------------------------------
# complement (tiling regime) and compact layouts
# --------------------------------------------------------------------------------------

def sorted_modes(L: Layout) -> list[tuple[int, int]]:
    """Modes (stride, size) with size > 1, sorted by stride (then size)."""
    return sorted((d, s) for s, d in zip(L.shape, L.stride) if s > 1)


def is_complementable(L: Layout, M: int) -> bool:
    """Shah/Colfax admissibility: sorted modes satisfy s_i d_i | d_{i+1}, d_0 >= 1, and
    s_last d_last | M."""
    cur = 1
    for d, s in sorted_modes(L):
        if d == 0 or d % cur != 0:
            return False
        cur = d * s
    return M % cur == 0


def complement(L: Layout, M: int) -> Layout | None:
    """The complement of an N-valued layout in [0, M): the unique coalesced, increasing
    layout C such that (L, C) is a bijection onto [0, M).  None if it does not exist."""
    if not is_complementable(L, M):
        return None
    shape, stride, cur = [], [], 1
    for d, s in sorted_modes(L):
        shape.append(d // cur)
        stride.append(cur)
        cur = d * s
    shape.append(M // cur)
    stride.append(cur)
    return coalesce(Layout(tuple(shape), tuple(stride)))


def is_direct_sum(I: Iterable[int], J: Iterable[int], M: int) -> bool:
    sums = sorted(a + b for a in I for b in J)
    return sums == list(range(M))


def is_compact(L: Layout) -> bool:
    return sorted(L.values()) == list(range(L.size))


def all_factorizations(n: int, minimum: int = 2) -> list[tuple[int, ...]]:
    """All ordered factorizations of n into factors >= 2 (the empty tuple for n = 1)."""
    if n == 1:
        return [()]
    out = []
    for f in range(minimum, n + 1):
        if n % f == 0:
            for rest in all_factorizations(n // f):
                out.append((f,) + rest)
    return out
