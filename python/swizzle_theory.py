"""Swizzles in the carry calculus: reference definitions (independent of CuTe).

A swizzle is CuTe's ``Swizzle<B, M, S>``: the map on offsets that XORs the B-bit field YYY into
the B-bit field ZZZ ("ZZZ ^= YYY").  Offsets carry two group laws, integer addition and bitwise
XOR; the binary expansion is a layout with values in the XOR group, and a swizzle is linear for
XOR.  Bit vectors are represented by Python ints under ``^``.
"""
from __future__ import annotations

from itertools import permutations, product
from math import prod
from typing import Callable, Sequence

from layout_theory import Layout, digits, prefix_products, recognize


def bit(x: int, b: int) -> int:
    return (x >> b) & 1


def supp(x: int) -> frozenset:
    return frozenset(b for b in range(x.bit_length()) if (x >> b) & 1)


class Swizzle:
    """sigma(x) = x ^ ((x & ymask) >> S)   (a left shift if S < 0); pairs = [(y_I, z_I)]."""

    def __init__(self, B: int, M: int, S: int):
        assert B >= 0 and M >= 0 and abs(S) >= B, "not a valid CuTe swizzle"
        self.B, self.M, self.S = B, M, S
        msk = (1 << B) - 1
        self.ymask = msk << (M + max(0, S))
        self.zmask = msk << (M - min(0, S))
        self.pairs = [(M + max(0, S) + i, M - min(0, S) + i) for i in range(B)]
        self.top = M + B + abs(S)          # first bit above both fields

    def __call__(self, x: int) -> int:
        t = x & self.ymask
        return x ^ (t >> self.S if self.S >= 0 else t << -self.S)

    def __repr__(self):
        return f"Sw<{self.B},{self.M},{self.S}>"

    def closed_form(self, x: int) -> int:
        """sigma(x) = x + sum_I 2^{z_I} * bit_{y_I}(x) * (1 - 2 bit_{z_I}(x))."""
        return x + sum((1 << z) * bit(x, y) * (1 - 2 * bit(x, z)) for y, z in self.pairs)

    def field_shape(self) -> tuple:
        B, M, S = self.B, self.M, abs(self.S)
        return (1 << M, 1 << B, 1 << (S - B), 1 << B, 1)

    def only_zy(self) -> Layout:
        """The layout CuTe calls swizzle_only_zy: projection on the two fields."""
        M, S = self.M, abs(self.S)
        return Layout(self.field_shape(), (0, 1 << M, 0, 1 << (M + S), 0))

    def anti_zy(self) -> Layout:
        """The layout CuTe calls swizzle_anti_zy: projection on the other bits."""
        B, M, S = self.B, self.M, abs(self.S)
        return Layout(self.field_shape(), (1, 0, 1 << (M + B), 0, 1 << (M + B + S)))

    def boundaries(self) -> list[int]:
        """Scales at which the field projections have a jump."""
        B, M, S = self.B, self.M, abs(self.S)
        return sorted({1 << M, 1 << (M + B), 1 << (M + S), 1 << (M + B + S)})


def make_swizzle(Y: int, Z: int) -> Swizzle | None:
    """CuTe's make_swizzle<Y, Z>: None where its static assertions fail."""
    if Y == 0 and Z == 0:
        return Swizzle(0, 0, 0)
    BY, BZ = bin(Y).count("1"), bin(Z).count("1")
    if BY != BZ or Y == 0 or Z == 0:
        return None
    tzy = (Y & -Y).bit_length() - 1
    tzz = (Z & -Z).bit_length() - 1
    M, S = min(tzy, tzz), tzy - tzz
    if abs(S) < BY:
        return None
    sw = Swizzle(BY, M, S)
    return sw if (sw.ymask | sw.zmask) == (Y | Z) else None


def merge_swizzles(a: Swizzle, b: Swizzle) -> Swizzle | None:
    """CuTe's composition(Swizzle, Swizzle): XOR of the masks; requires equal shifts."""
    if a.S != b.S:
        return None
    return make_swizzle(a.ymask ^ b.ymask, a.zmask ^ b.zmask)


def upcast_pairs(pairs, n: int):
    """The pairing of the paper's casting rule: keep the pairs with y, z >= n, shifted down by n."""
    return [(y - n, z - n) for y, z in pairs if y >= n and z >= n]


def upcast_swizzle(sw: Swizzle, n: int) -> Swizzle:
    """CuTe's upcast<2^n>(Swizzle)."""
    newM = sw.M - n
    return Swizzle(sw.B, newM, sw.S) if newM >= 0 else Swizzle(max(sw.B + newM, 0), 0, sw.S)


def downcast_swizzle(sw: Swizzle, n: int) -> Swizzle:
    return Swizzle(sw.B, sw.M + n, sw.S)


# --------------------------------------------------------------------------------------
# swizzled layouts  h(c) = sigma(o + f(c))
# --------------------------------------------------------------------------------------

def swizzled_values(sw: Callable[[int], int], o: int, f: Layout) -> list[int]:
    return [sw(o + v) for v in f.values()]


def binary_carry_counts(o: int, coord: Sequence[int], strides: Sequence[int], nbits: int) -> list[int]:
    """C_b = floor(((o mod 2^b) + sum_j c_j (r_j mod 2^b)) / 2^b) for b = 1..nbits."""
    return [((o % (1 << b)) + sum(c * (r % (1 << b)) for c, r in zip(coord, strides))) >> b
            for b in range(1, nbits + 1)]


def is_bit_disjoint(f: Layout) -> bool:
    """Every mode with non-zero stride has power-of-two size (or size 1), and the numbers
    2^b * r_j (b < log2 t_j) have pairwise disjoint binary supports."""
    seen = 0
    for t, r in zip(f.shape, f.stride):
        if t == 1 or r == 0:
            continue
        if t & (t - 1):
            return False
        for b in range(t.bit_length() - 1):
            v = r << b
            if v & seen:
                return False
            seen |= v
    return True


def binary_refinement(f: Layout) -> Layout:
    """The binary refinement of a bit-disjoint layout: every mode (2^a, r) with r != 0 becomes a
    binary modes with strides r, 2r, ..., 2^(a-1) r; modes with stride 0 (and unit modes) are
    kept, so the domain is unchanged."""
    shape, stride = [], []
    for t, r in zip(f.shape, f.stride):
        if t == 1 or r == 0:
            shape.append(t)
            stride.append(0)
            continue
        assert t & (t - 1) == 0
        for b in range(t.bit_length() - 1):
            shape.append(2)
            stride.append(r << b)
    return Layout(tuple(shape), tuple(stride))


def ordered_prime_factorizations(t: int) -> list[tuple]:
    ps, n, p = [], t, 2
    while n > 1:
        while n % p == 0:
            ps.append(p)
            n //= p
        p += 1
    return sorted(set(permutations(ps))) if ps else [(1,)]


def is_xor_layout_over_refinement(values: Sequence[int], shape: Sequence[int]) -> bool:
    """Brute force: is the function (value list over `shape`, colex) a layout with values in the
    XOR group over some by-mode refinement of `shape`?  It suffices to try the finest ones."""
    if values[0] != 0:
        return False
    for combo in product(*[ordered_prime_factorizations(t) for t in shape]):
        fine = tuple(s for part in combo for s in part)
        P = prefix_products(fine)
        ok = True
        for x in range(len(values)):
            want = 0
            for c, Pi in zip(digits(x, fine, False), P):
                if c & 1:
                    want ^= values[Pi]
            if want != values[x]:
                ok = False
                break
        if ok:
            return True
    return False


def decay(sw: Callable[[int], int], o: int, f: Layout) -> list[Layout] | None:
    """Exact decay: is  c |-> sigma(o + f(c)) - sigma(o)  an integer layout over a by-mode
    refinement of the shape of f?  Returns the per-mode coalesced layouts, or None."""
    base = sw(o)
    g = lambda x: sw(o + x) - base                       # noqa: E731
    per_mode = []
    for t, r in zip(f.shape, f.stride):
        Rj = recognize([g(r * x) for x in range(t)], 0)
        if Rj is None:
            return None
        per_mode.append(Rj)
    for coord in product(*[range(t) for t in f.shape]):
        ys = [c * r for c, r in zip(coord, f.stride)]
        if g(sum(ys)) != sum(g(y) for y in ys):
            return None
    return per_mode


def split_pairs(sw: Swizzle, supports: Sequence[frozenset]) -> list[int]:
    """Indices I whose bits y_I and z_I lie in two different modes."""
    out = []
    for i, (y, z) in enumerate(sw.pairs):
        my = [k for k, U in enumerate(supports) if y in U]
        mz = [k for k, U in enumerate(supports) if z in U]
        if my and mz and my[0] != mz[0]:
            out.append(i)
    return out


def cute_slice_model(sw: Swizzle, o: int, fd_value: int, fs: Layout) -> list[int]:
    """What CuTe's slice_and_offset computes when it does not decay: the swizzled layout with
    offset  o ^ P_zy(fd)  plus the constant  P_anti(fd)."""
    Pzy, Panti = sw.only_zy(), sw.anti_zy()
    o2 = o ^ Pzy(fd_value)
    return [sw(o2 + v) + Panti(fd_value) for v in fs.values()]


def transport(a: Layout, sw: Swizzle) -> Swizzle | None:
    """CuTe's composition(Layout, Swizzle): the swizzle with masks a(Y) and a(Z)."""
    return make_swizzle(a(sw.ymask), a(sw.zmask))
