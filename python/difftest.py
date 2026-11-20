"""Differential test: theory (layout_theory.py) versus a Python implementation of CuTe.

    python3 -I python/difftest.py <label> <path> [seed] [n]

<label> is "cutlass" (CUTLASS python/pycute) or "nvlabs" (PyCuTe).  Prints a JSON report.

Ground truth is always computed pointwise by this file and layout_theory.py; an
implementation's answer is "correct" iff it is a layout of the right size whose values agree
with the pointwise function (and whose shape refines the inner layout's shape, for composition).
Layouts are small; distributions are arbitrary, so the counts show that phenomena exist and how
they split by cause, not how often they occur in real kernels.
"""
from __future__ import annotations

import json
import random
import sys
from itertools import product
from math import prod

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
from impl_adapter import Impl, coarsens, flatten, tsize  # noqa: E402
from layout_theory import (  # noqa: E402
    Layout, all_factorizations, coalesce, complement, compose, concat, digits, prefix_products,
    recognize, sorted_modes,
)

# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------

def nest(items: list, rng: random.Random, p=0.5):
    """Randomly parenthesise a list (order preserved) into a nested tuple of rank >= 1."""
    if len(items) == 1:
        return items[0] if rng.random() < 0.7 else (items[0],)
    if rng.random() > p:
        return tuple(items)
    k = rng.randint(1, len(items) - 1)
    cuts = sorted(rng.sample(range(1, len(items)), min(k, len(items) - 1)))
    parts, prev = [], 0
    for c in cuts + [len(items)]:
        parts.append(items[prev:c])
        prev = c
    return tuple(nest(part, rng, p * 0.6) if len(part) > 1 else part[0] for part in parts)


def nest_pair(shape, stride, rng):
    """Apply the same random nesting to a shape and a stride tuple."""
    idx = nest(list(range(len(shape))), rng)

    def sub(t, vals):
        return tuple(sub(x, vals) for x in t) if isinstance(t, tuple) else vals[t]

    if not isinstance(idx, tuple):
        return shape[idx], stride[idx]
    return sub(idx, shape), sub(idx, stride)


def coalesce_ext(L: Layout) -> Layout:
    """Normal form of the *extended* function: non-zero jumps only, a trailing unit mode kept
    when there is a jump at scale |L|."""
    a, jumps = L.slope_and_jumps()
    agg: dict[int, int] = {}
    for P, g in jumps:
        if P == 1:
            a += g
        else:
            agg[P] = agg.get(P, 0) + g
    scales = sorted(P for P, g in agg.items() if g != 0)
    N = L.size
    bounds = scales + [max(N, scales[-1]) if scales else N]
    shape = [bounds[0]] + [bounds[i + 1] // bounds[i] for i in range(len(scales))]
    stride, d = [a], a
    for i, P in enumerate(scales):
        d = agg[P] + shape[i] * d
        stride.append(d)
    return Layout(tuple(shape), tuple(stride))


def why_rejected(Af: Layout, t: int, r: int) -> str:
    """Classify a single-mode pair that composes although the monomial rule may reject it."""
    Ac = coalesce_ext(Af)
    dg = digits(r, Ac.shape)
    nz = [i for i, c in enumerate(dg) if c]
    if len(nz) >= 2:
        return "sum_stride"            # stride has several non-zero digits
    if not nz:
        return "zero_stride"
    k, c = nz[0], dg[nz[0]]
    if k < Ac.rank - 1 and Ac.shape[k] % c != 0:
        return "non_dividing"          # single digit c that does not divide s_k
    return "other"


def as_layout(Rs, Rd):
    """The implementation's result as a theory Layout, or None if its shape has an entry < 1."""
    fs = flatten(Rs)
    if any(x < 1 for x in fs):
        return None
    return Layout(fs, flatten(Rd))


class Tally:
    def __init__(self):
        self.counts: dict[str, int] = {}
        self.examples: dict[str, list] = {}

    def add(self, cat: str, example=None, weight: int = 0):
        self.counts[cat] = self.counts.get(cat, 0) + 1
        if example is not None:
            ex = self.examples.setdefault(cat, [])
            ex.append((weight, example))
            ex.sort(key=lambda p: p[0])
            del ex[4:]

    def report(self):
        return {"counts": dict(sorted(self.counts.items())),
                "examples": {k: [e for _, e in v] for k, v in sorted(self.examples.items())}}


def show(shape, stride) -> str:
    return f"{shape}:{stride}".replace(" ", "")


SIZES_A = (1, 2, 2, 3, 4, 5, 6, 8)
STRIDES_A = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 24)
SIZES_B = (1, 2, 2, 3, 4, 6)
STRIDES_B = (0, 1, 2, 3, 4, 5, 6, 8, 9, 12)


def rand_flat(rng, max_rank, sizes, strides):
    r = rng.randint(1, max_rank)
    return tuple(rng.choice(sizes) for _ in range(r)), tuple(rng.choice(strides) for _ in range(r))


# ------------------------------------------------------------------------------------------
# suites
# ------------------------------------------------------------------------------------------

def suite_composition(impl: Impl, rng, n: int, nested: bool, structured: bool = False) -> dict:
    T = Tally()
    for _ in range(n):
        if structured:
            As, Ad, Bs, Bd = structured_pair(rng)
        else:
            As, Ad = rand_flat(rng, 3, SIZES_A, STRIDES_A)
            Bs, Bd = rand_flat(rng, 3, SIZES_B, STRIDES_B)
        Af, Bf = Layout(As, Ad), Layout(Bs, Bd)
        An = nest_pair(As, Ad, rng) if nested else ((As, Ad) if len(As) > 1 else (As[0], Ad[0]))
        Bn = nest_pair(Bs, Bd, rng) if nested else ((Bs, Bd) if len(Bs) > 1 else (Bs[0], Bd[0]))
        A, B = impl.mk(*An), impl.mk(*Bn)
        want = [Af(b) for b in Bf.values()]
        truth = compose(Af, Bf)
        ex = f"{show(*An)} o {show(*Bn)}"
        wgt = Af.size * Bf.size + 10 * (Af.rank + Bf.rank)
        st, R = impl.composition(A, B)
        if st == "ok":
            Rshape, Rstride = impl.tup(R)
            if as_layout(Rshape, Rstride) is None:
                T.add("silent_wrong", None)
                T.add("silent_wrong/result_has_zero_extent", f"{ex} -> {show(Rshape, Rstride)}", wgt)
                continue
            got = impl.values(R) if impl.size(R) <= 4096 else None
            mine = Layout(flatten(Rshape), flatten(Rstride)).values() if tsize(Rshape) <= 4096 else None
            if got != mine:
                T.add("impl_evaluation_differs_from_flat_semantics", f"{ex} -> {show(Rshape, Rstride)}", wgt)
            if tsize(Rshape) != Bf.size:
                T.add("silent_wrong", None)
                T.add("silent_wrong/result_has_wrong_size", f"{ex} -> {show(Rshape, Rstride)}; A(B(i))={want}", wgt)
            elif got == want:
                T.add("correct")
                if not coarsens(Bn[0], Rshape):
                    T.add("correct_values_but_shape_not_refining_B", f"{ex} -> {show(Rshape, Rstride)}", wgt)
            else:
                T.add("silent_wrong", None)
                # is every single mode handled correctly, so that only interference is to blame?
                per_mode_ok = True
                for t, r in zip(Bs, Bd):
                    s1, R1 = impl.composition(A, impl.mk(t, r))
                    if s1 != "ok" or impl.size(R1) != t or impl.values(R1) != [Af(r * x) for x in range(t)]:
                        per_mode_ok = False
                sub = "interference_between_modes" if per_mode_ok and len(Bs) > 1 else "other"
                T.add(f"silent_wrong/{sub}", f"{ex} -> {show(Rshape, Rstride)}={got}; A(B(i))={want}", wgt)
                T.add("silent_wrong/a_layout_exists" if truth is not None else "silent_wrong/no_layout_exists")
        else:
            if truth is None:
                T.add("correctly_rejected")
            else:
                T.add("rejected_but_composable", None)
                cause = "other"
                for t, r in zip(Bs, Bd):
                    s1, _ = impl.composition(A, impl.mk(t, r))
                    if s1 != "ok":
                        cause = why_rejected(Af, t, r)
                        break
                res = " , ".join(repr(Rj) for Rj in truth)
                T.add(f"rejected_but_composable/{cause}", f"{ex} = ({res})  [{R}]", wgt)
    return T.report()


def structured_pair(rng):
    """'Realistic' pairs: A a matrix-like data layout of a power-of-two-ish size (column-major,
    row-major, padded, tiled), B a compact permutation or a tile of that size."""
    M = rng.choice([8, 12, 16, 24, 32, 36, 48, 64])
    facA = rng.choice([f for f in all_factorizations(M) if 2 <= len(f) <= 3])
    kind = rng.choice(["col", "row", "padded", "perm"])
    Pp = prefix_products(facA)
    if kind == "col":
        Ad = tuple(Pp[:-1])
    elif kind == "row":
        Ad = tuple(M // Pp[i + 1] for i in range(len(facA)))
    elif kind == "padded":
        pad, d, Ad = rng.randint(1, 3), 1, []
        for s in facA:
            Ad.append(d)
            d = d * s + pad
        Ad = tuple(Ad)
    else:
        perm = list(range(len(facA)))
        rng.shuffle(perm)
        ranked = [facA[i] for i in perm]
        pp = prefix_products(ranked)
        Ad = [0] * len(facA)
        for pos, i in enumerate(perm):
            Ad[i] = pp[pos]
        Ad = tuple(Ad)
    facB = list(rng.choice(all_factorizations(M)))
    PpB = prefix_products(facB)
    perm = list(range(len(facB)))
    rng.shuffle(perm)
    Bs = tuple(facB[i] for i in perm)
    Bd = tuple(PpB[i] for i in perm)
    if rng.random() < 0.4 and len(Bs) > 1:       # a tile: drop some modes
        keep = sorted(rng.sample(range(len(Bs)), rng.randint(1, len(Bs) - 1)))
        Bs, Bd = tuple(Bs[i] for i in keep), tuple(Bd[i] for i in keep)
    return tuple(facA), Ad, Bs, Bd


def suite_coalesce(impl: Impl, rng, n: int) -> dict:
    T = Tally()
    for _ in range(n):
        s, d = rand_flat(rng, 5, SIZES_A, STRIDES_A)
        Ln = nest_pair(s, d, rng)
        want = coalesce(Layout(s, d))
        st, R = impl.coalesce(impl.mk(*Ln))
        if st != "ok":
            T.add("error", f"{show(*Ln)}: {R}")
            continue
        Rs, Rd = impl.tup(R)
        got = as_layout(Rs, Rd)
        if got is None:
            T.add("result_has_zero_extent", f"{show(*Ln)} -> {show(Rs, Rd)}", prod(s))
        elif got == want:
            T.add("identical_to_normal_form")
        elif got.size == 1 and got.rank == 1 and want.size == 1:
            T.add("identical_up_to_the_stride_of_a_size_1_layout")
        elif got.values() == want.values():
            T.add("same_function_but_not_normal_form", f"{show(*Ln)} -> {show(Rs, Rd)}; normal form {want}", prod(s))
        else:
            T.add("wrong_function", f"{show(*Ln)} -> {show(Rs, Rd)}", prod(s))
    return T.report()


def suite_complement(impl: Impl, rng, n: int) -> dict:
    T = Tally()
    for _ in range(n):
        s, d = rand_flat(rng, 3, (1, 2, 2, 3, 4), (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 24))
        M = rng.choice([4, 6, 8, 12, 16, 18, 24, 36, 48, 60, 72])
        L = Layout(s, d)
        I = L.values()
        ex = f"complement({show(s, d)}, {M})"
        want = complement(L, M) if len(set(I)) == len(I) else None
        st, R = impl.complement(impl.mk(*((s, d) if len(s) > 1 else (s[0], d[0]))), M)
        if st != "ok":
            T.add("rejected/although_a_complement_exists" if want is not None else "rejected/no_complement_exists",
                  f"{ex}: {R}" if want is not None else None, L.size)
            continue
        Rs, Rd = impl.tup(R)
        C = as_layout(Rs, Rd)
        if C is None:
            T.add("returned_layout_with_zero_extent/" + ("although_a_complement_exists" if want is not None
                                                          else ("non_injective_input" if len(set(I)) != len(I) else "no_complement_exists")),
                  f"{ex} -> {show(Rs, Rd)}", L.size)
            continue
        if want is not None:
            if C.values() == want.values():
                T.add("tiling/identical" if coalesce(C) == want else "tiling/same_function")
            else:
                T.add("tiling/different_from_theory", f"{ex} -> {C}; theory {want}", L.size)
            continue
        # no complement in [0, M) exists; what did the implementation return?
        vals = C.values()
        both = concat(L, C).values()
        if len(set(I)) != len(I):
            T.add("returned_for_non_injective_layout", f"{ex} -> {C}", L.size)
        elif sorted(both) == list(range(len(both))):
            T.add("non_tiling/tiles_a_longer_interval")       # rounding M up to a multiple
        else:
            ok = vals == sorted(set(vals)) and len(set(both)) == len(both)
            T.add("non_tiling/returned_disjoint_ordered_layout" if ok else "non_tiling/postconditions_violated",
                  f"{ex} -> {C}" if not ok else None, L.size)
    return T.report()


def rand_complementable(rng, M):
    """A random injective layout that has a complement in [0, M)."""
    fac = list(rng.choice(all_factorizations(M)))
    Pp = prefix_products(fac)
    pos = [i for i in range(len(fac)) if rng.random() < 0.5] or [0]
    rng.shuffle(pos)
    return tuple(fac[i] for i in pos), tuple(Pp[i] for i in pos)


def suite_divide(impl: Impl, rng, n: int) -> dict:
    T = Tally()
    for _ in range(n):
        M = rng.choice([4, 6, 8, 12, 16, 18, 24, 30, 36, 48])
        fac = rng.choice(all_factorizations(M))
        As, Ad = tuple(fac), tuple(rng.choice(STRIDES_A) for _ in fac)
        Bs, Bd = rand_complementable(rng, M)
        Af, Bf = Layout(As, Ad), Layout(Bs, Bd)
        C = complement(Bf, M)
        E = concat(Bf, C)
        want = [Af(e) for e in E.values()]
        truth = compose(Af, E)
        # the criterion of the division theorem, computed independently of `compose`
        scalesA = set(prefix_products(coalesce(Af).shape)[1:-1]) if M > 1 else set()
        scalesU = {x for dd, s in sorted_modes(Bf) for x in (dd, dd * s) if 1 < x < M}
        crit = all(a % u == 0 or u % a == 0 for a in scalesA for u in scalesU)
        if crit != (truth is not None):
            T.add("THEORY_CRITERION_MISMATCH", f"{Af} / {Bf}")
        ex = f"logical_divide({Af}, {Bf})"
        st, R = impl.logical_divide(impl.mk(*((As, Ad) if len(As) > 1 else (As[0], Ad[0]))),
                                    impl.mk(*((Bs, Bd) if len(Bs) > 1 else (Bs[0], Bd[0]))))
        if st == "ok":
            Rs, Rd = impl.tup(R)
            RL = as_layout(Rs, Rd)
            got = RL.values() if RL is not None else None
            if got is None:
                T.add("result_has_zero_extent", f"{ex} -> {show(Rs, Rd)}", M)
            elif got == want:
                T.add("correct")
            elif len(got) != M:
                # the implementation tiled a different number of elements than |A|
                C2 = complement(Bf, len(got))
                same = C2 is not None and got == [Af(e) for e in concat(Bf, C2).values()]
                kind = "more" if len(got) > M else "fewer"
                T.add(f"covers_{kind}_elements_than_A/" + ("values_consistent" if same else "values_wrong"),
                      f"{ex} -> {show(Rs, Rd)} has size {len(got)}, |A|={M}", M)
            else:
                T.add("silent_wrong", f"{ex} -> {show(Rs, Rd)}={got}; truth {want}", M)
        elif truth is None:
            T.add("correctly_rejected")
        else:
            T.add("rejected_but_divisible", f"{ex}: exists as {truth}  [{R}]", M)
    return T.report()


def suite_inverse(impl: Impl, rng, n: int) -> dict:
    T = Tally()
    for _ in range(n):
        if rng.random() < 0.5:
            M = rng.choice([4, 6, 8, 12, 16, 24, 36])
            s, d = rand_complementable(rng, M)
        else:
            s, d = rand_flat(rng, 3, (1, 2, 3, 4), (0, 1, 2, 3, 4, 6, 8, 12))
        L = Layout(s, d)
        A = impl.mk(*((s, d) if len(s) > 1 else (s[0], d[0])))
        # largest K such that the modes sorted by stride start a column-major layout of size K
        K, cur = 1, 1
        for dd, ss in sorted(sorted_modes(L), key=lambda m: (m[0], -m[1])):   # ties: larger mode first
            if dd == 0:
                continue
            if dd != cur:
                break
            K, cur = K * ss, dd * ss
        st, R = impl.right_inverse(A)
        if st != "ok":
            T.add("right_inverse/error", f"right_inverse({L}): {R}", L.size)
        else:
            Rs, Rd = impl.tup(R)
            RL = as_layout(Rs, Rd)
            Rv = RL.values() if RL is not None else []
            ok = RL is not None and all(L(Rv[i]) == i for i in range(len(Rv)))
            if not ok:
                T.add("right_inverse/postcondition_violated", f"right_inverse({L}) -> {show(Rs, Rd)}", L.size)
            elif len(Rv) == K:
                T.add("right_inverse/correct_and_maximal")
            else:
                T.add("right_inverse/correct_but_size_differs_from_compact_part", f"right_inverse({L}) -> {show(Rs, Rd)}, K={K}", L.size)
        I = L.values()
        if len(set(I)) == len(I) and all(dd > 0 for dd, _ in sorted_modes(L)):
            st, R = impl.left_inverse(A)
            if st != "ok":
                chained = complement(L, max(dd * ss for dd, ss in sorted_modes(L)) if sorted_modes(L) else 1) is not None
                T.add("left_inverse/rejected_injective_layout_" + ("with_stride_chain" if chained else "without_stride_chain"),
                      f"left_inverse({L}): {R}" if chained else None, L.size)
            else:
                Rs, Rd = impl.tup(R)
                Rl = as_layout(Rs, Rd)
                ok = Rl is not None and all(Rl(L(i)) == i for i in range(L.size))
                T.add("left_inverse/correct" if ok else "left_inverse/postcondition_violated",
                      None if ok else f"left_inverse({L}) -> {show(Rs, Rd)}", L.size)
    return T.report()


def suite_product(impl: Impl, rng, n: int) -> dict:
    T = Tally()
    for _ in range(n):
        M = rng.choice([4, 6, 8, 12, 16, 24])
        As, Ad = rand_complementable(rng, M)
        Bs, Bd = rand_flat(rng, 2, (1, 2, 3, 4), (0, 1, 2, 3, 4, 5, 6))
        Af, Bf = Layout(As, Ad), Layout(Bs, Bd)
        cosB = 1 + sum((t - 1) * r for t, r in zip(Bs, Bd))
        period = max(dd * ss for dd, ss in sorted_modes(Af)) if sorted_modes(Af) else 1
        Cstar = complement(Af, period * (cosB + 1))           # long enough to cover B's values
        if Cstar is None:
            continue
        want = [Af(x) + Cstar(b) for b in Bf.values() for x in range(Af.size)]
        truth = compose(Cstar, Bf)
        ex = f"logical_product({Af}, {Bf})"
        st, R = impl.logical_product(impl.mk(*((As, Ad) if len(As) > 1 else (As[0], Ad[0]))),
                                     impl.mk(*((Bs, Bd) if len(Bs) > 1 else (Bs[0], Bd[0]))))
        if st == "ok":
            Rs, Rd = impl.tup(R)
            RL = as_layout(Rs, Rd)
            got = RL.values() if RL is not None else None
            T.add("correct" if got == want else "silent_wrong",
                  None if got == want else f"{ex} -> {show(Rs, Rd)}={got}; truth {want}", Af.size * Bf.size)
        elif truth is None:
            T.add("correctly_rejected")
        else:
            T.add("rejected_but_exists", f"{ex}  [{R}]", Af.size * Bf.size)
    return T.report()


if __name__ == "__main__":
    label, path = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
    impl = Impl(label, path)
    out = {"implementation": label, "seed": seed, "n": n}
    out["composition_random_flat"] = suite_composition(impl, random.Random(seed), n, nested=False)
    out["composition_random_nested"] = suite_composition(impl, random.Random(seed + 1), n, nested=True)
    out["composition_structured"] = suite_composition(impl, random.Random(seed + 2), n, nested=False, structured=True)
    out["coalesce"] = suite_coalesce(impl, random.Random(seed + 3), n)
    out["complement"] = suite_complement(impl, random.Random(seed + 4), n)
    out["logical_divide"] = suite_divide(impl, random.Random(seed + 5), n)
    out["logical_product"] = suite_product(impl, random.Random(seed + 6), n // 2)
    out["inverse"] = suite_inverse(impl, random.Random(seed + 7), n // 2)
    print(json.dumps(out, indent=1, ensure_ascii=False))
