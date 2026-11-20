"""Differential test of PyCuTe's F2-stride layouts (its representation of swizzles).

    python3 -I python/difftest_f2.py <path-to-NVlabs-CuTe> [seed] [n]

An F2 stride acts on a coordinate through the bits of the coordinate (carry-less product), so a
mode of size 2^p with stride d is p binary modes with strides d, d<<1, ..., d<<(p-1): an
XOR-valued layout over a binary shape.  We test

  composition(A, B)   A: F2 strides, power-of-two extents;  B: integer strides
  right_inverse(A)    A: F2 strides  (post-condition of the implementation)

against pointwise evaluation, and decide existence of a composite by brute force over all
by-mode refinements of the shape of B.  Output: JSON on stdout.
"""
from __future__ import annotations

import json
import random
import sys
from itertools import product
from math import prod

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
from layout_theory import digits, prefix_products  # noqa: E402
from layout_theory import Layout  # noqa: E402
from swizzle_theory import binary_carry_counts, is_bit_disjoint  # noqa: E402

IMPL = sys.argv[1]
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
N = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
sys.path.insert(0, IMPL)
import pycute as pc  # noqa: E402

rng = random.Random(SEED)


def clmul(c: int, d: int) -> int:
    """Carry-less product: XOR of d << b over the set bits b of c."""
    out, b = 0, 0
    while c >> b:
        if (c >> b) & 1:
            out ^= d << b
        b += 1
    return out


def f2_eval(shape, stride, x: int) -> int:
    """Reference evaluation of an F2-stride layout at an integer index (last mode unbounded)."""
    out = 0
    for c, d in zip(digits(x, shape, True), stride):
        out ^= clmul(c, d)
    return out


def ordered_factorizations(t: int) -> list[tuple]:
    if t == 1:
        return [(1,)]
    out = [(t,)]
    for u in range(2, t):
        if t % u == 0:
            out += [(u,) + rest for rest in ordered_factorizations(t // u)]
    return out


def composable(g: list[int], shapeB) -> bool:
    """Is g (values over shapeB, colex) the function of an F2-stride layout over some by-mode
    refinement of shapeB?  The strides are forced: the values at the unit coordinates."""
    if g[0] != 0:
        return False
    for combo in product(*[ordered_factorizations(t) for t in shapeB]):
        fine = tuple(s for part in combo for s in part)
        P = prefix_products(fine)
        strides = [g[Pi] if s > 1 else 0 for s, Pi in zip(fine, P)]
        if all(g[x] == _xor(clmul(c, d) for c, d in zip(digits(x, fine, False), strides))
               for x in range(len(g))):
            return True
    return False


def _xor(items) -> int:
    out = 0
    for v in items:
        out ^= v
    return out


def mk(shape, stride, f2: bool):
    st = tuple(pc.F2(d) for d in stride) if f2 else tuple(stride)
    return pc.Layout(shape[0], st[0]) if len(shape) == 1 else pc.Layout(tuple(shape), st)


def rand_f2_layout():
    fam = rng.choice(["swizzle", "swizzle", "random"])
    if fam == "swizzle":            # (2^a, 2^b):(F2(1), F2(2^a + low bits)):  ZZZ ^= YYY
        a, b = rng.randint(1, 3), rng.randint(1, 3)
        low = rng.randrange(1 << a)
        return (1 << a, 1 << b), (1, (1 << a) | low), fam
    n = rng.randint(1, 3)
    shape = tuple(1 << rng.randint(1, 2) for _ in range(n))
    return shape, tuple(rng.randrange(1, 32) for _ in range(n)), fam


def rand_int_layout(nbits: int):
    """Returns (shape, stride, family).  'disjoint': power-of-two sizes and strides on disjoint bit
    ranges (the modes of a tile of a power-of-two array); 'random': anything small."""
    if rng.random() < 0.5:
        nb = max(2, nbits)
        cuts = sorted(rng.sample(range(1, nb), min(nb - 1, rng.randint(0, 2))))
        segs = [(a, b) for a, b in zip([0] + cuts, cuts + [nb])]
        rng.shuffle(segs)
        segs = segs[:rng.randint(1, len(segs))]
        return tuple(1 << (b - a) for a, b in segs), tuple(1 << a for a, b in segs), "disjoint"
    n = rng.randint(1, 3)
    shape = tuple(rng.choice((2, 2, 2, 3, 4, 4, 8)) for _ in range(n))
    stride = tuple(rng.choice((0, 1, 1, 2, 2, 4, 8, 16, 3, 5, 6)) for _ in range(n))
    return shape, stride, "random"


def main():
    comp = dict(cases=0, correct=0, correctly_rejected=0, silently_wrong=0, rejected_but_composable=0,
                wrong_size=0, examples_wrong=[], examples_rejected=[],
                wrong_with_odd_binary_carry=0)
    for _ in range(N):
        sA, dA, fam = rand_f2_layout()
        sB, dB, famB = rand_int_layout(prod(sA).bit_length() - 1)
        fam_t = comp.setdefault("by_family_of_B", {}).setdefault(
            famB, dict(cases=0, correct=0, correctly_rejected=0, silently_wrong=0, rejected_but_composable=0))
        fam_t["cases"] += 1
        # classification by Definition 8.5 (not by the sampling label)
        cls = "B_bit_disjoint" if is_bit_disjoint(Layout(sB, dB)) else "B_not_bit_disjoint"
        cls_t = comp.setdefault("by_definition", {}).setdefault(
            cls, dict(cases=0, correct=0, correctly_rejected=0, silently_wrong=0, rejected_but_composable=0))
        cls_t["cases"] += 1
        A, B = mk(sA, dA, True), mk(sB, dB, False)
        nB = prod(sB)
        vals_B = [sum(c * r for c, r in zip(digits(i, sB, False), dB)) for i in range(nB)]
        g = [f2_eval(sA, dA, v) for v in vals_B]
        exists = composable(g, sB)
        comp["cases"] += 1
        try:
            R = pc.composition(A, B)
        except Exception as e:                        # noqa: BLE001
            fam_t["rejected_but_composable" if exists else "correctly_rejected"] += 1
            cls_t["rejected_but_composable" if exists else "correctly_rejected"] += 1
            if exists:
                comp["rejected_but_composable"] += 1
                if len(comp["examples_rejected"]) < 5:
                    comp["examples_rejected"].append(dict(A=f"{sA}:F2{dA}", B=f"{sB}:{dB}", error=str(e)[:60]))
            else:
                comp["correctly_rejected"] += 1
            continue
        if pc.size(R) != nB:
            comp["wrong_size"] += 1
            continue
        got = [int(R(i)) for i in range(nB)]
        if got == g:
            comp["correct"] += 1
            fam_t["correct"] += 1
            cls_t["correct"] += 1
            continue
        comp["silently_wrong"] += 1
        fam_t["silently_wrong"] += 1
        cls_t["silently_wrong"] += 1
        # each mode on its own is right?  then the error is interference between modes
        per_mode_ok = all(
            got[c * P] == g[c * P] for t, P in zip(sB, prefix_products(sB)) for c in range(t))
        kind = ("a single mode is wrong" if not per_mode_ok else
                "interference, no composite exists" if not exists else "interference, a composite exists")
        comp.setdefault("wrong_by_cause", {}).setdefault(kind, 0)
        comp["wrong_by_cause"][kind] += 1
        # the theory: a wrong value needs a binary carry (of odd count at some bit)
        odd = False
        for i in range(nB):
            if got[i] != g[i]:
                odd |= any(C & 1 for C in binary_carry_counts(0, digits(i, sB, False), dB, 12))
        comp["wrong_with_odd_binary_carry"] += odd
        comp["examples_wrong"].append(dict(A=f"{sA}:F2{dA}", B=f"{sB}:{dB}", cause=kind, result=str(R), got=got, truth=g))
    comp["examples_wrong"] = sorted(comp["examples_wrong"], key=lambda e: len(e["truth"]))[:5]

    inv = dict(cases=0, returned=0, postcondition_holds=0, true_right_inverse=0, errors=0)
    for _ in range(N // 5):
        sA, dA, fam = rand_f2_layout()
        A = mk(sA, dA, True)
        inv["cases"] += 1
        try:
            R = pc.right_inverse(A)
        except Exception:                             # noqa: BLE001
            inv["errors"] += 1
            continue
        inv["returned"] += 1
        n = pc.size(R)
        Ri = [int(R(i)) for i in range(n)]
        post = all(int(R(int(A(Ri[i])))) == Ri[i] for i in range(n))
        inv["postcondition_holds"] += post
        inv["true_right_inverse"] += all(int(A(Ri[i])) == i for i in range(n))
    print(json.dumps(dict(seed=SEED, composition=comp, right_inverse=inv), indent=1))


if __name__ == "__main__":
    main()
