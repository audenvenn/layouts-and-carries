"""Checks of the statements about swizzles (Section 8 of the paper).

    python3 -I python/verify_swizzle.py [seed]

Each check is exhaustive over a small universe or randomised; all must pass.
"""
from __future__ import annotations

import random
import sys
from itertools import product

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
from layout_theory import Layout  # noqa: E402
from swizzle_theory import (  # noqa: E402
    Swizzle, binary_carry_counts, binary_refinement, bit, cute_slice_model, decay, downcast_swizzle,
    is_bit_disjoint, is_xor_layout_over_refinement, make_swizzle, merge_swizzles, split_pairs, supp,
    swizzled_values, transport, upcast_pairs, upcast_swizzle,
)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = random.Random(SEED)
RESULTS = []


def check(name):
    def deco(fn):
        info = fn()
        RESULTS.append(name)
        print(f"ok  {name}: {info}")
        return fn
    return deco


def all_swizzles(maxB=3, maxM=3, maxS=4):
    out = []
    for B in range(0, maxB + 1):
        for M in range(0, maxM + 1):
            for S in range(-maxS, maxS + 1):
                if abs(S) >= B and (S != 0 or B == 0):
                    out.append(Swizzle(B, M, S))
    return out


def rand_swizzle():
    B = rng.randint(1, 3)
    S = rng.choice([-1, 1]) * rng.randint(B, B + 2)
    return Swizzle(B, rng.randint(0, 3), S)


def rand_layout(max_rank=3, sizes=(1, 2, 2, 2, 3, 4, 4, 5, 6, 8), max_stride=24):
    n = rng.randint(1, max_rank)
    return Layout(tuple(rng.choice(sizes) for _ in range(n)),
                  tuple(rng.randint(0, max_stride) for _ in range(n)))


# --------------------------------------------------------------------------------------

@check("W1  structure of a swizzle (involution, closed form, period, XOR-linearity, projections)")
def w1():
    n = 0
    for sw in all_swizzles():
        T = sw.top
        Pzy, Panti = sw.only_zy(), sw.anti_zy()
        lim = 1 << (T + 2)
        for x in range(lim):
            y = sw(x)
            assert sw(y) == x                                   # involution
            assert y == sw.closed_form(x)                       # closed form
            assert y % (1 << sw.M) == x % (1 << sw.M)           # low M bits fixed
            assert sw(x + 3 * (1 << T)) == y + 3 * (1 << T)     # period 2^T
            assert Pzy(x) == x & (sw.ymask | sw.zmask)          # projections are layouts
            assert Pzy(x) + Panti(x) == x
            assert y == sw(Pzy(x)) + Panti(x)
            n += 1
        for _ in range(200):                                    # linear for XOR
            a, b = rng.randrange(lim), rng.randrange(lim)
            assert sw(a ^ b) == sw(a) ^ sw(b)
        if sw.S > 0:                                            # preserves every [0, 2^t)
            for t in range(T + 3):
                assert sorted(sw(x) for x in range(1 << t)) == list(range(1 << t))
        for t in (T, T + 1):                                    # and every whole period
            assert sorted(sw(x) for x in range(1 << t)) == list(range(1 << t))
        # Latin square: for fixed other bits, (y, z) -> z ^ y is a bijection in each argument
        if sw.S > 0 and sw.B > 0:
            for z in range(1 << sw.B):
                col = {(sw((yy << (sw.M + sw.S)) | (z << sw.M)) >> sw.M) & ((1 << sw.B) - 1)
                       for yy in range(1 << sw.B)}
                assert len(col) == 1 << sw.B
    return f"{len(all_swizzles())} swizzles, {n} points"


@check("W2  swizzled composition formula (XOR-linear part + parity of binary carries)")
def w2():
    n = 0
    for _ in range(3000):
        sw, f, o = rand_swizzle(), rand_layout(), rng.randint(0, 40)
        nb = 12
        for coord in product(*[range(t) for t in f.shape]):
            x = o + sum(c * r for c, r in zip(coord, f.stride))
            want = sw(o)
            for c, r in zip(coord, f.stride):
                if c & 1:
                    want ^= sw(r)
            for b, C in enumerate(binary_carry_counts(o, coord, f.stride, nb), start=1):
                if C & 1:
                    want ^= sw(1 << b)
            assert sw(x) == want, (sw, f, o, coord)
            n += 1
    return f"{n} points"


@check("W3  sigma o f is an XOR-linear layout over a refinement  <=>  f is bit-disjoint")
def w3():
    yes = no = 0
    for _ in range(6000):
        sw = rand_swizzle()
        f = rand_layout(max_rank=3, sizes=(1, 2, 2, 2, 3, 4, 4, 6, 8), max_stride=12)
        got = is_xor_layout_over_refinement(swizzled_values(sw, 0, f), f.shape)
        assert got == is_bit_disjoint(f), (sw, f, got)
        yes += got
        no += not got
    return f"{yes} linear, {no} not"


def rand_bit_disjoint(nbits):
    """Random binary layout whose strides have pairwise disjoint supports, plus a disjoint offset."""
    owner = [rng.randint(-1, 3) for _ in range(nbits)]          # -1: unused, 3: offset
    strides = [sum(1 << b for b in range(nbits) if owner[b] == k) for k in range(3)]
    strides = [r for r in strides if r]
    o = sum(1 << b for b in range(nbits) if owner[b] == 3)
    if not strides:
        strides = [0]
    return Layout((2,) * len(strides), tuple(strides)), o


@check("W4  decay to an integer layout  <=>  no swizzle pair is split between two modes")
def w4():
    yes = no = 0
    for _ in range(20000):
        sw = rand_swizzle()
        f, o = rand_bit_disjoint(sw.top + 1)
        h = swizzled_values(sw, o, f)
        strides = [sw(o + r) - sw(o) for r in f.stride]
        # colex enumeration of the binary box
        ok = True
        for x in range(f.size):
            coord = [(x >> k) & 1 for k in range(f.rank)]
            if h[x] - h[0] != sum(c * d for c, d in zip(coord, strides)):
                ok = False
                break
        crit = not split_pairs(sw, [supp(r) for r in f.stride])
        assert ok == crit, (sw, f, o)
        assert ok == (decay(sw, o, f) is not None)
        if ok:
            # the sign table of the strides
            for r, d in zip(f.stride, strides):
                want = r
                for y, z in sw.pairs:
                    ry, rz = bit(r, y), bit(r, z)
                    if rz and not ry:
                        want += -(1 << z) * 2 * bit(o, y)             # +Z or -Z
                    elif ry and not rz:
                        want += (1 << z) * (1 - 2 * bit(o, z))        # Y+Z or Y-Z
                    elif ry and rz:
                        want -= (1 << z)                              # Y only
                assert d == want, (sw, f, o, r, d, want)
        yes += ok
        no += not ok
    return f"{yes} decay, {no} do not"


@check("W5  offset bookkeeping of slicing: valid when no carry crosses a field boundary")
def w5():
    good = bad_boundary = bad_overlap = 0
    for _ in range(30000):
        sw = rand_swizzle()
        Pzy = sw.only_zy()
        o = Pzy(rng.randrange(1 << sw.top))                     # offsets kept inside the fields
        a = rng.randrange(1 << (sw.top + 1))                    # value of the diced part
        fs = rand_layout(max_rank=2, sizes=(1, 2, 2, 4, 3), max_stride=1 << sw.top)
        truth = [sw(o + a + v) for v in fs.values()]
        model = cute_slice_model(sw, o, a, fs)
        no_carry = all(
            (o % Q + a % Q + v % Q) < Q for Q in sw.boundaries() for v in fs.values())
        disjoint = (o & Pzy(a)) == 0
        if no_carry and disjoint:
            assert model == truth, (sw, o, a, fs)
            good += 1
        elif model != truth:
            if not disjoint:
                bad_overlap += 1
            else:
                bad_boundary += 1
    assert bad_boundary > 0 and bad_overlap > 0
    return f"{good} valid; wrong without the hypotheses: {bad_boundary} (carry), {bad_overlap} (overlap)"


@check("W6  pushing a swizzle through a layout: sufficient condition, and CuTe's rule")
def w6():
    right = wrong = rejected = cond_ok = 0
    example = None
    for _ in range(20000):
        nb = rng.randint(2, 5)
        # a: binary shape with strides that are powers of two (possibly repeated) or small numbers
        if rng.random() < 0.7:
            strides = [1 << e for e in rng.sample(range(nb + 2), nb)]
        else:
            strides = [rng.randint(0, 12) for _ in range(nb)]
        a = Layout((2,) * nb, tuple(strides))
        B = rng.randint(1, max(1, nb // 2))
        S = rng.choice([-1, 1]) * rng.randint(B, max(B, nb - B))
        M = rng.randint(0, max(0, nb - B - abs(S)))
        if M + B + abs(S) > nb:
            continue
        sw = Swizzle(B, M, S)
        N = 1 << nb
        truth = [a(sw(x)) for x in range(N)]
        sw2 = transport(a, sw)
        if sw2 is None:
            rejected += 1
            continue
        got = [sw2(a(x)) for x in range(N)]
        # the sufficient condition: bits are transported pair by pair, and a is additive in z
        t1 = all(bit(a(x), y2) == bit(x, y) and bit(a(x), z2) == bit(x, z)
                 for (y, z), (y2, z2) in zip(sw.pairs, sw2.pairs) for x in range(N))
        t2 = all(a(x + (1 << z)) == a(x) + (1 << z2)
                 for (y, z), (y2, z2) in zip(sw.pairs, sw2.pairs) for x in range(N) if not bit(x, z))
        if t1 and t2:
            assert got == truth, (a, sw, sw2)
            cond_ok += 1
        if got == truth:
            right += 1
        else:
            wrong += 1
            if example is None or a.size < example[0].size:
                example = (a, sw, sw2)
    assert wrong > 0
    return (f"{cond_ok} satisfy the condition (all correct); CuTe's rule: {right} right, {wrong} wrong, "
            f"{rejected} rejected; e.g. {example[0]} o {example[1]} -> {example[2]}")


@check("W7  composition(Swizzle, Swizzle): the merge rule is sound")
def w7():
    merged = 0
    sws = [s for s in all_swizzles(3, 3, 4) if s.B > 0]
    for s0 in sws:
        for s1 in sws:
            m = merge_swizzles(s0, s1)
            if m is None:
                continue
            for x in range(1 << (max(s0.top, s1.top) + 1)):
                assert s0(s1(x)) == m(x), (s0, s1, m, x)
            merged += 1
    return f"{merged} admissible pairs"


@check("W8  casting: floor(sigma(x)/2^n) = upcast(sigma)(floor(x/2^n)) for S > 0, and on multiples of 2^n")
def w8():
    n_ok = n_neg_fail = 0
    for sw in all_swizzles(3, 3, 4):
        for n in range(0, 4):
            up, down = upcast_swizzle(sw, n), downcast_swizzle(sw, n)
            lim = 1 << (sw.top + n + 1)
            ok = all(sw(x) >> n == up(x >> n) for x in range(lim))
            if sw.S >= 0:
                assert ok, (sw, n)
                n_ok += 1
            elif not ok:
                n_neg_fail += 1
            for x in range(lim):
                assert down(x) == (sw(x >> n) << n) + (x & ((1 << n) - 1))
            for X in range(lim >> n):                           # on multiples of 2^n: always
                assert sw(X << n) >> n == up(X)
    assert n_neg_fail > 0
    return f"{n_ok} cases with S >= 0 hold; {n_neg_fail} cases with S < 0 fail"


@check("W9  cosize: sigma(o + f) can exceed the cosize of f")
def w9():
    sw, f = Swizzle(1, 0, 1), Layout((2,), (2,))
    assert max(swizzled_values(sw, 0, f)) + 1 == 4 and max(f.values()) + 1 == 3
    bigger = total = 0
    for _ in range(5000):
        sw, f = rand_swizzle(), rand_layout()
        total += 1
        bigger += max(swizzled_values(sw, 0, f)) > max(f.values())
    return f"example Sw<1,0,1> o 2:2; {bigger} of {total} random cases"


def rand_pairing(nbits, max_pairs=3):
    pos = rng.sample(range(nbits), min(nbits - nbits % 2, 2 * rng.randint(1, max_pairs)))
    return [(pos[2 * i], pos[2 * i + 1]) for i in range(len(pos) // 2)]


def sigma(pairs):
    def f(x):
        out = x
        for y, z in pairs:
            out ^= bit(x, y) << z
        return out
    return f


@check("W10 general pairings: involution, closed form, XOR-linearity, merging (Lemma 8.3, Section 8.3)")
def w10():
    n = merged = 0
    for _ in range(4000):
        nb = rng.randint(2, 8)
        pi, pj = rand_pairing(nb), rand_pairing(nb)
        s1, s2 = sigma(pi), sigma(pj)
        for x in range(1 << nb):
            y = s1(x)
            assert s1(y) == x
            assert y == x + sum((1 << z) * bit(x, yy) * (1 - 2 * bit(x, z)) for yy, z in pi)
            n += 1
        a, b = rng.randrange(1 << nb), rng.randrange(1 << nb)
        assert s1(a ^ b) == s1(a) ^ s1(b)
        # merging: sigma_pi o sigma_pj = sigma of the symmetric difference, if that is a pairing
        # and no z of pj is a y of pi
        sym = sorted(set(pi) ^ set(pj))
        positions = [q for pr in sym for q in pr]
        is_pairing = len(positions) == len(set(positions))
        no_chain = not ({z for _, z in pj} & {y for y, _ in pi})
        if is_pairing and no_chain:
            s3 = sigma(sym)
            assert all(s1(s2(x)) == s3(x) for x in range(1 << nb)), (pi, pj)
            merged += 1
    assert merged > 100
    return f"{n} points, {merged} merges"


@check("W11 decay criterion (Theorem 8.7) for general pairings")
def w11():
    yes = no = 0
    for _ in range(20000):
        nb = rng.randint(2, 8)
        pi = rand_pairing(nb)
        sg = sigma(pi)
        owner = [rng.randint(-1, 3) for _ in range(nb)]
        strides = [sum(1 << b for b in range(nb) if owner[b] == k) for k in range(3)]
        strides = [r for r in strides if r] or [0]
        o = sum(1 << b for b in range(nb) if owner[b] == 3)
        h = [sg(o + sum(r for k, r in enumerate(strides) if (x >> k) & 1)) for x in range(1 << len(strides))]
        deltas = [sg(o + r) - sg(o) for r in strides]
        ok = all(h[x] - h[0] == sum(d for k, d in enumerate(deltas) if (x >> k) & 1) for x in range(len(h)))
        supports = [supp(r) for r in strides]
        split = any(
            any(y in U for U in supports) and any(z in U for U in supports)
            and [k for k, U in enumerate(supports) if y in U] != [k for k, U in enumerate(supports) if z in U]
            for y, z in pi)
        assert ok == (not split), (pi, strides, o)
        yes += ok
        no += not ok
    return f"{yes} decay, {no} do not"


@check("W12 Corollary 8.6: the stated XOR layout over the binary refinement (zero-stride modes kept)")
def w12():
    n = 0
    for _ in range(4000):
        sw = rand_swizzle()
        f = rand_layout(max_rank=3, sizes=(1, 2, 2, 3, 4, 5, 8), max_stride=6)
        if rng.random() < 0.5:                         # force some zero strides
            k = rng.randrange(f.rank)
            f = Layout(f.shape, f.stride[:k] + (0,) + f.stride[k + 1:])
        if not is_bit_disjoint(f):
            continue
        fb = binary_refinement(f)
        assert fb.size == f.size and fb.values() == f.values()
        strides = [sw(r) for r in fb.stride]
        P = [1]
        for t in fb.shape:
            P.append(P[-1] * t)
        for x in range(f.size):
            want = 0
            for t, Pi, d in zip(fb.shape, P, strides):
                if ((x // Pi) % t) & 1:
                    want ^= d
            assert want == sw(f(x)), (sw, f)
        n += 1
    return f"{n} bit-disjoint layouts"


@check("W13 casting: floor(sigma(x)/2^n) = sigma^(n)(floor(x/2^n)) for all x iff no pair has y < n <= z")
def w13():
    n_all = n_fail = 0
    for _ in range(3000):
        nb = rng.randint(2, 8)
        pi = rand_pairing(nb)
        n = rng.randint(0, nb)
        sg, up = sigma(pi), sigma(upcast_pairs(pi, n))
        holds = all(sg(x) >> n == up(x >> n) for x in range(1 << nb))
        crossing = any(y < n <= z for y, z in pi)
        assert holds == (not crossing), (pi, n)
        assert all(sg(X << n) >> n == up(X) for X in range(1 << max(0, nb - n)))   # multiples: always
        n_all += holds
        n_fail += not holds
    # CuTe's upcast keeps exactly the pairs with y, z >= n
    for sw in all_swizzles(3, 3, 4):
        for n in range(0, 5):
            assert sorted(upcast_swizzle(sw, n).pairs) == sorted(upcast_pairs(sw.pairs, n)), (sw, n)
    return f"{n_all} hold for all x, {n_fail} only on multiples of 2^n"


@check("W14 merging with B = 0, and the reviewer's counterexamples")
def w14():
    # merging with the identity swizzle
    for sw in all_swizzles(3, 3, 4):
        for ident in (Swizzle(0, 0, sw.S), Swizzle(0, sw.M, sw.S)):
            m = merge_swizzles(sw, ident)
            if m is not None:
                assert all(m(x) == sw(x) for x in range(1 << (sw.top + 1)))
    # pi = {2->1}, pi' = {1->0}: the symmetric difference is not a pairing and the composite
    # is not an involution
    s1, s2 = sigma([(2, 1)]), sigma([(1, 0)])
    assert s1(s2(4)) == 6 and s1(s2(6)) == 5
    # a non-zero offset that overlaps the fixed coordinate: Sw<1,0,1>, o = 1, f = (2,2):(1,2)
    sw = Swizzle(1, 0, 1)
    truth = [sw(1 + 1 + 2 * k) for k in range(2)]
    model = cute_slice_model(sw, 1, 1, Layout((2,), (2,)))
    assert truth == [3, 4] and model != truth
    # compact layout of size 3: cosize too small although S > 0
    assert [sw(v) for v in Layout((3,), (1,)).values()] == [0, 1, 3]
    return "ok"


print(f"\nall {len(RESULTS)} swizzle checks passed")
