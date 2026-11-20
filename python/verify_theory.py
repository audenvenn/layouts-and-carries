"""Brute-force checks of the main statements and examples of the paper (Sections 3-6).

Run:  python3 -I python/verify_theory.py [seed]

Each check either exhausts a small universe or samples at random; a failure raises
AssertionError with the offending instance.  These are sanity checks, not proofs.
"""
from __future__ import annotations

import random
import sys
from itertools import product
from math import ceil, gcd, prod

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from layout_theory import (  # noqa: E402
    Layout, all_factorizations, carries, carry_defect, carry_free, coalesce, complement, compose,
    compose_function, concat, digits, floor_spectrum, g_add, g_is_zero, g_smul, g_sub, g_sum,
    is_additive_over, is_compact, is_complementable, is_direct_sum, naive_composite,
    prefix_products, recognize, recognize_greedy, sorted_modes,
)

rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
REPORT: list[str] = []


def report(name: str, detail: str):
    REPORT.append(f"[ok] {name}: {detail}")
    print(REPORT[-1], flush=True)


def rand_shape(max_rank=4, sizes=(1, 2, 2, 3, 3, 4, 5, 6)):
    return tuple(rng.choice(sizes) for _ in range(rng.randint(1, max_rank)))


def rand_layout(max_rank=4, strides=range(-6, 25), sizes=(1, 2, 2, 3, 3, 4, 5, 6), vec=0):
    shape = rand_shape(max_rank, sizes)
    if vec:
        stride = tuple(tuple(rng.choice(list(strides)) for _ in range(vec)) for _ in shape)
    else:
        stride = tuple(rng.choice(list(strides)) for _ in shape)
    return Layout(shape, stride)


# ---------------------------------------------------------------------------------------
# T1. floor-sum (jump) form and the carry identity, over Z and Z^2, negative strides included
# ---------------------------------------------------------------------------------------
def check_floor_sum_and_carry(n=4000):
    for _ in range(n):
        L = rand_layout(vec=rng.choice([0, 0, 2]))
        for x in range(0, 3 * L.size + 5):
            assert g_is_zero(g_sub(L(x), L.eval_floor_sum(x))), (L, x)
        _, jumps = L.slope_and_jumps()
        for _ in range(20):
            xs = [rng.randrange(0, 2 * L.size + 3) for _ in range(rng.randint(2, 4))]
            lhs = g_sub(L(sum(xs)), g_sum((L(x) for x in xs), L.zero))
            rhs = g_sum((g_smul(carries(xs, P), gam) for P, gam in jumps), L.zero)
            assert g_is_zero(g_sub(lhs, rhs)), (L, xs)
            for P, _g in jumps:
                c = carries(xs, P)
                assert 0 <= c <= len(xs) - 1 and c == sum(x % P for x in xs) // P
    report("T1 jump form + carry identity", f"{n} random layouts (Z and Z^2 strides, negative strides allowed)")


# ---------------------------------------------------------------------------------------
# T2. normal form: spectral criterion == definition; coalesce is the normal form; the shapes
#     over which a layout function is additive are exactly the refinements of its coalesced shape
# ---------------------------------------------------------------------------------------
def check_recognition_exhaustive():
    total = layouts = 0
    for N in range(1, 10):
        vals = range(0, 3) if N >= 8 else range(-1, 3)
        for tail in product(vals, repeat=N - 1):
            g = [0, *tail]
            a, b = recognize(g), recognize_greedy(g)
            assert (a is None) == (b is None), g
            total += 1
            if a is not None:
                layouts += 1
                assert a == b, (g, a, b)
                assert a.values() == g and coalesce(a) == a
                # brute force: is g additive over SOME shape of size N?
            else:
                assert not any(is_additive_over(g, sh) for sh in all_factorizations(N) if sh), g
    report("T2a spectral criterion == digit-additivity", f"all {total} functions on [N], N<=9, small value sets; {layouts} are layouts")


def check_coalesce_normal_form(n=3000):
    for _ in range(n):
        L = rand_layout()
        C = coalesce(L)
        assert C.values() == L.values()
        assert recognize(L.values()) == C, (L, C, recognize(L.values()))
        # shapes over which f_L is additive == refinements of C.shape (prefix products superset)
        N = L.size
        if N <= 48:
            scalesC = set(prefix_products(C.shape)[1:-1]) if N > 1 else set()
            for sh in all_factorizations(N):
                if not sh:
                    continue
                refines = scalesC <= set(prefix_products(sh))
                assert is_additive_over(L.values(), sh) == refines, (L, C, sh)
    report("T2b coalesce = unique normal form; additive shapes = refinements", f"{n} random layouts")


def check_first_difference_formula(n=1500):
    # Delta f(n) = a + sum of jumps whose scale divides n
    for _ in range(n):
        L = rand_layout()
        a, jumps = L.slope_and_jumps()
        for x in range(1, 2 * L.size + 2):
            want = a + sum(g for P, g in jumps if x % P == 0)
            assert L(x) - L(x - 1) == want, (L, x)
    report("T2c odometer formula for first differences", f"{n} random layouts")


# ---------------------------------------------------------------------------------------
# T3/T4. composition: evaluation + carry defect; exact criterion; forced distributivity
# ---------------------------------------------------------------------------------------
def rand_nat_layout(max_rank=3, strides=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16), sizes=(1, 2, 2, 3, 4, 5, 6)):
    return rand_layout(max_rank, strides, sizes)


def check_composition_defect(n=6000):
    exist = 0
    for _ in range(n):
        A = rand_layout(vec=rng.choice([0, 0, 0, 2]))
        B = rand_nat_layout()
        naive = naive_composite(A, B)
        g = compose_function(A, B)
        allzero = True
        for idx, coord in enumerate(product(*[range(t) for t in reversed(B.shape)])):
            coord = coord[::-1]  # colex enumeration
            x = sum(c * P for c, P in zip(coord, prefix_products(B.shape)))
            dfc = carry_defect(A, B, coord)
            assert g_is_zero(g_sub(g[x], g_add(naive.at(coord), dfc))), (A, B, coord)
            allzero &= g_is_zero(dfc)
        assert allzero == (naive.values() == g)
        # exact composition vs. brute force over every refinement of B's shape
        R = compose(A, B)
        brute = None
        for refs in product(*[[sh or (1,) for sh in all_factorizations(t)] for t in B.shape]):
            flat = sum(refs, ())
            if is_additive_over(g, flat):
                brute = flat
                break
        assert (R is None) == (brute is None), (A, B, R, brute)
        if R is not None:
            exist += 1
            assert concat(*R).values() == g
            for Rj, t, r in zip(R, B.shape, B.stride):
                assert Rj.size == t and Rj.values() == [A(r * x) for x in range(t)]
    report("T3/T4 composition = evaluation + carry defect; exact criterion", f"{n} random pairs, {exist} composable")


def check_carry_free_sufficient_and_generic_exact(n=6000):
    suff = exact = 0
    for _ in range(n):
        B = rand_nat_layout()
        shapeA = rand_shape(3, (2, 3, 4, 5, 6, 8))
        # (a) sufficiency for arbitrary strides
        A = Layout(shapeA, tuple(rng.randrange(-5, 30) for _ in shapeA))
        _, jumps = A.slope_and_jumps()
        active = [P for P, gam in jumps if gam != 0]
        if carry_free(shapeA, B, active):
            suff += 1
            assert naive_composite(A, B).values() == compose_function(A, B), (A, B)
        # (b) exactness when all jumps are positive (no cancellation possible)
        d, strides = rng.randrange(0, 4), []
        for i, s in enumerate(shapeA):
            strides.append(d)
            d = s * d + rng.randrange(1, 5)
        Apos = Layout(shapeA, tuple(strides))
        scales = prefix_products(shapeA)[1:-1]
        lhs = naive_composite(Apos, B).values() == compose_function(Apos, B)
        assert lhs == carry_free(shapeA, B, scales), (Apos, B)
        exact += lhs
        # (c) universal A (digit layout): composable for all strides iff digit map is a layout
        m = len(shapeA)
        U = Layout(shapeA, tuple(tuple(int(i == j) for j in range(m)) for i in range(m)))
        assert (naive_composite(U, B).values() == compose_function(U, B)) == carry_free(shapeA, B, scales)
    report("T4b carry-free: sufficient always, exact for positive jumps and for the digit layout", f"{n} samples ({suff} carry-free, {exact} positive-jump composable)")


def refinements(B: Layout):
    """All by-mode refinements B' of B: same function, each mode t_j:r_j split along an
    ordered factorization of t_j (strides are B's values at the new units)."""
    per_mode = []
    for t, r in zip(B.shape, B.stride):
        opts = []
        for fac in all_factorizations(t) or [()]:
            fac = fac or (1,)
            Pp = prefix_products(fac)
            opts.append((fac, tuple(r * Pp[i] for i in range(len(fac)))))
        per_mode.append(opts)
    for choice in product(*per_mode):
        yield Layout(sum((c[0] for c in choice), ()), sum((c[1] for c in choice), ()))


def check_exactness_with_refinement(n=3000):
    ex = 0
    for _ in range(n):
        B = rand_nat_layout(2, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), (1, 2, 3, 4, 6, 8, 12))
        shapeA = rand_shape(3, (2, 3, 4, 5, 6, 7, 8, 11))
        scales = prefix_products(shapeA)[1:-1]
        d, strides = rng.randrange(0, 4), []
        for s in shapeA:
            strides.append(d)
            d = s * d + rng.randrange(1, 5)
        Apos = Layout(shapeA, tuple(strides))               # all jumps positive
        m = len(shapeA)
        Auni = Layout(shapeA, tuple(tuple(int(i == j) for j in range(m)) for i in range(m)))  # digit layout
        some_carry_free = any(carry_free(shapeA, Bp, scales) for Bp in refinements(B))
        for A in (Apos, Auni):
            R = compose(A, B)
            assert (R is not None) == some_carry_free, (A, B, R, some_carry_free)
        ex += some_carry_free
        # functional form of the criterion (any A): exists iff g is a layout function whose
        # jump scales are comparable with every proper prefix product of T
        A = Layout(shapeA, tuple(rng.randrange(-3, 12) for _ in shapeA))
        g = compose_function(A, B)
        C = recognize(g)
        ok = False
        if C is not None:
            sg = set(prefix_products(C.shape)[1:-1]) if len(g) > 1 else set()
            sT = {P for P in prefix_products(B.shape) if 1 < P < B.size}
            ok = all(a % b == 0 or b % a == 0 for a in sg for b in sT)
        assert (compose(A, B) is not None) == ok, (A, B, C)
    report("T4c exists <=> some refinement of B is carry-free (positive jumps / digit layout); functional criterion", f"{n} samples, {ex} composable")


def check_single_floor_term():
    # k(x) = floor(rho*x/q) on [t], gcd(rho,q)=1, 1<=rho<q: layout iff t<=u or t in {2u..Ju}
    cnt = 0
    for q in range(2, 40):
        for rho in range(1, q):
            if gcd(rho, q) != 1:
                continue
            u = ceil(q / rho)
            e = u * rho - q
            for t in range(1, 4 * q + 3):
                is_layout = recognize([rho * x // q for x in range(t)]) is not None
                if e == 0:
                    pred = t <= u or t % u == 0
                else:
                    J = ceil(rho / e) - 1
                    pred = t <= u or (t % u == 0 and t <= J * u)
                assert is_layout == pred, (q, rho, t, is_layout, pred)
                cnt += 1
    report("T5 single dilated floor term", f"{cnt} triples (q, rho, t) exhaustively")


def check_cute_conditions_sound(n=20000):
    """Stride/shape divisibility (in the form of Cecka 2026, eqs. for a single-mode B after
    coalescing and truncating A) imply that A o B exists and is given by CuTe's formula."""
    hit = 0
    for _ in range(n):
        A = coalesce(rand_layout(4, range(0, 25), (2, 2, 3, 4, 5, 6, 8)))
        t, r = rng.choice([1, 2, 3, 4, 5, 6, 8, 9, 12]), rng.choice([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 24])
        P = prefix_products(A.shape)
        active = [P[i] for i in range(1, A.rank) if P[i] <= (t - 1) * r]
        if not all(Pi % r == 0 or r % Pi == 0 for Pi in active):
            continue
        new_scales = sorted({Pi // r for Pi in active if Pi % r == 0 and Pi // r > 1})
        if not all(t % Q == 0 for Q in new_scales):
            continue
        hit += 1
        g = [A(r * x) for x in range(t)]
        R = recognize(g)
        assert R is not None, (A, t, r)
        assert set(prefix_products(R.shape)[1:-1]) <= set(new_scales), (A, t, r, R)
        assert g[1] == A(r) if t > 1 else True
    report("T5b CuTe's divisibility conditions are sufficient", f"{hit} admissible cases out of {n}")


# ---------------------------------------------------------------------------------------
# T6. complement = direct-sum decomposition (de Bruijn / Long)
# ---------------------------------------------------------------------------------------
def check_long_theorem():
    """Exhaustive: for M <= 24, I and J subsets of [M] containing 0 with I (+) J = [M]
    are exactly the alternating digit sets of ordered factorizations of M."""
    cnt = 0
    for M in range(1, 25):
        # all complementing pairs by direct search over I (J is forced)
        pairs = set()
        elems = list(range(1, M))
        def forced_J(I):
            J, covered = [], set()
            for c in range(M):
                if c in covered:
                    continue
                # c is the smallest uncovered element; it must be 0 + j
                J.append(c)
                new = {a + c for a in I}
                if new & covered or max(new) >= M:
                    return None
                covered |= new
            return tuple(J) if len(covered) == M else None
        # enumerate I as subsets containing 0 whose size divides M
        from itertools import combinations
        for k in range(1, M + 1):
            if M % k:
                continue
            for comb in combinations(elems, k - 1):
                I = (0, *comb)
                J = forced_J(I)
                if J is not None and is_direct_sum(I, J, M):
                    pairs.add((I, J))
        # alternating digit sets
        predicted = set()
        for fac in all_factorizations(M):
            Pp = prefix_products(fac)
            for parity in (0, 1):
                I = sorted(sum(c * Pp[i] for i, c in zip(idx, cs))
                           for idx in [[i for i in range(len(fac)) if i % 2 == parity]]
                           for cs in product(*[range(fac[i]) for i in idx]))
                Jd = sorted(sum(c * Pp[i] for i, c in zip(idx, cs))
                            for idx in [[i for i in range(len(fac)) if i % 2 != parity]]
                            for cs in product(*[range(fac[i]) for i in idx]))
                predicted.add((tuple(I), tuple(Jd)))
        assert pairs == predicted, (M, pairs ^ predicted)
        cnt += len(pairs)
    report("T6a Long's theorem (complementing pairs = alternating digit sets)", f"M<=24, {cnt} pairs")


def check_complement(n=4000):
    ex = 0
    for _ in range(n):
        L = rand_layout(3, (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 24), (1, 2, 2, 3, 4))
        M = rng.choice([4, 6, 8, 12, 16, 18, 24, 36, 48, 60, 72])
        I = L.values()
        injective = len(set(I)) == len(I)
        C = complement(L, M)
        # existence of ANY set J with I (+) J = [M]  (J is forced greedily)
        J, covered, ok = [], set(), injective and max(I) < M
        if ok:
            for c in range(M):
                if c in covered:
                    continue
                J.append(c)
                new = {a + c for a in I}
                if new & covered or max(new) >= M:
                    ok = False
                    break
                covered |= new
        exists = ok and len(covered) == M
        assert exists == (C is not None), (L, M, exists, C)
        if C is not None:
            ex += 1
            assert sorted(C.values()) == J and C.values() == J, (L, M, C)        # increasing, forced image
            assert sorted(concat(L, C).values()) == list(range(M))
            # double complement = sorted coalesced form of L
            CC = complement(C, M)
            assert CC is not None and CC.values() == sorted(I), (L, M, C, CC)
    report("T6b complement exists iff admissible; unique image; double complement", f"{n} samples, {ex} complementable")


# ---------------------------------------------------------------------------------------
# T7. division: composing with a compact layout
# ---------------------------------------------------------------------------------------
def rand_compact(M):
    fac = list(rng.choice(all_factorizations(M)))
    Pp = prefix_products(fac)
    perm = list(range(len(fac)))
    rng.shuffle(perm)
    return Layout(tuple(fac[i] for i in perm), tuple(Pp[i] for i in perm)), set(Pp[1:-1])


def check_division(n=8000):
    """Composition with a compact layout E of size M (Theorem 6.8, corrected after review):
    A o E exists iff spec(A^|[M]) u Sc(U) is a chain all of whose members divide M.
    The size of A is allowed to differ from M (truncated or extended evaluation)."""
    ex = other_size = 0
    for _ in range(n):
        M = rng.choice([4, 6, 8, 10, 12, 16, 18, 24, 30, 36])
        if rng.random() < 0.5:
            fac = rng.choice(all_factorizations(M))                    # |A| = M
        else:
            fac = rand_shape(3, (2, 2, 3, 4, 5, 6))                    # |A| arbitrary
            other_size += 1
        A = Layout(fac, tuple(rng.randrange(-4, 20) for _ in fac))
        B, scalesU = rand_compact(M)
        assert is_compact(B)
        specA = set(floor_spectrum([A(x) for x in range(M)])) - {1}
        chain = sorted(specA | scalesU)
        ok = all(M % q == 0 for q in chain) and all(b % a == 0 for a, b in zip(chain, chain[1:]))
        R = compose(A, B)
        assert (R is not None) == ok, (A, B, specA, scalesU, R)
        ex += ok
    report("T7 A o (compact E) exists iff spec(A^|[M]) u Sc(U) is a chain dividing M", f"{n} samples ({other_size} with |A| != M), {ex} composable")


# ---------------------------------------------------------------------------------------
# T8. associativity (Proposition 5.14(d), corrected after review)
# ---------------------------------------------------------------------------------------
def check_associativity(n=40000):
    both = left_only = right_only = 0
    for _ in range(n):
        A = rand_layout(3, range(-3, 14), (1, 2, 2, 3, 4, 6))
        B = rand_nat_layout(3, (0, 1, 2, 3, 4, 6, 8, 12), (1, 2, 2, 3, 4))
        C = rand_nat_layout(2, (0, 1, 2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 6))
        if max(C.values()) >= B.size:
            continue                                   # in-bounds hypothesis
        AB, BC = compose(A, B), compose(B, C)
        if AB is None or BC is None:
            continue
        L = compose(concat(*AB), C)                    # (A o B) o C
        R = compose(A, concat(*BC))                    # A o (B o C)
        h = [A(B(c)) for c in C.values()]
        if L is not None:
            assert concat(*L).values() == h
        if R is not None:
            assert concat(*R).values() == h
            assert L is not None, (A, B, C)            # right exists => left exists
        both += L is not None and R is not None
        left_only += L is not None and R is None
        right_only += L is None and R is not None
    assert right_only == 0
    # the reviewer's example: only the left bracketing exists
    A, B, C = Layout((12, 3), (0, 1)), Layout((2, 2, 3, 3), (1, 6, 2, 12)), Layout(6, 7)
    assert compose(concat(*compose(A, B)), C) is not None and compose(A, concat(*compose(B, C))) is None
    report("T8 associativity: same function; right bracketing exists => left exists; converse fails",
           f"{both} both, {left_only} left only, {right_only} right only")


# ---------------------------------------------------------------------------------------
# T9. the published sufficient conditions for multi-mode composition
# ---------------------------------------------------------------------------------------
def shah_left_divisible(fact, d, weak):
    """Shah 2024, Def. 2.11.  `fact` lists the finite factors M_0..M_{alpha-1}; the last factor is
    infinity.  Returns (i, c, induced finite factors) or None."""
    alpha, P = len(fact), 1
    for i in range(alpha + 1):
        if d % P == 0:
            c = d // P
            if i == alpha:
                return (i, c, [])
            if 1 <= c < fact[i] and (weak or fact[i] % c == 0):
                return (i, c, [fact[i] // c] + list(fact[i + 1:]))
        if i < alpha:
            P *= fact[i]
    return None


def shah_single(A: Layout, N: int, r: int):
    """Shah 2024, Def. 2.12 + 2.13: the composite A o (N:r), or None if {S, B} is not admissible."""
    fact = list(A.shape[:-1])
    ld = shah_left_divisible(fact, r, weak=False)
    if ld is None:
        return None
    i, c, induced = ld
    if shah_left_divisible(induced, N, weak=True) is None:
        return None
    alpha = len(fact)
    if i == alpha:
        return Layout((N,), (c * A.stride[alpha],))
    if N <= A.shape[i] // c:
        return Layout((N,), (c * A.stride[i],))
    shape, stride, rest, j = [A.shape[i] // c], [c * A.stride[i]], N // (A.shape[i] // c), i + 1
    while j < alpha and rest >= A.shape[j] and rest % A.shape[j] == 0:
        shape.append(A.shape[j]); stride.append(A.stride[j]); rest //= A.shape[j]; j += 1
    if rest > 1:
        shape.append(rest); stride.append(A.stride[j])
    return Layout(tuple(shape), tuple(stride))


def shah_admissible(A: Layout, B: Layout):
    """Shah 2024, Def. 2.17: every mode admissible and the intervals of definition pairwise disjoint.
    Returns the concatenated composite, or None."""
    Mp = prod(A.shape[:-1])
    parts, intervals = [], []
    for N, r in zip(B.shape, B.stride):
        R = shah_single(A, N, r)
        if R is None:
            return None
        parts.append(R)
        lo, hi = max(r, 1), min(r * (N - 1), Mp - 1)
        if lo <= hi:
            intervals.append((lo, hi))
    intervals.sort()
    if any(a[1] >= b[0] for a, b in zip(intervals, intervals[1:])):
        return None
    return concat(*parts)


def cecka_sufficient(A: Layout, B: Layout) -> bool:
    """Cecka 2026, Sec. 3.3.2 'Reductive case: distributive': every stride of B divides or is
    divisible by every prefix product of A, and the modes of B are mutually segregated."""
    Pp = prefix_products(A.shape)[:-1]
    if not all(P % r == 0 or r % P == 0 for r in B.stride for P in Pp):
        return False
    modes = list(zip(B.shape, B.stride))
    return all(si * di <= dj or sj * dj <= di
               for a, (si, di) in enumerate(modes) for (sj, dj) in modes[a + 1:])


def check_published_conditions(n=60000):
    stats = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
    cecka = [0, 0]
    smallest = None
    for _ in range(n):
        shapeA = tuple(rng.choice((2, 3, 4, 6, 8, 12)) for _ in range(rng.randint(1, 3)))
        A = Layout(shapeA, tuple(rng.randrange(1, 30) for _ in shapeA))
        k = rng.randint(1, 3)
        B = Layout(tuple(rng.choice((2, 2, 3, 4)) for _ in range(k)), tuple(rng.choice((1, 2, 3, 4, 6, 8, 12, 24)) for _ in range(k)))
        g = compose_function(A, B)
        R = shah_admissible(A, B)
        if R is not None:
            stats[k][0] += 1
            assert R.size == B.size
            if R.values() != g:
                stats[k][1] += 1
                assert k >= 3, ("Shah's condition fails with fewer than three modes", A, B, R)
                if smallest is None or A.size * B.size < smallest[0]:
                    smallest = (A.size * B.size, f"{A} o {B}: Shah composite {R} = {R.values()}, truth {g}")
        if cecka_sufficient(A, B):
            cecka[0] += 1
            add = all(A(sum(c * r for c, r in zip(cs, B.stride))) == sum(A(c * r) for c, r in zip(cs, B.stride))
                      for cs in product(*[range(t) for t in B.shape]))
            cecka[1] += not add
    # the reviewer's three-mode example
    A, B = Layout((12, 2), (1, 13)), Layout((2, 2, 2), (3, 4, 6))
    R = shah_admissible(A, B)
    assert R is not None and R.values() != compose_function(A, B) and compose(A, B) is None
    assert stats[1][1] == 0 and stats[2][1] == 0 and stats[3][1] > 0 and cecka[1] == 0
    report("T9 published sufficient conditions",
           f"Shah Def 2.17 admissible/wrong by #modes: 1: {stats[1]}, 2: {stats[2]}, 3: {stats[3]}; "
           f"Cecka segregation admissible/non-additive: {cecka}; smallest 3-mode failure: {smallest[1]}")


def check_review_remarks(n=6000):
    """Statements corrected after the second review: Remark 3.7 (extended additivity), Remark 4.7
    (sums of layouts), and the two small examples for Theorem 5.5(iv)."""
    # Remark 3.7: the extended function is additive on N iff the total jump at every scale >= 2 is 0
    add = non = 0
    for _ in range(n):
        L = rand_layout(max_rank=3, strides=range(-3, 6))
        _, jumps = L.slope_and_jumps()
        totals = {}
        for P, gam in jumps:
            totals[P] = g_add(totals[P], gam) if P in totals else gam
        crit = all(P == 1 or g_is_zero(t) for P, t in totals.items())
        lim = 3 * L.size + 2
        additive = all(L(x + y) == L(x) + L(y) for x in range(lim) for y in range(lim))
        assert additive == crit, (L, totals)
        add += additive
        non += not additive
    # Remark 4.7: g1 + g2 is a layout function iff the union of the spectra is a chain
    yes = no = 0
    for _ in range(n):
        N = rng.choice((4, 6, 8, 12, 16, 18, 24))
        fac = all_factorizations(N)
        shapes = [tuple(rng.sample(f, len(f))) for f in (rng.choice(fac), rng.choice(fac))]
        ls = [Layout(sh, tuple(rng.randrange(-3, 6) for _ in sh)) for sh in shapes]
        v1, v2 = ls[0].values(), ls[1].values()
        tot = [a + b for a, b in zip(v1, v2)]
        sp = sorted((set(floor_spectrum(v1)) | set(floor_spectrum(v2))) - {1})
        chain = all(b % a == 0 for a, b in zip(sp, sp[1:]))
        assert (recognize(tot) is not None) == chain, (ls, sp)
        yes += chain
        no += not chain
    # Theorem 5.5(iv): both requirements are needed
    A, B = Layout((2, 3), (0, 1)), Layout((3, 2), (1, 3))
    g = compose_function(A, B)
    assert g == [0, 0, 1, 1, 2, 2] and recognize(g) is not None and compose(A, B) is None
    assert compose(A, Layout((6,), (1,))) is not None
    A, B = Layout((2, 2), (1, 3)), Layout((3,), (1,))
    g = compose_function(A, B)
    assert g == [0, 1, 3] and recognize(g) is None and compose(A, B) is None
    # the segregated but interfering pair, and the single-digit stride that does not divide
    A, B = Layout((5, 2), (1, 6)), Layout((2, 2), (2, 4))
    assert compose_function(A, B) == [0, 2, 4, 7] and compose(A, B) is None
    assert compose_function(Layout((5, 2), (1, 8)), Layout((4,), (4,))) == [0, 4, 11, 18]
    assert compose(Layout((5, 2), (1, 8)), Layout((4,), (4,))) is None
    report("T10 remarks 3.7 and 4.7, examples for Theorem 5.5(iv)",
           f"extended additivity: {add} additive / {non} not; sums of layouts: {yes} layouts / {no} not")


if __name__ == "__main__":
    check_floor_sum_and_carry()
    check_recognition_exhaustive()
    check_coalesce_normal_form()
    check_first_difference_formula()
    check_composition_defect()
    check_carry_free_sufficient_and_generic_exact()
    check_exactness_with_refinement()
    check_single_floor_term()
    check_cute_conditions_sound()
    check_long_theorem()
    check_complement()
    check_division()
    check_associativity()
    check_published_conditions()
    check_review_remarks()
    print(f"\nall {len(REPORT)} checks passed")
