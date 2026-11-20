"""Differential test of CuTe's Swizzle / ComposedLayout algebra (C++ headers) against the theory.

    python3 -I python/difftest_swizzle_cpp.py <cutlass-include-dir> <cuda-header-root> <workdir> [seed] [n]

Suites
  slice      slice_and_offset(coord, composition(Swizzle, f))   vs  sigma(f(c0, c))
  transport  composition(a, Swizzle)                             vs  a(sigma(i))
  merge      composition(Swizzle, Swizzle)                       vs  sigma0(sigma1(x))
  cosize     cosize(composition(Swizzle, f))                     vs  1 + max sigma(f(c))
  upcast     upcast<N>(composition(Swizzle, f))                  vs  floor(sigma(f(c)) / N)
  rinv       right_inverse(composition(Swizzle, f))              vs  the defining property

As in difftest_cpp.py every case is one source line; a case counts as rejected iff a diagnostic
points at its line.  Output: JSON on stdout.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from itertools import product

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
from layout_theory import Layout, digits  # noqa: E402
from swizzle_theory import (  # noqa: E402
    Swizzle, decay, is_bit_disjoint, merge_swizzles, split_pairs, supp, binary_refinement,
    upcast_swizzle,
)

INC, CUDA, WORK = sys.argv[1], sys.argv[2], sys.argv[3]
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
N = int(sys.argv[5]) if len(sys.argv) > 5 else 200
CHUNK = 40
FLAGS = ["-std=c++17", "-w", "-ferror-limit=0",
         "-isystem", f"{CUDA}/cuda_cudart/include", "-isystem", f"{CUDA}/cuda_crt/include",
         "-isystem", f"{CUDA}/cccl/include/cccl", "-I", INC]
PRELUDE = (
    "#include <iostream>\n#include <cute/layout.hpp>\n#include <cute/swizzle.hpp>\n"
    "#include <cute/swizzle_layout.hpp>\n#include <cute/layout_composed.hpp>\nusing namespace cute;\n"
    "template <class T> long val(T const& t) { return long(int(t)); }\n"
    "template <class Shape, class Stride> const char* kind(Layout<Shape,Stride> const&) { return \"layout\"; }\n"
    "template <class A, class O, class B> const char* kind(ComposedLayout<A,O,B> const&) { return \"composed\"; }\n"
    "template <int B, int M, int S> const char* kind(Swizzle<B,M,S> const&) { return \"swizzle\"; }\n"
)
NPRE = PRELUDE.count("\n")
rng = random.Random(SEED)


def cpp_layout(L: Layout) -> str:
    def tup(kind, vals):
        return f"make_{kind}(" + ",".join(f"Int<{v}>{{}}" for v in vals) + ")"
    if L.rank == 1:
        return f"make_layout(Int<{L.shape[0]}>{{}}, Int<{L.stride[0]}>{{}})"
    return f"make_layout({tup('shape', L.shape)}, {tup('stride', L.stride)})"


def cpp_sw(sw: Swizzle) -> str:
    return f"Swizzle<{sw.B},{sw.M},{sw.S}>{{}}"


def show(L: Layout) -> str:
    return f"{L.shape[0]}:{L.stride[0]}" if L.rank == 1 else f"{L.shape}:{L.stride}".replace(" ", "")


def run_lines(bodies: dict) -> tuple[dict, dict]:
    """bodies: id -> C++ statements (one line) that print 'CASE id kind text | values'."""
    os.makedirs(WORK, exist_ok=True)
    ids = sorted(bodies)
    results, reasons = {}, {}
    for c in range(0, len(ids), CHUNK):
        good = ids[c:c + CHUNK]
        src, exe = f"{WORK}/chunk.cpp", f"{WORK}/chunk"
        while good:
            line_of = {}
            with open(src, "w") as f:
                f.write(PRELUDE)
                for n, i in enumerate(good):
                    line_of[NPRE + 1 + n] = i
                    f.write(f"void case_{i}() {{ {bodies[i]} }}\n")
                f.write("int main() {\n" + "".join(f"  case_{i}();\n" for i in good) + "  return 0; }\n")
            r = subprocess.run(["clang++", *FLAGS, "-fsyntax-only", src], capture_output=True, text=True)
            if r.returncode == 0:
                break
            bad, last_msg = {}, "compile error"
            for line in r.stderr.splitlines():
                m = re.search(r'static assertion failed[^\n]*?: (.*)$', line)
                if m:
                    last_msg = m.group(1).strip().strip('"')
                elif re.search(r"\berror: ", line):
                    last_msg = line.split("error: ", 1)[1][:90]
                m = re.match(rf"{re.escape(src)}:(\d+):\d+: (error|note)", line)
                if m and int(m.group(1)) in line_of:
                    bad.setdefault(line_of[int(m.group(1))], last_msg)
            assert bad, r.stderr[:3000]
            reasons.update(bad)
            good = [i for i in good if i not in bad]
        if not good:
            continue
        r = subprocess.run(["clang++", *FLAGS, "-O0", src, "-o", exe], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[:3000]
        for line in subprocess.run([exe], capture_output=True, text=True).stdout.splitlines():
            m = re.match(r"CASE (\d+) (\w+) (.*?) \|(.*)", line)
            if m:
                results[int(m.group(1))] = (m.group(2), m.group(3),
                                            [int(x.lstrip("_")) for x in m.group(4).split()])
    return results, reasons


PRINT_VALUES = ("std::cout << \"CASE {i} \" << kind(R) << \" \" << R << \" |\"; "
                "for (int k = 0; k < int(size(R)); ++k) std::cout << \" \" << val(R(k)); std::cout << \"\\n\";")


# --------------------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------------------

def rand_swizzle(maxB=2, maxM=2):
    B = rng.randint(1, maxB)
    S = rng.choice([1, 1, 1, -1]) * rng.randint(B, B + 1)
    return Swizzle(B, rng.randint(0, maxM), S)


def rand_aligned_layout(nbits: int, max_rank=3) -> Layout:
    """Power-of-two mode sizes, power-of-two strides, modes on disjoint bit ranges."""
    cuts = sorted(rng.sample(range(1, nbits), min(nbits - 1, rng.randint(0, max_rank - 1))))
    segs = [(a, b) for a, b in zip([0] + cuts, cuts + [nbits])]
    rng.shuffle(segs)
    return Layout(tuple(1 << (b - a) for a, b in segs), tuple(1 << a for a, b in segs))


def rand_small_layout(max_stride: int, max_rank=3) -> Layout:
    n = rng.randint(2, max_rank)
    return Layout(tuple(rng.choice((2, 2, 2, 3, 4, 4)) for _ in range(n)),
                  tuple(rng.randint(0, max_stride) for _ in range(n)))


def gen_slice(n):
    cases = {}
    for i in range(n):
        sw = rand_swizzle()
        fam = rng.choice(["aligned", "aligned", "overlap", "random", "small"])
        if fam == "aligned":
            f = rand_aligned_layout(sw.top + rng.randint(0, 1))
        elif fam == "overlap":          # power-of-two sizes and strides, modes may share bits
            n = rng.randint(2, 3)
            f = Layout(tuple(rng.choice((2, 2, 4)) for _ in range(n)),
                       tuple(1 << rng.randint(0, sw.top) for _ in range(n)))
        elif fam == "small":            # small strides, not necessarily powers of two
            n = rng.randint(2, 3)
            f = Layout(tuple(rng.choice((2, 2, 4)) for _ in range(n)),
                       tuple(rng.choice((1, 2, 3, 4, 5, 6, 8)) for _ in range(n)))
        else:
            f = rand_small_layout(1 << sw.top)
        if f.rank < 2:
            f = Layout(f.shape + (2,), f.stride + (1 << sw.top,))
        fixed = [rng.random() < 0.5 for _ in range(f.rank)]
        if all(fixed) or not any(fixed):
            fixed = [k == 0 for k in range(f.rank)]
            rng.shuffle(fixed)
        coord = [rng.randrange(t) if fx else None for t, fx in zip(f.shape, fixed)]
        dynamic = rng.random() < 0.3
        cs = ", ".join("_" if c is None else (str(c) if dynamic else f"Int<{c}>{{}}") for c in coord)
        body = (f"auto f = {cpp_layout(f)}; auto L = composition({cpp_sw(sw)}, f); "
                f"auto so = slice_and_offset(make_coord({cs}), L); auto R = get<0>(so); long off = val(get<1>(so)); "
                f"std::cout << \"CASE {i} \" << kind(R) << \" x |\"; "
                f"for (int k = 0; k < int(size(R)); ++k) std::cout << \" \" << (val(R(k)) + off); std::cout << \"\\n\";")
        cases[i] = dict(body=body, sw=sw, f=f, coord=coord, dynamic=dynamic, family=fam)
    return cases


def judge_slice(case, res):
    sw, f, coord = case["sw"], case["f"], case["coord"]
    free = [k for k, c in enumerate(coord) if c is None]
    truth = []
    for idx in product(*[range(f.shape[k]) for k in reversed(free)]):
        full = list(coord)
        for k, v in zip(reversed(free), idx):
            full[k] = v
        truth.append(sw(f.at(full)))
    kind, _, vals = res
    # the hypotheses under which the theory guarantees CuTe's bookkeeping
    hyp = is_bit_disjoint(f)
    fs = Layout(tuple(f.shape[k] for k in free), tuple(f.stride[k] for k in free))
    o = sum(c * f.stride[k] for k, c in enumerate(coord) if c is not None)
    exact = decay(sw, o, fs) is not None
    return dict(ok=vals == truth, kind=kind, hyp=hyp, exact_decay=exact, truth=truth, got=vals)


def gen_transport(n):
    cases = {}
    for i in range(n):
        nb = rng.randint(2, 5)
        fam = rng.choice(["bitperm", "bitperm", "aligned", "random"])
        if fam == "bitperm":
            a = Layout((2,) * nb, tuple(1 << e for e in rng.sample(range(nb + 1), nb)))
        elif fam == "aligned":
            a = rand_aligned_layout(nb)
        else:
            a = Layout((2,) * nb, tuple(rng.randint(0, 9) for _ in range(nb)))
        B = rng.randint(1, max(1, nb // 2))
        S = rng.choice([-1, 1]) * rng.randint(B, max(B, nb - B))
        M = rng.randint(0, max(0, nb - B - abs(S)))
        if M + B + abs(S) > nb:
            B, S, M = 1, 1, 0
        sw = Swizzle(B, M, S)
        body = f"auto a = {cpp_layout(a)}; auto R = composition(a, {cpp_sw(sw)}); " + PRINT_VALUES.replace("{i}", str(i))
        cases[i] = dict(body=body, sw=sw, a=a, family=fam)
    return cases


def gen_merge():
    cases, i = {}, 0
    sws = [Swizzle(B, M, S) for B in (1, 2) for M in (0, 1, 2) for S in (-3, -2, -1, 1, 2, 3) if abs(S) >= B]
    for s0 in sws:
        for s1 in sws:
            if s0.S != s1.S or rng.random() > 0.35:
                continue
            body = (f"auto R = composition({cpp_sw(s0)}, {cpp_sw(s1)}); "
                    f"std::cout << \"CASE {i} \" << kind(R) << \" \" << R << \" |\"; "
                    f"for (int k = 0; k < 256; ++k) std::cout << \" \" << val(R(k)); std::cout << \"\\n\";")
            cases[i] = dict(body=body, s0=s0, s1=s1)
            i += 1
    return cases


def gen_cosize(n):
    cases = {}
    for i in range(n):
        sw = rand_swizzle()
        fam = rng.choice(["compact", "random"])
        f = rand_aligned_layout(max(1, sw.top + rng.randint(-1, 1))) if fam == "compact" else rand_small_layout(1 << sw.top)
        body = (f"auto f = {cpp_layout(f)}; auto R = composition({cpp_sw(sw)}, f); "
                f"std::cout << \"CASE {i} \" << kind(R) << \" \" << val(cosize(R)) << \" |\"; "
                f"for (int k = 0; k < int(size(R)); ++k) std::cout << \" \" << val(R(k)); std::cout << \"\\n\";")
        cases[i] = dict(body=body, sw=sw, f=f, family=fam)
    return cases


def gen_upcast(n):
    cases = {}
    for i in range(n):
        sw = rand_swizzle(maxB=2, maxM=3)
        nlog = rng.randint(1, 2)
        Nn = 1 << nlog
        rank = rng.randint(1, 3)
        # strides are multiples of N, so that upcast keeps the shape and divides the strides
        f = Layout(tuple(rng.choice((2, 2, 4)) for _ in range(rank)),
                   tuple(Nn * rng.choice((1, 2, 4, 8, 16, 3)) for _ in range(rank)))
        body = (f"auto f = {cpp_layout(f)}; auto R = upcast<{Nn}>(composition({cpp_sw(sw)}, f)); "
                + PRINT_VALUES.replace("{i}", str(i)))
        cases[i] = dict(body=body, sw=sw, f=f, nlog=nlog)
    return cases


def gen_rinv(n):
    cases = {}
    for i in range(n):
        sw = rand_swizzle()
        nb = sw.top + rng.randint(0, 1)
        f = rand_aligned_layout(nb)                     # compact: a bijection onto [0, 2^nb)
        body = (f"auto f = {cpp_layout(f)}; auto L = composition({cpp_sw(sw)}, f); auto R = right_inverse(L); "
                f"std::cout << \"CASE {i} \" << kind(R) << \" \" << R << \" |\"; "
                f"for (int k = 0; k < int(size(R)); ++k) std::cout << \" \" << val(L(val(R(k)))); std::cout << \"\\n\";")
        cases[i] = dict(body=body, sw=sw, f=f)
    return cases


# --------------------------------------------------------------------------------------

def main():
    report = {"seed": SEED}

    # ---- slice
    cases = gen_slice(N)
    res, rej = run_lines({i: c["body"] for i, c in cases.items()})
    t = dict(cases=len(cases), rejected=len(rej), correct_layout=0, correct_composed=0, wrong_layout=0,
             wrong_composed=0, wrong_under_hypothesis=0, correct_under_hypothesis=0,
             decayable_kept_composed=0, examples=[])
    t["bit_disjoint_inner_layout"] = sum(is_bit_disjoint(c["f"]) for c in cases.values())
    t["rejected_under_hypothesis"] = 0
    for i, c in cases.items():
        if i in rej:
            t["rejected_under_hypothesis"] += is_bit_disjoint(c["f"])
            continue
        j = judge_slice(c, res[i])
        key = ("correct_" if j["ok"] else "wrong_") + j["kind"]
        t[key] += 1
        if j["hyp"]:
            t["correct_under_hypothesis" if j["ok"] else "wrong_under_hypothesis"] += 1
        if j["ok"] and j["kind"] == "composed" and j["exact_decay"]:
            t["decayable_kept_composed"] += 1
        fam = t.setdefault("by_family", {}).setdefault(c["family"], dict(correct=0, wrong=0))
        fam["correct" if j["ok"] else "wrong"] += 1
        if not j["ok"]:
            t["wrong_static_coord"] = t.get("wrong_static_coord", 0) + (not c["dynamic"])
            inj = len(set(c["f"].values())) == c["f"].size
            t["wrong_with_injective_inner_layout"] = t.get("wrong_with_injective_inner_layout", 0) + inj
            if inj:
                t.setdefault("examples_injective", []).append(
                    dict(swizzle=repr(c["sw"]), f=show(c["f"]), coord=c["coord"], dynamic=c["dynamic"],
                         kind=j["kind"], got=j["got"], truth=j["truth"]))
            t["examples"].append(dict(swizzle=repr(c["sw"]), f=show(c["f"]), coord=c["coord"],
                                      dynamic=c["dynamic"], kind=j["kind"], got=j["got"], truth=j["truth"]))
    t["examples"] = sorted(t["examples"], key=lambda e: (len(e["truth"]), e["f"]))[:6]
    t["examples_injective"] = sorted(t.get("examples_injective", []), key=lambda e: (len(e["truth"]), e["f"]))[:6]
    t["reject_reasons"] = sorted(set(rej.values()))[:5]
    report["slice"] = t

    # ---- transport
    cases = gen_transport(N)
    res, rej = run_lines({i: c["body"] for i, c in cases.items()})
    t = dict(cases=len(cases), rejected=len(rej), correct=0, wrong=0, rejected_but_transportable=0,
             examples=[])
    for i, c in cases.items():
        a, sw = c["a"], c["sw"]
        truth = [a(sw(x)) for x in range(a.size)]
        vals = [a(x) for x in range(a.size)]
        representable = any(
            [Swizzle(B2, M2, S2)(v) for v in vals] == truth
            for B2 in range(0, 4) for M2 in range(0, 7) for S2 in range(-6, 7)
            if abs(S2) >= B2 and (S2 != 0 or B2 == 0))
        if i in rej:
            t["rejected_but_transportable"] += representable
            continue
        ok = res[i][2] == truth
        t["correct" if ok else "wrong"] += 1
        fam = t.setdefault("by_family", {}).setdefault(c["family"], dict(correct=0, wrong=0))
        fam["correct" if ok else "wrong"] += 1
        if not ok:
            t["wrong_but_representable"] = t.get("wrong_but_representable", 0) + representable
            t["examples"].append(dict(a=show(a), swizzle=repr(sw), result=res[i][1], got=res[i][2], truth=truth))
    t["examples"] = sorted(t["examples"], key=lambda e: (len(e["truth"]), e["a"]))[:6]
    t["reject_reasons"] = sorted(set(rej.values()))[:5]
    report["transport"] = t

    # ---- merge
    cases = gen_merge()
    res, rej = run_lines({i: c["body"] for i, c in cases.items()})
    t = dict(cases=len(cases), rejected=len(rej), correct=0, wrong=0, model_agrees=0)
    for i, c in cases.items():
        m = merge_swizzles(c["s0"], c["s1"])
        t["model_agrees"] += (m is None) == (i in rej)
        if i in rej:
            continue
        truth = [c["s0"](c["s1"](x)) for x in range(256)]
        t["correct" if res[i][2] == truth else "wrong"] += 1
    report["merge"] = t

    # ---- cosize
    cases = gen_cosize(N)
    res, rej = run_lines({i: c["body"] for i, c in cases.items()})
    t = dict(cases=len(cases), rejected=len(rej), exact=0, too_small=0, too_large=0, examples=[])
    for i, c in cases.items():
        if i in rej:
            continue
        got, true = int(res[i][1]), max(res[i][2]) + 1
        assert res[i][2] == [c["sw"](v) for v in c["f"].values()]
        key = "exact" if got == true else ("too_small" if got < true else "too_large")
        t[key] += 1
        fam = t.setdefault("by_family", {}).setdefault(c["family"], dict(exact=0, too_small=0, too_large=0))
        fam[key] += 1
        if got < true and len(t["examples"]) < 4:
            t["examples"].append(dict(swizzle=repr(c["sw"]), f=show(c["f"]), cosize=got, true=true))
    report["cosize"] = t

    # ---- upcast
    cases = gen_upcast(N)
    res, rej = run_lines({i: c["body"] for i, c in cases.items()})
    t = dict(cases=len(cases), rejected=len(rej), correct=0, wrong=0, wrong_with_positive_shift=0, examples=[])
    for i, c in cases.items():
        if i in rej:
            continue
        sw, f, n = c["sw"], c["f"], c["nlog"]
        truth = [sw(v) >> n for v in f.values()]
        ok = res[i][2] == truth
        t["correct" if ok else "wrong"] += 1
        if not ok:
            t["wrong_with_positive_shift"] += sw.S > 0
            if len(t["examples"]) < 4:
                t["examples"].append(dict(swizzle=repr(sw), f=show(f), N=1 << n, result=res[i][1],
                                          got=res[i][2], truth=truth))
    t["reject_reasons"] = sorted(set(rej.values()))[:5]
    report["upcast"] = t

    # ---- right inverse
    cases = gen_rinv(N)
    res, rej = run_lines({i: c["body"] for i, c in cases.items()})
    t = dict(cases=len(cases), rejected=len(rej), correct=0, wrong=0, examples=[])
    for i, c in cases.items():
        if i in rej:
            continue
        ok = res[i][2] == list(range(len(res[i][2])))
        t["correct" if ok else "wrong"] += 1
        if not ok and len(t["examples"]) < 4:
            t["examples"].append(dict(swizzle=repr(c["sw"]), f=show(c["f"]), result=res[i][1], L_of_R=res[i][2]))
    t["reject_reasons"] = sorted(set(rej.values()))[:5]
    report["right_inverse"] = t

    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
