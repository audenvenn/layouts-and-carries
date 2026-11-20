"""Differential test of the C++ CuTe headers (static integers) against the theory.

    python3 -I python/difftest_cpp.py <cutlass-include-dir> <cuda-header-root> <workdir> [seed] [n] [suite]

CuTe checks composition admissibility with static_asserts, so an inadmissible pair is a
compile error.  Cases are compiled in chunks: every case is one line in its own function, clang
is asked to report all errors (-ferror-limit=0), and a case counts as rejected iff a diagnostic
(or the instantiation backtrace of one) points at its line.  The surviving cases are then built
into a runner that prints each resulting layout and its values.  Chunks are small and compiled
one at a time to keep memory use low.

The CUDA headers are NVIDIA's redistributable cccl / cuda_crt / cuda_cudart archives; nothing
from them is executed -- the layout algebra is host-side constexpr code.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
from difftest import (  # noqa: E402
    SIZES_A, SIZES_B, STRIDES_A, STRIDES_B, Tally, coalesce_ext, rand_flat, structured_pair,
    why_rejected,
)
from layout_theory import Layout, compose, digits  # noqa: E402

INC, CUDA, WORK = sys.argv[1], sys.argv[2], sys.argv[3]
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
N = int(sys.argv[5]) if len(sys.argv) > 5 else 300
SUITE = sys.argv[6] if len(sys.argv) > 6 else "random"
CHUNK = 80
FLAGS = ["-std=c++17", "-w", "-ferror-limit=0",
         "-isystem", f"{CUDA}/cuda_cudart/include", "-isystem", f"{CUDA}/cuda_crt/include",
         "-isystem", f"{CUDA}/cccl/include/cccl", "-I", INC]
PRELUDE = "#include <iostream>\n#include <cute/layout.hpp>\nusing namespace cute;\n"   # 3 lines


def cpp_layout(shape, stride) -> str:
    def tup(kind, vals):
        return f"make_{kind}(" + ",".join(f"Int<{v}>{{}}" for v in vals) + ")"
    if len(shape) == 1:
        return f"make_layout(Int<{shape[0]}>{{}}, Int<{stride[0]}>{{}})"
    return f"make_layout({tup('shape', shape)}, {tup('stride', stride)})"


def case_line(i, A, B) -> str:
    """One case = one source line (so diagnostics can be attributed by line number)."""
    return (f"void case_{i}() {{ auto A = {cpp_layout(*A)}; auto B = {cpp_layout(*B)}; auto R = composition(A, B); "
            f"std::cout << \"CASE {i} \" << R << \" |\"; "
            f"for (int k = 0; k < int(size(R)); ++k) std::cout << \" \" << R(k); std::cout << \"\\n\"; }}\n")


def show(shape, stride) -> str:
    return f"{shape[0]}:{stride[0]}" if len(shape) == 1 else f"{shape}:{stride}".replace(" ", "")


def run_cases(cases: dict) -> tuple[dict, dict]:
    """cases: id -> (A, B) with A, B = (shape, stride).  Returns (results, reasons):
    results[id] = (printed layout, values) for cases that compile; reasons[id] = assertion text."""
    os.makedirs(WORK, exist_ok=True)
    ids = sorted(cases)
    results, reasons = {}, {}
    for c in range(0, len(ids), CHUNK):
        good = ids[c:c + CHUNK]
        src, exe = f"{WORK}/chunk.cpp", f"{WORK}/chunk"
        # Clang diagnoses a failing template specialization only where it is first instantiated,
        # so two cases that share one (same outer layout and inner mode) are not both reported
        # in a single pass.  Repeat until the remaining cases compile cleanly.
        while good:
            line_of = {}
            with open(src, "w") as f:
                f.write(PRELUDE)
                for n, i in enumerate(good):
                    line_of[4 + n] = i
                    f.write(case_line(i, *cases[i]))
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
                    last_msg = line.split("error: ", 1)[1][:80]
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
            m = re.match(r"CASE (\d+) (\S+) \|(.*)", line)
            results[int(m.group(1))] = (m.group(2), [int(x.lstrip("_")) for x in m.group(3).split()])
    return results, reasons


def single_mode_cause(Af: Layout, t: int, r: int) -> str:
    """Why a single-mode composite returned by C++ can be wrong: describe the stride r in the
    radix of the (extended-)coalesced outer layout."""
    Ac = coalesce_ext(Af)
    dg = digits(r, Ac.shape)
    nz = [i for i, c in enumerate(dg) if c]
    if len(nz) == 1 and nz[0] < Ac.rank - 1 and Ac.shape[nz[0]] % dg[nz[0]] != 0:
        return "stride_digit_does_not_divide_mode_size"
    if len(nz) >= 2:
        return "stride_has_several_digits"
    return "other"


def main():
    rng = random.Random(SEED)
    cases = {}
    for i in range(N):
        if SUITE == "structured":
            As, Ad, Bs, Bd = structured_pair(rng)
        else:
            As, Ad = rand_flat(rng, 3, SIZES_A, STRIDES_A)
            Bs, Bd = rand_flat(rng, 3, SIZES_B, STRIDES_B)
        cases[i] = ((As, Ad), (Bs, Bd))
    out, why = run_cases(cases)

    # second pass: the single modes of every multi-mode case whose result is wrong
    wrong_multi = [i for i in out if len(cases[i][1][0]) > 1
                   and out[i][1] != [Layout(*cases[i][0])(b) for b in Layout(*cases[i][1]).values()]]
    sub = {}
    for i in wrong_multi:
        (A, (Bs, Bd)) = cases[i]
        for j, (t, r) in enumerate(zip(Bs, Bd)):
            sub[(i + 1) * 1000 + j] = (A, ((t,), (r,)))
    sub_out, _ = run_cases(sub) if sub else ({}, {})

    T = Tally()
    for i, ((As, Ad), (Bs, Bd)) in cases.items():
        Af, Bf = Layout(As, Ad), Layout(Bs, Bd)
        want = [Af(b) for b in Bf.values()]
        truth = compose(Af, Bf)
        ex = f"{show(As, Ad)} o {show(Bs, Bd)}"
        wgt = Af.size * Bf.size + 10 * (Af.rank + Bf.rank)
        if i in out:
            text, got = out[i]
            if got == want:
                T.add("correct")
                continue
            T.add("silent_wrong", None)
            T.add("silent_wrong/a_layout_exists" if truth is not None else "silent_wrong/no_layout_exists")
            if len(got) != Bf.size:
                kind = "result_has_wrong_size"
            elif len(Bs) == 1:
                kind = "single_mode/" + single_mode_cause(Af, Bs[0], Bd[0])
            else:
                modes_ok = all(
                    (i + 1) * 1000 + j in sub_out and sub_out[(i + 1) * 1000 + j][1] == [Af(r * x) for x in range(t)]
                    for j, (t, r) in enumerate(zip(Bs, Bd)))
                kind = "interference_between_modes" if modes_ok else "multi_mode_with_a_wrong_single_mode"
            T.add(f"silent_wrong/{kind}", f"{ex} -> {text}={got}; A(B(i))={want}", wgt)
        elif truth is None:
            T.add("correctly_rejected")
            T.add(f"correctly_rejected/[{why[i]}]")
        else:
            T.add("rejected_but_composable", None)
            cause = "other"
            for t, r in zip(Bs, Bd):
                if compose(Af, Layout(t, r)) is not None:
                    c = why_rejected(Af, t, r)
                    if c != "other":
                        cause = c
                        break
            res = " , ".join(repr(Rj) for Rj in truth)
            T.add(f"rejected_but_composable/{cause}", f"{ex} = ({res})  [{why[i]}]", wgt)
    rep = T.report()
    rep.update({"implementation": "cute-c++ (static integers)", "seed": SEED, "n": N, "suite": SUITE})
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
