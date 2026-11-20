# Layouts and Carries

A mathematical foundation for the layout algebra of CuTe, the core of NVIDIA CUTLASS:
the paper, a Lean 4 formalisation of its core results, and the programs that check the theory
and compare it with three CuTe implementations.

**Paper:** [`paper/main.pdf`](paper/main.pdf) —
*Layouts and Carries: A Mathematical Foundation for the CuTe Layout Algebra*, Auden Venn, 2026.

## The idea

A CuTe layout is a pair `shape:stride`, read as a function: expand the index in the mixed radix
given by the shape and take the inner product of the digits with the strides. Write
`P_i` for the prefix products of the shape and `γ_i = d_i − s_{i−1}·d_{i−1}` for the *jumps* of
the strides. Then

    f(x)     = x·d_0 + Σ_i ⌊x / P_i⌋·γ_i
    f(x + y) = f(x) + f(y) + Σ_i c_i(x, y)·γ_i ,     c_i = the carry into digit i

A layout is additive up to carries, and each carry is charged a jump. The paper derives the
layout algebra from this identity:

* **Which functions are layouts.** A function is a layout exactly when the scales that occur in
  its expansion in the functions `⌊x/Q⌋` form a divisibility chain and divide its size;
  `coalesce` is the resulting normal form.
* **Composition.** The composite, if it exists, is obtained by evaluating the outer layout at
  the strides of the inner one, and it exists exactly when a *carry defect* vanishes. For common
  outer layouts this is one inequality, `Σ_j (t_j − 1)(r_j mod P) < P`.
* **Complements** are the classical direct-sum decompositions of an interval, and logical
  division has an exact criterion.
* **Swizzles** are layouts for the other addition on offsets, XOR, which differs from integer
  addition by the carries of the binary radix.

## Contents

| Path | What it is |
|---|---|
| `paper/` | LaTeX source and PDF of the paper |
| `lean/` | Lean 4 + Mathlib formalisation of the core results (about 1500 lines, no `sorry`) |
| `python/layout_theory.py`, `python/swizzle_theory.py` | reference implementation of the definitions, independent of CuTe |
| `python/verify_theory.py`, `python/verify_swizzle.py` | exhaustive and randomised checks of the criteria and examples of the paper |
| `python/difftest.py`, `python/difftest_f2.py` | differential tests against the two Python implementations of CuTe |
| `python/difftest_cpp.py`, `python/difftest_swizzle_cpp.py` | differential tests against the C++ CuTe headers |
| `tests/cpp/` | the C++ instances quoted in the paper |
| `results/` | the JSON reports behind the tables of the paper |
| `scripts/` | `fetch_refs.sh` (third-party sources), `setup_lean.sh` |

Which statements are machine-checked, and which rest on paper proofs and program checks, is
listed in Section 9.1 of the paper and in [`lean/README.md`](lean/README.md).

## Reproducing

All commands are run from the repository root. The Python programs need no packages (they were
run with Python 3.15).

```sh
# 0. third-party sources (needs git and curl; about 100 MB on disk)
sh scripts/fetch_refs.sh

# 1. checks of the theory (about 40 s)
python3 python/verify_theory.py 0
python3 python/verify_swizzle.py 0

# 2. differential tests against the two Python implementations (about 20 s each)
python3 python/difftest.py nvlabs  refs/repos/nvlabs-cute 0 20000 > results/difftest_nvlabs.json
python3 python/difftest.py cutlass refs/cutlass/python    0 20000 > results/difftest_cutlass.json
python3 python/difftest_f2.py refs/repos/nvlabs-cute 0 20000     > results/difftest_f2.json

# 3. differential tests against C++ CuTe (clang++ with C++17; no CUDA toolchain or GPU needed;
#    a few minutes each)
python3 python/difftest_cpp.py refs/cutlass/include refs/cuda-headers /tmp/cute-cpp 0 3000 random     > results/difftest_cpp_random.json
python3 python/difftest_cpp.py refs/cutlass/include refs/cuda-headers /tmp/cute-cpp 2 3000 structured > results/difftest_cpp_structured.json
python3 python/difftest_swizzle_cpp.py refs/cutlass/include refs/cuda-headers /tmp/cute-swz 0 600     > results/difftest_swizzle_cpp.json

# 4. the C++ instances quoted in the paper
clang++ -std=c++17 -w -isystem refs/cuda-headers/cuda_cudart/include \
  -isystem refs/cuda-headers/cuda_crt/include -isystem refs/cuda-headers/cccl/include/cccl \
  -I refs/cutlass/include tests/cpp/examples.cpp -o /tmp/examples && /tmp/examples
# (and the same for tests/cpp/swizzle_examples.cpp)

# 5. the paper (tectonic; any recent LaTeX distribution works as well)
cd paper && tectonic -X compile main.tex && cd ..

# 6. the Lean development (needs elan; see scripts/setup_lean.sh).  `lake exe cache get`
#    downloads the prebuilt Mathlib, several GB.
cd lean && lake exe cache get && lake build
```

## Third-party software

The tests run against the following, none of which is redistributed here:

* CUTLASS, <https://github.com/NVIDIA/cutlass>, commit `af95104` (the CuTe headers and the bundled
  Python module `pycute`);
* PyCuTe, <https://github.com/NVlabs/CuTe>, commit `56cc3d0`;
* NVIDIA's redistributable CUDA 13.5 header archives `cccl`, `cuda_crt` and `cuda_cudart`
  (headers only; they let the CuTe headers compile with a host compiler).

The differential tests use small random layouts; the counts they produce show that certain
behaviours exist and how they split by cause, not how often they occur in practice.

## How this work was produced

This work was carried out with substantial assistance from an AI system (Claude, developed by
Anthropic), which, under the author's direction, surveyed the literature, proposed and developed
the framework, wrote the proofs, the Lean formalisation and the test programs, and drafted the
manuscript. Drafts were independently reviewed by a second AI system (Codex, developed by
OpenAI), which found a number of errors that have been corrected and contributed several
examples and counterexamples. The author takes responsibility for the content. The AI systems
are not authors.

Only the statements listed as machine-checked have been verified by Lean; the others have paper
proofs and have been tested by the programs in `python/`.

## License

The software (`python/`, `lean/`, `tests/`, `scripts/`) is licensed under the Apache License 2.0
(see [`LICENSE`](LICENSE)); the paper (`paper/`) under CC BY 4.0 (see
[`paper/LICENSE`](paper/LICENSE)).

## Citation

```bibtex
@misc{venn2026layouts,
  author = {Auden Venn},
  title  = {Layouts and Carries: A Mathematical Foundation for the {CuTe} Layout Algebra},
  year   = {2026}
}
```
