# Lean formalisation

Lean 4 with Mathlib (`v4.37.0-rc1`, see `lean-toolchain` and `lakefile.toml`). The development is
about 1500 lines, contains no `sorry`, and depends only on the standard axioms `propext`,
`Classical.choice` and `Quot.sound`.

    lake exe cache get    # prebuilt Mathlib
    lake build

A flat layout is a list of pairs (size, stride) with strides in an arbitrary additive commutative
group; `eval` is the extended layout function.

| File | Contents (numbers refer to the paper) |
|---|---|
| `CuteLayout/Basic.lean` | floor-sum form (Prop. 3.2), carry identity (Thm. 3.4), composition formula (Thm. 5.3), Cor. 5.4, carry-free composition (Thm. 5.7(a)), the merge step of coalesce |
| `CuteLayout/Additive.lean` | layouts are the carry-free additive maps (Prop. 3.6) |
| `CuteLayout/NormalForm.lean` | uniqueness of the reduced floor-sum form |
| `CuteLayout/Spectrum.lean` | floor expansion (Lemma 4.1), layouts have chain spectra (Thm. 4.3 and 4.3(a)) |
| `CuteLayout/Definable.lean` | the composite over the shape of the inner layout is forced, and exists iff the carry defect vanishes (Lemma 5.2 for `T' = T`, Cor. 5.4) |
| `CuteLayout/Exact.lean` | converse of the carry-free criterion for independent jumps (Thm. 5.7(b), first assertion) |
| `CuteLayout/Swizzle.lean` | a swizzle is an involution and is linear for XOR (Lemma 8.3(a), (b); one field, right shift) |
| `CuteLayout/Examples.lean` | Examples 5.9, 5.11, 5.12 by evaluation |

Not formalised: the uniqueness statements of Prop. 3.6 and Cor. 4.4, Thm. 5.5, the last
assertion of Thm. 5.7(b), Sections 6 and 7, and the rest of Section 8. Table 2 of the paper gives
the exact correspondence.
