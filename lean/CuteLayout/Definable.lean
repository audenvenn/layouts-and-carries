/-
  When is a composite definable over the shape of the inner layout?

  `comp_over_shape_iff`: for a layout `A` with positive mode sizes and a non-empty `ℕ`-valued
  layout `B` with positive mode sizes, there is a layout `R` with the shape of `B` and
  `R(x) = A(B(x))` on the domain of `B` if and only if the carry defect vanishes on that domain;
  and then `R` agrees with the naive composite (Lemma 5.2 for `T' = T` and Corollary 5.4 of the
  paper).
-/
import CuteLayout.Basic

namespace CuTe

section Monoid

variable {D : Type*} [AddCommMonoid D]

lemma eval_zero : ∀ (L : Layout D), eval L 0 = 0
  | [] => by simp
  | [(_, d)] => by simp
  | (s, d) :: m :: L => by
      rw [eval_cons_cons, Nat.zero_mod, Nat.zero_div, zero_smul, zero_add]
      exact eval_zero (m :: L)

/-- The value at the unit of the first mode is its stride (if the mode is not a unit mode). -/
lemma eval_cons_cons_one (t : ℕ) (d : D) (m : ℕ × D) (L : Layout D) (ht : 1 < t) :
    eval ((t, d) :: m :: L) 1 = d := by
  rw [eval_cons_cons, Nat.mod_eq_of_lt ht, Nat.div_eq_of_lt ht, one_smul, eval_zero, add_zero]

/-- Indices that are multiples of the first mode size see only the remaining modes. -/
lemma eval_cons_cons_mul (t : ℕ) (d : D) (m : ℕ × D) (L : Layout D) (ht : 0 < t) (y : ℕ) :
    eval ((t, d) :: m :: L) (t * y) = eval (m :: L) y := by
  rw [eval_cons_cons, Nat.mul_mod_right, Nat.mul_div_cancel_left y ht, zero_smul, zero_add]

end Monoid

lemma size_pos_of_pos : ∀ (B : Layout ℕ), (∀ n ∈ B, 0 < n.1) → 0 < size B
  | [], _ => by simp [size]
  | n :: B, h => by
      have h1 : 0 < n.1 := h n (by simp)
      have h2 : 0 < size B := size_pos_of_pos B (fun m hm => h m (List.mem_cons_of_mem _ hm))
      simp only [size, List.map_cons, List.prod_cons] at h2 ⊢
      exact Nat.mul_pos h1 h2

variable {D : Type*} [AddCommGroup D]

/-! Two layouts with a common shape are bundled as a list of triples
`(size, stride of the inner layout, stride of the outer layout)`. -/

/-- The `ℕ`-valued (inner) layout of a bundle. -/
def innerOf (P : List (ℕ × ℕ × D)) : Layout ℕ := P.map fun m => (m.1, m.2.1)

/-- The `D`-valued (outer) layout of a bundle. -/
def outerOf (P : List (ℕ × ℕ × D)) : Layout D := P.map fun m => (m.1, m.2.2)

omit [AddCommGroup D] in
lemma exists_bundle : ∀ (R : Layout D) (B : Layout ℕ), R.map Prod.fst = B.map Prod.fst →
    ∃ P : List (ℕ × ℕ × D), outerOf P = R ∧ innerOf P = B
  | [], [], _ => ⟨[], rfl, rfl⟩
  | (s, d) :: R, (t, r) :: B, h => by
      simp only [List.map_cons, List.cons.injEq] at h
      obtain ⟨hst, h'⟩ := h
      have hst' : s = t := hst
      subst hst'
      obtain ⟨P, hR, hB⟩ := exists_bundle R B h'
      refine ⟨(s, r, d) :: P, ?_, ?_⟩
      · rw [← hR]; rfl
      · rw [← hB]; rfl
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h

/-- If the outer layout of a bundle composes `A` with the inner one on the whole domain, then its
strides at the non-unit modes are those of the naive composite (the composite is forced). -/
lemma agree_of_eq (A : Layout D) : ∀ (P : List (ℕ × ℕ × D)), (∀ m ∈ P, 0 < m.1) →
    (∀ x, x < size (innerOf P) → eval (outerOf P) x = eval A (eval (innerOf P) x)) →
    ∀ m ∈ P, m.1 = 1 ∨ m.2.2 = eval A m.2.1
  | [], _, _ => by simp
  | [(t, r, d)], hpos, h => by
      intro m hm
      rw [List.mem_singleton] at hm
      subst hm
      by_cases ht : t = 1
      · exact Or.inl ht
      · right
        have ht0 : 0 < t := hpos (t, r, d) (by simp)
        have h1 : 1 < size (innerOf [(t, r, d)]) := by
          simp only [size, innerOf, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
            mul_one]
          omega
        have h2 := h 1 h1
        simpa [innerOf, outerOf] using h2
  | (t, r, d) :: n :: P, hpos, h => by
      have ht0 : 0 < t := hpos (t, r, d) (by simp)
      have hpos' : ∀ m ∈ n :: P, 0 < m.1 := fun m hm => hpos m (List.mem_cons_of_mem _ hm)
      have hsz : 0 < size (innerOf (n :: P)) := size_pos_of_pos _ (by
        intro k hk
        simp only [innerOf, List.mem_map] at hk
        obtain ⟨m, hm, rfl⟩ := hk
        exact hpos' m hm)
      have hsize : size (innerOf ((t, r, d) :: n :: P)) = t * size (innerOf (n :: P)) := by
        simp [size, innerOf]
      have htail : ∀ m ∈ n :: P, m.1 = 1 ∨ m.2.2 = eval A m.2.1 :=
        agree_of_eq A (n :: P) hpos' (by
          intro y hy
          have hx : t * y < size (innerOf ((t, r, d) :: n :: P)) := by
            rw [hsize]
            nlinarith [hy, ht0]
          have h2 := h (t * y) hx
          have e1 : eval (outerOf ((t, r, d) :: n :: P)) (t * y) = eval (outerOf (n :: P)) y :=
            eval_cons_cons_mul t d _ _ ht0 y
          have e2 : eval (innerOf ((t, r, d) :: n :: P)) (t * y) = eval (innerOf (n :: P)) y :=
            eval_cons_cons_mul t r _ _ ht0 y
          rw [e1, e2] at h2
          exact h2)
      intro m hm
      rcases List.mem_cons.mp hm with rfl | hm
      · by_cases ht : t = 1
        · exact Or.inl ht
        · right
          have ht1 : 1 < t := by omega
          have hx : 1 < size (innerOf ((t, r, d) :: n :: P)) := by
            rw [hsize]
            calc 1 < t := ht1
              _ = t * 1 := (Nat.mul_one t).symm
              _ ≤ t * size (innerOf (n :: P)) := Nat.mul_le_mul_left t (Nat.succ_le_of_lt hsz)
          have h2 := h 1 hx
          have e1 : eval (outerOf ((t, r, d) :: n :: P)) 1 = d := eval_cons_cons_one t d _ _ ht1
          have e2 : eval (innerOf ((t, r, d) :: n :: P)) 1 = r := eval_cons_cons_one t r _ _ ht1
          rw [e1, e2] at h2
          exact h2
      · exact htail m hm

/-- If the strides of the outer layout agree with the naive composite at all non-unit modes, the
two layouts have the same function on the domain. -/
lemma eval_outer_eq_naive (A : Layout D) : ∀ (P : List (ℕ × ℕ × D)) (x : ℕ), P ≠ [] →
    (∀ m ∈ P, 0 < m.1) → (∀ m ∈ P, m.1 = 1 ∨ m.2.2 = eval A m.2.1) → x < size (innerOf P) →
    eval (outerOf P) x = eval (naive A (innerOf P)) x
  | [], _, h, _, _, _ => absurd rfl h
  | [(t, r, d)], x, _, _, hag, hx => by
      have hx' : x < t := by simpa [size, innerOf] using hx
      show eval [(t, d)] x = eval [(t, eval A r)] x
      rw [eval_singleton, eval_singleton]
      rcases hag (t, r, d) (by simp) with h1 | h1
      · have h1' : t = 1 := h1
        have hx0 : x = 0 := by omega
        rw [hx0, zero_smul, zero_smul]
      · have h1' : d = eval A r := h1
        rw [h1']
  | (t, r, d) :: n :: P, x, _, hpos, hag, hx => by
      have hsize : size (innerOf ((t, r, d) :: n :: P)) = t * size (innerOf (n :: P)) := by
        simp [size, innerOf]
      have hx' : x / t < size (innerOf (n :: P)) := by
        rw [hsize] at hx
        exact Nat.div_lt_of_lt_mul hx
      have ih := eval_outer_eq_naive A (n :: P) (x / t) (List.cons_ne_nil _ _)
        (fun m hm => hpos m (List.mem_cons_of_mem _ hm))
        (fun m hm => hag m (List.mem_cons_of_mem _ hm)) hx'
      have e1 : eval (outerOf ((t, r, d) :: n :: P)) x
          = (x % t) • d + eval (outerOf (n :: P)) (x / t) := eval_cons_cons t d _ _ x
      have e2 : eval (naive A (innerOf ((t, r, d) :: n :: P))) x
          = (x % t) • eval A r + eval (naive A (innerOf (n :: P))) (x / t) :=
        eval_cons_cons t (eval A r) _ _ x
      rw [e1, e2, ih]
      rcases hag (t, r, d) (by simp) with h1 | h1
      · have h1' : t = 1 := h1
        subst h1'
        rw [Nat.mod_one, zero_smul, zero_smul]
      · have h1' : d = eval A r := h1
        rw [h1']

/-- **Definability over the shape of `B`.**  A layout with the shape of `B` whose function is
`A ∘ B` on the domain of `B` exists if and only if the carry defect vanishes on that domain. -/
theorem comp_over_shape_iff (L : Layout D) (s : ℕ) (d : D) (hs : 0 < s) (hL : Pos L)
    (B : Layout ℕ) (hB : B ≠ []) (hBpos : ∀ n ∈ B, 0 < n.1) :
    (∃ R : Layout D, R.map Prod.fst = B.map Prod.fst ∧
        ∀ x, x < size B → eval R x = eval ((s, d) :: L) (eval B x))
      ↔ ∀ x, x < size B → defect (jumpList s (s • d) L) (digitPairs B x) = 0 := by
  constructor
  · rintro ⟨R, hshape, hR⟩ x hx
    obtain ⟨P, hPo, hPi⟩ := exists_bundle R B hshape
    subst hPo
    subst hPi
    have hpos : ∀ m ∈ P, 0 < m.1 := fun m hm =>
      hBpos (m.1, m.2.1) (List.mem_map.mpr ⟨m, hm, rfl⟩)
    have hne : P ≠ [] := fun h => hB (by rw [h]; rfl)
    have hag := agree_of_eq ((s, d) :: L) P hpos hR
    have h1 := eval_outer_eq_naive ((s, d) :: L) P x hne hpos hag hx
    rw [← eval_comp_eq_naive_iff L s d hs hL (innerOf P) x, ← hR x hx, h1]
  · intro h
    refine ⟨naive ((s, d) :: L) B, ?_, ?_⟩
    · simp [naive, Function.comp_def]
    · intro x hx
      exact ((eval_comp_eq_naive_iff L s d hs hL B x).mpr (h x hx)).symm

/-- In that case the composite is the naive composite: any layout with the shape of `B` that
realises `A ∘ B` has the same function as `naive A B` (Lemma 5.2 for `T' = T`). -/
theorem comp_over_shape_unique (A : Layout D) (R : Layout D) (B : Layout ℕ) (hB : B ≠ [])
    (hBpos : ∀ n ∈ B, 0 < n.1) (hshape : R.map Prod.fst = B.map Prod.fst)
    (hR : ∀ x, x < size B → eval R x = eval A (eval B x)) (x : ℕ) (hx : x < size B) :
    eval R x = eval (naive A B) x := by
  obtain ⟨P, hPo, hPi⟩ := exists_bundle R B hshape
  subst hPo
  subst hPi
  have hpos : ∀ m ∈ P, 0 < m.1 := fun m hm =>
    hBpos (m.1, m.2.1) (List.mem_map.mpr ⟨m, hm, rfl⟩)
  have hne : P ≠ [] := fun h => hB (by rw [h]; rfl)
  exact eval_outer_eq_naive A P x hne hpos (agree_of_eq A P hpos hR) hx

end CuTe
