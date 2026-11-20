/-
  Layouts are the carry-free additive maps (Definition 3.5 and Proposition 3.6 of the paper).

  `additive_iff_layout`: for a non-empty shape `S` with positive mode sizes, a function `g` is
  `S`-additive -- `g (x + y) = g x + g y` whenever the digits of `x` and `y` add without carry --
  if and only if it agrees on `{0, …, |S| - 1}` with the function of a layout over `S`.  The
  strides are then the values of `g` at the scales (`stridesOf`).
-/
import CuteLayout.Basic

namespace CuTe

variable {D : Type*} [AddCommGroup D]

/-- A pair of indices is carry-free for the shape `S` if its digits add without carry
(Definition 3.5); in particular both indices and their sum lie below `|S|`. -/
def CarryFree : List ℕ → ℕ → ℕ → Prop
  | [], x, y => x + y < 1
  | s :: S, x, y => x % s + y % s < s ∧ CarryFree S (x / s) (y / s)

lemma list_prod_pos : ∀ (S : List ℕ), (∀ s ∈ S, 0 < s) → 0 < S.prod
  | [], _ => by simp
  | s :: S, h => by
      rw [List.prod_cons]
      exact Nat.mul_pos (h s (by simp))
        (list_prod_pos S (fun t ht => h t (List.mem_cons_of_mem _ ht)))

lemma carryFree_lt : ∀ (S : List ℕ) (x y : ℕ), CarryFree S x y → x + y < S.prod
  | [], x, y, h => by
      have h' : x + y < 1 := h
      simpa using h'
  | s :: S, x, y, h => by
      have h' : x % s + y % s < s ∧ CarryFree S (x / s) (y / s) := h
      obtain ⟨h1, h2⟩ := h'
      have ih : x / s + y / s + 1 ≤ S.prod := carryFree_lt S (x / s) (y / s) h2
      have h3 : s * (x / s + y / s + 1) ≤ s * S.prod := Nat.mul_le_mul_left s ih
      rw [Nat.mul_add, Nat.mul_add, Nat.mul_one] at h3
      have hx := Nat.div_add_mod x s
      have hy := Nat.div_add_mod y s
      rw [List.prod_cons]
      omega

lemma carryFree_zero_left : ∀ (S : List ℕ), (∀ s ∈ S, 0 < s) → ∀ u, u < S.prod →
    CarryFree S 0 u
  | [], _, u, hu => by
      simp only [List.prod_nil] at hu
      show 0 + u < 1
      omega
  | s :: S, hS, u, hu => by
      have hs : 0 < s := hS s (by simp)
      rw [List.prod_cons] at hu
      show 0 % s + u % s < s ∧ CarryFree S (0 / s) (u / s)
      rw [Nat.zero_mod, Nat.zero_add, Nat.zero_div]
      exact ⟨Nat.mod_lt u hs,
        carryFree_zero_left S (fun t ht => hS t (List.mem_cons_of_mem _ ht)) (u / s)
          (Nat.div_lt_of_lt_mul hu)⟩

/-- The layout over `S` whose strides are the values of `g` at the scales `P, P * s₀, …`. -/
def stridesOf (g : ℕ → D) : ℕ → List ℕ → Layout D
  | _, [] => []
  | P, s :: S => (s, g P) :: stridesOf g (P * s) S

omit [AddCommGroup D] in
lemma stridesOf_fst (g : ℕ → D) : ∀ (S : List ℕ) (P : ℕ), (stridesOf g P S).map Prod.fst = S
  | [], _ => rfl
  | s :: S, P => by
      simp only [stridesOf, List.map_cons]
      rw [stridesOf_fst g S]

/-- Along one digit an additive map is linear. -/
lemma additive_digit (g : ℕ → D) (P s : ℕ) (S : List ℕ) (hS : ∀ t ∈ s :: S, 0 < t)
    (H : ∀ x y, CarryFree (s :: S) x y → g (P * (x + y)) = g (P * x) + g (P * y)) :
    ∀ r, r < s → g (P * r) = r • g P := by
  have hS' : ∀ t ∈ S, 0 < t := fun t ht => hS t (List.mem_cons_of_mem _ ht)
  have hprod : 0 < S.prod := list_prod_pos S hS'
  have h00 : CarryFree S 0 0 := carryFree_zero_left S hS' 0 hprod
  intro r
  induction r with
  | zero =>
    intro _
    have h := H 0 0 (carryFree_zero_left (s :: S) hS 0 (list_prod_pos (s :: S) hS))
    rw [Nat.add_zero] at h
    have h' : g (P * 0) + 0 = g (P * 0) + g (P * 0) := by
      rw [add_zero]
      exact h
    rw [zero_smul]
    exact (add_left_cancel h').symm
  | succ r ih =>
    intro hr
    have hcf : CarryFree (s :: S) r 1 := by
      show r % s + 1 % s < s ∧ CarryFree S (r / s) (1 / s)
      rw [Nat.mod_eq_of_lt (by omega : r < s), Nat.mod_eq_of_lt (by omega : 1 < s),
        Nat.div_eq_of_lt (by omega : r < s), Nat.div_eq_of_lt (by omega : 1 < s)]
      exact ⟨hr, h00⟩
    have h := H r 1 hcf
    rw [h, ih (by omega), Nat.mul_one, succ_nsmul]

/-- **Proposition 3.6, first half**: an `S`-additive map is the function of the layout over `S`
whose strides are its values at the scales. -/
theorem eval_stridesOf (g : ℕ → D) : ∀ (S : List ℕ) (P : ℕ), (∀ s ∈ S, 0 < s) → S ≠ [] →
    (∀ x y, CarryFree S x y → g (P * (x + y)) = g (P * x) + g (P * y)) →
    ∀ x, x < S.prod → eval (stridesOf g P S) x = g (P * x)
  | [], _, _, hne, _, _, _ => absurd rfl hne
  | [s], P, hS, _, H, x, hx => by
      have hx' : x < s := by simpa using hx
      show eval [(s, g P)] x = g (P * x)
      rw [eval_singleton, additive_digit g P s [] hS H x hx']
  | s :: t :: S, P, hS, _, H, x, hx => by
      have hs : 0 < s := hS s (by simp)
      have hS' : ∀ u ∈ t :: S, 0 < u := fun u hu => hS u (List.mem_cons_of_mem _ hu)
      have hx' : x / s < (t :: S).prod := by
        rw [List.prod_cons] at hx
        exact Nat.div_lt_of_lt_mul hx
      have H' : ∀ u v, CarryFree (t :: S) u v →
          g (P * s * (u + v)) = g (P * s * u) + g (P * s * v) := by
        intro u v huv
        have hcf : CarryFree (s :: t :: S) (s * u) (s * v) := by
          show (s * u) % s + (s * v) % s < s ∧ CarryFree (t :: S) (s * u / s) (s * v / s)
          rw [Nat.mul_mod_right, Nat.mul_mod_right, Nat.mul_div_cancel_left u hs,
            Nat.mul_div_cancel_left v hs]
          exact ⟨hs, huv⟩
        have h := H (s * u) (s * v) hcf
        rw [← Nat.mul_add, ← Nat.mul_assoc, ← Nat.mul_assoc, ← Nat.mul_assoc] at h
        exact h
      have ih := eval_stridesOf g (t :: S) (P * s) hS' (List.cons_ne_nil _ _) H' (x / s) hx'
      have hcf : CarryFree (s :: t :: S) (x % s) (s * (x / s)) := by
        show (x % s) % s + (s * (x / s)) % s < s
          ∧ CarryFree (t :: S) (x % s / s) (s * (x / s) / s)
        rw [Nat.mod_mod, Nat.mul_mod_right, Nat.add_zero, Nat.div_eq_of_lt (Nat.mod_lt x hs),
          Nat.mul_div_cancel_left (x / s) hs]
        exact ⟨Nat.mod_lt x hs, carryFree_zero_left (t :: S) hS' (x / s) hx'⟩
      have h := H (x % s) (s * (x / s)) hcf
      rw [Nat.mod_add_div] at h
      have hrest : stridesOf g (P * s) (t :: S) = (t, g (P * s)) :: stridesOf g (P * s * t) S :=
        rfl
      show eval ((s, g P) :: stridesOf g (P * s) (t :: S)) x = g (P * x)
      rw [hrest, eval_cons_cons, ← hrest, ih, h,
        additive_digit g P s (t :: S) hS H (x % s) (Nat.mod_lt x hs), ← Nat.mul_assoc]

/-- **Proposition 3.6, second half**: a layout function is additive on carry-free pairs. -/
theorem eval_add_of_carryFree : ∀ (L : Layout D) (x y : ℕ), CarryFree (L.map Prod.fst) x y →
    eval L (x + y) = eval L x + eval L y
  | [], _, _, _ => by simp
  | [(s, d)], x, y, _ => by rw [eval_singleton, eval_singleton, eval_singleton, add_smul]
  | (s, d) :: m :: L, x, y, h => by
      have h' : x % s + y % s < s ∧ CarryFree ((m :: L).map Prod.fst) (x / s) (y / s) := h
      obtain ⟨h1, h2⟩ := h'
      have hs : 0 < s := by omega
      have hn : ¬ s ≤ x % s + y % s := by omega
      have hmod : (x + y) % s = x % s + y % s := by
        rw [Nat.add_mod, Nat.mod_eq_of_lt h1]
      have hdiv : (x + y) / s = x / s + y / s := by
        rw [Nat.add_div hs, ite_eq_right hn, Nat.add_zero]
      rw [eval_cons_cons, eval_cons_cons, eval_cons_cons, hmod, hdiv,
        eval_add_of_carryFree (m :: L) (x / s) (y / s) h2, add_smul]
      abel

/-- **Proposition 3.6** (layouts are the carry-free additive maps).  For a non-empty shape `S`
with positive mode sizes, `g` is `S`-additive if and only if it agrees on `{0, …, |S| - 1}` with
the function of some layout over `S`. -/
theorem additive_iff_layout (S : List ℕ) (hS : ∀ s ∈ S, 0 < s) (hne : S ≠ []) (g : ℕ → D) :
    (∀ x y, CarryFree S x y → g (x + y) = g x + g y)
      ↔ ∃ L : Layout D, L.map Prod.fst = S ∧ ∀ x, x < S.prod → eval L x = g x := by
  constructor
  · intro H
    refine ⟨stridesOf g 1 S, stridesOf_fst g S 1, ?_⟩
    intro x hx
    have H1 : ∀ x y, CarryFree S x y → g (1 * (x + y)) = g (1 * x) + g (1 * y) := by
      intro x y hxy
      rw [Nat.one_mul, Nat.one_mul, Nat.one_mul]
      exact H x y hxy
    have h := eval_stridesOf g S 1 hS hne H1 x hx
    rw [Nat.one_mul] at h
    exact h
  · rintro ⟨L, hL, hg⟩ x y hxy
    have hlt := carryFree_lt S x y hxy
    have hxy' : CarryFree (L.map Prod.fst) x y := by
      rw [hL]
      exact hxy
    rw [← hg (x + y) hlt, ← hg x (by omega), ← hg y (by omega)]
    exact eval_add_of_carryFree L x y hxy'

end CuTe
