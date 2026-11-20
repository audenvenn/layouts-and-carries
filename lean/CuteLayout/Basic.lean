/-
  The carry calculus of CuTe layouts: core definitions and theorems.

  A flat layout is a list of modes `(size, stride)`, least significant first.  Strides live in an
  additive commutative group `D` (offsets `ℤ`, coordinates `ℤ^k`, bit vectors under XOR, ...).

  Main results (numbers refer to the paper "Layouts and Carries"):
  * `eval_eq_floorSum`  – Proposition 3.2 (floor-sum form);
  * `eval_add`          – Theorem 3.4 (carry identity, two summands);
  * `eval_dot`, `eval_comp` – Theorem 5.3 (composition = evaluation + carry defect);
  * `eval_comp_eq_naive_iff` – Corollary 5.4;
  * `eval_comp_of_carryFree`, `eval_comp_of_carryFree_ineq` – Theorem 5.7(a);
  * `eval_merge`        – the coalescing step preserves the layout function.
-/
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Algebra.Order.Ring.Nat
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

namespace CuTe

/-- A flat layout: modes `(size, stride)`, least significant first. -/
abbrev Layout (D : Type*) := List (ℕ × D)

/-! ### Floor sums: functions `x ↦ Σ_j ⌊x / Q_j⌋ • γ_j` -/

section FloorSum

variable {D : Type*} [AddCommGroup D]

/-- `floorSum J x = Σ_{(Q, γ) ∈ J} ⌊x / Q⌋ • γ` for a list `J` of (scale, jump) pairs. -/
def floorSum (J : List (ℕ × D)) (x : ℕ) : D := (J.map fun j => (x / j.1) • j.2).sum

/-- The carry out of the digits below scale `P` when `x` and `y` are added. -/
def carry (P x y : ℕ) : ℕ := if P ≤ x % P + y % P then 1 else 0

/-- `Σ_{(Q, γ) ∈ J} carry(Q, x, y) • γ`. -/
def carrySum (J : List (ℕ × D)) (x y : ℕ) : D := (J.map fun j => carry j.1 x y • j.2).sum

/-- `dot cs = Σ c_j * r_j` for a list of (coefficient, stride) pairs. -/
def dot (cs : List (ℕ × ℕ)) : ℕ := (cs.map fun p => p.1 * p.2).sum

/-- Total carry across scale `P` in the schoolbook evaluation of `Σ c_j * r_j`. -/
def carryN (P : ℕ) (cs : List (ℕ × ℕ)) : ℕ := (cs.map fun p => p.1 * (p.2 % P)).sum / P

/-- The carry defect `Σ_{(Q, γ) ∈ J} carryN(Q, cs) • γ`. -/
def defect (J : List (ℕ × D)) (cs : List (ℕ × ℕ)) : D := (J.map fun j => carryN j.1 cs • j.2).sum

@[simp] lemma floorSum_nil (x : ℕ) : floorSum ([] : List (ℕ × D)) x = 0 := by simp [floorSum]
@[simp] lemma floorSum_cons (j : ℕ × D) (J : List (ℕ × D)) (x : ℕ) :
    floorSum (j :: J) x = (x / j.1) • j.2 + floorSum J x := by simp [floorSum]
@[simp] lemma carrySum_nil (x y : ℕ) : carrySum ([] : List (ℕ × D)) x y = 0 := by simp [carrySum]
@[simp] lemma carrySum_cons (j : ℕ × D) (J : List (ℕ × D)) (x y : ℕ) :
    carrySum (j :: J) x y = carry j.1 x y • j.2 + carrySum J x y := by simp [carrySum]
@[simp] lemma defect_nil (cs : List (ℕ × ℕ)) : defect ([] : List (ℕ × D)) cs = 0 := by simp [defect]
@[simp] lemma defect_cons (j : ℕ × D) (J : List (ℕ × D)) (cs : List (ℕ × ℕ)) :
    defect (j :: J) cs = carryN j.1 cs • j.2 + defect J cs := by simp [defect]

lemma add_div_carry {P : ℕ} (hP : 0 < P) (x y : ℕ) :
    (x + y) / P = x / P + y / P + carry P x y := by
  unfold carry
  exact Nat.add_div hP

/-- The carry identity for floor sums. -/
theorem floorSum_add (J : List (ℕ × D)) (hJ : ∀ j ∈ J, 0 < j.1) (x y : ℕ) :
    floorSum J (x + y) = floorSum J x + floorSum J y + carrySum J x y := by
  induction J with
  | nil => simp
  | cons j J ih =>
    have hj : 0 < j.1 := hJ j (by simp)
    have hJ' : ∀ k ∈ J, 0 < k.1 := fun k hk => hJ k (List.mem_cons_of_mem _ hk)
    rw [floorSum_cons, floorSum_cons, floorSum_cons, carrySum_cons, ih hJ', add_div_carry hj,
      add_smul, add_smul]
    abel

lemma dot_div {P : ℕ} (hP : 0 < P) (cs : List (ℕ × ℕ)) :
    dot cs / P = (cs.map fun p => p.1 * (p.2 / P)).sum + carryN P cs := by
  have key : dot cs
      = P * (cs.map fun p => p.1 * (p.2 / P)).sum + (cs.map fun p => p.1 * (p.2 % P)).sum := by
    induction cs with
    | nil => simp [dot]
    | cons p cs ih =>
      have h2 : p.1 * p.2 = P * (p.1 * (p.2 / P)) + p.1 * (p.2 % P) := by
        have h := Nat.div_add_mod p.2 P
        calc p.1 * p.2 = p.1 * (P * (p.2 / P) + p.2 % P) := by rw [h]
          _ = P * (p.1 * (p.2 / P)) + p.1 * (p.2 % P) := by ring
      simp only [dot, List.map_cons, List.sum_cons] at ih ⊢
      rw [ih, h2]
      ring
  rw [key, carryN, Nat.mul_add_div hP]

/-- Evaluating a floor sum at `Σ c_j r_j`: term-wise evaluation plus the carry defect. -/
theorem floorSum_dot (J : List (ℕ × D)) (hJ : ∀ j ∈ J, 0 < j.1) (cs : List (ℕ × ℕ)) :
    floorSum J (dot cs) = (cs.map fun p => p.1 • floorSum J p.2).sum + defect J cs := by
  induction J with
  | nil =>
    have h0 : ∀ l : List (ℕ × ℕ),
        (l.map fun p => p.1 • floorSum ([] : List (ℕ × D)) p.2).sum = 0 := by
      intro l
      induction l with
      | nil => simp
      | cons p l ihl =>
        simp only [List.map_cons, List.sum_cons]
        rw [ihl, floorSum_nil, smul_zero, add_zero]
    rw [floorSum_nil, defect_nil, h0, add_zero]
  | cons j J ih =>
    have hj : 0 < j.1 := hJ j (by simp)
    have hJ' : ∀ k ∈ J, 0 < k.1 := fun k hk => hJ k (List.mem_cons_of_mem _ hk)
    have hsum : ∀ l : List (ℕ × ℕ), (l.map fun p => p.1 • floorSum (j :: J) p.2).sum
        = (l.map fun p => p.1 * (p.2 / j.1)).sum • j.2
          + (l.map fun p => p.1 • floorSum J p.2).sum := by
      intro l
      induction l with
      | nil => simp
      | cons p l ihl =>
        simp only [List.map_cons, List.sum_cons]
        rw [ihl, floorSum_cons]
        simp only [smul_add, add_smul, mul_smul]
        abel
    rw [floorSum_cons, defect_cons, ih hJ', dot_div hj, add_smul, hsum]
    abel

lemma carryN_eq_zero {P : ℕ} {cs : List (ℕ × ℕ)}
    (h : (cs.map fun p => p.1 * (p.2 % P)).sum < P) : carryN P cs = 0 := by
  unfold carryN
  exact Nat.div_eq_of_lt h

/-- If every scale with a non-zero jump sees no carry, the defect vanishes. -/
lemma defect_eq_zero (J : List (ℕ × D)) (cs : List (ℕ × ℕ))
    (h : ∀ j ∈ J, j.2 = 0 ∨ carryN j.1 cs = 0) : defect J cs = 0 := by
  induction J with
  | nil => simp
  | cons j J ih =>
    have hJ' : ∀ k ∈ J, k.2 = 0 ∨ carryN k.1 cs = 0 := fun k hk => h k (List.mem_cons_of_mem _ hk)
    rw [defect_cons, ih hJ']
    rcases h j (by simp) with h0 | h0
    · rw [h0, smul_zero, add_zero]
    · rw [h0, zero_smul, add_zero]

end FloorSum

/-! ### Layout functions -/

section Eval

variable {D : Type*} [AddCommMonoid D]

/-- The extended layout function: mixed-radix digits of `x` times the strides; the last mode is
unbounded (CuTe's convention outside the domain). -/
def eval : Layout D → ℕ → D
  | [], _ => 0
  | [(_, d)], x => x • d
  | (s, d) :: m :: L, x => (x % s) • d + eval (m :: L) (x / s)

@[simp] lemma eval_nil (x : ℕ) : eval ([] : Layout D) x = 0 := by simp [eval]
@[simp] lemma eval_singleton (s : ℕ) (d : D) (x : ℕ) : eval [(s, d)] x = x • d := by simp [eval]
lemma eval_cons_cons (s : ℕ) (d : D) (m : ℕ × D) (L : Layout D) (x : ℕ) :
    eval ((s, d) :: m :: L) x = (x % s) • d + eval (m :: L) (x / s) := by simp [eval]

/-- Every mode has positive size. -/
def Pos (L : Layout D) : Prop := ∀ m ∈ L, 0 < m.1

/-- (digit, stride) pairs of the index `x` in the radix of an `ℕ`-valued layout. -/
def digitPairs : Layout ℕ → ℕ → List (ℕ × ℕ)
  | [], _ => []
  | [(_, r)], x => [(x, r)]
  | (t, r) :: m :: B, x => (x % t, r) :: digitPairs (m :: B) (x / t)

lemma eval_eq_dot : ∀ (B : Layout ℕ) (x : ℕ), eval B x = dot (digitPairs B x)
  | [], _ => by simp [digitPairs, dot]
  | [(_, r)], x => by simp [digitPairs, dot]
  | (t, r) :: m :: B, x => by
      rw [eval_cons_cons, eval_eq_dot (m :: B) (x / t)]
      simp [digitPairs, dot]

/-- The naive composite of `A` and `B`: the shape of `B` with strides `A(r_j)`. -/
def naive (A : Layout D) (B : Layout ℕ) : Layout D := B.map fun m => (m.1, eval A m.2)

lemma eval_naive (A : Layout D) : ∀ (B : Layout ℕ) (x : ℕ),
    eval (naive A B) x = ((digitPairs B x).map fun p => p.1 • eval A p.2).sum
  | [], _ => by simp [naive, digitPairs]
  | [(_, r)], x => by simp [naive, digitPairs]
  | (t, r) :: m :: B, x => by
      have ih := eval_naive A (m :: B) (x / t)
      simp only [naive, List.map_cons] at ih ⊢
      rw [eval_cons_cons, ih]
      simp [digitPairs]

end Eval

section Group

variable {D : Type*} [AddCommGroup D]

/-- The jumps of the modes of `L` that follow a mode with prefix product `P` and "linear
continuation" `c = s • d`: pairs (scale `P_i`, jump `d_i - s_{i-1} • d_{i-1}`). -/
def jumpList : ℕ → D → Layout D → List (ℕ × D)
  | _, _, [] => []
  | P, c, (s, d) :: L => (P, d - c) :: jumpList (P * s) (s • d) L

lemma floorSum_jumpList_div (s : ℕ) : ∀ (L : Layout D) (P : ℕ) (c : D) (x : ℕ),
    floorSum (jumpList P c L) (x / s) = floorSum (jumpList (s * P) c L) x
  | [], _, _, _ => by simp [jumpList]
  | (t, d) :: L, P, c, x => by
      simp only [jumpList, floorSum_cons]
      rw [floorSum_jumpList_div s L (P * t) (t • d) x, Nat.div_div_eq_div_mul, mul_assoc]

/-- **Floor-sum form** (Proposition 3.2): `f(x) = x • d₀ + Σ_i ⌊x / P_i⌋ • γ_i`. -/
theorem eval_eq_floorSum : ∀ (L : Layout D) (s : ℕ) (d : D) (x : ℕ),
    eval ((s, d) :: L) x = x • d + floorSum (jumpList s (s • d) L) x
  | [], s, d, x => by simp [jumpList]
  | (t, e) :: L, s, d, x => by
      have hx : x • d = (x % s) • d + (x / s) • (s • d) := by
        have h := Nat.mod_add_div x s
        calc x • d = (x % s + s * (x / s)) • d := by rw [h]
          _ = (x % s) • d + (x / s) • (s • d) := by rw [add_smul, mul_comm, mul_smul]
      rw [eval_cons_cons, eval_eq_floorSum L t e (x / s), floorSum_jumpList_div]
      simp only [jumpList, floorSum_cons]
      rw [hx, smul_sub]
      abel

lemma jumpList_pos : ∀ (L : Layout D) (P : ℕ) (c : D), 0 < P → Pos L →
    ∀ j ∈ jumpList P c L, 0 < j.1
  | [], _, _, _, _ => by simp [jumpList]
  | (s, d) :: L, P, c, hP, hL => by
      have hs : 0 < s := hL (s, d) (by simp)
      have hL' : Pos L := fun m hm => hL m (List.mem_cons_of_mem _ hm)
      intro j hj
      simp only [jumpList, List.mem_cons] at hj
      rcases hj with rfl | hj
      · exact hP
      · exact jumpList_pos L (P * s) (s • d) (Nat.mul_pos hP hs) hL' j hj

/-- **Carry identity** (Theorem 3.4): a layout function is additive up to carries, and each carry
is charged the jump at its scale. -/
theorem eval_add (L : Layout D) (s : ℕ) (d : D) (hs : 0 < s) (hL : Pos L) (x y : ℕ) :
    eval ((s, d) :: L) (x + y)
      = eval ((s, d) :: L) x + eval ((s, d) :: L) y + carrySum (jumpList s (s • d) L) x y := by
  rw [eval_eq_floorSum L s d (x + y), eval_eq_floorSum L s d x, eval_eq_floorSum L s d y,
    floorSum_add _ (jumpList_pos L s (s • d) hs hL), add_smul]
  abel

/-- **Composition formula** (Theorem 5.3), coordinate form: evaluating `A` at `Σ c_j r_j` gives
`Σ c_j • A(r_j)` plus the carry defect. -/
theorem eval_dot (L : Layout D) (s : ℕ) (d : D) (hs : 0 < s) (hL : Pos L) (cs : List (ℕ × ℕ)) :
    eval ((s, d) :: L) (dot cs)
      = (cs.map fun p => p.1 • eval ((s, d) :: L) p.2).sum + defect (jumpList s (s • d) L) cs := by
  have hsum : (cs.map fun p => p.1 • eval ((s, d) :: L) p.2).sum
      = dot cs • d + (cs.map fun p => p.1 • floorSum (jumpList s (s • d) L) p.2).sum := by
    induction cs with
    | nil => simp [dot]
    | cons p cs ih =>
      simp only [List.map_cons, List.sum_cons]
      rw [ih, eval_eq_floorSum]
      simp only [dot, List.map_cons, List.sum_cons, smul_add, add_smul, mul_smul]
      abel
  rw [eval_eq_floorSum, floorSum_dot _ (jumpList_pos L s (s • d) hs hL), hsum]
  abel

/-- **Composition formula** (Theorem 5.3): `A(B(x))` is the value of the naive composite plus the
carry defect of the digits of `x`. -/
theorem eval_comp (L : Layout D) (s : ℕ) (d : D) (hs : 0 < s) (hL : Pos L) (B : Layout ℕ) (x : ℕ) :
    eval ((s, d) :: L) (eval B x)
      = eval (naive ((s, d) :: L) B) x + defect (jumpList s (s • d) L) (digitPairs B x) := by
  rw [eval_eq_dot B x, eval_dot L s d hs hL, eval_naive]

/-- **Corollary 5.4**: at an index `x`, the naive composite is correct if and only if the carry
defect vanishes there. -/
theorem eval_comp_eq_naive_iff (L : Layout D) (s : ℕ) (d : D) (hs : 0 < s) (hL : Pos L)
    (B : Layout ℕ) (x : ℕ) :
    eval ((s, d) :: L) (eval B x) = eval (naive ((s, d) :: L) B) x
      ↔ defect (jumpList s (s • d) L) (digitPairs B x) = 0 := by
  rw [eval_comp L s d hs hL B x]
  constructor
  · intro h
    simpa using h
  · intro h
    rw [h, add_zero]

/-- **Carry-free composition** (Theorem 5.7(a)): if no scale of `A` with a non-zero jump sees a
carry, then `A ∘ B` is the naive composite at `x`. -/
theorem eval_comp_of_carryFree (L : Layout D) (s : ℕ) (d : D) (hs : 0 < s) (hL : Pos L)
    (B : Layout ℕ) (x : ℕ)
    (h : ∀ j ∈ jumpList s (s • d) L, j.2 = 0 ∨ carryN j.1 (digitPairs B x) = 0) :
    eval ((s, d) :: L) (eval B x) = eval (naive ((s, d) :: L) B) x := by
  rw [eval_comp L s d hs hL, defect_eq_zero _ _ h, add_zero]

/-- The coalescing step: merging two modes whose jump vanishes does not change the function. -/
theorem eval_merge (L : Layout D) (s t : ℕ) (d : D) (x : ℕ) :
    eval ((s, d) :: (t, s • d) :: L) x = eval ((s * t, d) :: L) x := by
  have h : t • (s • d) = (s * t) • d := by rw [mul_comm, mul_smul]
  rw [eval_eq_floorSum, eval_eq_floorSum]
  simp only [jumpList, floorSum_cons, sub_self, smul_zero, zero_add, h]

/-! ### The carry-free inequality -/

/-- The size of a layout: the product of its mode sizes. -/
def size (B : Layout ℕ) : ℕ := (B.map Prod.fst).prod

/-- For an index inside the domain, every digit is smaller than the size of its mode. -/
lemma digitPairs_bounded : ∀ (B : Layout ℕ) (x : ℕ), B ≠ [] → x < size B →
    List.Forall₂ (fun p m => p.1 < m.1 ∧ p.2 = m.2) (digitPairs B x) B
  | [], _, h, _ => absurd rfl h
  | [(t, r)], x, _, hx => by
      simp only [size, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one] at hx
      simp only [digitPairs]
      exact List.Forall₂.cons ⟨hx, rfl⟩ List.Forall₂.nil
  | (t, r) :: m :: B, x, _, hx => by
      have hx2 : x < t * size (m :: B) := by
        simpa only [size, List.map_cons, List.prod_cons] using hx
      have ht : 0 < t := by
        rcases Nat.eq_zero_or_pos t with h | h
        · rw [h, Nat.zero_mul] at hx2
          exact absurd hx2 (Nat.not_lt_zero _)
        · exact h
      have hx' : x / t < size (m :: B) := Nat.div_lt_of_lt_mul hx2
      simp only [digitPairs]
      exact List.Forall₂.cons ⟨Nat.mod_lt _ ht, rfl⟩
        (digitPairs_bounded (m :: B) (x / t) (List.cons_ne_nil _ _) hx')

lemma residues_le {P : ℕ} {cs : List (ℕ × ℕ)} {B : Layout ℕ}
    (h : List.Forall₂ (fun p m => p.1 < m.1 ∧ p.2 = m.2) cs B) :
    (cs.map fun p => p.1 * (p.2 % P)).sum ≤ (B.map fun m => (m.1 - 1) * (m.2 % P)).sum := by
  induction h with
  | nil => simp
  | cons hpm _ ih =>
    obtain ⟨h1, h2⟩ := hpm
    simp only [List.map_cons, List.sum_cons]
    rw [h2]
    exact Nat.add_le_add (Nat.mul_le_mul_right _ (Nat.le_sub_one_of_lt h1)) ih

/-- **Carry-free composition** (Theorem 5.7(a)) with the explicit inequality: if
`Σ_j (t_j - 1) * (r_j mod Q) < Q` for every scale `Q` of `A` carrying a non-zero jump, then on the
whole domain of `B` the composite `A ∘ B` is the naive composite. -/
theorem eval_comp_of_carryFree_ineq (L : Layout D) (s : ℕ) (d : D) (hs : 0 < s) (hL : Pos L)
    (B : Layout ℕ) (hB : B ≠ []) (x : ℕ) (hx : x < size B)
    (h : ∀ j ∈ jumpList s (s • d) L,
      j.2 = 0 ∨ (B.map fun m => (m.1 - 1) * (m.2 % j.1)).sum < j.1) :
    eval ((s, d) :: L) (eval B x) = eval (naive ((s, d) :: L) B) x := by
  apply eval_comp_of_carryFree L s d hs hL B x
  intro j hj
  rcases h j hj with h0 | h0
  · exact Or.inl h0
  · exact Or.inr
      (carryN_eq_zero (lt_of_le_of_lt (residues_le (digitPairs_bounded B x hB hx)) h0))

end Group

end CuTe
