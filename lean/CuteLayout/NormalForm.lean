/-
  Uniqueness of the reduced floor-sum representation (the uniqueness half of Lemma 4.1 /
  Theorem 4.3 of the paper): a function on `{0, …, N-1}` determines the list of its scales and
  non-zero jumps.  A slope is a jump at scale 1, since `⌊x / 1⌋ • a = x • a`.
-/
import CuteLayout.Basic

namespace CuTe

variable {D : Type*} [AddCommGroup D]

/-- A floor sum vanishes at `x` if all its scales exceed `x`. -/
lemma floorSum_eq_zero_of_lt (J : List (ℕ × D)) (x : ℕ) (h : ∀ j ∈ J, x < j.1) :
    floorSum J x = 0 := by
  induction J with
  | nil => simp
  | cons j J ih =>
    rw [floorSum_cons, ih (fun k hk => h k (List.mem_cons_of_mem _ hk)),
      Nat.div_eq_of_lt (h j (by simp)), zero_smul, add_zero]

/-- The value of a floor sum at its smallest scale is the jump at that scale. -/
lemma floorSum_head (Q : ℕ) (γ : D) (J : List (ℕ × D)) (hQ : 0 < Q) (h : ∀ j ∈ J, Q < j.1) :
    floorSum ((Q, γ) :: J) Q = γ := by
  rw [floorSum_cons, floorSum_eq_zero_of_lt J Q h]
  show (Q / Q) • γ + 0 = γ
  rw [Nat.div_self hQ, one_smul, add_zero]

/-- **Uniqueness of the reduced floor-sum form.**  Two lists of (scale, jump) pairs with strictly
increasing positive scales below `N` and non-zero jumps that define the same function on
`{0, …, N-1}` are equal. -/
theorem floorSum_injective (N : ℕ) : ∀ (J J' : List (ℕ × D)),
    J.Pairwise (fun a b => a.1 < b.1) → J'.Pairwise (fun a b => a.1 < b.1) →
    (∀ j ∈ J, 0 < j.1 ∧ j.1 < N ∧ j.2 ≠ 0) → (∀ j ∈ J', 0 < j.1 ∧ j.1 < N ∧ j.2 ≠ 0) →
    (∀ x, x < N → floorSum J x = floorSum J' x) → J = J'
  | [], [], _, _, _, _, _ => rfl
  | [], (Q', γ') :: J', _, hp', _, hm', h => by
      exfalso
      obtain ⟨hpos, hlt, hne⟩ := hm' (Q', γ') (by simp)
      have hgt : ∀ j ∈ J', Q' < j.1 := (List.pairwise_cons.mp hp').1
      have h1 := h Q' hlt
      rw [floorSum_nil, floorSum_head Q' γ' J' hpos hgt] at h1
      exact hne h1.symm
  | (Q, γ) :: J, [], hp, _, hm, _, h => by
      exfalso
      obtain ⟨hpos, hlt, hne⟩ := hm (Q, γ) (by simp)
      have hgt : ∀ j ∈ J, Q < j.1 := (List.pairwise_cons.mp hp).1
      have h1 := h Q hlt
      rw [floorSum_nil, floorSum_head Q γ J hpos hgt] at h1
      exact hne h1
  | (Q, γ) :: J, (Q', γ') :: J', hp, hp', hm, hm', h => by
      obtain ⟨hpos, hlt, hne⟩ := hm (Q, γ) (by simp)
      obtain ⟨hpos', hlt', hne'⟩ := hm' (Q', γ') (by simp)
      have hgt : ∀ j ∈ J, Q < j.1 := (List.pairwise_cons.mp hp).1
      have hgt' : ∀ j ∈ J', Q' < j.1 := (List.pairwise_cons.mp hp').1
      rcases Nat.lt_trichotomy Q Q' with hlt2 | heq | hgt2
      · exfalso
        have hall : ∀ j ∈ (Q', γ') :: J', Q < j.1 := by
          intro j hj
          rcases List.mem_cons.mp hj with rfl | hj
          · exact hlt2
          · exact lt_trans hlt2 (hgt' j hj)
        have h1 := h Q hlt
        rw [floorSum_head Q γ J hpos hgt, floorSum_eq_zero_of_lt _ Q hall] at h1
        exact hne h1
      · subst heq
        have h1 := h Q hlt
        rw [floorSum_head Q γ J hpos hgt, floorSum_head Q γ' J' hpos hgt'] at h1
        subst h1
        have htail : ∀ x, x < N → floorSum J x = floorSum J' x := by
          intro x hx
          have h2 := h x hx
          rw [floorSum_cons, floorSum_cons] at h2
          exact add_left_cancel h2
        rw [floorSum_injective N J J' (List.pairwise_cons.mp hp).2 (List.pairwise_cons.mp hp').2
          (fun j hj => hm j (List.mem_cons_of_mem _ hj))
          (fun j hj => hm' j (List.mem_cons_of_mem _ hj)) htail]
      · exfalso
        have hall : ∀ j ∈ (Q, γ) :: J, Q' < j.1 := by
          intro j hj
          rcases List.mem_cons.mp hj with rfl | hj
          · exact hgt2
          · exact lt_trans hgt2 (hgt j hj)
        have h1 := h Q' hlt'
        rw [floorSum_head Q' γ' J' hpos' hgt', floorSum_eq_zero_of_lt _ Q' hall] at h1
        exact hne' h1.symm

end CuTe
