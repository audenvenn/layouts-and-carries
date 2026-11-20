/-
  The carry-free criterion is exact for independent jumps (Theorem 5.7(b) of the paper).

  `carryFree_of_naive_correct`: if the jumps of `A` are independent and the naive composite of
  `A` and `B` is correct on the whole domain of `B`, then `B` is carry-free at every scale of `A`
  with a non-zero jump.  `naive_correct_iff_carryFree` combines this with Theorem 5.7(a).
-/
import CuteLayout.Definable

namespace CuTe

/-- The digits of the last index of a layout are all maximal. -/
lemma digitPairs_last : ∀ (B : Layout ℕ), B ≠ [] → (∀ n ∈ B, 0 < n.1) →
    digitPairs B (size B - 1) = B.map fun m => (m.1 - 1, m.2)
  | [], h, _ => absurd rfl h
  | [(t, r)], _, _ => by
      simp [digitPairs, size]
  | (t, r) :: m :: B, _, hpos => by
      have ht : 0 < t := hpos (t, r) (by simp)
      have hpos' : ∀ n ∈ m :: B, 0 < n.1 := fun n hn => hpos n (List.mem_cons_of_mem _ hn)
      have hN : 0 < size (m :: B) := size_pos_of_pos (m :: B) hpos'
      have hsize : size ((t, r) :: m :: B) = t * size (m :: B) := by simp [size]
      obtain ⟨k, hk⟩ : ∃ k, size (m :: B) = k + 1 := ⟨size (m :: B) - 1, by omega⟩
      have hx : size ((t, r) :: m :: B) - 1 = t * k + (t - 1) := by
        rw [hsize, hk, Nat.mul_add, Nat.mul_one]
        omega
      have hmod : (t * k + (t - 1)) % t = t - 1 := by
        rw [Nat.mul_add_mod]
        exact Nat.mod_eq_of_lt (by omega)
      have hdiv : (t * k + (t - 1)) / t = k := by
        rw [Nat.mul_add_div ht, Nat.div_eq_of_lt (by omega), Nat.add_zero]
      have ih := digitPairs_last (m :: B) (List.cons_ne_nil _ _) hpos'
      have hk' : size (m :: B) - 1 = k := by omega
      rw [hk'] at ih
      rw [hx]
      simp only [digitPairs, hmod, hdiv, ih, List.map_cons]

variable {D : Type*} [AddCommGroup D]

/-- **Theorem 5.7(b)**.  Suppose the jumps of `A = (s, d) :: L` are independent: a combination
`Σ ν(P_i) • γ_i` with natural coefficients vanishes only if `ν(P_i) = 0` wherever `γ_i ≠ 0`.
If the naive composite of `A` and `B` is correct on the whole domain of `B`, then `B` is
carry-free at every scale of `A` with a non-zero jump. -/
theorem carryFree_of_naive_correct (L : Layout D) (s : ℕ) (d : D) (hs : 0 < s) (hL : Pos L)
    (B : Layout ℕ) (hB : B ≠ []) (hBpos : ∀ n ∈ B, 0 < n.1)
    (hind : ∀ ν : ℕ → ℕ, ((jumpList s (s • d) L).map fun j => ν j.1 • j.2).sum = 0 →
      ∀ j ∈ jumpList s (s • d) L, j.2 ≠ 0 → ν j.1 = 0)
    (hcorrect : ∀ x, x < size B →
      eval ((s, d) :: L) (eval B x) = eval (naive ((s, d) :: L) B) x) :
    ∀ j ∈ jumpList s (s • d) L, j.2 ≠ 0 →
      (B.map fun m => (m.1 - 1) * (m.2 % j.1)).sum < j.1 := by
  intro j hj hj2
  have hN : 0 < size B := size_pos_of_pos B hBpos
  have hdef : defect (jumpList s (s • d) L) (digitPairs B (size B - 1)) = 0 :=
    (eval_comp_eq_naive_iff L s d hs hL B (size B - 1)).mp (hcorrect (size B - 1) (by omega))
  rw [digitPairs_last B hB hBpos] at hdef
  have h0 : carryN j.1 (B.map fun m : ℕ × ℕ => (m.1 - 1, m.2)) = 0 :=
    hind (fun P => carryN P (B.map fun m : ℕ × ℕ => (m.1 - 1, m.2))) hdef j hj hj2
  have hjpos : 0 < j.1 := jumpList_pos L s (s • d) hs hL j hj
  have hsum : ((B.map fun m => (m.1 - 1, m.2)).map fun p => p.1 * (p.2 % j.1)).sum
      = (B.map fun m => (m.1 - 1) * (m.2 % j.1)).sum := by
    rw [List.map_map]
    rfl
  have h0' : (B.map fun m => (m.1 - 1) * (m.2 % j.1)).sum / j.1 = 0 := by
    rw [← hsum]
    exact h0
  by_contra hlt
  have hpos := Nat.div_pos (not_lt.mp hlt) hjpos
  omega

/-- **Theorem 5.7** for independent jumps: the naive composite is correct on the whole domain if
and only if `B` is carry-free at every scale of `A` with a non-zero jump. -/
theorem naive_correct_iff_carryFree (L : Layout D) (s : ℕ) (d : D) (hs : 0 < s) (hL : Pos L)
    (B : Layout ℕ) (hB : B ≠ []) (hBpos : ∀ n ∈ B, 0 < n.1)
    (hind : ∀ ν : ℕ → ℕ, ((jumpList s (s • d) L).map fun j => ν j.1 • j.2).sum = 0 →
      ∀ j ∈ jumpList s (s • d) L, j.2 ≠ 0 → ν j.1 = 0) :
    (∀ x, x < size B → eval ((s, d) :: L) (eval B x) = eval (naive ((s, d) :: L) B) x)
      ↔ ∀ j ∈ jumpList s (s • d) L, j.2 ≠ 0 →
          (B.map fun m => (m.1 - 1) * (m.2 % j.1)).sum < j.1 := by
  constructor
  · exact carryFree_of_naive_correct L s d hs hL B hB hBpos hind
  · intro h x hx
    apply eval_comp_of_carryFree_ineq L s d hs hL B hB x hx
    intro j hj
    by_cases hj2 : j.2 = 0
    · exact Or.inl hj2
    · exact Or.inr (h j hj hj2)

end CuTe
