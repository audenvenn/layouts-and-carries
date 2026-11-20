/-
  Swizzles (Section 8 of the paper).  CuTe's `Swizzle<B, M, S>` with `S > 0` is the map
  `x ↦ x ^^^ ((x &&& Y) >>> S)` ("ZZZ ^= YYY") for a mask `Y`; it is an involution as soon as the
  field `Y` does not meet its own image `Y >>> S` (Lemma 8.2(a)).
-/
import CuteLayout.Basic

namespace CuTe

/-- A swizzle with mask `Y` and (positive) shift `S`: bit `i` is replaced by
`bit i XOR (bit (S + i) AND Y (S + i))`. -/
def swizzle (Y S x : ℕ) : ℕ := x ^^^ ((x &&& Y) >>> S)

lemma testBit_swizzle (Y S x i : ℕ) :
    (swizzle Y S x).testBit i = (x.testBit i ^^ (x.testBit (S + i) && Y.testBit (S + i))) := by
  simp [swizzle, Nat.testBit_xor, Nat.testBit_and, Nat.testBit_shiftRight]

/-- **Lemma 8.2(a)**: a swizzle is an involution, provided no bit of the mask is the image of
another bit of the mask under the shift. -/
theorem swizzle_involutive (Y S : ℕ) (h : ∀ i, Y.testBit i = true → Y.testBit (S + i) = false)
    (x : ℕ) : swizzle Y S (swizzle Y S x) = x := by
  apply Nat.eq_of_testBit_eq
  intro i
  rw [testBit_swizzle, testBit_swizzle, testBit_swizzle]
  cases hY : Y.testBit (S + i) with
  | false => simp
  | true =>
    rw [h (S + i) hY]
    cases x.testBit i <;> cases x.testBit (S + i) <;> simp

/-- The swizzle is linear for XOR (Lemma 8.2(b)). -/
theorem swizzle_xor (Y S x y : ℕ) : swizzle Y S (x ^^^ y) = swizzle Y S x ^^^ swizzle Y S y := by
  apply Nat.eq_of_testBit_eq
  intro i
  simp only [testBit_swizzle, Nat.testBit_xor]
  cases x.testBit i <;> cases y.testBit i <;> cases x.testBit (S + i) <;> cases y.testBit (S + i) <;>
    cases Y.testBit (S + i) <;> rfl

/-- The 128-byte swizzle `Swizzle<3,4,3>` (mask `0b1110000000`, shift 3) satisfies the hypothesis. -/
example (x : ℕ) : swizzle 0b1110000000 3 (swizzle 0b1110000000 3 x) = x := by
  apply swizzle_involutive
  intro i hi
  have hlt : i < 10 := by
    by_contra hge
    have h0 : (0b1110000000 : ℕ).testBit i = false :=
      Nat.testBit_lt_two_pow
        (lt_of_lt_of_le (by norm_num) (Nat.pow_le_pow_right (by norm_num) (not_lt.mp hge)))
    rw [h0] at hi
    exact Bool.noConfusion hi
  have key : ∀ j, j < 10 → (0b1110000000 : ℕ).testBit j = true →
      (0b1110000000 : ℕ).testBit (3 + j) = false := by decide
  exact key i hlt hi

end CuTe
