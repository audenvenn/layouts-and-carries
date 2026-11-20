/-
  Concrete instances quoted in the paper, checked by evaluation.
-/
import CuteLayout.Basic

namespace CuTe

/-- Shah's example: `A = (6,2):(1,7)`, `B = (3,2):(2,3)`. -/
def A₁ : Layout ℤ := [(6, 1), (2, 7)]
def B₁ : Layout ℕ := [(3, 2), (2, 3)]

/-- At index 5 (coordinate `(2,1)`) the true composite is 8, the naive composite gives 7. -/
example : eval A₁ (eval B₁ 5) = 8 := by decide
example : eval (naive A₁ B₁) 5 = 7 := by decide

/-- Three modes: `A = (12,2):(1,13)`, `B = (2,2,2):(3,4,6)`; no two strides carry, all three do. -/
def A₂ : Layout ℤ := [(12, 1), (2, 13)]
def B₂ : Layout ℕ := [(2, 3), (2, 4), (2, 6)]

example : eval A₂ (eval B₂ 7) = 14 := by decide
example : eval (naive A₂ B₂) 7 = 13 := by decide

/-- A row-major 4×3 matrix viewed through the column-major shape (3,4): index 4 is offset 1,
while the naive composite `(3,4):(3,9)` gives 12. -/
def A₃ : Layout ℤ := [(4, 3), (3, 1)]
def B₃ : Layout ℕ := [(3, 1), (4, 3)]

example : eval A₃ (eval B₃ 4) = 1 := by decide

/-- The documentation example `A = (6,2):(8,2)`, refined `B' = (2,2,3):(3,6,1)`: the naive
composite `(2,2,3):(24,2,8)` agrees with `A ∘ B'` on all twelve indices. -/
def A₄ : Layout ℤ := [(6, 8), (2, 2)]
def B₄ : Layout ℕ := [(2, 3), (2, 6), (3, 1)]

example : ∀ x ∈ List.range 12, eval A₄ (eval B₄ x) = eval (naive A₄ B₄) x := by decide

end CuTe
