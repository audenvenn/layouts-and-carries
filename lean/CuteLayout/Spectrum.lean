/-
  Which functions are layouts?  (Section 4 of the paper.)

  * `fexp_coeff`, `fexp_unique` – Lemma 4.1: every function with `g 0 = 0` has a floor expansion
    `g x = Σ_{1 ≤ q ≤ x} ⌊x / q⌋ • c q`, and the values on `{0, …, N-1}` determine the
    coefficients at the scales `1 ≤ q < N`.
  * `layout_over_shape_iff`, `additive_iff_spec` – Theorem 4.3: `g` is a layout over the shape
    `S` (equivalently, `S`-additive) if and only if its floor spectrum (the scales `q ≥ 2` with
    `c q ≠ 0`) lies in the scales of `S`.
  * `layout_iff_chain` – Theorem 4.3(a): `g` is the function of some layout of size `N` if and
    only if its floor spectrum is a chain under divisibility whose members divide `N`.
-/
import CuteLayout.Basic
import CuteLayout.Additive

namespace CuTe

variable {D : Type*} [AddCommGroup D]

/-! ### Sums over `List.range` -/

lemma sum_range_succ' (f : ℕ → D) (n : ℕ) :
    ((List.range (n + 1)).map f).sum = ((List.range n).map f).sum + f n := by
  simp [List.range_succ]

lemma sum_range_congr (f f' : ℕ → D) (n : ℕ) (h : ∀ p, p < n → f p = f' p) :
    ((List.range n).map f).sum = ((List.range n).map f').sum := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [sum_range_succ', sum_range_succ', ih (fun p hp => h p (by omega)), h n (by omega)]

lemma sum_range_add (f f' : ℕ → D) (n : ℕ) :
    ((List.range n).map fun q => f q + f' q).sum
      = ((List.range n).map f).sum + ((List.range n).map f').sum := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [sum_range_succ', sum_range_succ', sum_range_succ', ih]
    abel

lemma sum_range_zero (n : ℕ) : ((List.range n).map fun _ => (0 : D)).sum = 0 := by
  induction n with
  | zero => simp
  | succ n ih => rw [sum_range_succ', ih, add_zero]

/-! ### Floor expansions (Lemma 4.1) -/

/-- The floor expansion with coefficients `c`: `x ↦ Σ_{q ≤ x} ⌊x / q⌋ • c q`.  The term `q = 0`
vanishes because `x / 0 = 0`. -/
def fexp (c : ℕ → D) (x : ℕ) : D := ((List.range (x + 1)).map fun q => (x / q) • c q).sum

/-- Auxiliary function: `coeffAux g n q` is the `q`-th floor-expansion coefficient of `g` for
`q < n`. -/
def coeffAux (g : ℕ → D) : ℕ → ℕ → D
  | 0, _ => 0
  | n + 1, q =>
      if q = n then g n - ((List.range n).map fun p => (n / p) • coeffAux g n p).sum
      else coeffAux g n q

/-- The floor-expansion coefficients of `g`: `c q = g q - Σ_{p < q} ⌊q / p⌋ • c p`. -/
def coeff (g : ℕ → D) (q : ℕ) : D := coeffAux g (q + 1) q

lemma coeffAux_of_lt (g : ℕ → D) : ∀ (n q : ℕ), q < n → coeffAux g n q = coeff g q
  | 0, _, h => absurd h (Nat.not_lt_zero _)
  | n + 1, q, h => by
      by_cases hq : q = n
      · subst hq
        rfl
      · have h' : q < n := by omega
        rw [← coeffAux_of_lt g n q h']
        simp [coeffAux, hq]

theorem coeff_eq (g : ℕ → D) (q : ℕ) :
    coeff g q = g q - ((List.range q).map fun p => (q / p) • coeff g p).sum := by
  have h : coeff g q = g q - ((List.range q).map fun p => (q / p) • coeffAux g q p).sum := by
    simp [coeff, coeffAux]
  rw [h, sum_range_congr (fun p => (q / p) • coeffAux g q p) (fun p => (q / p) • coeff g p) q
    (fun p hp => by simp only [coeffAux_of_lt g q p hp])]

/-- **Existence of the floor expansion** (Lemma 4.1). -/
theorem fexp_coeff (g : ℕ → D) (hg : g 0 = 0) : ∀ x, fexp (coeff g) x = g x
  | 0 => by
      unfold fexp
      rw [sum_range_succ']
      simp [hg]
  | n + 1 => by
      unfold fexp
      rw [sum_range_succ', Nat.div_self (by omega : 0 < n + 1), one_smul, coeff_eq g (n + 1)]
      abel

/-- **Uniqueness of the floor expansion** (Lemma 4.1): the values on `{0, …, N-1}` determine the
coefficients at the scales `1 ≤ q < N`. -/
theorem fexp_unique (N : ℕ) (c c' : ℕ → D) (h : ∀ x, x < N → fexp c x = fexp c' x) :
    ∀ q, 1 ≤ q → q < N → c q = c' q := by
  intro q
  induction q using Nat.strong_induction_on with
  | _ q ih =>
    intro hq1 hqN
    have h1 := h q hqN
    unfold fexp at h1
    rw [sum_range_succ', sum_range_succ'] at h1
    have h2 : ((List.range q).map fun p => (q / p) • c p).sum
        = ((List.range q).map fun p => (q / p) • c' p).sum := by
      apply sum_range_congr
      intro p hp
      rcases Nat.eq_zero_or_pos p with h0 | h0
      · subst h0
        simp
      · show (q / p) • c p = (q / p) • c' p
        rw [ih p hp h0 (lt_trans hp hqN)]
    rw [h2, Nat.div_self (by omega : 0 < q), one_smul, one_smul] at h1
    exact add_left_cancel h1

lemma fexp_congr (c c' : ℕ → D) (x : ℕ) (h : ∀ q, 1 ≤ q → q ≤ x → c q = c' q) :
    fexp c x = fexp c' x := by
  unfold fexp
  apply sum_range_congr
  intro q hq
  rcases Nat.eq_zero_or_pos q with h0 | h0
  · subst h0
    simp
  · show (x / q) • c q = (x / q) • c' q
    rw [h q h0 (by omega)]

/-! ### Floor sums as floor expansions -/

/-- The total jump at scale `q` of a list of (scale, jump) pairs. -/
def total : List (ℕ × D) → ℕ → D
  | [], _ => 0
  | j :: J, q => (if j.1 = q then j.2 else 0) + total J q

lemma sum_single (x Q : ℕ) (γ : D) : ∀ n : ℕ,
    ((List.range n).map fun q => (x / q) • (if Q = q then γ else 0)).sum
      = if Q < n then (x / Q) • γ else 0
  | 0 => by simp
  | n + 1 => by
      rw [sum_range_succ', sum_single x Q γ n]
      by_cases h1 : Q < n
      · have h2 : Q < n + 1 := by omega
        have h3 : Q ≠ n := by omega
        simp [h1, h2, h3]
      · by_cases h2 : Q = n
        · subst h2
          simp
        · have h3 : ¬ Q < n + 1 := by omega
          simp [h1, h2, h3]

lemma fexp_single (x Q : ℕ) (γ : D) :
    ((List.range (x + 1)).map fun q => (x / q) • (if Q = q then γ else 0)).sum = (x / Q) • γ := by
  rw [sum_single x Q γ (x + 1)]
  by_cases h : Q < x + 1
  · rw [ite_eq_left h]
  · rw [ite_eq_right h, Nat.div_eq_of_lt (by omega), zero_smul]

/-- A floor sum is the floor expansion of its totals. -/
lemma floorSum_eq_fexp : ∀ (J : List (ℕ × D)) (x : ℕ), floorSum J x = fexp (total J) x
  | [], x => by
      unfold fexp
      rw [floorSum_nil, sum_range_congr (fun q => (x / q) • total ([] : List (ℕ × D)) q)
        (fun _ => 0) (x + 1) (fun q _ => by simp [total]), sum_range_zero]
  | j :: J, x => by
      rw [floorSum_cons, floorSum_eq_fexp J x]
      unfold fexp
      have h : ∀ q, (x / q) • total (j :: J) q
          = (x / q) • (if j.1 = q then j.2 else 0) + (x / q) • total J q := by
        intro q
        simp only [total, smul_add]
      rw [sum_range_congr (fun q => (x / q) • total (j :: J) q)
          (fun q => (x / q) • (if j.1 = q then j.2 else 0) + (x / q) • total J q) (x + 1)
          (fun q _ => h q),
        sum_range_add (fun q => (x / q) • (if j.1 = q then j.2 else 0))
          (fun q => (x / q) • total J q) (x + 1),
        fexp_single]

lemma total_eq_zero_of_not_mem : ∀ (J : List (ℕ × D)) (q : ℕ), q ∉ J.map Prod.fst →
    total J q = 0
  | [], _, _ => rfl
  | j :: J, q, h => by
      simp only [List.map_cons, List.mem_cons, not_or] at h
      simp only [total]
      rw [ite_eq_right (Ne.symm h.1), zero_add]
      exact total_eq_zero_of_not_mem J q h.2

lemma total_map_nodup (c : ℕ → D) : ∀ (K : List ℕ) (q : ℕ), K.Nodup →
    total (K.map fun p => (p, c p)) q = if q ∈ K then c q else 0
  | [], _, _ => by simp [total]
  | p :: K, q, h => by
      have hp : p ∉ K := (List.nodup_cons.mp h).1
      have ih := total_map_nodup c K q (List.nodup_cons.mp h).2
      simp only [List.map_cons, total]
      rw [ih]
      by_cases hpq : p = q
      · subst hpq
        simp [hp]
      · have hqp : ¬ q = p := fun e => hpq e.symm
        simp [hpq, hqp]

/-! ### Scales of a shape -/

/-- The scales `P, P * s₀, P * s₀ * s₁, …` at which the modes of a shape start (one per mode).
`scalesFrom 1 S` is the list `P₀ = 1, P₁, …, P_{m-1}` of the paper. -/
def scalesFrom : ℕ → List ℕ → List ℕ
  | _, [] => []
  | P, s :: S => P :: scalesFrom (P * s) S

lemma jumpList_fst : ∀ (L : Layout D) (P : ℕ) (c : D),
    (jumpList P c L).map Prod.fst = scalesFrom P (L.map Prod.fst)
  | [], _, _ => rfl
  | (s, d) :: L, P, c => by
      simp only [jumpList, List.map_cons, scalesFrom]
      rw [jumpList_fst L (P * s) (s • d)]

/-- A layout function is the floor sum of its slope (a jump at scale `1`) and its jumps. -/
lemma eval_eq_floorSum_all (L : Layout D) (x : ℕ) : eval L x = floorSum (jumpList 1 0 L) x := by
  cases L with
  | nil => simp [jumpList]
  | cons m L =>
    obtain ⟨s, d⟩ := m
    rw [eval_eq_floorSum L s d x]
    simp [jumpList]

lemma scalesFrom_dvd : ∀ (S : List ℕ) (P q : ℕ), q ∈ scalesFrom P S → P ∣ q ∧ q ∣ P * S.prod
  | [], _, _, h => by simp [scalesFrom] at h
  | s :: S, P, q, h => by
      simp only [scalesFrom, List.mem_cons] at h
      rcases h with h | h
      · rw [h]
        exact ⟨dvd_rfl, Dvd.intro _ rfl⟩
      · obtain ⟨h1, h2⟩ := scalesFrom_dvd S (P * s) q h
        refine ⟨dvd_trans (Dvd.intro _ rfl) h1, ?_⟩
        rw [List.prod_cons, ← mul_assoc]
        exact h2

/-- Any two scales of a shape are comparable under divisibility. -/
lemma scalesFrom_comparable : ∀ (S : List ℕ) (P q q' : ℕ), q ∈ scalesFrom P S →
    q' ∈ scalesFrom P S → q ∣ q' ∨ q' ∣ q
  | [], _, _, _, h, _ => by simp [scalesFrom] at h
  | s :: S, P, q, q', h, h' => by
      simp only [scalesFrom, List.mem_cons] at h h'
      rcases h with h | h
      · rcases h' with h' | h'
        · rw [h, h']
          exact Or.inl dvd_rfl
        · rw [h]
          exact Or.inl (dvd_trans (Dvd.intro _ rfl) (scalesFrom_dvd S (P * s) q' h').1)
      · rcases h' with h' | h'
        · rw [h']
          exact Or.inr (dvd_trans (Dvd.intro _ rfl) (scalesFrom_dvd S (P * s) q h).1)
        · exact scalesFrom_comparable S (P * s) q q' h h'

/-- A shape without unit modes has pairwise distinct scales. -/
lemma scalesFrom_nodup : ∀ (S : List ℕ) (P : ℕ), 0 < P → (∀ s ∈ S, 2 ≤ s) →
    (scalesFrom P S).Nodup
  | [], _, _, _ => List.nodup_nil
  | s :: S, P, hP, h => by
      have hs : 2 ≤ s := h s (by simp)
      have hS : ∀ t ∈ S, 2 ≤ t := fun t ht => h t (List.mem_cons_of_mem _ ht)
      have hPs : 0 < P * s := Nat.mul_pos hP (by omega)
      simp only [scalesFrom]
      refine List.nodup_cons.mpr ⟨?_, scalesFrom_nodup S (P * s) hPs hS⟩
      intro hmem
      have h1 : P * s ∣ P := (scalesFrom_dvd S (P * s) P hmem).1
      have h2 : P * s ≤ P := Nat.le_of_dvd hP h1
      have h3 : P * 2 ≤ P * s := Nat.mul_le_mul_left P hs
      omega

/-! ### The layout with prescribed jumps -/

/-- The layout over the shape `S` whose jump at each scale `P` is `c P` (`e` is the linear
continuation of the preceding mode). -/
def ofCoeff (c : ℕ → D) : ℕ → D → List ℕ → Layout D
  | _, _, [] => []
  | P, e, s :: S => (s, c P + e) :: ofCoeff c (P * s) (s • (c P + e)) S

lemma ofCoeff_fst (c : ℕ → D) : ∀ (S : List ℕ) (P : ℕ) (e : D),
    (ofCoeff c P e S).map Prod.fst = S
  | [], _, _ => rfl
  | s :: S, P, e => by
      simp only [ofCoeff, List.map_cons]
      rw [ofCoeff_fst c S]

lemma jumpList_ofCoeff (c : ℕ → D) : ∀ (S : List ℕ) (P : ℕ) (e : D),
    jumpList P e (ofCoeff c P e S) = (scalesFrom P S).map fun q => (q, c q)
  | [], _, _ => rfl
  | s :: S, P, e => by
      simp only [ofCoeff, jumpList, scalesFrom, List.map_cons, add_sub_cancel_right]
      rw [jumpList_ofCoeff c S]

/-! ### Theorem 4.3 -/

/-- **Theorem 4.3, first half**: the floor spectrum of a layout function lies in the scales of
its shape.  (`c` is any coefficient function of the layout function on `{0, …, N-1}`.) -/
theorem spec_subset_scales (L : Layout D) (N : ℕ) (c : ℕ → D)
    (h : ∀ x, x < N → eval L x = fexp c x) (q : ℕ) (hq1 : 1 ≤ q) (hqN : q < N) (hc : c q ≠ 0) :
    q ∈ scalesFrom 1 (L.map Prod.fst) := by
  have hfe : ∀ x, x < N → fexp c x = fexp (total (jumpList 1 0 L)) x := by
    intro x hx
    rw [← h x hx, eval_eq_floorSum_all, floorSum_eq_fexp]
  have h1 : c q = total (jumpList 1 0 L) q := fexp_unique N c _ hfe q hq1 hqN
  by_contra hmem
  rw [← jumpList_fst L 1 0] at hmem
  exact hc (h1.trans (total_eq_zero_of_not_mem _ q hmem))

/-- **Theorem 4.3, second half**: if the coefficients vanish outside the scales of `S`, the layout
over `S` with those jumps has the floor expansion `c` on `{0, …, N-1}`. -/
theorem eval_ofCoeff (S : List ℕ) (hnd : (scalesFrom 1 S).Nodup) (N : ℕ) (c : ℕ → D)
    (h : ∀ q, 1 ≤ q → q < N → c q ≠ 0 → q ∈ scalesFrom 1 S) (x : ℕ) (hx : x < N) :
    eval (ofCoeff c 1 0 S) x = fexp c x := by
  rw [eval_eq_floorSum_all, jumpList_ofCoeff, floorSum_eq_fexp]
  apply fexp_congr
  intro q hq1 hqx
  rw [total_map_nodup c _ q hnd]
  by_cases hmem : q ∈ scalesFrom 1 S
  · rw [ite_eq_left hmem]
  · rw [ite_eq_right hmem]
    by_contra hne
    exact hmem (h q hq1 (by omega) (fun e => hne e.symm))

/-- **Theorem 4.3** (layouts have chain spectra).  Let `S` be a non-empty shape with pairwise
distinct scales (for instance without unit modes, `scalesFrom_nodup`), and let `g` have the floor
expansion `c` on `{0, …, N-1}`.  Then `g` is the function of a layout over `S` there if and only
if every scale `q ≥ 2` with `c q ≠ 0` is a scale of `S`. -/
theorem layout_over_shape_iff (S : List ℕ) (hne : S ≠ []) (hnd : (scalesFrom 1 S).Nodup) (N : ℕ)
    (g c : ℕ → D) (hc : ∀ x, x < N → g x = fexp c x) :
    (∃ L : Layout D, L.map Prod.fst = S ∧ ∀ x, x < N → eval L x = g x)
      ↔ ∀ q, 2 ≤ q → q < N → c q ≠ 0 → q ∈ scalesFrom 1 S := by
  constructor
  · rintro ⟨L, hS, hL⟩ q hq2 hqN hcq
    rw [← hS]
    exact spec_subset_scales L N c (fun x hx => by rw [hL x hx, hc x hx]) q (by omega) hqN hcq
  · intro H
    refine ⟨ofCoeff c 1 0 S, ofCoeff_fst c S 1 0, ?_⟩
    intro x hx
    rw [hc x hx]
    apply eval_ofCoeff S hnd N c _ x hx
    intro q hq1 hqN hcq
    rcases Nat.lt_or_ge q 2 with h2 | h2
    · have hq : q = 1 := by omega
      rw [hq]
      cases S with
      | nil => exact absurd rfl hne
      | cons s S => simp [scalesFrom]
    · exact H q h2 hqN hcq

/-- **Theorem 4.3 as stated in the paper**, for shapes without unit modes: a function with
`g 0 = 0` is `S`-additive if and only if its floor spectrum lies in the scales of `S`. -/
theorem additive_iff_spec (S : List ℕ) (hne : S ≠ []) (h2 : ∀ s ∈ S, 2 ≤ s) (g : ℕ → D)
    (hg : g 0 = 0) :
    (∀ x y, CarryFree S x y → g (x + y) = g x + g y)
      ↔ ∀ q, 2 ≤ q → q < S.prod → coeff g q ≠ 0 → q ∈ scalesFrom 1 S := by
  have hpos : ∀ s ∈ S, 0 < s := fun s hs => lt_of_lt_of_le (by norm_num) (h2 s hs)
  rw [additive_iff_layout S hpos hne g]
  exact layout_over_shape_iff S hne (scalesFrom_nodup S 1 Nat.one_pos h2) S.prod g (coeff g)
    (fun x _ => (fexp_coeff g hg x).symm)

/-! ### Theorem 4.3(a): chains -/

/-- The shape of size `N` whose scales are `P` followed by the chain `K`. -/
def shapeOf (N : ℕ) : ℕ → List ℕ → List ℕ
  | P, [] => [N / P]
  | P, Q :: K => (Q / P) :: shapeOf N Q K

lemma scalesFrom_shapeOf (N : ℕ) : ∀ (K : List ℕ) (P : ℕ),
    (P :: K).Pairwise (fun a b => a ∣ b) → scalesFrom P (shapeOf N P K) = P :: K
  | [], P, _ => by simp [shapeOf, scalesFrom]
  | Q :: K, P, h => by
      have hPQ : P ∣ Q := (List.pairwise_cons.mp h).1 Q (by simp)
      have hK : (Q :: K).Pairwise (fun a b => a ∣ b) := (List.pairwise_cons.mp h).2
      simp only [shapeOf, scalesFrom]
      rw [Nat.mul_div_cancel' hPQ, scalesFrom_shapeOf N K Q hK]

lemma prod_shapeOf (N : ℕ) : ∀ (K : List ℕ) (P : ℕ),
    (P :: K).Pairwise (fun a b => a ∣ b) → (∀ q ∈ P :: K, q ∣ N) →
    P * (shapeOf N P K).prod = N
  | [], P, _, hN => by
      simp only [shapeOf, List.prod_cons, List.prod_nil, mul_one]
      exact Nat.mul_div_cancel' (hN P (by simp))
  | Q :: K, P, h, hN => by
      have hPQ : P ∣ Q := (List.pairwise_cons.mp h).1 Q (by simp)
      have hK : (Q :: K).Pairwise (fun a b => a ∣ b) := (List.pairwise_cons.mp h).2
      simp only [shapeOf, List.prod_cons]
      rw [← mul_assoc, Nat.mul_div_cancel' hPQ]
      exact prod_shapeOf N K Q hK (fun q hq => hN q (List.mem_cons_of_mem _ hq))

/-- **Theorem 4.3(a)**: a function with `g 0 = 0` is the function of some layout of size `N` if
and only if its floor spectrum `{q : 2 ≤ q < N, coeff g q ≠ 0}` is a chain under divisibility all
of whose members divide `N`. -/
theorem layout_iff_chain (N : ℕ) (g : ℕ → D) (hg : g 0 = 0) :
    (∃ L : Layout D, (L.map Prod.fst).prod = N ∧ ∀ x, x < N → eval L x = g x)
      ↔ (∀ q, 2 ≤ q → q < N → coeff g q ≠ 0 → q ∣ N) ∧
        (∀ q q', 2 ≤ q → q < q' → q' < N → coeff g q ≠ 0 → coeff g q' ≠ 0 → q ∣ q') := by
  have hc : ∀ x, x < N → g x = fexp (coeff g) x := fun x _ => (fexp_coeff g hg x).symm
  constructor
  · rintro ⟨L, hN, hL⟩
    have hmem : ∀ q, 2 ≤ q → q < N → coeff g q ≠ 0 → q ∈ scalesFrom 1 (L.map Prod.fst) :=
      fun q hq2 hqN hcq =>
        spec_subset_scales L N (coeff g) (fun x hx => by rw [hL x hx, hc x hx]) q (by omega)
          hqN hcq
    constructor
    · intro q hq2 hqN hcq
      have h1 := (scalesFrom_dvd _ 1 q (hmem q hq2 hqN hcq)).2
      rwa [one_mul, hN] at h1
    · intro q q' hq2 hlt hq'N hcq hcq'
      rcases scalesFrom_comparable _ 1 q q' (hmem q hq2 (by omega) hcq)
          (hmem q' (by omega) hq'N hcq') with h1 | h1
      · exact h1
      · have h2 := Nat.le_of_dvd (by omega) h1
        omega
  · rintro ⟨hdiv, hchain⟩
    classical
    obtain ⟨K, hK⟩ : ∃ K : List ℕ, K = (List.range N).filter
        (fun q => decide (2 ≤ q ∧ coeff g q ≠ 0)) := ⟨_, rfl⟩
    have hKmem : ∀ q, q ∈ K ↔ q < N ∧ 2 ≤ q ∧ coeff g q ≠ 0 := by
      intro q
      rw [hK]
      simp only [List.mem_filter, List.mem_range, decide_eq_true_eq]
    have hKlt : K.Pairwise (· < ·) := by
      rw [hK]
      exact List.Pairwise.filter _ List.pairwise_lt_range
    have hKdvd : K.Pairwise (fun a b => a ∣ b) :=
      hKlt.imp_of_mem (fun {a b} ha hb hab =>
        hchain a b ((hKmem a).mp ha).2.1 hab ((hKmem b).mp hb).1 ((hKmem a).mp ha).2.2
          ((hKmem b).mp hb).2.2)
    have hchainK : (1 :: K).Pairwise (fun a b => a ∣ b) :=
      List.pairwise_cons.mpr ⟨fun q _ => one_dvd q, hKdvd⟩
    have hdivK : ∀ q ∈ 1 :: K, q ∣ N := by
      intro q hq
      rcases List.mem_cons.mp hq with h1 | h1
      · rw [h1]
        exact one_dvd N
      · exact hdiv q ((hKmem q).mp h1).2.1 ((hKmem q).mp h1).1 ((hKmem q).mp h1).2.2
    have hscales : scalesFrom 1 (shapeOf N 1 K) = 1 :: K := scalesFrom_shapeOf N K 1 hchainK
    have hnd : (scalesFrom 1 (shapeOf N 1 K)).Nodup := by
      rw [hscales]
      refine List.nodup_cons.mpr ⟨?_, hKlt.imp (fun hab => ne_of_lt hab)⟩
      intro h1
      have h2 := ((hKmem 1).mp h1).2.1
      omega
    refine ⟨ofCoeff (coeff g) 1 0 (shapeOf N 1 K), ?_, ?_⟩
    · rw [ofCoeff_fst]
      have h1 := prod_shapeOf N K 1 hchainK hdivK
      rwa [one_mul] at h1
    · intro x hx
      rw [hc x hx]
      apply eval_ofCoeff _ hnd N (coeff g) _ x hx
      intro q hq1 hqN hcq
      rw [hscales]
      rcases Nat.lt_or_ge q 2 with h2 | h2
      · have hq : q = 1 := by omega
        rw [hq]
        exact List.mem_cons_self
      · exact List.mem_cons_of_mem _ ((hKmem q).mpr ⟨hqN, h2, hcq⟩)

end CuTe
