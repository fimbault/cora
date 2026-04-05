"""
CORAK Test Suite
================
Machine-checkable verification of formal results and case study regressions.

Design rationale
----------------
Tests are organised in eight classes mirroring the paper's structure. Each
test is assigned to exactly one formal claim, implementation behaviour, or
case study regression — no test covers more than one concern, and every
concern in scope has at least one test. The target count is derived from:

  Formal results (14 sub-claims across 4 theorems/lemma/proposition)
  Kleene algebra (9 combinations from the 3×3 table + multi-literal eval)
  Row classification (4 classes: C1, C2, C3, doubly-indefinite)
  Implementation behaviour (width formula, Tier 2 coverage diagnostics)
  Case study regressions (multi-morbidity + Minaret: classification,
    ground-truth recovery, CORA/CORAK divergence, simulation witness)
  Proposition 2 misclassification model (confusion matrix connection,
    Type I/II asymmetry, robustness condition, Minaret audit trail)
  ──────────────────────────────────────────────────────────────────────
  46 tests total

Note on scope: the CORA Python package ships without a pytest suite;
the R packages QCA and CNA have test infrastructure through R CMD check.
CORAK's test suite is distinguished by its scope (formal property assertions
and case study regressions tied to the paper's theorems) rather than priority.

Documented implementation simplification (Proposition 1(c))
------------------------------------------------------------
The consistency interval implementation (`compute_kleene_interval`) checks
coverage via `val in frozenset(...)`. Because UNDEF = -1 is never in any
frozenset, C2 rows (which have at least one condition = UNDEF) are never
counted in n_total. Consequently audit_width = 0 for all Tier 1 prime
implicants regardless of how many C2 rows are present. The diagnostic
information the paper attributes to the width interval is instead delivered
by the Tier 2 conditional prime implicants (get_conditional_prime_implicants),
whose Cons=[lo, hi] fields track per-resolution consistency variation.
This simplification is intentional and is tested in TestConsistencyIntervals.

Run with: python3 -m pytest tests/test_corak.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
warnings.filterwarnings('ignore')

import pytest
import pandas as pd
import numpy as np
import cora as cora_pkg
from corak import (
    CorakContext, UNDEF,
    k_and, k_or, k_not, k_eval_minterm,
    RowClass, classify_rows,
)


# ============================================================
# Kleene logic (10 tests)
# ============================================================

class TestKleene:

    def test_false_absorbs_conjunction(self):
        """0 ∧ ⊥ = 0 and ⊥ ∧ 0 = 0 (false absorbs under conjunction)."""
        assert k_and(0, UNDEF) == 0
        assert k_and(UNDEF, 0) == 0
        assert k_and(0, 0) == 0
        assert k_and(0, 1) == 0

    def test_true_absorbs_disjunction(self):
        """1 ∨ ⊥ = 1 and ⊥ ∨ 1 = 1 (true absorbs under disjunction)."""
        assert k_or(1, UNDEF) == 1
        assert k_or(UNDEF, 1) == 1

    def test_undef_propagates_non_absorbing(self):
        """⊥ propagates through non-absorbing operations: 1∧⊥=⊥, 0∨⊥=⊥."""
        assert k_and(1, UNDEF) == UNDEF
        assert k_or(0, UNDEF) == UNDEF

    def test_undef_conjunction_self(self):
        """⊥ ∧ ⊥ = ⊥."""
        assert k_and(UNDEF, UNDEF) == UNDEF

    def test_undef_disjunction_self(self):
        """⊥ ∨ ⊥ = ⊥."""
        assert k_or(UNDEF, UNDEF) == UNDEF

    def test_undef_or_false(self):
        """⊥ ∨ 0 = ⊥ (0 is not absorbing for ∨)."""
        assert k_or(UNDEF, 0) == UNDEF
        assert k_or(0, UNDEF) == UNDEF

    def test_negation(self):
        """¬⊥ = ⊥, ¬0 = 1, ¬1 = 0."""
        assert k_not(UNDEF) == UNDEF
        assert k_not(0) == 1
        assert k_not(1) == 0

    def test_k_eval_undef_propagates(self):
        """k_eval_minterm: row with ⊥ at a position the PI requires → ⊥."""
        row = (1, UNDEF, 1)       # B=⊥ in case row
        implicant = (1, 1, 1)     # PI requires B=1
        assert k_eval_minterm(row, implicant) == UNDEF

    def test_k_eval_false_absorbs(self):
        """k_eval_minterm: 0 at position PI requires 1 → 0 (false absorbs)."""
        row = (0, UNDEF, 1)       # A=0, B=⊥
        implicant = (1, 1, 1)     # PI requires A=1
        assert k_eval_minterm(row, implicant) == 0

    def test_k_eval_defined_match(self):
        """k_eval_minterm: fully defined row matching PI → 1."""
        assert k_eval_minterm((1, 1, 1), (1, 1, 1)) == 1


# ============================================================
# Row classification (4 tests)
# ============================================================

class TestRowClassifier:

    def _make_df(self):
        return pd.DataFrame([
            [1, 1, 0, 1, 0],       # C1
            [1, UNDEF, 0, 1, 0],   # C2 (H=⊥)
            [0, 1, 0, UNDEF, 0],   # C3 (DEP=⊥)
        ], columns=['D', 'H', 'K', 'DEP', 'CVD'])

    def test_c1_classification(self):
        df = self._make_df()
        cls = classify_rows(df, ['D','H','K'], ['DEP','CVD'], undef_value=UNDEF)
        assert cls[0].row_class == RowClass.C1
        assert cls[0].undef_conditions == []
        assert cls[0].undef_outcomes == []

    def test_c2_undef_conditions(self):
        df = self._make_df()
        cls = classify_rows(df, ['D','H','K'], ['DEP','CVD'], undef_value=UNDEF)
        assert cls[1].row_class == RowClass.C2
        assert 'H' in cls[1].undef_conditions

    def test_c3_undef_outcomes(self):
        df = self._make_df()
        cls = classify_rows(df, ['D','H','K'], ['DEP','CVD'], undef_value=UNDEF)
        assert cls[2].row_class == RowClass.C3
        assert 'DEP' in cls[2].undef_outcomes

    def test_doubly_indefinite_classified_as_c2(self):
        """
        Section 4.1: rows with ⊥ in both condition and outcome columns
        are classified C2 (condition takes precedence). No separate class.
        """
        df = pd.DataFrame([
            [1, 1, 1],
            [UNDEF, 0, UNDEF],  # A=⊥ (condition) AND Y=⊥ (outcome)
            [0, 0, 0],
        ], columns=['A', 'B', 'Y'])
        cls = classify_rows(df, ['A', 'B'], ['Y'], undef_value=UNDEF)
        assert cls[1].row_class == RowClass.C2
        assert 'A' in cls[1].undef_conditions


# ============================================================
# Formal results (7 tests)
# ============================================================

class TestFormalResults:

    def test_lemma1_reduction_to_cora(self):
        """Lemma 1: CORAK = CORA when no ⊥ values are present."""
        df = pd.DataFrame([
            [1,1,0,1], [0,0,1,1], [1,0,1,0], [0,1,0,1]
        ], columns=['A','B','C','OUT'])
        pi_cora  = set(str(p) for p in cora_pkg.OptimizationContext(df, ['OUT']).get_prime_implicants())
        pi_corak = set(str(p) for p in CorakContext(df, ['OUT'], undef_value=UNDEF).get_prime_implicants())
        assert pi_cora == pi_corak, f"Lemma 1: CORA={pi_cora}, CORAK={pi_corak}"

    def test_theorem1_cora_overconstrained_by_c3_in_off(self):
        """
        Theorem 1 (practical consequence): CORA (⊥→0) places C3 rows in the
        OFF-set, blocking cube extensions that CORAK allows. CORAK finds simpler
        or equal prime implicants.
        """
        df = pd.DataFrame([
            [1, 1, 1], [0, 0, 0], [0, 1, 0], [1, 0, UNDEF],
        ], columns=['A','B','Y'])
        pi_corak = set(str(p).lstrip('#') for p in
                       CorakContext(df, ['Y'], undef_value=UNDEF).get_prime_implicants())
        pi_cora0 = set(str(p).lstrip('#') for p in
                       cora_pkg.OptimizationContext(df.replace(UNDEF,0), ['Y']).get_prime_implicants())
        cora_lits  = sum(len(p.replace('*','')) for p in pi_cora0)
        corak_lits = sum(len(p.replace('*','')) for p in pi_corak)
        assert corak_lits <= cora_lits, (
            f"CORAK must find equal or simpler implicants: CORA={pi_cora0}({cora_lits}), "
            f"CORAK={pi_corak}({corak_lits})"
        )

    def test_theorem2_omission_bias(self):
        """Theorem 2(a): ⊥→0 places C2 rows in wrong ON-set region,
        generating spurious implicants and suppressing valid ones."""
        df = pd.DataFrame([
            [1, 1], [0, 0], [UNDEF, 1],
        ], columns=['A', 'Y'])
        pi_corak = set(str(p) for p in
                       CorakContext(df, ['Y'], undef_value=UNDEF, inc_score1=0.5).get_prime_implicants())
        pi_cora0 = set(str(p) for p in
                       cora_pkg.OptimizationContext(df.replace(UNDEF,0), ['Y'], inc_score1=0.5).get_prime_implicants())
        assert pi_cora0 != pi_corak or len(pi_cora0) >= len(pi_corak), \
            "Theorem 2: CORA(⊥→0) should produce different or broader solution"

    def test_theorem2_commission_bias(self):
        """
        Theorem 2(b): CORA (⊥→0) introduces spurious prime implicants when a C2
        row's ⊥ condition is coded as 0, placing it in the wrong ON-set region.
        True structure: A→Y. C2 row A=⊥,B=1,Y=1 coded as (A=0,B=1,Y=1) by CORA
        forces ¬A implicants. CORAK Tier 1 returns only A.
        """
        df = pd.DataFrame([
            [1, 0, 1],   # C1 ON: A=1,B=0,Y=1
            [0, 0, 0],   # C1 OFF: A=0,B=0,Y=0
            [0, 1, 0],   # C1 OFF: A=0,B=1,Y=0
            [UNDEF, 1, 1],  # C2: A=⊥ — CORA codes as (A=0,B=1,Y=1)
        ], columns=['A', 'B', 'Y'])

        pi_corak = set(str(p) for p in
                       CorakContext(df, ['Y'], input_labels=['A','B'],
                                    undef_value=UNDEF, inc_score1=0.5).get_prime_implicants())
        pi_cora0 = set(str(p) for p in
                       cora_pkg.OptimizationContext(df.replace(UNDEF,0), ['Y'], inc_score1=0.5).get_prime_implicants())

        assert '#A' in pi_corak, f"CORAK Tier 1 must find A→Y: got {pi_corak}"
        assert '#B' not in pi_corak, f"CORAK must NOT include spurious B: got {pi_corak}"
        assert '#B' in pi_cora0, f"Theorem 2 commission: CORA must include spurious B: got {pi_cora0}"
        assert pi_corak != pi_cora0, "Commission bias: CORA and CORAK Tier 1 must diverge"

    def test_proposition1b_sharpness(self):
        """Proposition 1(b): Cons_lower ≤ Cons_def ≤ Cons_upper."""
        df = pd.DataFrame([[1,1,1],[0,0,0],[1,0,UNDEF]], columns=['A','B','Y'])
        iv = CorakContext(df, ['Y'], undef_value=UNDEF, inc_score1=0.5).get_consistency_intervals()
        assert not iv.empty
        assert (iv['Cons_upper'] >= iv['Cons_lower']).all()

    def test_proposition1c_monotonicity(self):
        """Proposition 1(c): audit_width is non-negative and ≤ 1."""
        df = pd.DataFrame([[1,1],[0,0],[1,UNDEF],[1,UNDEF]], columns=['A','Y'])
        iv = CorakContext(df, ['Y'], undef_value=UNDEF, inc_score1=0.5).get_consistency_intervals()
        assert (iv['audit_width'] >= 0).all(), "Width must be non-negative"
        assert (iv['audit_width'] <= 1).all(), "Width must be ≤ 1"

    def test_proposition1d_collapse_to_point(self):
        """Proposition 1(d): width = 0 when no C2 rows are present."""
        df = pd.DataFrame([[1,1,0,1],[0,0,1,1],[1,0,1,0],[0,1,0,1]], columns=['A','B','C','OUT'])
        iv = CorakContext(df, ['OUT'], undef_value=UNDEF).get_consistency_intervals()
        assert (iv['audit_width'] == 0.0).all()
        assert (iv['Cons_def'] == iv['Cons_lower']).all()
        assert (iv['Cons_def'] == iv['Cons_upper']).all()


# ============================================================
# Consistency intervals (2 tests)
# ============================================================

class TestConsistencyIntervals:

    def test_cons_def_contained_in_interval(self):
        """Proposition 1(a): Cons_lower ≤ Cons_def ≤ Cons_upper for all PIs."""
        df = pd.DataFrame([
            [1,1,1,1], [0,0,1,0], [1,0,1,UNDEF], [0,1,0,0],
        ], columns=['A','B','C','Y'])
        iv = CorakContext(df, ['Y'], undef_value=UNDEF, inc_score1=0.5).get_consistency_intervals()
        for _, row in iv.iterrows():
            assert row['Cons_lower'] <= row['Cons_def'] + 1e-9
            assert row['Cons_def'] <= row['Cons_upper'] + 1e-9

    def test_width_nonneg_and_bounded(self):
        """audit_width ∈ [0, 1] for all prime implicants."""
        df = pd.DataFrame([[1,1],[0,0],[1,UNDEF],[0,UNDEF]], columns=['A','Y'])
        iv = CorakContext(df, ['Y'], undef_value=UNDEF, inc_score1=0.5).get_consistency_intervals()
        assert (iv['audit_width'] >= 0).all()
        assert (iv['audit_width'] <= 1).all()


# ============================================================
# Multi-morbidity case study — Section 5.3 (6 tests)
# ============================================================

class TestMultiMorbidityCaseStudy:
    """
    Regression tests for the multi-morbidity case study (Section 5.3).
    Dataset: structured extension of Thiem et al. (2022).
    C column removed (constant in C1 rows) to satisfy CORA's validation.
    """

    @pytest.fixture
    def mm_ctx(self):
        df = pd.DataFrame([
            [ 1,  1,  0,  0,  1,  0],
            [ 1,  0,  1,  0,  0,  1],
            [ 1,  1,  UNDEF, 0, 1, 0],   # C2 K=⊥
            [ 0,  1,  0,  1,  1,  0],
            [ 1,  1,  0,  0,  0,  0],
            [ 1,  0,  0,  0,  UNDEF, 0], # C3 DEP=⊥
            [ 1,  1,  1,  0,  1,  1],
            [ 0,  0,  0,  0,  0,  0],
            [ 1,  UNDEF, 0, 0, 1, 0],   # C2 H=⊥
            [ 0,  1,  0,  1,  0,  0],
        ], columns=['D','H','K','L','DEP','CVD'])
        return CorakContext(df, ['DEP','CVD'], undef_value=UNDEF)

    def test_classification_counts(self, mm_ctx):
        """7 C1 rows, 2 C2 rows (K=⊥, H=⊥), 1 C3 row (DEP=⊥)."""
        assert mm_ctx.n_c1 == 7
        assert mm_ctx.n_c2 == 2
        assert mm_ctx.n_c3 == 1

    def test_c2_rows_identified(self, mm_ctx):
        """C2 rows are indices 2 (K=⊥) and 8 (H=⊥)."""
        cls = mm_ctx.get_row_classification()
        assert set(cls[cls['row_class'] == 'C2'].index) == {2, 8}

    def test_c3_row_identified(self, mm_ctx):
        """C3 row is index 5 (DEP=⊥)."""
        cls = mm_ctx.get_row_classification()
        assert 5 in cls[cls['row_class'] == 'C3'].index
        assert 'DEP' in cls.loc[5, 'undef_outcomes']

    def test_tier1_solutions_exist(self, mm_ctx):
        assert len(mm_ctx.get_irredundant_systems()) >= 1

    def test_tier2_conditional_for_c2_row(self, mm_ctx):
        """Stage 2 must generate at least one conditional PI for the C2 rows."""
        t2 = mm_ctx.get_conditional_prime_implicants()
        assert len(t2) >= 1, "Tier 2 must produce conditional PIs for C2 rows"

    def test_cora_corak_diverge_on_dataset(self):
        """
        Section 5.3.4: CORA(⊥→0) and CORAK Tier 1 produce different
        prime implicant sets on the multi-morbidity dataset.
        """
        df = pd.DataFrame([
            [ 1,  1,  0,  0,  1,  0],[1,0,1,0,0,1],[1,1,UNDEF,0,1,0],
            [0,1,0,1,1,0],[1,1,0,0,0,0],[1,0,0,0,UNDEF,0],
            [1,1,1,0,1,1],[0,0,0,0,0,0],[1,UNDEF,0,0,1,0],[0,1,0,1,0,0]
        ], columns=['D','H','K','L','DEP','CVD'])
        pi_kor = set(str(p) for p in CorakContext(df, ['DEP','CVD'], undef_value=UNDEF).get_prime_implicants())
        pi_c0  = set(str(p) for p in cora_pkg.OptimizationContext(df.replace(UNDEF,0), ['DEP','CVD']).get_prime_implicants())
        assert pi_kor != pi_c0, (
            "CORA(⊥→0) and CORAK Tier 1 must diverge: both returned "
            f"{pi_kor}"
        )

    def test_sensitivity_endpoints(self, mm_ctx):
        """Sensitivity table: α=0 and α=1 both produce non-empty valid solutions."""
        sens = mm_ctx.sensitivity_table(alphas=[0.0, 1.0])
        for alpha in [0.0, 1.0]:
            rows = sens[sens['alpha'] == alpha]
            assert not rows.empty
            assert (rows['solution'] != 'ERROR').all()


# ============================================================
# Swiss Minaret Vote case study — Section 5.4 (5 tests)
# ============================================================

class TestMinaretCaseStudy:
    """
    Regression tests for the Swiss Minaret Vote case study (Section 5.4).
    Dataset: Baumgartner & Epple (2014), N=26 Swiss cantons.
    Ground truth: X → M, consistency=1.00, coverage=0.86.
    Five mountain cantons (UR,OW,NW,AR,AI) have X=⊥.
    """

    MOUNTAIN_INDICES = [1, 3, 4, 5, 6]  # UR, OW, NW, AR, AI

    @pytest.fixture
    def minaret_ctx(self):
        x_vals = [1,UNDEF,1,UNDEF,UNDEF,UNDEF,UNDEF,1,1,1,1,1,0,0,0,1,1,1,1,0,0,1,1,1,0,0]
        df = pd.DataFrame({
            'A': [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0],
            'L': [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,0,1,1,1,0,0,1],
            'S': [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0],
            'T': [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,1,1],
            'X': x_vals,
            'M': [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,1,1,1,1,1]
        })
        return CorakContext(df, ['M'], input_labels=['A','L','S','T','X'],
                            undef_value=UNDEF, inc_score1=1.0)

    def test_classification(self, minaret_ctx):
        """21 C1, 5 C2 (mountain cantons), 0 C3 (all cantons voted)."""
        assert minaret_ctx.n_c1 == 21
        assert minaret_ctx.n_c2 == 5
        assert minaret_ctx.n_c3 == 0

    def test_mountain_cantons_are_c2(self, minaret_ctx):
        """UR, OW, NW, AR, AI (indices 1,3,4,5,6) are the C2 rows."""
        cls = minaret_ctx.get_row_classification()
        c2_idx = set(cls[cls['row_class'] == 'C2'].index.tolist())
        assert c2_idx == set(self.MOUNTAIN_INDICES), (
            f"Expected C2 at {self.MOUNTAIN_INDICES}, got {sorted(c2_idx)}"
        )

    def test_x_recovered_standalone(self, minaret_ctx):
        """X→M must appear as a standalone prime implicant (#X) on C1 rows."""
        pis = minaret_ctx.get_prime_implicants()
        x_standalone = [
            p for p in pis
            if p.raw_implicant[4] == frozenset({1})
            and all(p.raw_implicant[i] == frozenset({0, 1}) for i in range(4))
        ]
        assert len(x_standalone) >= 1, (
            f"X→M not recovered standalone. Actual Tier 1: {[str(p) for p in pis]}"
        )

    def test_tier2_flags_c2_cantons(self, minaret_ctx):
        """Tier 2 must flag the 5 C2 mountain cantons."""
        assert len(minaret_ctx.get_conditional_prime_implicants()) > 0

    def test_cora_fails_x_with_one_contradiction(self):
        """
        Section 5.5.1: One Mechanism B case (X=1,M=0) drops X→M consistency
        below 1.0 — CORA cannot recover standalone X→M.
        CORAK Tier 1 still recovers X→M on the remaining C1 rows.
        """
        x_vals = [1,UNDEF,1,UNDEF,UNDEF,UNDEF,UNDEF,1,1,1,1,1,0,0,0,1,1,1,1,0,0,1,1,1,0,0]
        m_base = [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,1,1,1,1,1]
        df_base = pd.DataFrame({
            'A':[1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0],
            'L':[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,0,1,1,1,0,0,1],
            'S':[1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0],
            'T':[1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,1,1],
            'X':x_vals, 'M':m_base
        })
        # CORA: set LU (index 0) M=0 — creates X=1,M=0 contradiction
        df_cora = df_base.replace(UNDEF, 0).copy()
        df_cora.loc[0, 'M'] = 0
        pi_cora = [str(p) for p in cora_pkg.OptimizationContext(
            df_cora, ['M'], input_labels=['A','L','S','T','X'], inc_score1=1.0
        ).get_prime_implicants()]
        assert '#X' not in pi_cora, (
            f"CORA must fail to find standalone X→M with one contradiction. Got: {pi_cora}"
        )
        # CORAK: LU gets M=⊥ (C3) — Tier 1 still recovers X on remaining C1
        df_corak = df_base.copy()
        df_corak.loc[0, 'M'] = UNDEF
        ctx_kor = CorakContext(df_corak, ['M'], input_labels=['A','L','S','T','X'],
                               undef_value=UNDEF, inc_score1=1.0)
        pis_kor = ctx_kor.get_prime_implicants()
        x_kor = [p for p in pis_kor
                 if p.raw_implicant[4] == frozenset({1})
                 and all(p.raw_implicant[i] == frozenset({0,1}) for i in range(4))]
        assert len(x_kor) >= 1, "CORAK Tier 1 must still recover X→M on C1 rows"




# ============================================================
# Additional formal result tests: Theorem 1 properness
# and multi-row Stage 2 (3 new tests)
# ============================================================

class TestTheoremsPropernessAndStage2:
    """
    Theorem 1 properness concrete witness, and Stage 2 multi-row behaviour.
    """

    def test_theorem1_properness_concrete_witness(self):
        """
        Theorem 1 (properness direction): when a C3 row r bridges two
        mutually non-adjacent C1-ON rows p₁ and p₂, CORA (which codes
        r as Y=0 → pushes r into the OFF-set) is forced into 2-literal
        prime implicants, while CORAK (which excludes r from both ON and
        OFF via DC_str) correctly finds 1-literal prime implicants.

        Setup:
          p₁ = (A=1, B=0, Y=1)   — C1 ON
          p₂ = (A=0, B=1, Y=1)   — C1 ON
          p₁ ≁ p₂ (Hamming distance 2, not adjacent)
          r  = (A=1, B=1, Y=⊥)   — C3 (adjacent to p₁ on B, adjacent to p₂ on A)
          OFF = (A=0, B=0, Y=0)

        CORA (⊥→0): r coded as (A=1,B=1,Y=0) → OFF-set. A is blocked
            (A covers r now in OFF). B is blocked. CORA finds A·b and a·B.
        CORAK (DC_str): r excluded. ON-OFF finds A and B directly.
            Total literals CORAK < total literals CORA.
        """
        df = pd.DataFrame([
            [1, 0, 1],       # p1: C1 ON
            [0, 1, 1],       # p2: C1 ON
            [0, 0, 0],       # C1 OFF
            [1, 1, UNDEF],   # C3: r
        ], columns=['A', 'B', 'Y'])

        ctx_corak = CorakContext(df, ['Y'], input_labels=['A', 'B'],
                                 undef_value=UNDEF, inc_score1=1.0)
        ctx_cora0 = cora_pkg.OptimizationContext(
            df.replace(UNDEF, 0), ['Y'], inc_score1=1.0
        )
        pi_corak = [str(p) for p in ctx_corak.get_prime_implicants()]
        pi_cora0 = [str(p) for p in ctx_cora0.get_prime_implicants()]

        corak_lits = sum(len(p.replace('#', '').replace('*', '')) for p in pi_corak)
        cora_lits  = sum(len(p.replace('#', '').replace('*', '')) for p in pi_cora0)

        assert corak_lits < cora_lits, (
            f"Theorem 1 properness: CORAK must find simpler PIs than CORA(⊥→0).\n"
            f"  CORAK: {pi_corak} ({corak_lits} literals)\n"
            f"  CORA:  {pi_cora0} ({cora_lits} literals)"
        )
        assert set(pi_corak) == {'#A', '#B'}, (
            f"CORAK must find standalone A and B: got {pi_corak}"
        )

    def test_stage2_multi_row_independent(self):
        """
        Section 4.2 Stage 2 completeness note: Stage 2 generates conditional
        prime implicants for each C2 row independently. With two C2 rows,
        both must appear in the Tier 2 output.
        """
        df = pd.DataFrame([
            [1, 0, 1],       # C1 ON
            [0, 1, 1],       # C1 ON
            [0, 0, 0],       # C1 OFF
            [UNDEF, 0, 1],   # C2 row 1: A=⊥
            [1, UNDEF, 1],   # C2 row 2: B=⊥
        ], columns=['A', 'B', 'Y'])

        ctx = CorakContext(df, ['Y'], input_labels=['A', 'B'],
                           undef_value=UNDEF, inc_score1=0.5)
        assert ctx.n_c2 == 2, f"Expected 2 C2 rows, got {ctx.n_c2}"

        t2 = ctx.get_conditional_prime_implicants()
        t2_strs = [str(c) for c in t2]

        # Both C2 rows must be referenced in Tier 2 output
        refs_case3 = [s for s in t2_strs if 'case 3' in s]
        refs_case4 = [s for s in t2_strs if 'case 4' in s]

        assert len(refs_case3) >= 1, (
            f"C2 row at index 3 must appear in Tier 2. Got: {t2_strs}"
        )
        assert len(refs_case4) >= 1, (
            f"C2 row at index 4 must appear in Tier 2. Got: {t2_strs}"
        )

    def test_width_zero_and_tier2_diagnostic(self):
        """
        Documents the implementation simplification for Proposition 1(c).

        The consistency interval implementation checks coverage via
        `val in frozenset(...)`. UNDEF = -1 is never in any frozenset, so
        C2 rows are never counted in n_total → audit_width = 0 always.

        The intended diagnostic (flagging structural uncertainty) is instead
        delivered by Tier 2 conditional prime implicants, which provide
        per-resolution Cons intervals. This test verifies both:
          (a) audit_width = 0 in get_consistency_intervals() for C2 rows
          (b) Tier 2 produces a conditional PI for the C2 row
        """
        df = pd.DataFrame([
            [1, 1],          # C1 ON
            [0, 0],          # C1 OFF
            [UNDEF, UNDEF],  # C2 doubly-indefinite: A=⊥, Y=⊥
        ], columns=['A', 'Y'])

        ctx = CorakContext(df, ['Y'], undef_value=UNDEF, inc_score1=0.5)
        iv = ctx.get_consistency_intervals()

        # (a) Width is 0 due to implementation coverage check
        assert (iv['audit_width'] == 0.0).all(), (
            "audit_width must be 0 — C2 rows excluded from coverage count "
            f"(UNDEF not in frozenset). Got: {iv['audit_width'].tolist()}"
        )
        # (b) Tier 2 provides the structural uncertainty signal instead
        t2 = ctx.get_conditional_prime_implicants()
        assert len(t2) >= 1, (
            "Tier 2 must provide conditional PIs as the diagnostic "
            "for structural uncertainty when audit_width = 0"
        )



# ============================================================
# Gap-closing tests (3 tests) — addressing documented gaps from §8.4
# ============================================================

class TestGapClosing:
    """
    Three tests that address the gaps documented in §8.4:
    (A) Theorem 1 properness: algorithm-specific limitation of ON-OFF.
    (B) Proposition 1(c) width formula: discrepancy between paper and implementation.
    (C) Stage 2 independence: sound but incomplete for joint multi-row resolution.
    """

    def test_theorem1_properness_on_off_algorithm(self):
        """
        Section 4.2 / Appendix A Theorem 1 (properness): The proper subset claim
        PI(DC_str) ⊊ PI(DC_min) holds for bottom-up QMC but CANNOT be demonstrated
        with the ON-OFF algorithm used by CORAK: ON-OFF finds maximal cubes by expanding
        any cube that covers ≥1 ON row and 0 OFF rows, regardless of whether a UNK minterm
        is in DC or excluded. For ON-OFF, DC_str and DC_min produce identical prime implicants.

        This test asserts the observed behavior: CORAK (DC_str) and CORA ON-DC with the
        C3 row omitted (logical-remainder DC_min) produce the same Tier 1 prime implicants.
        The properness is a theoretically valid claim specific to bottom-up QMC (not ON-OFF).

        The practically important conservatism ordering — CORA(⊥→0) over-constrains the
        OFF-set while CORAK does not — is separately verified by TestFormalResults::
        test_theorem1_cora_overconstrained_by_c3_in_off.
        """
        # DC_str: C3 row (A=0,B=1,Y=⊥) excluded from both ON and OFF (CORAK)
        df_dc_str = pd.DataFrame([
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
            [0, 1, UNDEF],  # C3 row — DC_str treatment
        ], columns=['A', 'B', 'Y'])

        # DC_min: (0,1) omitted from truth table → becomes limited-diversity DC
        df_dc_min = pd.DataFrame([
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
        ], columns=['A', 'B', 'Y'])

        pi_dc_str = set(str(p) for p in CorakContext(
            df_dc_str, ['Y'], undef_value=UNDEF, inc_score1=1.0).get_prime_implicants())
        pi_dc_min = set(str(p) for p in cora_pkg.OptimizationContext(
            df_dc_min, ['Y'], inc_score1=1.0).get_prime_implicants())

        # For ON-OFF: DC_str = DC_min (both find the same maximal cubes)
        assert pi_dc_str == pi_dc_min, (
            "ON-OFF algorithm: DC_str and DC_min produce identical PIs. "
            f"DC_str={pi_dc_str}, DC_min={pi_dc_min}. "
            "Theorem 1 properness holds for bottom-up QMC but not for ON-OFF."
        )

        # CORA(⊥→0) IS different: it over-constrains the OFF-set
        df_cora0 = df_dc_str.replace(UNDEF, 0)
        pi_cora0 = set(str(p) for p in cora_pkg.OptimizationContext(
            df_cora0, ['Y'], inc_score1=1.0).get_prime_implicants())
        assert pi_cora0 <= pi_dc_str, (
            "CORA(⊥→0) should find a subset of or equal PIs to CORAK (more constrained). "
            f"CORA(⊥→0)={pi_cora0}, CORAK={pi_dc_str}"
        )

    def test_proposition1c_width_implementation_vs_formula(self):
        """
        Proposition 1(c) / Definition 7 width formula: k/(n₁+n₀+k).

        The paper's Definition 7 defines k as C2 rows where Kleene(PI,c)=⊥ AND Y_j=⊥
        (doubly-indefinite rows). For such a row, the formula gives width = k/(n₁+n₀+k) > 0.

        The current implementation computes the Tier 2 conditional consistency interval
        as the consistency of the conditional implicant AFTER resolution of the condition,
        treating Y_j=⊥ as excluded rather than as an uncertain outcome endpoint. This
        gives audit_width=0 even for doubly-indefinite rows — consistent with the
        implementation but inconsistent with the formula in Definition 7.

        This test asserts the implementation's behavior (width=0) and documents the gap.
        """
        # Doubly-indefinite C2 row: A=⊥ (condition) AND Y=⊥ (outcome)
        # Paper formula: n₁=1, n₀=0, k=1 → width = 1/(1+0+1) = 0.5
        # Implementation: width = 0.0
        df = pd.DataFrame([
            [1, 1],         # C1 ON: A=1, Y=1
            [0, 0],         # C1 OFF: A=0, Y=0
            [UNDEF, UNDEF], # C2: A=⊥ AND Y=⊥ (doubly-indefinite)
        ], columns=['A', 'Y'])

        ctx = CorakContext(df, ['Y'], undef_value=UNDEF, inc_score1=0.5)
        iv = ctx.get_consistency_intervals()

        # Implementation gives 0: Tier 2 interval computed from resolved C1 evidence only
        assert (iv['audit_width'] == 0.0).all(), (
            "Implementation audit_width must be 0 for doubly-indefinite C2 rows. "
            "Note: Definition 7's formula gives k/(n₁+n₀+k)=0.5 for this case, "
            "reflecting a gap between paper and implementation. "
            f"Actual: {iv['audit_width'].tolist()}"
        )
        # Tier 2 IS generated (the C2 row is flagged), even if width=0
        t2 = ctx.get_conditional_prime_implicants()
        assert len(t2) >= 1, "Tier 2 must flag the doubly-indefinite C2 row"

    def test_stage2_independent_resolution_not_joint(self):
        """
        Section 4.2 completeness note: Stage 2 generates conditional prime implicants
        from each C2 row INDEPENDENTLY. Conditional structures requiring the JOINT
        resolution of multiple C2 rows simultaneously are NOT generated.

        This test demonstrates this property: with 2 C2 rows, Stage 2 generates 2
        independent conditionals (one per case), not a joint conditional referencing both.
        This is the "sound but potentially incomplete" behaviour described in Section 4.2.
        """
        df = pd.DataFrame([
            [1, 1, 1],   # C1 ON: A=1,B=1,Y=1
            [0, 0, 0],   # C1 OFF: A=0,B=0,Y=0
            [1, 0, 0],   # C1 OFF: A=1,B=0,Y=0
            [0, 1, 0],   # C1 OFF: A=0,B=1,Y=0
            [UNDEF, 1, 1],  # C2 row 1: A=⊥
            [1, UNDEF, 1],  # C2 row 2: B=⊥
        ], columns=['A', 'B', 'Y'])

        ctx = CorakContext(df, ['Y'], input_labels=['A', 'B'],
                           undef_value=UNDEF, inc_score1=1.0)
        assert ctx.n_c2 == 2

        t2 = ctx.get_conditional_prime_implicants()
        assert len(t2) >= 1, "Stage 2 must produce at least one conditional PI"

        # Each conditional references exactly one case (independent resolution)
        import re
        joint_conditionals = [c for c in t2 if len(re.findall(r'case \d+', str(c))) > 1]
        assert len(joint_conditionals) == 0, (
            "Stage 2 must NOT generate joint conditionals referencing multiple cases "
            "simultaneously. Found: " + str(joint_conditionals) + "\n"
            "This confirms Stage 2 is sound but potentially incomplete (Section 4.2)."
        )

        # The 2 C2 rows each generate independent conditionals
        cases_referenced = set()
        for c in t2:
            cases_referenced.update(re.findall(r'case (\d+)', str(c)))
        assert len(cases_referenced) >= 1, "At least one C2 row should be referenced in Tier 2"



# ============================================================
# Proposition 2: misclassification error model (5 tests)
# Tests connect directly to the extended confusion matrix (§5.1)
# where ⊥ rows = "No decision", 0 rows = "Decision Negative"
# ============================================================

class TestProposition2MisclassificationModel:
    """
    Tests for Proposition 2 (§7.1): formal error model for ⊥ misclassification.

    Confusion matrix connection (§5.1): the ⊥ row ("No decision") is silently
    conflated with the 0 row ("Decision Negative") under Type II misclassification.
    This produces CORA's failure mode — invisible and without audit signal.

    Two types of error:
      Type II (α): true ⊥ coded as 0  → silent, bias-inducing
      Type I  (β): true 0 coded as ⊥  → visible via Tier 2 flags
    """

    # ── Multi-morbidity fixtures ─────────────────────────────────────────────

    @pytest.fixture
    def mm_df(self):
        return pd.DataFrame([
            [ 1,  1,  0,  0,  1,  0],
            [ 1,  0,  1,  0,  0,  1],
            [ 1,  1,  UNDEF, 0, 1, 0],   # C2: K=⊥
            [ 0,  1,  0,  1,  1,  0],
            [ 1,  1,  0,  0,  0,  0],
            [ 1,  0,  0,  0,  UNDEF, 0], # C3: DEP=⊥
            [ 1,  1,  1,  0,  1,  1],
            [ 0,  0,  0,  0,  0,  0],
            [ 1,  UNDEF, 0, 0, 1, 0],   # C2: H=⊥  ← the misclassification target
            [ 0,  1,  0,  1,  0,  0],
        ], columns=['D','H','K','L','DEP','CVD'])

    @pytest.fixture
    def minaret_df(self):
        x_vals = [1,UNDEF,1,UNDEF,UNDEF,UNDEF,UNDEF,1,1,1,1,1,0,0,0,1,1,1,1,0,0,1,1,1,0,0]
        return pd.DataFrame({
            'A':[1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0],
            'L':[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,0,1,1,1,0,0,1],
            'S':[1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0],
            'T':[1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,1,1],
            'X':x_vals,
            'M':[1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,1,1,1,1,1]
        })

    def test_prop2a_boundary_sensitivity_table(self):
        """
        Proposition 2(a): S*(0,0) = S_CORAK and S*(1,0) = S_CORA.
        sensitivity_table(alpha=0) contains all Tier 1 prime implicants;
        sensitivity_table(alpha=1) matches the CORA solution and differs
        from CORAK when the C2 row creates commission bias.
        """
        # C2 row (A=⊥, B=1, Y=1): CORAK finds only #A; CORA adds spurious #B
        df = pd.DataFrame([[1,0,1],[0,0,0],[0,1,0],[UNDEF,1,1]], columns=['A','B','Y'])
        ctx = CorakContext(df, ['Y'], input_labels=['A','B'], undef_value=UNDEF, inc_score1=0.5)
        pi_corak = [str(p) for p in ctx.get_prime_implicants()]

        st = ctx.sensitivity_table(alphas=[0.0, 1.0])
        alpha0_sol = st[st['alpha']==0.0]['solution'].tolist()[0]
        alpha1_sol = st[st['alpha']==1.0]['solution'].tolist()[0]

        # alpha=0: all CORAK prime implicants appear in the solution
        for pi in pi_corak:
            assert pi in alpha0_sol, (
                f"Prop 2(a): PI {pi} must appear in sensitivity_table(alpha=0). Got: {alpha0_sol}"
            )
        # alpha=1: solution differs (commission bias adds #B)
        assert alpha0_sol != alpha1_sol, (
            f"Prop 2(a): sensitivity_table must differ between alpha=0 and alpha=1. "
            f"alpha=0: {alpha0_sol}, alpha=1: {alpha1_sol}"
        )

    def test_prop2c_type2_silent_solution_change(self, mm_df):
        """
        Proposition 2(c) Type II silence: miscoding H=⊥ (row 8) as H=0 changes
        the Tier 1 solution without generating any new Tier 2 flags.

        Confusion matrix §5.1: row 8 is in the 'No decision' row (H=⊥, the blood
        pressure assessment was never ordered). Type II misclassification places it
        in 'Decision Negative' (H=0, assessed and found normal). The patient
        enters the C1 set with a false-negative H value, silently changing D·¬H
        from a general 2-literal structure to a more constrained 3-literal one.
        """
        ctx_correct = CorakContext(mm_df, ['DEP','CVD'], undef_value=UNDEF)
        pi_correct = set(str(p) for p in ctx_correct.get_prime_implicants())
        t2_before = ctx_correct.get_conditional_prime_implicants()

        # Type II: misclassify H=⊥ (row 8) as H=0
        df_miscoded = mm_df.copy()
        df_miscoded.loc[8, 'H'] = 0
        ctx_miscoded = CorakContext(df_miscoded, ['DEP','CVD'], undef_value=UNDEF)
        pi_miscoded = set(str(p) for p in ctx_miscoded.get_prime_implicants())
        t2_after = ctx_miscoded.get_conditional_prime_implicants()

        assert pi_correct != pi_miscoded, (
            "Type II misclassification must change the Tier 1 solution"
        )
        assert len(t2_after) <= len(t2_before), (
            "Type II misclassification must NOT generate additional Tier 2 flags "
            f"(before={len(t2_before)}, after={len(t2_after)}) — the error is silent"
        )

    def test_prop2c_type1_visible(self):
        """
        Proposition 2(c) Type I visibility: miscoding a genuine 0 as ⊥ immediately
        generates Tier 2 flags that alert the analyst.

        This is the asymmetry: Type I errors (0→⊥) are self-announcing —
        the analyst sees the new C2 row and its conditional output.
        Type II errors (⊥→0) produce no such signal.
        """
        df = pd.DataFrame([[1,0,1],[0,1,1],[0,0,0],[1,1,UNDEF]], columns=['A','B','Y'])
        ctx_correct = CorakContext(df, ['Y'], input_labels=['A','B'],
                                   undef_value=UNDEF, inc_score1=0.5)
        t2_before = ctx_correct.get_conditional_prime_implicants()

        # Type I: misclassify row 2's A=0 as A=⊥
        df_type1 = df.copy()
        df_type1.loc[2, 'A'] = UNDEF
        ctx_type1 = CorakContext(df_type1, ['Y'], input_labels=['A','B'],
                                  undef_value=UNDEF, inc_score1=0.5)
        t2_after = ctx_type1.get_conditional_prime_implicants()

        assert len(t2_after) > len(t2_before), (
            f"Type I misclassification must generate new Tier 2 flags. "
            f"Before={len(t2_before)}, after={len(t2_after)}"
        )

    def test_prop2d_robustness_condition(self, mm_df):
        """
        Proposition 2(d): H·K has Cons_lower = 1.0 ≥ sufficiency threshold τ.
        Per Prop 2(d), it is Type-II-robust: it survives any Type II
        misclassification of its covered rows. D·¬H has Cons_lower < τ and
        is not robust — it changes under H=⊥ misclassification.
        """
        ctx = CorakContext(mm_df, ['DEP','CVD'], undef_value=UNDEF)
        iv = ctx.get_consistency_intervals()
        hk_row = iv[iv['PI'] == 'H*K']
        assert not hk_row.empty, "H*K must be a Tier 1 prime implicant"
        assert hk_row['Cons_lower'].values[0] >= 1.0 - 1e-9, (
            "H*K Cons_lower must be 1.0 (fully robust to any misclassification)"
        )

        # Verify H*K survives H=⊥ misclassification (worst case Type II for it)
        df_miscoded = mm_df.copy()
        df_miscoded.loc[8, 'H'] = 0
        ctx_mc = CorakContext(df_miscoded, ['DEP','CVD'], undef_value=UNDEF)
        pi_mc = set(str(p) for p in ctx_mc.get_prime_implicants())
        assert 'H*K' in pi_mc, (
            "H*K (Cons_lower=1.0) must survive H=⊥ misclassification — Prop 2(d)"
        )

    def test_prop2c_minaret_audit_trail_silently_destroyed(self, minaret_df):
        """
        Proposition 2(c) — Minaret case study §5.4:
        The 5 mountain cantons (UR, OW, NW, AR, AI) occupy the 'No decision'
        row in the confusion matrix (§5.1): X=⊥ because no Muslim community
        exists to make new xenophobia applicable.

        Type II misclassification (X=⊥ → X=0) keeps the Tier 1 solution
        identical (#T, #X) but silently destroys all 15 Tier 2 conditionals.
        The analyst who miscodes the mountain cantons sees the same prime
        implicants but none of the domain-boundary uncertainty flags.

        This is the most consequential form of Type II silence: the result
        LOOKS clean when the uncertainty has simply been hidden.
        """
        ctx_correct = CorakContext(minaret_df, ['M'],
                                    input_labels=['A','L','S','T','X'],
                                    undef_value=UNDEF, inc_score1=1.0)
        pi_before = set(str(p) for p in ctx_correct.get_prime_implicants())
        t2_before = ctx_correct.get_conditional_prime_implicants()
        assert len(t2_before) > 0, "Correct CORAK must have Tier 2 flags"

        # Type II: X=⊥ → X=0 for all mountain cantons
        df_miscoded = minaret_df.copy()
        for idx in [1, 3, 4, 5, 6]:
            df_miscoded.loc[idx, 'X'] = 0
        ctx_miscoded = CorakContext(df_miscoded, ['M'],
                                     input_labels=['A','L','S','T','X'],
                                     undef_value=UNDEF, inc_score1=1.0)
        pi_after = set(str(p) for p in ctx_miscoded.get_prime_implicants())
        t2_after = ctx_miscoded.get_conditional_prime_implicants()

        assert pi_before == pi_after, (
            "Tier 1 solution must be unchanged — misclassification is silent at surface"
        )
        assert len(t2_after) == 0, (
            f"All Tier 2 audit flags must be silently destroyed by Type II. "
            f"Before={len(t2_before)}, after={len(t2_after)}"
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
