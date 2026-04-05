"""
corak.context
=============
CorakContext: CORAK analysis using composition over CORA's OptimizationContext.

Architecture: CorakContext holds a raw DataFrame (including ⊥ values) and
creates an internal CORA OptimizationContext restricted to C1 rows.
All standard CORA methods are delegated to the internal context.
CORAK-specific methods (classification, intervals, conditional PIs,
sensitivity table) operate on the full raw DataFrame.
"""

from __future__ import annotations

import warnings

import pandas as pd

from cora.prime_implicants import OptimizationContext

from .kleene import UNDEF
from .row_classifier import (
    RowClass,
    RowInfo,
    classify_rows,
    classification_summary,
)
from .conditional_pi import (
    ConditionalImplicant,
    consistency_intervals_table,
    generate_conditional_prime_implicants,
)


class CorakContext:
    """
    CORAK analysis context.

    Wraps CORA's OptimizationContext, restricting Tier 1 analysis to C1
    (fully defined) rows and adding Kleene-valued Tier 2 analysis.

    Parameters
    ----------
    data : pd.DataFrame
    output_labels : list[str]
    input_labels : list[str] | None
    case_col : str | None
    n_cut : int
    inc_score1 : float
    algorithm : str  — "ON-DC" or "ON-OFF"
    undef_value : int  — sentinel for ⊥ (default -1)
    return_conditional : bool
    cons_threshold_conditional : float | None
    """

    def __init__(
        self,
        data: pd.DataFrame,
        output_labels: list[str],
        input_labels: list[str] | None = None,
        case_col: str | None = None,
        n_cut: int = 1,
        inc_score1: float = 1.0,
        inc_score2: float | None = None,
        U: int | None = None,
        rename_columns: bool = False,
        algorithm: str = "ON-DC",
        undef_value: int = UNDEF,
        return_conditional: bool = True,
        cons_threshold_conditional: float | None = None,
    ) -> None:
        if undef_value in (0, 1):
            raise ValueError(f"undef_value must not be 0 or 1 (got {undef_value}).")

        self._undef_value = undef_value
        self._return_conditional = return_conditional
        self._cons_threshold = (
            cons_threshold_conditional
            if cons_threshold_conditional is not None
            else inc_score1
        )
        self._algorithm = algorithm
        self._inc_score1 = inc_score1
        self._inc_score2 = inc_score2
        self._U = U

        # Raw data (with ⊥ values)
        self._raw_data = data.copy().reset_index(drop=True)
        self._output_labels = list(output_labels)

        # Infer input labels
        if input_labels is not None:
            self._input_labels = list(input_labels)
        else:
            exclude = set(output_labels) | ({case_col} if case_col else set())
            self._input_labels = [c for c in data.columns if c not in exclude]

        # Classify rows
        self._classification: dict[int, RowInfo] = classify_rows(
            self._raw_data,
            self._input_labels,
            self._output_labels,
            self._undef_value,
        )
        self._c2_rows = {i: v for i, v in self._classification.items() if v.is_c2}
        self._c3_rows = {i: v for i, v in self._classification.items() if v.is_c3}

        # Build C1-only data for CORA, dropping constant condition columns
        c1_data, self._active_inputs = self._build_c1_data()

        # Internal CORA context (Tier 1)
        self._cora = OptimizationContext(
            data=c1_data,
            output_labels=output_labels,
            input_labels=self._active_inputs,
            case_col=case_col,
            n_cut=n_cut,
            inc_score1=inc_score1,
            inc_score2=inc_score2,
            U=U,
            rename_columns=rename_columns,
            algorithm=algorithm,
        )

        # Cached CORAK results
        self._conditional_pis: list[ConditionalImplicant] | None = None
        self._intervals: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _build_c1_data(self) -> tuple[pd.DataFrame, list[str]]:
        c1_idx = [i for i, v in self._classification.items() if v.is_c1]
        if not c1_idx:
            raise ValueError("No fully defined (C1) rows in dataset.")
        c1 = self._raw_data.loc[c1_idx].copy()

        # Drop constant condition columns (CORA rejects them)
        active = []
        for col in self._input_labels:
            if c1[col].nunique() > 1:
                active.append(col)
            else:
                warnings.warn(
                    f"Condition '{col}' is constant across C1 rows — "
                    f"excluded from Tier 1 minimisation.",
                    UserWarning, stacklevel=3,
                )
        if not active:
            raise ValueError("All condition columns constant in C1 rows.")
        return c1, active

    # ------------------------------------------------------------------
    # Delegation to CORA (standard interface)
    # ------------------------------------------------------------------

    def get_prime_implicants(self) -> tuple:
        """Tier 1: prime implicants on C1 rows (identical to CORA on C1)."""
        return self._cora.get_prime_implicants()

    def get_irredundant_sums(self, max_depth: int | None = None):
        """Tier 1: irredundant sums (single-output)."""
        return self._cora.get_irredundant_sums(max_depth=max_depth)

    def get_irredundant_systems(self):
        """Tier 1: irredundant systems (multi-output)."""
        return self._cora.get_irredundant_systems()

    def prime_implicant_chart(self) -> pd.DataFrame:
        return self._cora.prime_implicant_chart()

    def pi_details(self) -> pd.DataFrame:
        return self._cora.pi_details()

    def system_details(self) -> pd.DataFrame:
        return self._cora.system_details()

    def get_preprocessed_data(self, raw: bool = False) -> pd.DataFrame:
        """Truth table derived from C1 rows."""
        return self._cora.get_preprocessed_data(raw=raw)

    # ------------------------------------------------------------------
    # CORAK-specific methods
    # ------------------------------------------------------------------

    def get_row_classification(self) -> pd.DataFrame:
        """DataFrame classifying each row as C1, C2, or C3."""
        return classification_summary(self._classification)

    def get_consistency_intervals(self) -> pd.DataFrame:
        """
        Kleene consistency intervals for all Tier 1 prime implicants.

        Columns: PI, Cons_def, Cons_lower, Cons_upper, audit_width

        Proposition 1:
          (a) Cons_lower ≤ Cons_def ≤ Cons_upper
          (b) Both bounds are sharp
          (c) audit_width = k / n_total, non-decreasing in k
          (d) audit_width = 0 when no C2 rows are covered
        """
        if self._intervals is not None:
            return self._intervals
        pis = list(self.get_prime_implicants())
        if not pis:
            self._intervals = pd.DataFrame(
                columns=["PI","Cons_def","Cons_lower","Cons_upper","audit_width"])
            return self._intervals
        self._intervals = consistency_intervals_table(
            pis, self._raw_data, self._active_inputs,
            self._output_labels, self._classification, self._undef_value,
        )
        return self._intervals

    def get_conditional_prime_implicants(self) -> list[ConditionalImplicant]:
        """
        Tier 2: conditional INUS structures (Pass 2 over C2 rows).

        Each ConditionalImplicant:
          - implicant:         INUS expression
          - resolution_clauses: C2 row + ⊥→{0,1} assignment
          - cons_lower/upper:  Kleene consistency interval
          - audit_width:       k/n_total sensitivity signal
        """
        if self._conditional_pis is not None:
            return self._conditional_pis
        if not self._return_conditional or not self._c2_rows:
            self._conditional_pis = []
            return []
        pis = list(self.get_prime_implicants())
        if not pis:
            self._conditional_pis = []
            return []

        off_indices = [
            i for i, info in self._classification.items()
            if info.is_c1
            and all(self._raw_data.loc[i, c] == 0 for c in self._output_labels)
        ]
        self._conditional_pis = generate_conditional_prime_implicants(
            c2_rows=self._c2_rows,
            tier1_implicants=pis,
            data=self._raw_data,
            input_labels=self._active_inputs,
            output_labels=self._output_labels,
            classification=self._classification,
            off_set_indices=off_indices,
            cons_threshold=self._cons_threshold,
            undef_value=self._undef_value,
        )
        return self._conditional_pis

    def sensitivity_table(
        self, alphas: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Sensitivity of Tier 1 solutions to α-reclassification (⊥ → 0).

        α = 0.0 → CORAK Tier 1 (no recoding)
        α = 1.0 → standard CORA (all ⊥ recoded to 0)

        Returns DataFrame: alpha, output, solution, cons_def
        """
        if alphas is None:
            alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

        all_cols = self._active_inputs + self._output_labels
        undef_positions = [
            (i, col)
            for i in self._raw_data.index
            for col in all_cols
            if self._raw_data.loc[i, col] == self._undef_value
        ]

        records = []
        for alpha in alphas:
            recoded = self._raw_data.copy()
            n_recode = int(round(alpha * len(undef_positions)))
            for i, col in undef_positions[:n_recode]:
                recoded.loc[i, col] = 0

            mask = (recoded[all_cols] == self._undef_value).any(axis=1)
            clean = recoded[~mask].copy()

            if clean.empty:
                for out in self._output_labels:
                    records.append({"alpha": alpha, "output": out,
                                    "solution": "—", "cons_def": float("nan")})
                continue

            active = [c for c in self._active_inputs if clean[c].nunique() > 1]
            if not active:
                for out in self._output_labels:
                    records.append({"alpha": alpha, "output": out,
                                    "solution": "—", "cons_def": float("nan")})
                continue

            try:
                ctx_a = OptimizationContext(
                    clean, output_labels=self._output_labels,
                    input_labels=active, algorithm=self._algorithm,
                    inc_score1=self._inc_score1,
                )
                is_multi = len(self._output_labels) > 1
                if is_multi:
                    systems = ctx_a.get_irredundant_systems()
                    sol = systems[0] if systems else None
                    for j, out in enumerate(self._output_labels):
                        if sol is None:
                            records.append({"alpha":alpha,"output":out,
                                            "solution":"—","cons_def":float("nan")})
                        else:
                            impls = sol.system_multiple[j]
                            sol_str = " + ".join(im.implicant for im in impls) or "—"
                            records.append({
                                "alpha": alpha, "output": out,
                                "solution": sol_str,
                                "cons_def": round(sol.inclusion_score(), 3),
                            })
                else:
                    sums = ctx_a.get_irredundant_sums()
                    out = self._output_labels[0]
                    if sums:
                        s = sums[0]
                        sol_str = " + ".join(str(im.implicant) for im in s.system)
                        records.append({
                            "alpha": alpha, "output": out,
                            "solution": sol_str,
                            "cons_def": round(s.inclusion_score(), 3),
                        })
                    else:
                        records.append({"alpha":alpha,"output":out,
                                        "solution":"—","cons_def":float("nan")})
            except Exception as exc:
                warnings.warn(f"sensitivity_table α={alpha}: {exc}")
                for out in self._output_labels:
                    records.append({"alpha":alpha,"output":out,
                                    "solution":"ERROR","cons_def":float("nan")})

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def undef_value(self) -> int:
        return self._undef_value

    @property
    def input_labels(self) -> list[str]:
        return self._active_inputs

    @property
    def output_labels(self) -> list[str]:
        return self._output_labels

    @property
    def n_c1(self) -> int:
        return sum(1 for v in self._classification.values() if v.is_c1)

    @property
    def n_c2(self) -> int:
        return len(self._c2_rows)

    @property
    def n_c3(self) -> int:
        return len(self._c3_rows)

    def __repr__(self) -> str:
        return (
            f"CorakContext(n={len(self._raw_data)}, "
            f"C1={self.n_c1}, C2={self.n_c2}, C3={self.n_c3}, "
            f"outputs={self._output_labels})"
        )
