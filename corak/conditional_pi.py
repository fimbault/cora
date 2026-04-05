"""
corak.conditional_pi
====================
Generation of conditional INUS structures (Tier 2) and
Kleene consistency intervals for Tier 1 prime implicants.

Conditional INUS structure:
    An INUS structure that holds if all ⊥-valued conditions in C2 rows
    contributing to its derivation resolve to specific values.

Kleene consistency interval [Cons_lower, Cons_upper]:
    Lower bound: fraction of covered cases with defined positive outcome.
    Upper bound: fraction of covered cases with defined OR ⊥ positive outcome.
    Width = k / n_total where k = count of covered C2 rows with ⊥ outcomes.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd

from .kleene import UNDEF, k_eval_minterm
from .row_classifier import RowClass, RowInfo


@dataclass
class ResolutionClause:
    """A single ⊥→{0,1} resolution for a C2 row."""
    case_index: int | str
    resolutions: dict[str, int]  # {column_name: resolved_value}

    def __str__(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in self.resolutions.items())
        return f"case {self.case_index}: [{parts}]"


@dataclass
class ConditionalImplicant:
    """
    A prime implicant valid under a specific resolution of ⊥-valued conditions.

    Attributes
    ----------
    implicant : str
        String representation of the INUS expression (e.g. "D*h").
    raw_implicant : tuple
        Raw internal representation from CORA.
    resolution_clauses : list[ResolutionClause]
        The specific ⊥→{0,1} assignments that make this implicant valid.
    cons_lower : float
        Worst-case consistency (all ⊥ outcomes resolve to 0).
    cons_upper : float
        Best-case consistency (all ⊥ outcomes resolve to 1).
    outputs : list[int]
        Output indices this implicant applies to (1-indexed, multi-output).
    output_labels : list[str]
        Output column names.
    """
    implicant: str
    raw_implicant: tuple
    resolution_clauses: list[ResolutionClause] = field(default_factory=list)
    cons_lower: float = 0.0
    cons_upper: float = 0.0
    outputs: list[int] = field(default_factory=list)
    output_labels: list[str] = field(default_factory=list)

    @property
    def audit_width(self) -> float:
        """Width of the Kleene consistency interval."""
        return round(self.cons_upper - self.cons_lower, 4)

    @property
    def is_structurally_sensitive(self) -> bool:
        """True when audit_width > 0.1 — substantial structural uncertainty."""
        return self.audit_width > 0.1

    def __str__(self) -> str:
        clause_str = "; ".join(str(r) for r in self.resolution_clauses)
        return (
            f"{self.implicant} "
            f"[if {clause_str}] "
            f"Cons=[{self.cons_lower:.2f}, {self.cons_upper:.2f}] "
            f"Δ={self.audit_width:.2f}"
        )

    def __repr__(self) -> str:
        return str(self)


# ---------------------------------------------------------------------------
# Consistency interval computation
# ---------------------------------------------------------------------------

def _pi_covers_row(raw_implicant: tuple, row_values: list[int]) -> bool:
    """
    Return True if *raw_implicant* covers *row_values*.

    raw_implicant is a tuple of frozensets (CORA's internal representation).
    A position is covered if row_values[i] is in raw_implicant[i].
    """
    for val, imp_set in zip(row_values, raw_implicant):
        if val not in imp_set:
            return False
    return True


def compute_kleene_interval(
    raw_implicant: tuple,
    data: pd.DataFrame,
    input_labels: list[str],
    output_labels: list[str],
    classification: dict[int | str, RowInfo],
    undef_value: int = UNDEF,
) -> tuple[float, float]:
    """
    Compute [Cons_lower, Cons_upper] for a prime implicant over C1 ∪ C2 rows.

    Cons_lower = |{covered C1∪C2 rows with outcome = 1}| / |covered C1∪C2 rows|
    Cons_upper = |{covered C1∪C2 rows with outcome ∈ {1, ⊥}}| / |covered C1∪C2 rows|

    For multi-output: all outcomes must be positive (conservative treatment).
    """
    n_total = 0
    n_lower = 0
    n_upper = 0

    for idx, info in classification.items():
        if info.row_class == RowClass.C3:
            continue  # C3 rows excluded from interval computation

        row = data.loc[idx]
        input_vals = [row[c] for c in input_labels]

        if not _pi_covers_row(raw_implicant, input_vals):
            continue

        n_total += 1

        # Check outcomes
        out_vals = [row[c] for c in output_labels]

        # Lower bound: all outcomes must be defined 1
        if all(v == 1 for v in out_vals):
            n_lower += 1

        # Upper bound: all outcomes must be 1 or ⊥
        if all(v == 1 or v == undef_value for v in out_vals):
            n_upper += 1

    if n_total == 0:
        return 0.0, 0.0

    return round(n_lower / n_total, 4), round(n_upper / n_total, 4)


def consistency_intervals_table(
    prime_implicants: list,
    data: pd.DataFrame,
    input_labels: list[str],
    output_labels: list[str],
    classification: dict[int | str, RowInfo],
    undef_value: int = UNDEF,
) -> pd.DataFrame:
    """
    Return a DataFrame of consistency intervals for all Tier 1 prime implicants.

    Columns: PI, Cons_def (CORA point estimate), Cons_lower, Cons_upper, audit_width
    """
    records = []
    for pi in prime_implicants:
        lo, hi = compute_kleene_interval(
            pi.raw_implicant, data, input_labels, output_labels,
            classification, undef_value,
        )
        try:
            cons_def = round(pi.inclusion_score(), 3)
        except Exception:
            cons_def = float("nan")

        records.append({
            "PI": pi.implicant,
            "Cons_def": cons_def,
            "Cons_lower": lo,
            "Cons_upper": hi,
            "audit_width": round(hi - lo, 4),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Pass 2: conditional prime implicant generation
# ---------------------------------------------------------------------------

def generate_conditional_prime_implicants(
    c2_rows: dict[int | str, RowInfo],
    tier1_implicants: list,
    data: pd.DataFrame,
    input_labels: list[str],
    output_labels: list[str],
    classification: dict[int | str, RowInfo],
    off_set_indices: list[int | str],
    cons_threshold: float = 0.75,
    undef_value: int = UNDEF,
) -> list[ConditionalImplicant]:
    """
    Generate Tier 2 conditional INUS structures.

    For each C2 row, enumerate all possible 0/1 resolutions of its ⊥-valued
    conditions. For each resolution that does not land in the OFF-set, check
    whether any Tier 1 prime implicant covers the resolved row. If the
    implicant's consistency meets the threshold, emit a ConditionalImplicant.

    Parameters
    ----------
    c2_rows : dict
        Mapping of row index → RowInfo for C2 rows only.
    tier1_implicants : list
        Standard (Tier 1) prime implicants from CorakContext.
    data : pd.DataFrame
        Full dataset.
    input_labels, output_labels : list[str]
        Column names.
    classification : dict
        Full row classification (C1/C2/C3).
    off_set_indices : list
        Indices of C1 OFF-set rows (used to check resolution conflicts).
    cons_threshold : float
        Minimum Cons_lower required to emit a conditional implicant.
    undef_value : int
        Sentinel for ⊥ (default -1).

    Returns
    -------
    list[ConditionalImplicant]
    """
    pi_cond: list[ConditionalImplicant] = []
    seen: set[tuple] = set()  # deduplicate (implicant_str, resolution_key)

    # Build OFF-set input patterns for fast lookup
    off_patterns = []
    for idx in off_set_indices:
        row = data.loc[idx]
        off_patterns.append(tuple(row[c] for c in input_labels))

    for idx, info in c2_rows.items():
        row = data.loc[idx]
        undef_cols = info.undef_conditions
        k = len(undef_cols)

        for bits in itertools.product([0, 1], repeat=k):
            # Build resolved input vector
            resolved = list(row[c] for c in input_labels)
            for col, bit in zip(undef_cols, bits):
                col_idx = input_labels.index(col)
                resolved[col_idx] = bit

            resolved_tuple = tuple(resolved)

            # Skip if this resolved row is in the OFF-set
            if resolved_tuple in off_patterns:
                continue

            # Find Tier 1 PIs that cover this resolved row
            for pi in tier1_implicants:
                if not _pi_covers_row(pi.raw_implicant, list(resolved)):
                    continue

                # Compute consistency interval for this PI
                lo, hi = compute_kleene_interval(
                    pi.raw_implicant, data, input_labels, output_labels,
                    classification, undef_value,
                )

                if lo < cons_threshold:
                    continue

                # Deduplicate
                resolution_dict = dict(zip(undef_cols, bits))
                key = (pi.implicant, idx, tuple(sorted(resolution_dict.items())))
                if key in seen:
                    continue
                seen.add(key)

                clause = ResolutionClause(
                    case_index=idx,
                    resolutions=resolution_dict,
                )
                out_labels = getattr(pi, "output_labels", output_labels)
                out_indices = getattr(pi, "outputs", list(range(1, len(output_labels) + 1)))

                cpi = ConditionalImplicant(
                    implicant=pi.implicant,
                    raw_implicant=pi.raw_implicant,
                    resolution_clauses=[clause],
                    cons_lower=lo,
                    cons_upper=hi,
                    outputs=list(out_indices),
                    output_labels=list(out_labels),
                )
                pi_cond.append(cpi)

    return pi_cond
