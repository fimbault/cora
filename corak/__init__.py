"""
CORAK — Combinational Regularity Analysis with Kleene-Valued Conditions
=======================================================================
Extension of CORA (Thiem, Mkrtchyan & Sebechlebská 2022) for data containing
structurally indefinite conditions encoded as ⊥ (undef_value=-1 by default).

Quick start
-----------
    from corak import CorakContext
    import pandas as pd

    df = pd.DataFrame(...)          # include -1 for structurally indefinite values
    ctx = CorakContext(df, ["Y"], undef_value=-1)

    pi_std    = ctx.get_prime_implicants()           # Tier 1 (CORA-equivalent)
    classif   = ctx.get_row_classification()          # C1/C2/C3 per row
    intervals = ctx.get_consistency_intervals()       # Kleene [lo, hi] per PI
    pi_cond   = ctx.get_conditional_prime_implicants()# Tier 2 conditional INUS
    sens      = ctx.sensitivity_table()               # α-recoding sensitivity

References
----------
Thiem, A., Mkrtchyan, L., & Sebechlebská, Z. (2022). Combinational Regularity
Analysis (CORA). BMC Medical Research Methodology, 22(1), 333.
"""

from .kleene import UNDEF, k_and, k_or, k_not, k_eval_minterm
from .row_classifier import RowClass, RowInfo, classify_rows, classification_summary
from .conditional_pi import ResolutionClause, ConditionalImplicant
from .context import CorakContext

__all__ = [
    # Core class
    "CorakContext",
    # Kleene logic
    "UNDEF",
    "k_and",
    "k_or",
    "k_not",
    "k_eval_minterm",
    # Row classification
    "RowClass",
    "RowInfo",
    "classify_rows",
    "classification_summary",
    # Conditional implicants
    "ResolutionClause",
    "ConditionalImplicant",
]

__version__ = "0.1.0"
