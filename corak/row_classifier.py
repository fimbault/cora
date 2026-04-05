"""
corak.row_classifier
====================
Classify truth table rows into C1, C2, C3 based on ⊥ (UNDEF) presence.

Classes:
  C1 — Fully defined: no ⊥ in any column.
  C2 — Condition-indefinite: at least one ⊥ in condition columns.
       These rows cannot certify standard prime implicants.
  C3 — Outcome-indefinite: no ⊥ in conditions, at least one ⊥ in outcomes.
       These are structural don't-cares (DC_str ≠ DC_min).

Rows with ⊥ in both conditions AND outcomes are classified as C2
(the condition-indefiniteness dominates).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from .kleene import UNDEF


class RowClass(str, Enum):
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"


@dataclass
class RowInfo:
    """Classification result for a single truth-table row."""
    index: int | str
    row_class: RowClass
    undef_conditions: list[str] = field(default_factory=list)
    undef_outcomes: list[str] = field(default_factory=list)

    @property
    def is_c1(self) -> bool:
        return self.row_class == RowClass.C1

    @property
    def is_c2(self) -> bool:
        return self.row_class == RowClass.C2

    @property
    def is_c3(self) -> bool:
        return self.row_class == RowClass.C3


def classify_rows(
    data: pd.DataFrame,
    input_labels: list[str],
    output_labels: list[str],
    undef_value: int = UNDEF,
) -> dict[int | str, RowInfo]:
    """
    Classify every row of *data* into C1, C2, or C3.

    Parameters
    ----------
    data : pd.DataFrame
        The raw case-by-variable table. Must contain all input_labels and
        output_labels as columns.
    input_labels : list[str]
        Condition (input) column names.
    output_labels : list[str]
        Outcome (output) column names.
    undef_value : int
        The sentinel encoding structural indefiniteness (default -1).

    Returns
    -------
    dict mapping row index → RowInfo
    """
    result: dict[int | str, RowInfo] = {}

    for idx, row in data.iterrows():
        undef_conds = [
            col for col in input_labels
            if row[col] == undef_value
        ]
        undef_outs = [
            col for col in output_labels
            if row[col] == undef_value
        ]

        if undef_conds:
            cls = RowClass.C2
        elif undef_outs:
            cls = RowClass.C3
        else:
            cls = RowClass.C1

        result[idx] = RowInfo(
            index=idx,
            row_class=cls,
            undef_conditions=undef_conds,
            undef_outcomes=undef_outs,
        )

    return result


def classification_summary(
    classification: dict[int | str, RowInfo],
) -> pd.DataFrame:
    """
    Return a DataFrame summary of row classifications.

    Columns: index, row_class, undef_conditions, undef_outcomes, n_undef_conds, n_undef_outs
    """
    records = []
    for idx, info in classification.items():
        records.append({
            "index": idx,
            "row_class": info.row_class.value,
            "undef_conditions": ", ".join(info.undef_conditions) or "—",
            "undef_outcomes": ", ".join(info.undef_outcomes) or "—",
            "n_undef_conds": len(info.undef_conditions),
            "n_undef_outs": len(info.undef_outcomes),
        })
    return pd.DataFrame(records).set_index("index")


def filter_by_class(
    data: pd.DataFrame,
    classification: dict[int | str, RowInfo],
    row_class: RowClass,
) -> pd.DataFrame:
    """Return rows of *data* belonging to *row_class*."""
    indices = [
        idx for idx, info in classification.items()
        if info.row_class == row_class
    ]
    return data.loc[indices]
