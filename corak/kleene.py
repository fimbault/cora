"""
corak.kleene
============
Kleene strong three-valued logic operations for CORAK.

Values: 0 (false), 1 (true), UNDEF (-1, structural indefiniteness).

Key properties (used in Boolean minimisation):
  0 ∧ ⊥ = 0   (false absorbs under conjunction)
  1 ∨ ⊥ = 1   (true absorbs under disjunction)
  1 ∧ ⊥ = ⊥   (⊥ propagates through non-absorbing operations)
  ¬⊥    = ⊥   (negation preserves indefiniteness)
"""

# Sentinel for structural indefiniteness (⊥)
UNDEF: int = -1


def k_and(a: int, b: int) -> int:
    """Kleene conjunction: 0 absorbs, ⊥ propagates otherwise."""
    if a == 0 or b == 0:
        return 0
    if a == UNDEF or b == UNDEF:
        return UNDEF
    return 1


def k_or(a: int, b: int) -> int:
    """Kleene disjunction: 1 absorbs, ⊥ propagates otherwise."""
    if a == 1 or b == 1:
        return 1
    if a == UNDEF or b == UNDEF:
        return UNDEF
    return 0


def k_not(a: int) -> int:
    """Kleene negation: ⊥ maps to ⊥."""
    if a == UNDEF:
        return UNDEF
    return 1 - a


def k_eval_minterm(row: tuple[int, ...], implicant: tuple[int, ...]) -> int:
    """
    Evaluate whether *row* satisfies *implicant* under Kleene logic.

    An implicant is a partial assignment: UNDEF positions are don't-cares
    (the literal is dropped from the conjunction). Non-UNDEF positions require
    the row value to match.

    Returns:
        1   if the row definitely satisfies the implicant
        0   if the row definitely does not satisfy it
        UNDEF if the outcome is indeterminate (row has ⊥ in a required position)
    """
    result = 1
    for r_val, impl_val in zip(row, implicant):
        if impl_val == UNDEF:
            continue  # don't-care position, skip
        if r_val == UNDEF:
            # Row value is indefinite — result becomes indeterminate unless
            # already known false
            result = k_and(result, UNDEF)
        else:
            match = 1 if r_val == impl_val else 0
            result = k_and(result, match)
        if result == 0:
            return 0  # short-circuit
    return result


def is_defined(v: int) -> bool:
    """Return True iff v is a defined value (0 or 1)."""
    return v in (0, 1)
