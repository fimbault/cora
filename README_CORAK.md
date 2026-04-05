# CORAK

**Combinational Regularity Analysis with Kleene-Valued Conditions**

CORAK extends [CORA](https://github.com/PoliUniLu/cora) with a third truth
value ⊥ (*structural indefiniteness*) — the formal distinction between
*defined absence* (a condition is genuinely not present) and *inapplicability*
(a condition is outside the conceptual domain of a case). It implements
Kleene's strong three-valued logic and functions simultaneously as a
minimisation method and as a diagnostic for the *shared causal space
assumption*.

> **Integration note.** CORAK is designed as a composable extension of CORA
> and makes no modifications to the CORA codebase. It is intended as a
> proposal toward the upstream CORA repository at
> [PoliUniLu/cora](https://github.com/PoliUniLu/cora).

---

## Installation

Requires CORA ≥ 2.0.11:

```bash
pip install cora
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

---

## Quick start

```python
import pandas as pd
from corak import CorakContext, UNDEF

df = pd.DataFrame([
    [1, 1, 1], [0, 0, 0],
    [1, UNDEF, 1],   # C2: condition inapplicable to this case
    [1, 0, UNDEF],   # C3: outcome structurally unobservable
], columns=['A', 'B', 'Y'])

ctx = CorakContext(df, ['Y'], undef_value=UNDEF)

# Tier 1 — certified on fully defined (C1) rows
print(ctx.get_prime_implicants())

# Tier 2 — conditional structures for C2 rows
print(ctx.get_conditional_prime_implicants())

# Kleene consistency intervals
print(ctx.get_consistency_intervals())

# Sensitivity to ⊥ misclassification (Proposition 2)
print(ctx.sensitivity_table(alphas=[0.0, 0.25, 0.5, 0.75, 1.0]))
```

---

## Key concepts

| Row class | Condition columns | Outcome columns | Treatment |
|---|---|---|---|
| **C1** | All defined | All defined | Standard CORA — enters Tier 1 |
| **C2** | At least one ⊥ | Any | Excluded from Tier 1; generates Tier 2 conditional structures |
| **C3** | All defined | At least one ⊥ | Excluded from ON-set and OFF-set (structural don't-care) |

Doubly-indefinite rows (⊥ in both condition and outcome) are classified C2.

---

## Formal results

| Result | Statement |
|---|---|
| **Theorem 1** | Minimisation don't-cares (DC_min) ≠ structural don't-cares (DC_str) |
| **Theorem 2** | CORA under ⊥→0 is biased in both directions: omission and commission |
| **Lemma 1** | When no ⊥ values are present, CORAK ≡ CORA exactly |
| **Proposition 1** | Kleene consistency interval: containment, sharpness, monotonicity, collapse-to-point |
| **Proposition 2** | Misclassification error model: Type II (⊥→0) silent; Type I (0→⊥) visible via Tier 2 flags; robustness condition via Cons_lower ≥ τ |

---

## Tests

```bash
pytest tests/ -v
```

46 tests covering:
- Kleene algebra (10)
- C1/C2/C3 row classification (4)
- All five formal results (9)
- Multi-morbidity case study regressions (7)
- Swiss Minaret Vote case study regressions (5)
- Proposition 2 misclassification error model (5)
- Gap-closing / implementation behaviour (3)
- Theorem 1 properness and Stage 2 multi-row (3)

---

## Architecture

```
corak/
├── kleene.py           Kleene connectives (k_and, k_or, k_not, k_eval_minterm)
├── row_classifier.py   C1 / C2 / C3 row classification
├── conditional_pi.py   Tier 2: conditional prime implicants + consistency intervals
└── context.py          CorakContext — public API (wraps CORA OptimizationContext)
```

`CorakContext` uses **composition**: it holds an internal CORA
`OptimizationContext` that operates on C1 rows only. No upstream code is
modified. All standard CORA methods pass through unchanged.

---

## Examples

`examples/corak/` contains two simulation case studies:

- `simulation_study.py` — multi-morbidity (clinical data)
- `minaret_simulation_study.py` — Swiss Minaret Vote (political science)
- `corak_simulation_cases.ipynb` — unified notebook for both cases

Reproduce the multi-morbidity simulation:

```bash
python examples/corak/simulation_study.py --runs 500 --seed 2024
```

---

## License

GNU General Public License v3.0 — same licence as CORA.
