"""
examples/corak/minaret_simulation_study.py
==========================================
Monte Carlo simulation — Section 5.4 of the CORAK paper.
Dataset: Swiss Minaret Vote (Baumgartner & Epple 2014, SMR 43(2):280-312)
N = 26 Swiss cantons. Ground truth: X → M (new xenophobia → minaret ban)
Consistency = 1.00, Coverage = 0.86.

Design
------
Two mechanisms introduce ⊥ at rate r simultaneously:

  Mechanism A (C2 rows): X=⊥ for a fraction r of X=1 cantons.
    CORA codes X=⊥ as X=0 → spurious (X=0, M=1) ON-set entries.
    Blocks X→M by creating (X=1, M=0) contradictions via Mechanism B.

  Mechanism B (C3 rows): M=⊥ for a fraction r of X=1, M=1 cantons.
    CORA codes M=⊥ as M=0 → creates (X=1, M=0) OFF-set entries.
    Drops X→M consistency below threshold → CORA fails.

Recovery metric: X→M appears as a STANDALONE prime implicant (#X).
Compound implicants (a*X, T*x, etc.) are NOT counted as recovery.

Usage
-----
    python3 minaret_simulation_study.py                  # default: 200 runs
    python3 minaret_simulation_study.py --runs 500       # more runs
    python3 minaret_simulation_study.py --seed 42        # different seed
    python3 minaret_simulation_study.py --fast           # 50 runs (quick check)

Output: minaret_simulation_results.csv in the same directory.
"""

import sys, os, argparse
import pandas as pd
import numpy as np

# Allow import from installed package OR from local repo
try:
    from corak import CorakContext
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))
    from corak import CorakContext

# ── Dataset ───────────────────────────────────────────────────────────────────
CANTONS = [
    'LU','UR','SZ','OW','NW','AR','AI','GL','ZG','SO','SG','AG',
    'VD','NE','GE','GR','TG','ZH','BE','FR','BS','BL','SH','TI','VS','JU'
]

BASE_DATA = {
    'A': [1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,1,1,1,1,1,0,0,0,0],
    'L': [0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,0,0,1,1,0,1,1,1,0,0,1],
    'S': [1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,1,1,1,1,0,0,0,1,0,0,0],
    'T': [1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,1,1,0,1,1,0,0,0,0,1,1],
    'X': [1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,1,1,1,1,0,0,1,1,1,0,0],
    'M': [1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0,1,1,1,1,1,0,1,1,1,1,1]
}
CONDITIONS = ['A', 'L', 'S', 'T', 'X']
OUTCOME = ['M']
UNDEF = -1


def load_base():
    return pd.DataFrame(BASE_DATA, index=CANTONS)


def perturb(base_df, rate, rng):
    """Introduce ⊥ via Mechanism A (X=⊥) and Mechanism B (M=⊥)."""
    df = base_df.copy()
    x1 = df[df['X'] == 1].index.tolist()
    n_a = round(len(x1) * rate)
    if n_a > 0:
        df.loc[rng.choice(x1, size=n_a, replace=False), 'X'] = UNDEF

    x1m1 = df[(df['X'] == 1) & (df['M'] == 1)].index.tolist()
    n_b = round(len(x1m1) * rate)
    if n_b > 0:
        df.loc[rng.choice(x1m1, size=min(n_b, len(x1m1)), replace=False), 'M'] = UNDEF
    return df


def x_recovers_standalone(pis):
    """True iff X=1 alone — no other conditions — is a prime implicant for M."""
    for pi in pis:
        raw = pi.raw_implicant  # tuple of frozensets (one per condition)
        if (raw[4] == frozenset({1}) and
                all(raw[i] == frozenset({0, 1}) for i in range(4))):
            return True
    return False


def run_one(df_perturbed, undef_as_zero=False):
    df = df_perturbed.replace(UNDEF, 0) if undef_as_zero else df_perturbed
    try:
        ctx = CorakContext(
            df.reset_index(drop=True),
            output_labels=OUTCOME,
            input_labels=CONDITIONS,
            undef_value=UNDEF,
            inc_score1=1.0
        )
        pis = ctx.get_prime_implicants()
        recovered = x_recovers_standalone(pis)
        t2_count = 0 if undef_as_zero else len(ctx.get_conditional_prime_implicants())
        return recovered, t2_count > 0
    except Exception:
        return False, False


def run_simulation(n_sims=200, rates=None, seed=2024):
    if rates is None:
        rates = [0.00, 0.05, 0.10, 0.20, 0.30]

    base_df = load_base()
    rng = np.random.default_rng(seed)
    results = []

    print(f"Swiss Minaret simulation — {n_sims} runs/rate, seed={seed}")
    print(f"{'Rate':>6}  {'CORA X→M':>10}  {'CORAK T1':>10}  {'Tier2':>8}  {'AvgC2':>6}  {'AvgC3':>6}")
    print("-" * 58)

    for rate in rates:
        cora_rec = corak_rec = t2_count = 0
        c2s, c3s = [], []

        for _ in range(n_sims):
            df_p = perturb(base_df, rate, rng)
            c2s.append((df_p['X'] == UNDEF).sum())
            c3s.append((df_p['M'] == UNDEF).sum())
            c, _ = run_one(df_p, undef_as_zero=True)
            k, t2 = run_one(df_p, undef_as_zero=False)
            cora_rec += c
            corak_rec += k
            t2_count += t2

        row = {
            'rate': rate,
            'cora_recovery': round(cora_rec / n_sims, 3),
            'corak_tier1_recovery': round(corak_rec / n_sims, 3),
            'tier2_flag_rate': round(t2_count / n_sims, 3),
            'avg_c2': round(float(np.mean(c2s)), 1),
            'avg_c3': round(float(np.mean(c3s)), 1),
        }
        results.append(row)
        print(f"{rate*100:>5.0f}%  {row['cora_recovery']:>10.3f}  {row['corak_tier1_recovery']:>10.3f}"
              f"  {row['tier2_flag_rate']:>8.3f}  {row['avg_c2']:>6.1f}  {row['avg_c3']:>6.1f}")

    return pd.DataFrame(results)


def save_results(df, out_dir):
    path = os.path.join(out_dir, 'minaret_simulation_results.csv')
    df.to_csv(path, index=False)
    print(f"  → {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CORAK Minaret simulation — Section 5.4'
    )
    parser.add_argument('--runs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--rates', nargs='+', type=float,
                        default=[0.0, 0.05, 0.10, 0.20, 0.30])
    parser.add_argument('--fast', action='store_true',
                        help='Quick run: 50 sims/rate')
    parser.add_argument('--out', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    if args.fast:
        args.runs = 50
        print("[Fast mode: 50 simulations/rate]")

    results = run_simulation(n_sims=args.runs, rates=args.rates, seed=args.seed)
    print(f"\nSaving results to {args.out}/")
    save_results(results, args.out)
    print("Done.")
