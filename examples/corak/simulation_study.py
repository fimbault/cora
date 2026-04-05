"""
examples/corak/simulation_study.py
===================================
Reproducible Monte Carlo simulation comparing CORA and CORAK under
increasing rates of structural indefiniteness (⊥).

Generates Section 5.3 Table of the CORAK paper.

Design
------
Ground truth: DEP = D·H  (joint sufficiency of D and H)

Two ⊥ mechanisms are introduced simultaneously at rate *r*:

  C2 rows (condition-indefinite): H=⊥ for some cases.
    CORA codes H=⊥ as H=0.
    Affected cases: D=1, H=1 (true), DEP=1 — BP not measured.

  C3 rows (outcome-indefinite): DEP=⊥ for some cases.
    CORA codes DEP=⊥ as DEP=0.
    Affected cases: D=1, H=1, DEP=1 (true) — PHQ-9 not administered.

Bias mechanisms demonstrated:
  (A) C3 rows coded as DEP=0 deflate CORA's consistency score:
      CORA sees some true positives as negatives → lower inclusion score.
      CORAK Cons_def (C1 only) remains correct at 1.00.
      → Metric: consistency deflation = CORAK_Cons_def − CORA_score

  (B) Tier 2 flags: CORAK detects ⊥ and signals structural uncertainty.
      Flag rate = % of runs with any Tier 2 conditional PI.

Reproducibility
---------------
All results are saved to simulation_results.json with full metadata.
Re-run with the same seed to reproduce exactly.

Usage
-----
    python3 examples/corak/simulation_study.py           # 500 sims
    python3 examples/corak/simulation_study.py --fast    # 100 sims
    python3 examples/corak/simulation_study.py --seed 99 # custom seed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import cora as cora_pkg
from corak import CorakContext, UNDEF


# ─────────────────────────────────────────────────────────────────
# Ground truth & constants
# ─────────────────────────────────────────────────────────────────
CONDITIONS     = ['D', 'H', 'K', 'L', 'C']
OUTCOME        = ['DEP']
TARGET_PI      = 'DH'   # normalized: D·H is the true sufficient structure

GROUND_TRUTH_STR = "DEP = D·H  (D=1 and H=1 are jointly sufficient)"


# ─────────────────────────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────────────────────────

def generate_dataset(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Stratified dataset from ground truth DEP = D·H.
    K, L, C are noise conditions.
    Ensures each (D,H) cell has at least 2 cases.
    """
    rows = []
    per_cell = max(2, n // 4)
    for D in [0, 1]:
        for H in [0, 1]:
            count = per_cell + int(rng.integers(0, 3))
            for _ in range(count):
                K   = int(rng.integers(0, 2))
                L   = int(rng.integers(0, 2))
                C   = int(rng.integers(0, 2))
                DEP = int(D == 1 and H == 1)
                rows.append({'D': D, 'H': H, 'K': K, 'L': L, 'C': C, 'DEP': DEP})
    df = pd.DataFrame(rows)
    return df.sample(
        n=min(n, len(df)),
        random_state=int(rng.integers(0, 2**31)),
    ).reset_index(drop=True)


def introduce_undef(
    df: pd.DataFrame,
    rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Introduce structural indefiniteness at *rate* via two mechanisms:

    Mechanism A — C2 rows: replace H=1 with UNDEF for a fraction of cases.
      Simulates hypertensive patients whose BP was not measured.
      CORA error: H=⊥ → H=0, placing D=1 cases in wrong ON-set region.

    Mechanism B — C3 rows: replace DEP=1 with UNDEF for a fraction of
      D=1,H=1 cases. Simulates future DEP unknown (PHQ-9 not administered).
      CORA error: DEP=⊥ → DEP=0, true positives enter OFF-set.
      This directly deflates CORA's consistency score for D·H.
    """
    df_out = df.copy()
    if rate == 0.0:
        return df_out

    # --- Mechanism A: H=1 → H=⊥ ---
    eligible_h = df_out.index[df_out['H'] == 1].tolist()
    if eligible_h:
        n_h = max(1, int(round(rate * len(eligible_h))))
        n_h = min(n_h, len(eligible_h))
        targets_h = rng.choice(eligible_h, size=n_h, replace=False)
        df_out.loc[targets_h, 'H'] = UNDEF

    # --- Mechanism B: DEP=1 (D=1,H=1 cases) → DEP=⊥ ---
    # Only from remaining C1 ON rows (where H was not already set to UNDEF)
    eligible_dep = df_out.index[
        (df_out['D'] == 1) & (df_out['H'] == 1) & (df_out['DEP'] == 1)
    ].tolist()
    if eligible_dep:
        n_dep = max(1, int(round(rate * len(eligible_dep))))
        n_dep = min(n_dep, len(eligible_dep))
        targets_dep = rng.choice(eligible_dep, size=n_dep, replace=False)
        df_out.loc[targets_dep, 'DEP'] = UNDEF

    return df_out


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def normalize_pi(s: str) -> str:
    return s.lstrip('#').replace('*', '')


# ─────────────────────────────────────────────────────────────────
# Single simulation run
# ─────────────────────────────────────────────────────────────────

def run_one(
    n: int,
    rate: float,
    rng: np.random.Generator,
) -> dict:
    df_full  = generate_dataset(n, rng)
    df_undef = introduce_undef(df_full, rate, rng)

    result = dict(
        cora_dh_found=False,
        corak_dh_found=False,
        cora_cons=float('nan'),
        corak_cons_def=float('nan'),
        corak_t2_count=0,
        n_c2=0,
        n_c3=0,
    )

    # ── CORA (⊥→0) ───────────────────────────────────────────
    df_cora = df_undef.replace(UNDEF, 0)
    active = [c for c in CONDITIONS if df_cora[c].nunique() > 1]
    if active:
        try:
            ctx_c = cora_pkg.OptimizationContext(
                df_cora, OUTCOME, input_labels=active,
                inc_score1=0.75, algorithm='ON-OFF',
            )
            pis = list(ctx_c.get_prime_implicants())
            if pis:
                norm = {normalize_pi(str(p)) for p in pis}
                result['cora_dh_found'] = TARGET_PI in norm
                result['cora_cons'] = float(
                    np.mean([p.inclusion_score() for p in pis])
                )
        except Exception:
            pass

    # ── CORAK ────────────────────────────────────────────────
    try:
        ctx_k = CorakContext(
            df_undef, OUTCOME, input_labels=CONDITIONS,
            inc_score1=0.75, algorithm='ON-OFF',
            undef_value=UNDEF,
            return_conditional=True,
            cons_threshold_conditional=0.5,
        )
        result['n_c2'] = ctx_k.n_c2
        result['n_c3'] = ctx_k.n_c3

        pis_k = list(ctx_k.get_prime_implicants())
        if pis_k:
            norm_k = {normalize_pi(str(p)) for p in pis_k}
            result['corak_dh_found'] = TARGET_PI in norm_k

        intervals = ctx_k.get_consistency_intervals()
        if not intervals.empty:
            result['corak_cons_def'] = float(intervals['Cons_def'].mean())

        # Tier 2: count conditional implicants
        pi_cond = ctx_k.get_conditional_prime_implicants()
        result['corak_t2_count'] = len(pi_cond)

    except Exception:
        pass

    return result


# ─────────────────────────────────────────────────────────────────
# Main simulation loop
# ─────────────────────────────────────────────────────────────────

def run_simulation(
    n_cases: int = 40,
    n_sims: int = 500,
    rates: list[float] | None = None,
    seed: int = 2024,
) -> dict:
    """
    Run the full Monte Carlo simulation.

    Returns a dict with:
      - 'table': pd.DataFrame (main results)
      - 'metadata': dict (run configuration)
      - 'raw': list of per-run dicts (for reanalysis)
    """
    if rates is None:
        rates = [0.0, 0.05, 0.10, 0.20, 0.30]

    print(f"\n{'='*68}")
    print(f"CORAK Simulation Study — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Ground truth:  {GROUND_TRUTH_STR}")
    print(f"n={n_cases} cases/dataset | {n_sims} simulations/rate | seed={seed}")
    print(f"⊥ mechanisms: A=H→⊥ (C2 rows)  B=DEP→⊥ (C3 rows)")
    print(f"Metrics:")
    print(f"  D·H recovery  : % runs where CORA/CORAK finds D·H as PI")
    print(f"  Cons deflation: CORAK Cons_def − CORA inclusion score (signed)")
    print(f"  Tier 2 flag   : % runs with ≥1 conditional INUS structure")
    print('='*68)

    records     = []
    raw_results = []

    for rate in rates:
        rng = np.random.default_rng(seed)

        cora_dh = corak_dh = 0
        cora_cons_vals    = []
        corak_cons_vals   = []
        deflation_vals    = []
        t2_flag_count = 0
        avg_n_c2 = avg_n_c3 = 0.0

        for _ in range(n_sims):
            res = run_one(n_cases, rate, rng)
            raw_results.append({'rate': rate, **res})

            cora_dh  += int(res['cora_dh_found'])
            corak_dh += int(res['corak_dh_found'])

            if not np.isnan(res['cora_cons']):
                cora_cons_vals.append(res['cora_cons'])
            if not np.isnan(res['corak_cons_def']):
                corak_cons_vals.append(res['corak_cons_def'])

            if (not np.isnan(res['cora_cons'])
                    and not np.isnan(res['corak_cons_def'])):
                deflation_vals.append(
                    res['corak_cons_def'] - res['cora_cons']
                )

            t2_flag_count += int(res['corak_t2_count'] > 0)
            avg_n_c2 += res['n_c2']
            avg_n_c3 += res['n_c3']

        cr     = round(cora_dh  / n_sims, 2)
        kr     = round(corak_dh / n_sims, 2)
        cc     = round(float(np.mean(cora_cons_vals))   if cora_cons_vals   else float('nan'), 3)
        kc     = round(float(np.mean(corak_cons_vals))  if corak_cons_vals  else float('nan'), 3)
        defl   = round(float(np.mean(deflation_vals))   if deflation_vals   else float('nan'), 3)
        t2r    = round(t2_flag_count / n_sims, 2)
        nc2    = round(avg_n_c2 / n_sims, 1)
        nc3    = round(avg_n_c3 / n_sims, 1)

        print(
            f"⊥ {rate*100:4.0f}% | "
            f"CORA D·H={cr:.2f}  CORAK D·H={kr:.2f} | "
            f"CORA Cons={cc:.3f}  CORAK Cons={kc:.3f}  "
            f"Δ={defl:+.3f} | "
            f"Tier2={t2r:.2f} | "
            f"C2={nc2:.1f} C3={nc3:.1f}"
        )

        records.append({
            'undef_rate':        rate,
            'cora_dh_recovery':  cr,
            'corak_dh_recovery': kr,
            'cora_cons_mean':    cc,
            'corak_cons_def':    kc,
            'cons_deflation':    defl,
            'corak_t2_flag_rate': t2r,
            'avg_n_c2':          nc2,
            'avg_n_c3':          nc3,
        })

    print('='*68)
    df = pd.DataFrame(records)

    print("\n── Paper Table (Section 5.3) ──────────────────────────────")
    print(f"{'⊥ Rate':>8} | {'CORA Cons':>10} | {'CORAK Cons_def':>14} | "
          f"{'Deflation':>10} | {'Tier 2 flag':>12}")
    print("─" * 62)
    for _, row in df.iterrows():
        print(
            f"{row['undef_rate']*100:7.0f}% | "
            f"{row['cora_cons_mean']:>10.3f} | "
            f"{row['corak_cons_def']:>14.3f} | "
            f"{row['cons_deflation']:>+10.3f} | "
            f"{row['corak_t2_flag_rate']:>12.2f}"
        )

    metadata = {
        'timestamp':      datetime.now().isoformat(),
        'ground_truth':   GROUND_TRUTH_STR,
        'n_cases':        n_cases,
        'n_simulations':  n_sims,
        'undef_rates':    rates,
        'seed':           seed,
        'corak_version':  '0.1.0',
        'python_version': sys.version.split()[0],
    }

    return {'table': df, 'metadata': metadata, 'raw': raw_results}


# ─────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────

def save_results(results: dict, out_dir: str) -> None:
    """
    Save all simulation artefacts for reproducibility:
      simulation_results.csv   — main table (human-readable)
      simulation_raw.csv       — per-run data (for reanalysis)
      simulation_metadata.json — run configuration
    """
    os.makedirs(out_dir, exist_ok=True)

    # Main table
    csv_path = os.path.join(out_dir, 'simulation_results.csv')
    results['table'].to_csv(csv_path, index=False)
    print(f"  → {csv_path}")

    # Raw per-run data
    raw_path = os.path.join(out_dir, 'simulation_raw.csv')
    pd.DataFrame(results['raw']).to_csv(raw_path, index=False)
    print(f"  → {raw_path}")

    # Metadata (JSON)
    meta_path = os.path.join(out_dir, 'simulation_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(results['metadata'], f, indent=2)
    print(f"  → {meta_path}")


def load_results(out_dir: str) -> dict:
    """Load previously saved simulation results."""
    table = pd.read_csv(os.path.join(out_dir, 'simulation_results.csv'))
    raw   = pd.read_csv(os.path.join(out_dir, 'simulation_raw.csv'))
    with open(os.path.join(out_dir, 'simulation_metadata.json')) as f:
        meta = json.load(f)
    return {'table': table, 'metadata': meta, 'raw': raw.to_dict('records')}


# ─────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CORAK simulation study — Section 5.3'
    )
    parser.add_argument('--n',    type=int,   default=40,   help='Cases per dataset')
    parser.add_argument('--runs', type=int,   default=500,  help='Simulations per rate')
    parser.add_argument('--seed', type=int,   default=2024, help='Random seed')
    parser.add_argument('--rates', nargs='+', type=float,
                        default=[0.0, 0.05, 0.10, 0.20, 0.30],
                        help='⊥ rates to simulate')
    parser.add_argument('--fast', action='store_true',
                        help='Quick run: 100 sims/rate')
    parser.add_argument('--out',  type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='Output directory')
    args = parser.parse_args()

    if args.fast:
        args.runs = 100
        print("[Fast mode: 100 simulations/rate]")

    results = run_simulation(
        n_cases=args.n,
        n_sims=args.runs,
        rates=args.rates,
        seed=args.seed,
    )

    print(f"\nSaving results to {args.out}/")
    save_results(results, args.out)
    print("Done.")
