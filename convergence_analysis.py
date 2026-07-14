"""
Quantitative convergence diagnostics: fits competing decay-curve models
to the effect-size-vs-N sweep, and tracks the Hardy-Littlewood bump
ratio across N. Reuses sweep data already saved by run_all.py --sweep.

Usage:
    python convergence_analysis.py --moduli 6 30 \
        --sweep 1000000 10000000 100000000 --data-dir data
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import stats_tests as st
from stats_tests import hardy_littlewood_gap_weight


def fit_decay_curve(N_values, effect_sizes):
    """
    Fits two competing models to (N, effect_size):
      A: V ~ a + b*log(N)               [steady log-linear decay]
      B: V ~ a + b*(loglog(N)/log(N))   [Tao-style secondary-bias shape]
    Returns slope/intercept/R^2 for both, plus a verdict on which fits better.
    """
    N_values = np.asarray(N_values, dtype=float)
    effect_sizes = np.asarray(effect_sizes, dtype=float)

    log_N = np.log(N_values)
    fit_a = scipy_stats.linregress(log_N, effect_sizes)

    loglog_over_log = np.log(np.log(N_values)) / np.log(N_values)
    fit_b = scipy_stats.linregress(loglog_over_log, effect_sizes)

    r2_a = fit_a.rvalue ** 2
    r2_b = fit_b.rvalue ** 2

    if r2_b > r2_a + 0.02:
        verdict = "Data favors the Tao-style loglog(N)/log(N) decay shape."
    elif r2_a > r2_b + 0.02:
        verdict = "Data favors simple linear-in-log(N) decay."
    else:
        verdict = "The two models fit comparably well (R^2 within 0.02)."

    return {
        "model_linear_logN": {"slope": fit_a.slope, "intercept": fit_a.intercept, "r_squared": r2_a},
        "model_tao_loglogN_over_logN": {"slope": fit_b.slope, "intercept": fit_b.intercept, "r_squared": r2_b},
        "verdict": verdict,
    }


def hardy_littlewood_bump_ratio(df, q, boosted_gap, neighbor_gaps):
    """Ratio of boosted_gap's observed frequency to the average frequency
    of neighbor_gaps, alongside the theoretical HL-predicted ratio."""
    counts = df["gap"].value_counts()
    total = len(df)

    boosted_freq = counts.get(boosted_gap, 0) / total
    neighbor_freqs = [counts.get(g, 0) / total for g in neighbor_gaps]
    neighbor_avg = np.mean(neighbor_freqs) if neighbor_freqs else np.nan

    observed_ratio = boosted_freq / neighbor_avg if neighbor_avg > 0 else np.nan

    theoretical_boosted = hardy_littlewood_gap_weight(boosted_gap)
    theoretical_neighbors = [hardy_littlewood_gap_weight(g) for g in neighbor_gaps]
    theoretical_neighbor_avg = np.mean(theoretical_neighbors) if theoretical_neighbors else np.nan
    theoretical_ratio = (theoretical_boosted / theoretical_neighbor_avg
                          if theoretical_neighbor_avg > 0 else np.nan)

    return {
        "n_observations": total,
        "boosted_gap": boosted_gap,
        "neighbor_gaps": neighbor_gaps,
        "observed_ratio": observed_ratio,
        "theoretical_hl_ratio": theoretical_ratio,
    }


def bump_ratio_across_sweep(N_values, q, boosted_gap, neighbor_gaps, data_dir="data"):
    """Runs hardy_littlewood_bump_ratio at every N, using existing
    sweep_N{N}.csv files (no regeneration)."""
    rows = []
    for N in N_values:
        path = os.path.join(data_dir, f"sweep_N{N}.csv")
        if not os.path.exists(path):
            print(f"[convergence_analysis] WARNING: {path} not found, skipping N={N}")
            continue
        df = pd.read_csv(path)
        sub = st.filter_admissible(df, q)
        result = hardy_littlewood_bump_ratio(sub, q, boosted_gap, neighbor_gaps)
        result["N"] = N
        rows.append(result)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moduli", type=int, nargs="+", default=[6, 30])
    parser.add_argument("--sweep", type=int, nargs="+", required=True,
                         help="Same N values used in run_all.py --sweep.")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    os.makedirs("tables", exist_ok=True)

    for q in args.moduli:
        effect_path = f"tables/effect_size_vs_N_mod{q}.csv"
        if not os.path.exists(effect_path):
            print(f"[convergence_analysis] {effect_path} not found -- run run_all.py --sweep first.")
            continue
        effect_df = pd.read_csv(effect_path)
        fit_result = fit_decay_curve(effect_df["N"], effect_df["cramers_v"])

        print(f"\n=== Decay curve fit, mod {q} ===")
        print(f"Linear-in-log(N):     R^2={fit_result['model_linear_logN']['r_squared']:.4f}")
        print(f"Tao loglog(N)/log(N): R^2={fit_result['model_tao_loglogN_over_logN']['r_squared']:.4f}")
        print(f"Verdict: {fit_result['verdict']}")

        pd.DataFrame([
            {"model": "linear_logN", **fit_result["model_linear_logN"]},
            {"model": "tao_loglogN_over_logN", **fit_result["model_tao_loglogN_over_logN"]},
        ]).to_csv(f"tables/decay_curve_fit_mod{q}.csv", index=False)

    for q in args.moduli:
        print(f"\n=== Hardy-Littlewood bump ratio across N, mod {q} (gap=6 vs neighbors 4,8) ===")
        bump = bump_ratio_across_sweep(args.sweep, q, boosted_gap=6, neighbor_gaps=[4, 8], data_dir=args.data_dir)
        print(bump)
        bump.to_csv(f"tables/bump_ratio_mod{q}.csv", index=False)

    print("\n[convergence_analysis] Done. See tables/decay_curve_fit_mod{q}.csv "
          "and tables/bump_ratio_mod{q}.csv")


if __name__ == "__main__":
    main()
