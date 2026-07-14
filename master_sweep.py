"""
Runs the full analysis pipeline across a range of N (default 10^5 to
10^9) and consolidates every metric into one table with both N and
modulus as columns.

Usage:
    python master_sweep.py --moduli 6 30 --min-exponent 5 --max-exponent 9

Output:
  tables/master_trend.csv        <- one row per (N, modulus), all metrics
  tables/decay_verdicts.csv      <- log-linear vs. Tao-style decay fit, per modulus
  figures/*_combined.png         <- multi-modulus comparison figures

Raw per-N data is NOT saved by default (pass --keep-raw-data to save it);
it's not needed for the summary table and can be large at N=10^8-10^9.
"""

import argparse
import gc
import os
import time
import numpy as np
import pandas as pd

from gen_primes import generate_primes, build_gap_table, save_table
import stats_tests as st
import plots as pl
from cramer_model import build_synthetic_gap_table
from convergence_analysis import hardy_littlewood_bump_ratio, fit_decay_curve


def ensure_dirs():
    for d in ("data", "tables", "figures"):
        os.makedirs(d, exist_ok=True)


def analyze_one_N_one_modulus(df, df_sim, N, q):
    """Runs all summary-level tests for one (N, modulus) pair, returns
    one flat dict (a row of the master trend table)."""
    omnibus = st.omnibus_class_gap_test(df, q)
    pairwise = st.pairwise_ks_tests(df, q)
    n_pairwise_significant = int(pairwise["significant_at_0.05"].sum())
    n_pairwise_total = len(pairwise)
    del pairwise
    gc.collect()

    pair_pred = st.compare_pair_predictions(df, q)
    naive_chi2 = pair_pred.loc[pair_pred["prediction"] == "naive_uniform", "chi2"].iloc[0]
    hl_chi2 = pair_pred.loc[pair_pred["prediction"] == "los_style_simplified", "chi2"].iloc[0]

    excess_bias = st.real_vs_simulated_excess_bias(df, df_sim, q)
    del pair_pred
    gc.collect()

    sub = st.filter_admissible(df, q)
    bump = hardy_littlewood_bump_ratio(sub, q, boosted_gap=6, neighbor_gaps=[4, 8])
    n_primes = len(sub)
    mean_gap = sub["gap"].mean()
    del sub
    gc.collect()

    pair_table = st.residue_pair_frequency_table(df, q)
    diagonal_share = np.mean([pair_table.loc[r, r] for r in pair_table.index if r in pair_table.columns])
    off_diagonal_share = (pair_table.values.sum() - sum(
        pair_table.loc[r, r] for r in pair_table.index if r in pair_table.columns
    )) / (pair_table.shape[0] * pair_table.shape[1] - pair_table.shape[0])

    return {
        "N": N,
        "modulus": q,
        "n_primes": n_primes,
        "mean_gap": mean_gap,
        "omnibus_chi2": omnibus["chi2_statistic"],
        "omnibus_cramers_v": omnibus["cramers_v"],
        "omnibus_p_value": omnibus["p_value"],
        "n_pairwise_significant": n_pairwise_significant,
        "n_pairwise_total": n_pairwise_total,
        "naive_uniform_chi2": naive_chi2,
        "hardy_littlewood_chi2": hl_chi2,
        "hl_beats_naive": bool(hl_chi2 < naive_chi2),
        "excess_bias_vs_cramer_p": excess_bias["p_value"],
        "excess_bias_vs_cramer_v": excess_bias["cramers_v_excess_bias"],
        "hl_bump_observed_ratio": bump["observed_ratio"],
        "hl_bump_theoretical_ratio": bump["theoretical_hl_ratio"],
        "diagonal_share": diagonal_share,
        "off_diagonal_share": off_diagonal_share,
        "naive_expected_share": 1.0 / (int(pd.Series(pair_table.index).nunique()) ** 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moduli", type=int, nargs="+", default=[6, 30])
    parser.add_argument("--min-exponent", type=int, default=5,
                         help="Smallest N = 10^min_exponent (default 5)")
    parser.add_argument("--max-exponent", type=int, default=9,
                         help="Largest N = 10^max_exponent (default 9)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the Cramer-model simulation.")
    parser.add_argument("--keep-raw-data", action="store_true",
                         help="Save each N's raw prime/gap table to data/ (off by default).")
    args = parser.parse_args()

    ensure_dirs()
    N_values = [10 ** e for e in range(args.min_exponent, args.max_exponent + 1)]
    print(f"[master_sweep] Running N = {N_values}")
    print(f"[master_sweep] Moduli = {args.moduli}")

    rows = []
    for N in N_values:
        for q in args.moduli:
            t0 = time.time()
            primes = generate_primes(N)
            print(f"[master_sweep] N={N:,}, mod {q}: generation took {time.time()-t0:.1f}s")

            # Rebuild the dataframe for one modulus at a time (rather than
            # holding all moduli's columns, or a shared `primes` array,
            # alive across the loop) -- keeps peak memory manageable at
            # N=10^9; regenerating primes per modulus costs seconds.
            df = build_gap_table(primes, moduli=(q,))
            del primes
            gc.collect()
            if args.keep_raw_data:
                ext = "parquet" if N >= 10**8 else "csv"
                out_path = f"data/N{N}_mod{q}.{ext}"
                save_table(df, out_path)

            df_sim = build_synthetic_gap_table(N, q, seed=args.seed)
            row = analyze_one_N_one_modulus(df, df_sim, N, q)
            rows.append(row)
            print(f"[master_sweep]   mod {q}: omnibus V={row['omnibus_cramers_v']:.4f}, "
                  f"excess-bias-vs-cramer p={row['excess_bias_vs_cramer_p']:.2e}, "
                  f"HL bump ratio={row['hl_bump_observed_ratio']:.3f} "
                  f"(theoretical {row['hl_bump_theoretical_ratio']:.3f})")
            del df_sim, df
            gc.collect()

    trend_df = pd.DataFrame(rows)
    trend_df.to_csv("tables/master_trend.csv", index=False)
    print(f"\n[master_sweep] Saved tables/master_trend.csv ({len(trend_df)} rows: "
          f"{len(N_values)} N values x {len(args.moduli)} moduli)")

    verdict_rows = []
    for q in args.moduli:
        sub = trend_df[trend_df["modulus"] == q].sort_values("N")
        fit = fit_decay_curve(sub["N"], sub["omnibus_cramers_v"])
        verdict_rows.append({
            "modulus": q,
            "r2_linear_logN": fit["model_linear_logN"]["r_squared"],
            "r2_tao_loglogN_over_logN": fit["model_tao_loglogN_over_logN"]["r_squared"],
            "verdict": fit["verdict"],
        })
        print(f"\n[master_sweep] Decay verdict, mod {q}: {fit['verdict']}")
    pd.DataFrame(verdict_rows).to_csv("tables/decay_verdicts.csv", index=False)

    pl.plot_metric_vs_N_multi_modulus(
        trend_df, "omnibus_cramers_v",
        ylabel="Effect size (Cramer's V, class vs. gap)",
        title="Bias magnitude vs. N, all moduli",
        out_name="effect_size_vs_N_combined")

    pl.plot_metric_vs_N_multi_modulus(
        trend_df, "excess_bias_vs_cramer_v",
        ylabel="Excess bias effect size (real vs. Cramer model)",
        title="Excess bias beyond random baseline vs. N, all moduli",
        out_name="excess_bias_v_vs_N_combined")

    pl.plot_metric_vs_N_multi_modulus(
        trend_df, "hl_bump_observed_ratio",
        ylabel="Observed gap=6 / neighbor(4,8) frequency ratio",
        title="Hardy-Littlewood bump ratio vs. N, all moduli",
        out_name="bump_ratio_vs_N_combined")

    pl.plot_metric_vs_N_multi_modulus(
        trend_df, "diagonal_share",
        ylabel="Same-class-repeat share",
        title="Diagonal (same-class) share vs. N, all moduli",
        out_name="diagonal_share_vs_N_combined")

    print("\n[master_sweep] Done. Key files:")
    print("  tables/master_trend.csv       <- main consolidated table")
    print("  tables/decay_verdicts.csv     <- decay model comparison, per modulus")
    print("  figures/*_combined.png        <- multi-modulus comparison figures")


if __name__ == "__main__":
    main()
