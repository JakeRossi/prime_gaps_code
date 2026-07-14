"""
Orchestrates the full pipeline: generates primes, runs all statistical
tests, produces all tables/figures for a single primary N, plus an
optional effect-size-vs-N sweep.

Usage:
    python run_all.py --N 100000000 --moduli 6 30 \
        --sweep 1000000 10000000 100000000 1000000000

Output:
  data/primes_gap_table_N{N}.csv        <- raw prime/gap table
  tables/equidistribution_mod{q}.csv
  tables/omnibus_test_mod{q}.csv
  tables/pairwise_ks_mod{q}.csv
  tables/pair_predictions_mod{q}.csv
  tables/dominant_gaps_mod{q}.csv
  tables/cramer_model_comparison_mod{q}.csv
  tables/excess_bias_vs_cramer_mod{q}.csv
  tables/top_biased_pairs_mod{q}.csv
  tables/effect_size_vs_N_mod{q}.csv    <- if --sweep given
  figures/*.png
"""

import argparse
import os
import pandas as pd

from gen_primes import run as gen_run
import stats_tests as st
import plots as pl
from cramer_model import build_synthetic_gap_table


def ensure_dirs():
    for d in ("data", "tables", "figures"):
        os.makedirs(d, exist_ok=True)


def run_core_analysis(df, q):
    """Equidistribution, omnibus test, pairwise KS, pair predictions,
    dominant gaps -- tables and figures for one modulus."""
    equidist = st.equidistribution_table(df, q)
    equidist.to_csv(f"tables/equidistribution_mod{q}.csv", index=False)

    omnibus = st.omnibus_class_gap_test(df, q)
    pd.DataFrame([{k: v for k, v in omnibus.items() if k != "contingency_table"}]) \
        .to_csv(f"tables/omnibus_test_mod{q}.csv", index=False)

    pairwise = st.pairwise_ks_tests(df, q)
    pairwise.to_csv(f"tables/pairwise_ks_mod{q}.csv", index=False)

    pair_pred = st.compare_pair_predictions(df, q)
    pair_pred.to_csv(f"tables/pair_predictions_mod{q}.csv", index=False)

    dominant = st.dominant_gaps_by_class(df, q)
    dominant.to_csv(f"tables/dominant_gaps_mod{q}.csv", index=False)

    pl.plot_gap_histograms_by_class(df, q)
    pl.plot_residue_pair_heatmap(df, q)
    pl.plot_dominant_gaps(df, q)

    print(f"[run_all] modulus {q}: chi2={omnibus['chi2_statistic']:.1f}, "
          f"p={omnibus['p_value']:.3g}, Cramer's V={omnibus['cramers_v']:.4f}")
    return omnibus


def run_N_sweep(N_list, q, moduli_all):
    """Generates data at each N, tracks omnibus Cramer's V vs. N."""
    rows = []
    for N in N_list:
        out_path = f"data/sweep_N{N}.csv"
        df = gen_run(N, out_path, moduli=moduli_all)
        omnibus = st.omnibus_class_gap_test(df, q)
        rows.append({"N": N, "cramers_v": omnibus["cramers_v"], "p_value": omnibus["p_value"]})
        print(f"[run_all] sweep N={N:,}: Cramer's V (mod {q}) = {omnibus['cramers_v']:.4f}")

    effect_df = pd.DataFrame(rows)
    effect_df.to_csv(f"tables/effect_size_vs_N_mod{q}.csv", index=False)
    pl.plot_effect_size_vs_N(effect_df, q)
    return effect_df


def run_cramer_comparison(df_real, q, N, seed=0):
    """Builds the Cramer-model baseline, compares against real data."""
    df_sim = build_synthetic_gap_table(N, q, seed=seed)

    real_omnibus = st.omnibus_class_gap_test(df_real, q)
    sim_omnibus = st.omnibus_class_gap_test(df_sim, q)

    comparison = pd.DataFrame([
        {"source": "real_primes", "cramers_v": real_omnibus["cramers_v"],
         "p_value": real_omnibus["p_value"], "n": real_omnibus["n_observations"]},
        {"source": "cramer_model", "cramers_v": sim_omnibus["cramers_v"],
         "p_value": sim_omnibus["p_value"], "n": sim_omnibus["n_observations"]},
    ])
    comparison.to_csv(f"tables/cramer_model_comparison_mod{q}.csv", index=False)
    pl.plot_real_vs_cramer_model(df_real, df_sim, q)

    excess_bias = st.real_vs_simulated_excess_bias(df_real, df_sim, q)
    pd.DataFrame([excess_bias]).to_csv(f"tables/excess_bias_vs_cramer_mod{q}.csv", index=False)
    print(f"[run_all] modulus {q}: excess bias vs. Cramer model p={excess_bias['p_value']:.3g}, "
          f"Cramer's V={excess_bias['cramers_v_excess_bias']:.4f}")

    top_pairs = st.top_biased_pairs(df_real, q)
    top_pairs.to_csv(f"tables/top_biased_pairs_mod{q}.csv", index=False)

    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True,
                         help="Main N for the primary (largest) analysis.")
    parser.add_argument("--moduli", type=int, nargs="+", default=[6, 30])
    parser.add_argument("--sweep", type=int, nargs="*", default=None,
                         help="N values for the effect-size-vs-N sweep, "
                              "e.g. --sweep 1000000 10000000 100000000 1000000000")
    args = parser.parse_args()

    ensure_dirs()

    df = gen_run(args.N, f"data/primes_gap_table_N{args.N}.csv", moduli=tuple(args.moduli))

    for q in args.moduli:
        run_core_analysis(df, q)
        run_cramer_comparison(df, q, args.N)

    if args.sweep:
        for q in args.moduli:
            run_N_sweep(args.sweep, q, tuple(args.moduli))

    print("\n[run_all] Done. See tables/ and figures/.")


if __name__ == "__main__":
    main()
