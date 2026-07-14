# Prime Gaps Under Modular Constraints — data pipeline

Code that generates every table and figure in `prime_gaps_paper_draft.tex`.
Tested end to end at N up to 10^9 (not just small N) — see "Tested status"
below.

## Setup

```bash
pip install -r requirements.txt
```

`primesieve` and `pyarrow` are optional. In practice, the pure-Python
segmented sieve in `gen_primes.py` is fast enough on its own (N=10^9 in
under 10 seconds, thanks to NumPy vectorization) — `primesieve` is not
required even at N=10^9, and it's common for its install to fail on
Windows due to missing C build tools. Don't spend time fighting that
install; the code falls back automatically and works fine without it.
`pyarrow` (for Parquet output) is only relevant if you pass
`--keep-raw-data` to `master_sweep.py`, or for `run_all.py`'s largest runs.

## Files

| File | What it does |
|---|---|
| `gen_primes.py` | Segmented sieve, builds the prime-gap/residue table, saves to CSV/Parquet |
| `stats_tests.py` | Equidistribution check, omnibus chi-square + Cramér's V, pairwise KS with BH correction, naive-uniform and Hardy–Littlewood residue-pair predictions, dominant gaps |
| `cramer_model.py` | Residue-restricted Cramér-model simulation (random baseline) |
| `plots.py` | All figures, including multi-modulus combined comparison plots |
| `run_all.py` | Full analysis for one primary N (all tables/figures for that N), plus an optional effect-size-vs-N sweep on the side |
| `master_sweep.py` | Runs the full analysis across a *range* of N (default 10^5–10^9) and consolidates every metric into one table (`master_trend.csv`) with both N and modulus as columns — the tool actually used to generate the paper's Section 4.3–4.6 results |
| `convergence_analysis.py` | Post-hoc analysis of sweep data: fits log-linear vs. Tao-style decay curves, tracks the Hardy–Littlewood bump ratio across N |
| `dataset_profiler.py` | General-purpose tool (not paper-specific): point it at a folder of CSVs and it flags outliers, trend reversals, and correlations automatically |

## Tested status

Every script has been run successfully end to end at N up to 10^9,
including the full `master_sweep.py` sweep (10^5 through 10^9, both
moduli) — this is what actually produced the numbers in the paper's
Results section. No `primesieve` install was needed for any of it.

## Quick start — sanity check first

```bash
python run_all.py --N 1000000 --moduli 6 30 --sweep 100000 300000 1000000
```

Takes seconds, confirms everything works on your machine before a longer run.

## Getting the full paper results

This is what actually generated the numbers currently in the paper:

```bash
python master_sweep.py --moduli 6 30 --min-exponent 5 --max-exponent 9
```

Produces `tables/master_trend.csv` (every metric, every N, both moduli),
`tables/decay_verdicts.csv`, and four `figures/*_combined.png` files.
Raw per-N data is **not** saved by default (add `--keep-raw-data` if you
want it) — those files aren't needed downstream and can be large
(hundreds of MB per file at N=10^9), which is also why they're excluded
via `.gitignore` rather than committed to the repo.

For a deep dive on one specific N (all 64 pair cells, all 28 pairwise KS
tests, every figure for that one N), use `run_all.py` instead:

```bash
python run_all.py --N 1000000000 --moduli 6 30
```

Then, optionally, run the decay-curve/bump-ratio diagnostics on top of
either sweep's output:

```bash
python convergence_analysis.py --moduli 6 30 --sweep 1000000 10000000 100000000
```

## Where each output goes in the paper

- `tables/equidistribution_mod{q}.csv` → Section 4.1
- `tables/omnibus_test_mod{q}.csv`, `tables/pairwise_ks_mod{q}.csv`,
  `tables/pair_predictions_mod{q}.csv`, `figures/gap_histograms_mod{q}.png`,
  `figures/residue_pair_heatmap_mod{q}.png` → Section 4.2
- `tables/pair_predictions_mod{q}.csv` (naive vs. Hardy–Littlewood
  goodness-of-fit) → Section 4.3
- `tables/master_trend.csv`, `tables/decay_verdicts.csv`,
  `figures/effect_size_vs_N_combined.png` → Section 4.4 (bias persistence,
  the paper's main N-dependence result)
- `tables/dominant_gaps_mod{q}.csv`, `figures/dominant_gaps_mod{q}.png` → Section 4.5
- `tables/cramer_model_comparison_mod{q}.csv`,
  `tables/excess_bias_vs_cramer_mod{q}.csv`, `tables/top_biased_pairs_mod{q}.csv`,
  `figures/real_vs_cramer_mod{q}.png`,
  `figures/excess_bias_v_vs_N_combined.png`,
  `figures/bump_ratio_vs_N_combined.png` → Section 4.6

## Known limitation, as flagged in the paper

`stats_tests.los_style_prediction()` implements the *real* Hardy–Littlewood
singular-series formula (`hardy_littlewood_gap_weight`), not a made-up
heuristic — but only its **leading-order term**. That term alone predicts
same-residue-class pairs should be *more* common (the real, separate
"sexy primes outnumber twin primes" effect), which is the opposite sign
from the actual Lemke Oliver–Soundararajan bias. The true LOS effect comes
from a smaller secondary correction term (see Tao's derivation, cited in
the paper) that isn't implemented here — reproducing it was out of scope
for this project's timeframe. This is stated explicitly in the paper's
Limitations section; it's also exactly why the leading-order prediction
fits *worse* than naive-uniform at mod 6 but *better* at mod 30 (Section 4.3)
— a real, discussed finding, not a bug.
