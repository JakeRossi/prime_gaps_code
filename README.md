# Prime Gaps Under Modular Constraints — data pipeline

Code that generates every table and figure referenced in
`prime_gaps_paper_draft.tex`. All of it has been smoke-tested end to end
at small N (10^5–10^6) in this environment; running at your real target
N (10^8–10^9) has NOT been tested here since large-N runs are slow.

## Setup

```bash
pip install -r requirements.txt
```

`primesieve` and `pyarrow` are optional but strongly recommended once you
move past N ~ 10^7 (see comments in `gen_primes.py`). The code
auto-detects both and falls back gracefully if they aren't installed.

## Files

| File | Paper section it feeds | What it does |
|---|---|---|
| `gen_primes.py` | Methodology | Modules 1–4: segmented sieve, gap/residue table, save to CSV/Parquet |
| `stats_tests.py` | Sections 4.1, 4.2, 4.4 | Modules 6–7: equidistribution check, omnibus chi-square + Cramér's V, pairwise KS with BH correction, residue-pair vs. LOS/uniform prediction, dominant gaps |
| `cramer_model.py` | Background, Section 4.5 | Module 8: residue-restricted Cramér model simulation |
| `plots.py` | Figures throughout | Module 5: every figure in the paper |
| `run_all.py` | — | Orchestrates everything above into one command |

## Quick start (sanity check first!)

Run small before you run big — this takes seconds and confirms
everything works on your machine:

```bash
python run_all.py --N 1000000 --moduli 6 30 --sweep 100000 300000 1000000
```

This produces `data/`, `tables/`, and `figures/` directories with
everything the paper needs, just at toy scale. Open a couple of the
PNGs in `figures/` to eyeball that they look sane before committing to
a long run.

## Real run

```bash
python run_all.py --N 1000000000 --moduli 6 30 \
    --sweep 1000000 10000000 100000000 1000000000
```

This is the run whose output tables/figures you actually cite in the
paper. With `primesieve` installed, generating primes up to 10^9 should
take a few minutes; without it, budget more (the pure-Python sieve is
correct but slower — consider testing at 10^8 first if you're on a
deadline and haven't installed primesieve).

**Important:** `--sweep` reruns prime generation from scratch at every
N in the list (that's what Section 4.3 needs — one Cramér's V per N).
Including 10^9 in the sweep means generating it AGAIN in addition to
the main `--N` run. If you're tight on time, you can drop the largest
sweep value and just note in the paper that N=10^9 sweep point is your
`--N` run's own value (they're numerically identical, no need to
literally rerun it — feel free to hand-merge that data point into
`tables/effect_size_vs_N_mod{q}.csv` instead of waiting for a duplicate run).

## Where each output goes in the paper

See the top-of-file docstring in `run_all.py` for the exact mapping of
output filename → paper section. In short:

- `tables/equidistribution_mod{q}.csv` → Section 4.1
- `tables/omnibus_test_mod{q}.csv`, `tables/pairwise_ks_mod{q}.csv`,
  `tables/pair_predictions_mod{q}.csv`,
  `figures/gap_histograms_mod{q}.png`,
  `figures/residue_pair_heatmap_mod{q}.png` → Section 4.2
- `tables/effect_size_vs_N_mod{q}.csv`,
  `figures/effect_size_vs_N_mod{q}.png` → Section 4.3 (tests the paper's
  working conjecture directly)
- `tables/dominant_gaps_mod{q}.csv`,
  `figures/dominant_gaps_mod{q}.png` → Section 4.4
- `tables/cramer_model_comparison_mod{q}.csv`,
  `figures/real_vs_cramer_mod{q}.png` → Section 4.5

## Known simplification to flag in your Methodology section

`stats_tests.los_style_prediction()` is a **simplified stand-in**, not
the exact Hardy–Littlewood singular-series formula from Lemke Oliver &
Soundararajan (2016). It's flagged clearly in its docstring. If you have
time on Day 4–5, revisiting their equations (7)–(9) to implement the
exact constants would meaningfully strengthen the paper; if not,
reporting it as an explicit, acknowledged simplification is honest and
fine for a first pass.
