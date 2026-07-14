"""
Statistical tests for class-conditional prime gap analysis: equidistribution
checks, chi-square/KS tests, Hardy-Littlewood predictions, and Cramer-model
comparison.
"""

import itertools
import numpy as np
import pandas as pd
from scipy import stats
from sympy import totient, gcd, factorint


def admissible_residues(q):
    """Residues r mod q with gcd(r,q)=1."""
    return [r for r in range(1, q) if gcd(r, q) == 1]


def filter_admissible(df, q, min_prime=None):
    """Drop primes not in an admissible residue class mod q, and drop
    columns unrelated to modulus q (keeps memory down when df was built
    with multiple moduli)."""
    admissible = set(admissible_residues(q))
    floor = min_prime if min_prime is not None else (q + 1)
    mask = (df["p"] > floor) & (df[f"res_mod{q}"].isin(admissible))
    relevant_cols = ["p", "gap", f"res_mod{q}", f"next_res_mod{q}"]
    relevant_cols = [c for c in relevant_cols if c in df.columns]
    return df.loc[mask, relevant_cols].copy()


def fast_crosstab(row_values, col_values):
    """Equivalent to pd.crosstab(row_values, col_values) but uses
    numpy bincount instead of pandas groupby -- much lower memory
    overhead at tens of millions of rows."""
    row_values = np.asarray(row_values)
    col_values = np.asarray(col_values)

    row_unique, row_inverse = np.unique(row_values, return_inverse=True)
    col_unique, col_inverse = np.unique(col_values, return_inverse=True)

    n_rows, n_cols = len(row_unique), len(col_unique)
    flat_index = row_inverse.astype(np.int64) * n_cols + col_inverse.astype(np.int64)
    counts = np.bincount(flat_index, minlength=n_rows * n_cols).reshape(n_rows, n_cols)

    return pd.DataFrame(counts, index=row_unique, columns=col_unique)


def bh_correct(pvalues):
    """Benjamini-Hochberg correction. Returns q-values in input order."""
    pvalues = np.asarray(pvalues)
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    q_ranked = ranked * n / (np.arange(n) + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q_values = np.empty(n)
    q_values[order] = q_ranked
    return q_values


def equidistribution_table(df, q):
    """Observed share of primes in each admissible class vs. 1/phi(q)."""
    sub = filter_admissible(df, q)
    counts = sub[f"res_mod{q}"].value_counts().sort_index()
    total = counts.sum()
    phi_q = int(totient(q))
    out = pd.DataFrame({
        "residue": counts.index,
        "count": counts.values,
        "observed_share": counts.values / total,
        "expected_share_1_over_phi_q": 1.0 / phi_q,
    })
    out["deviation"] = out["observed_share"] - out["expected_share_1_over_phi_q"]
    return out


def make_gap_bins(gaps, max_explicit_gap=60):
    """Bin gap values: each even value up to max_explicit_gap gets its
    own bin, larger gaps pool into one tail bin (keeps expected chi-square
    cell counts >= 5)."""
    return np.where(gaps >= max_explicit_gap, max_explicit_gap, gaps)


def omnibus_class_gap_test(df, q, max_explicit_gap=60):
    """Chi-square test of independence between residue class and
    (binned) gap size, with Cramer's V as effect size."""
    sub = filter_admissible(df, q)
    sub["gap_bin"] = make_gap_bins(sub["gap"].values, max_explicit_gap)

    contingency = fast_crosstab(sub[f"res_mod{q}"], sub["gap_bin"])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    n = contingency.values.sum()
    r, k = contingency.shape
    cramers_v = np.sqrt((chi2 / n) / (min(r - 1, k - 1)))

    return {
        "modulus": q,
        "chi2_statistic": chi2,
        "dof": dof,
        "p_value": p_value,
        "cramers_v": cramers_v,
        "n_observations": int(n),
        "contingency_table": contingency,
    }


def pairwise_ks_tests(df, q):
    """Two-sample KS test between every pair of admissible classes'
    gap distributions, with Benjamini-Hochberg correction across pairs."""
    sub = filter_admissible(df, q)
    classes = sorted(sub[f"res_mod{q}"].unique())
    rows = []
    for a, b in itertools.combinations(classes, 2):
        gaps_a = sub.loc[sub[f"res_mod{q}"] == a, "gap"].values
        gaps_b = sub.loc[sub[f"res_mod{q}"] == b, "gap"].values
        ks_stat, p_val = stats.ks_2samp(gaps_a, gaps_b)
        rows.append({"class_a": a, "class_b": b, "ks_statistic": ks_stat, "p_value": p_val})

    result = pd.DataFrame(rows)
    result["q_value_bh"] = bh_correct(result["p_value"].values)
    result["significant_at_0.05"] = result["q_value_bh"] < 0.05
    return result


def residue_pair_frequency_table(df, q):
    """Observed frequency of (residue of p_n, residue of p_{n+1}) among
    consecutive primes -- the LOS-style pair table."""
    sub = filter_admissible(df, q)
    admissible = set(admissible_residues(q))
    sub = sub.loc[sub[f"next_res_mod{q}"].isin(admissible)]
    table = fast_crosstab(sub[f"res_mod{q}"], sub[f"next_res_mod{q}"])
    table = table / table.values.sum()
    return table


def naive_uniform_prediction(q):
    """1/phi(q)^2 for every permissible residue pair."""
    phi_q = int(totient(q))
    residues = admissible_residues(q)
    val = 1.0 / (phi_q ** 2)
    return pd.DataFrame(val, index=residues, columns=residues)


def hardy_littlewood_gap_weight(h):
    """Hardy-Littlewood singular series factor for a prime pair (p, p+h):
    f(h) = product over odd primes p | h of (p-1)/(p-2). Zero for odd h."""
    if h <= 0 or h % 2 != 0:
        return 0.0
    weight = 1.0
    for p in factorint(h):
        if p > 2:
            weight *= (p - 1) / (p - 2)
    return weight


def los_style_prediction(q, mean_log_scale=None, max_gap=400):
    """
    Predicted residue-pair frequency table using the leading-order
    Hardy-Littlewood weight, exponentially damped by gap size.

    NOTE: this is the LEADING-ORDER term only. It predicts same-class
    pairs (gap divisible by small factors of q) as MORE common -- the
    "sexy primes > twin primes" effect -- which is the opposite sign
    from the LOS same-class-avoidance bias itself. The actual LOS effect
    comes from a smaller secondary correction term not implemented here.
    """
    residues = admissible_residues(q)
    if mean_log_scale is None:
        mean_log_scale = 15.0

    weights = {}
    for a in residues:
        for b in residues:
            target = (b - a) % q
            total_weight = 0.0
            h = target if target != 0 else q
            while h <= max_gap:
                total_weight += hardy_littlewood_gap_weight(h) * np.exp(-h / mean_log_scale)
                h += q
            weights[(a, b)] = total_weight

    total = sum(weights.values())
    table = pd.DataFrame(index=residues, columns=residues, dtype=float)
    for (a, b), w in weights.items():
        table.loc[a, b] = w / total
    return table


def compare_pair_predictions(df, q):
    """Chi-square goodness-of-fit of the observed pair table against the
    naive-uniform and Hardy-Littlewood predictions."""
    residues = admissible_residues(q)

    sub = filter_admissible(df, q)
    admissible = set(admissible_residues(q))
    sub = sub.loc[sub[f"next_res_mod{q}"].isin(admissible)]
    n_obs = len(sub)
    observed_counts = fast_crosstab(sub[f"res_mod{q}"], sub[f"next_res_mod{q}"])

    mean_log_scale = float(sub["gap"].mean())

    def goodness_of_fit(pred_table):
        pred_counts = pred_table.reindex(index=residues, columns=residues) * n_obs
        chi2 = ((observed_counts.values - pred_counts.values) ** 2 / pred_counts.values).sum()
        dof = len(residues) ** 2 - 1
        p_val = 1 - stats.chi2.cdf(chi2, dof)
        return chi2, dof, p_val

    naive = naive_uniform_prediction(q)
    los = los_style_prediction(q, mean_log_scale=mean_log_scale)

    chi2_naive, dof_naive, p_naive = goodness_of_fit(naive)
    chi2_los, dof_los, p_los = goodness_of_fit(los)

    return pd.DataFrame([
        {"prediction": "naive_uniform", "chi2": chi2_naive, "dof": dof_naive, "p_value": p_naive},
        {"prediction": "los_style_simplified", "chi2": chi2_los, "dof": dof_los, "p_value": p_los},
    ])


def real_vs_simulated_excess_bias(df_real, df_sim, q):
    """
    Two-sample chi-square test comparing the real pair-count table
    against the simulated (Cramer-model) pair-count table directly.

    The Cramer model also shows a mild same-class dip (a property of
    geometric waiting times on an alternating admissible-residue
    sequence, not primes specifically), so this tests whether real
    primes are MORE biased than an equivalent random model, not just
    biased at all.
    """
    admissible = set(admissible_residues(q))

    real = filter_admissible(df_real, q)
    real = real.loc[real[f"next_res_mod{q}"].isin(admissible)]
    real_counts = fast_crosstab(real[f"res_mod{q}"], real[f"next_res_mod{q}"])

    sim = filter_admissible(df_sim, q)
    sim = sim.loc[sim[f"next_res_mod{q}"].isin(admissible)]
    sim_counts = fast_crosstab(sim[f"res_mod{q}"], sim[f"next_res_mod{q}"])

    residues = admissible_residues(q)
    real_counts = real_counts.reindex(index=residues, columns=residues, fill_value=0)
    sim_counts = sim_counts.reindex(index=residues, columns=residues, fill_value=0)

    combined = np.vstack([real_counts.values.flatten(), sim_counts.values.flatten()])
    chi2, p_value, dof, expected = stats.chi2_contingency(combined)

    n = combined.sum()
    r, k = combined.shape
    cramers_v = np.sqrt((chi2 / n) / (min(r - 1, k - 1)))

    return {
        "modulus": q,
        "chi2_statistic": chi2,
        "dof": dof,
        "p_value": p_value,
        "cramers_v_excess_bias": cramers_v,
        "n_real": int(real_counts.values.sum()),
        "n_simulated": int(sim_counts.values.sum()),
    }


def top_biased_pairs(df, q, n_top=5):
    """Ranks (this_class -> next_class) pairs by relative deviation
    from the naive-uniform expectation."""
    observed = residue_pair_frequency_table(df, q)
    phi_q = int(totient(q))
    expected = 1.0 / (phi_q ** 2)

    rows = []
    for a in observed.index:
        for b in observed.columns:
            obs_share = observed.loc[a, b]
            rows.append({
                "this_class": a,
                "next_class": b,
                "observed_share": obs_share,
                "expected_share": expected,
                "relative_deviation": (obs_share - expected) / expected,
                "same_class": a == b,
            })
    result = pd.DataFrame(rows).sort_values("relative_deviation", key=abs, ascending=False)
    return result.head(n_top)


def dominant_gaps_by_class(df, q, top_k=5):
    """Top-k most frequent gap values per residue class, with share."""
    sub = filter_admissible(df, q)
    rows = []
    for r, group in sub.groupby(f"res_mod{q}"):
        counts = group["gap"].value_counts(normalize=True).head(top_k)
        for rank, (gap_val, share) in enumerate(counts.items(), start=1):
            rows.append({"residue": r, "rank": rank, "gap": gap_val, "share": share})
    return pd.DataFrame(rows)
