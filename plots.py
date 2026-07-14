"""Figure-generating functions. All figures saved to figures/ by default."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import stats_tests as st


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_gap_histograms_by_class(df, q, out_dir="figures", max_gap=60):
    """Overlaid gap-distribution histograms, one per admissible class."""
    _ensure_dir(out_dir)
    sub = st.filter_admissible(df, q)
    classes = sorted(sub[f"res_mod{q}"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, max_gap + 2, 2)
    for r in classes:
        gaps = sub.loc[sub[f"res_mod{q}"] == r, "gap"]
        ax.hist(gaps, bins=bins, histtype="step", density=True,
                label=f"p ≡ {r} (mod {q})", linewidth=1.5)
    ax.set_xlabel("Gap to next prime")
    ax.set_ylabel("Density")
    ax.set_title(f"Gap distribution by residue class (mod {q})")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"gap_histograms_mod{q}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] saved {out_path}")
    return out_path


def plot_residue_pair_heatmap(df, q, out_dir="figures"):
    """Heatmap of observed (p_n mod q, p_{n+1} mod q) frequencies."""
    _ensure_dir(out_dir)
    table = st.residue_pair_frequency_table(df, q)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(table.values, cmap="viridis")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_xlabel("Residue of next prime (mod %d)" % q)
    ax.set_ylabel("Residue of prime (mod %d)" % q)
    ax.set_title(f"Observed consecutive-prime residue-pair frequency (mod {q})")
    fig.colorbar(im, ax=ax, label="Observed frequency")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"residue_pair_heatmap_mod{q}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] saved {out_path}")
    return out_path


def plot_dominant_gaps(df, q, out_dir="figures", top_k=5):
    """Grouped bar chart of the top_k most frequent gap values per class."""
    _ensure_dir(out_dir)
    table = st.dominant_gaps_by_class(df, q, top_k=top_k)
    classes = sorted(table["residue"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / len(classes)
    for i, r in enumerate(classes):
        sub = table[table["residue"] == r].sort_values("rank")
        x = np.arange(len(sub)) + i * width
        ax.bar(x, sub["share"], width=width, label=f"p ≡ {r} (mod {q})")
        for xi, gap_val in zip(x, sub["gap"]):
            ax.text(xi, 0.002, str(int(gap_val)), rotation=90, fontsize=7, va="bottom")
    ax.set_xlabel(f"Rank (bar labeled with the gap value, top {top_k} per class)")
    ax.set_ylabel("Share within class")
    ax.set_title(f"Most frequent gap sizes by residue class (mod {q})")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"dominant_gaps_mod{q}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] saved {out_path}")
    return out_path


def plot_effect_size_vs_N(effect_size_df, q, out_dir="figures"):
    """Effect size (Cramer's V) vs. N. Expects columns ["N", "cramers_v"]."""
    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(effect_size_df["N"], effect_size_df["cramers_v"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("N (upper limit of prime search)")
    ax.set_ylabel("Effect size (Cramer's V, class vs. gap)")
    ax.set_title(f"Bias magnitude vs. N (mod {q})")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"effect_size_vs_N_mod{q}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] saved {out_path}")
    return out_path


def plot_metric_vs_N_multi_modulus(trend_df, metric_col, ylabel, title, out_name, out_dir="figures"):
    """Plots metric_col vs. N, one line per modulus in trend_df."""
    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    for q, group in trend_df.groupby("modulus"):
        group = group.sort_values("N")
        ax.plot(group["N"], group[metric_col], marker="o", label=f"mod {q}")
    ax.set_xscale("log")
    ax.set_xlabel("N (upper limit of prime search)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{out_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] saved {out_path}")
    return out_path


def plot_real_vs_cramer_model(df_real, df_simulated, q, out_dir="figures", max_gap=60):
    """Overlays real vs. Cramer-model-simulated gap histograms, per class."""
    _ensure_dir(out_dir)
    sub_real = st.filter_admissible(df_real, q)
    sub_sim = st.filter_admissible(df_simulated, q)
    classes = sorted(sub_real[f"res_mod{q}"].unique())
    bins = np.arange(0, max_gap + 2, 2)

    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 4), sharey=True)
    if len(classes) == 1:
        axes = [axes]
    for ax, r in zip(axes, classes):
        real_gaps = sub_real.loc[sub_real[f"res_mod{q}"] == r, "gap"]
        sim_gaps = sub_sim.loc[sub_sim[f"res_mod{q}"] == r, "gap"]
        ax.hist(real_gaps, bins=bins, density=True, alpha=0.5, label="Real primes")
        ax.hist(sim_gaps, bins=bins, density=True, histtype="step", linewidth=2,
                label="Cramér model", color="black")
        ax.set_title(f"p ≡ {r} (mod {q})")
        ax.set_xlabel("Gap")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Density")
    fig.suptitle(f"Real vs. Cramér-model gap distributions (mod {q})")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"real_vs_cramer_mod{q}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] saved {out_path}")
    return out_path
