import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import CLUSTER_FEATURES


def plot_distributions(df_feat, path="figures/eda_distributions.png"):
    """Histogram (with KDE) of each clustering feature, in real units."""
    n = len(CLUSTER_FEATURES)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows))
    axes = axes.ravel()
    for ax, col in zip(axes, CLUSTER_FEATURES):
        sns.histplot(df_feat[col], bins=30, kde=True, ax=ax, color="#2c7fb8")
        ax.set_title(col)
        ax.set_xlabel("")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Distributions of the nine clustering features (after cleaning)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def plot_correlation(df_feat, path="figures/eda_correlation.png"):
    """Correlation heatmap of the nine clustering features (Pearson)."""
    corr = df_feat[CLUSTER_FEATURES].corr()
    plt.figure(figsize=(8.5, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, cbar_kws={"label": "Pearson r"})
    plt.title("Correlation between the nine clustering features")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")
    return corr


def run(df_feat):
    """Generate the EDA figures and print the strongest correlations."""
    print("=" * 70)
    print("EDA — feature distributions + correlations")
    print("=" * 70)
    plot_distributions(df_feat)
    corr = plot_correlation(df_feat)
    pairs = (corr.where(~(corr.values == 1.0))
                 .stack()
                 .sort_values(ascending=False))
    seen, top = set(), []
    for (a, b), v in pairs.items():
        key = frozenset((a, b))
        if key in seen:
            continue
        seen.add(key)
        top.append((a, b, v))
    print("\nStrongest feature correlations (|r|, deduplicated):")
    for a, b, v in sorted(top, key=lambda t: -abs(t[2]))[:8]:
        print(f"  {a:20s} vs {b:20s} r = {v:+.2f}")
    return corr


if __name__ == "__main__":
    from src.load_data import load
    from src.preprocess import clean, engineer

    df_feat = engineer(clean(load()))
    run(df_feat)
