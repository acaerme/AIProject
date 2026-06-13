import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.preprocess import (CLUSTER_FEATURES, DESCRIPTOR_COLS, N_CLUSTERS,
                            RANDOM_STATE)


def fit_final(X, k: int = N_CLUSTERS):
    """Step 8 — fit the final KMeans model (n_init=10 set explicitly)."""
    print("=" * 70)
    print(f"STEP 8 — FINAL sklearn KMeans (k={k})")
    print("=" * 70)
    model = KMeans(n_clusters=k, init="k-means++", n_init=10,
                   random_state=RANDOM_STATE)
    model.fit(X)
    print(f"Converged. Inertia={model.inertia_:.1f}, iterations={model.n_iter_}")
    print("Cluster sizes:", np.bincount(model.labels_).tolist())
    return model


def _value_tier_names(profile: pd.DataFrame) -> dict:
    """Name clusters by ascending mean Total_Spend (robust to label order)."""
    order = profile["Total_Spend"].sort_values().index.tolist()
    k = len(order)
    if k == 2:
        tiers = ["Budget Shoppers", "Premium Spenders"]
    elif k == 3:
        tiers = ["Budget / Low-Engagement", "Mid-Market Regulars",
                 "Premium High-Spenders"]
    else:
        tiers = [f"Tier {i+1} (spend rank {i+1}/{k})" for i in range(k)]
    return {cluster: tiers[rank] for rank, cluster in enumerate(order)}


def profile_clusters(df_feat: pd.DataFrame, labels: np.ndarray,
                     model: KMeans) -> pd.DataFrame:
    """Step 10 — build, print, and return the per-cluster profile table."""
    print("\n" + "=" * 70)
    print("STEP 10 — CLUSTER PROFILING")
    print("=" * 70)

    df = df_feat.copy()
    df["Cluster"] = labels

    # Mean of each clustering feature per cluster (real units).
    feat_profile = df.groupby("Cluster")[CLUSTER_FEATURES].mean()

    # Mean of each 0/1 descriptor per cluster = its rate within the cluster.
    desc_profile = df.groupby("Cluster")[DESCRIPTOR_COLS].mean()

    sizes = df["Cluster"].value_counts().sort_index()
    shares = (sizes / len(df) * 100).round(1)

    profile = feat_profile.copy()
    profile["Size"] = sizes
    profile["Share_%"] = shares
    for col in DESCRIPTOR_COLS:
        profile[col] = desc_profile[col]

    names = _value_tier_names(feat_profile)
    profile["Segment"] = profile.index.map(names)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("\nPer-cluster mean of CLUSTERING features (real units):")
    print(feat_profile.round(1).to_string())
    print("\nCluster sizes:")
    for c in sizes.index:
        print(f"  Cluster {c} -> {names[c]:28s} | n={sizes[c]:4d} ({shares[c]:.1f}%)")
    print("\nPer-cluster DESCRIPTOR rates (campaign accept / complaint), %:")
    print((desc_profile * 100).round(1).to_string())

    return profile, names


def plot_centroid_heatmap(model: KMeans, path="figures/cluster_heatmap.png"):
    """Heatmap of standardized centroids (clusters x 9 features)."""
    centers = pd.DataFrame(model.cluster_centers_, columns=CLUSTER_FEATURES)
    centers.index.name = "Cluster"
    plt.figure(figsize=(10, 0.9 * len(centers) + 2))
    sns.heatmap(centers, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, cbar_kws={"label": "standardized value (z)"})
    plt.title("Cluster centroids (standardized features)")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def plot_feature_bars(profile: pd.DataFrame, names: dict,
                      path="figures/cluster_feature_bars.png"):
    """Income, Total_Spend and Age per cluster, in real units."""
    cols = ["Income", "Total_Spend", "Age"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = [names[c] for c in profile.index]
    colors = sns.color_palette("Set2", len(profile))
    for ax, col in zip(axes, cols):
        ax.bar(range(len(profile)), profile[col], color=colors)
        ax.set_title(col)
        ax.set_xticks(range(len(profile)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Cluster means — headline features (real units)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def plot_campaign_rates(df_feat: pd.DataFrame, labels: np.ndarray, names: dict,
                        path="figures/cluster_campaign_rates.png"):
    """Per-cluster campaign acceptance + complaint rates (descriptors)."""
    df = df_feat.copy()
    df["Cluster"] = labels
    rates = df.groupby("Cluster")[DESCRIPTOR_COLS].mean() * 100
    ax = rates.plot(kind="bar", figsize=(11, 5), colormap="tab10")
    ax.set_ylabel("Rate within cluster (%)")
    ax.set_xlabel("Cluster")
    ax.set_title("Campaign acceptance & complaint rates by cluster (descriptors)")
    ax.set_xticklabels([names[c] for c in rates.index], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Descriptor", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def plot_cluster_sizes(profile: pd.DataFrame, names: dict,
                       path="figures/cluster_sizes.png"):
    """Bar chart of cluster sizes."""
    plt.figure(figsize=(7, 4.5))
    colors = sns.color_palette("Set2", len(profile))
    plt.bar([names[c] for c in profile.index], profile["Size"], color=colors)
    plt.ylabel("Number of customers")
    plt.title("Cluster sizes")
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def plot_pca_scatter(X, labels: np.ndarray, names: dict,
                     path="figures/cluster_pca.png"):
    """2-D PCA projection of the scaled data, coloured by cluster."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    proj = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_
    plt.figure(figsize=(7.5, 6))
    palette = sns.color_palette("Set2", len(np.unique(labels)))
    for c in np.unique(labels):
        m = labels == c
        plt.scatter(proj[m, 0], proj[m, 1], s=12, alpha=0.6,
                    color=palette[c], label=names[c])
    plt.xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
    plt.title(f"Clusters in 2-D PCA space "
              f"(total {ev.sum()*100:.1f}% variance shown)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")
    return ev


def profile_and_plot(df_feat, labels, model, X):
    """Run Step 10: profile table + all five plots. Returns (profile, names)."""
    profile, names = profile_clusters(df_feat, labels, model)
    print("\nSaving cluster plots to figures/ ...")
    plot_centroid_heatmap(model)
    plot_feature_bars(profile, names)
    plot_campaign_rates(df_feat, labels, names)
    plot_cluster_sizes(profile, names)
    plot_pca_scatter(X, labels, names)
    return profile, names


if __name__ == "__main__":
    from src.load_data import load
    from src.preprocess import run_preprocessing

    df_feat, X, scaler = run_preprocessing(load())
    model = fit_final(X)
    profile_and_plot(df_feat, model.labels_, model, X)
    print("\n[OK] Final model fit and clusters profiled.")
