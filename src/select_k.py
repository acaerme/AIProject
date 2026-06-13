import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.preprocess import RANDOM_STATE

K_RANGE = range(2, 11)


def sweep(X, k_range=K_RANGE):
    """Fit KMeans for each k and collect inertia + silhouette."""
    print("=" * 70)
    print("STEP 6 — CHOOSING k (elbow + silhouette sweep)")
    print("=" * 70)

    ks, inertias, silhouettes = [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                    random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        ks.append(k)
        inertias.append(km.inertia_)
        silhouettes.append(sil)
        print(f"  k={k:2d} | inertia={km.inertia_:12.1f} | silhouette={sil:.4f}")

    return ks, inertias, silhouettes


def plot_elbow(ks, inertias, path="figures/elbow.png"):
    """Save the inertia-vs-k elbow curve."""
    plt.figure(figsize=(7, 4.5))
    plt.plot(ks, inertias, "o-", color="#2c7fb8")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.title("Elbow method")
    plt.xticks(list(ks))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def plot_silhouette(ks, silhouettes, path="figures/silhouette.png"):
    """Save the silhouette-vs-k curve, marking the peak."""
    plt.figure(figsize=(7, 4.5))
    plt.plot(ks, silhouettes, "o-", color="#d95f0e")
    best_k = ks[int(max(range(len(silhouettes)), key=lambda i: silhouettes[i]))]
    best_s = max(silhouettes)
    plt.axvline(best_k, ls="--", color="gray", alpha=0.7,
                label=f"peak at k={best_k} ({best_s:.3f})")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Mean silhouette score")
    plt.title("Silhouette analysis")
    plt.xticks(list(ks))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def recommend_k(ks, silhouettes):
    """Report the silhouette-max k and a recommendation (not a blind argmax)."""
    sil_max_k = ks[int(max(range(len(silhouettes)), key=lambda i: silhouettes[i]))]
    print(f"\nSilhouette is maximised at k={sil_max_k}.")
    print("Recommendation: inspect the elbow + silhouette plots. If silhouette "
          "peaks at k=2, prefer k=3 or k=4 for richer, still well-separated and "
          "namable segments (set N_CLUSTERS in src/preprocess.py).")
    return sil_max_k


def run(X):
    """Full Step 6: sweep, save both plots, print recommendation."""
    ks, inertias, silhouettes = sweep(X)
    plot_elbow(ks, inertias)
    plot_silhouette(ks, silhouettes)
    recommend_k(ks, silhouettes)
    return ks, inertias, silhouettes


if __name__ == "__main__":
    from src.load_data import load
    from src.preprocess import run_preprocessing

    _, X, _ = run_preprocessing(load())
    run(X)
