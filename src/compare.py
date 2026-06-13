import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix

from src.kmeans_scratch import KMeansScratch
from src.preprocess import N_CLUSTERS, RANDOM_STATE


def best_match_accuracy(labels_a, labels_b):
    """Agreement between two labelings under the best cluster permutation.

    Returns (accuracy, mapping, confusion_matrix).
    """
    C = confusion_matrix(labels_a, labels_b)
    row_ind, col_ind = linear_sum_assignment(-C)   # Hungarian: maximise overlap
    matched = C[row_ind, col_ind].sum()
    accuracy = matched / C.sum()
    mapping = {int(b): int(a) for a, b in zip(row_ind, col_ind)}
    return accuracy, mapping, C


def compare(X, k: int = N_CLUSTERS):
    """Fit both implementations on the same data and report agreement."""
    print("=" * 70)
    print(f"STEP 9 — SCRATCH vs SKLEARN (k={k})")
    print("=" * 70)

    scratch = KMeansScratch(n_clusters=k, n_init=10,
                            random_state=RANDOM_STATE).fit(X)
    sklearn_km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                        random_state=RANDOM_STATE).fit(X)

    print(f"\nInertia  -> scratch: {scratch.inertia_:.1f} | "
          f"sklearn: {sklearn_km.inertia_:.1f} | "
          f"diff: {abs(scratch.inertia_ - sklearn_km.inertia_):.1f}")

    acc, mapping, C = best_match_accuracy(sklearn_km.labels_, scratch.labels_)
    ari = adjusted_rand_score(sklearn_km.labels_, scratch.labels_)

    print("\nConfusion matrix (rows = sklearn cluster, cols = scratch cluster):")
    print(C)
    print(f"\nBest-match cluster mapping (scratch -> sklearn): {mapping}")
    print(f"\nBest-match agreement : {acc*100:.2f}%  "
          f"({int(round(acc*C.sum()))}/{C.sum()} customers)")
    print(f"Adjusted Rand Index  : {ari:.4f}  (1.0 = identical partitions)")

    if acc >= 0.95 and ari >= 0.90:
        print("\n[OK] High agreement — the scratch implementation matches sklearn.")
    else:
        print("\n[!] Lower-than-expected agreement; inspect (different local optima "
              "are possible but worth checking).")
    return {"accuracy": acc, "ari": ari, "mapping": mapping,
            "confusion": C, "scratch": scratch, "sklearn": sklearn_km}


if __name__ == "__main__":
    from src.load_data import load
    from src.preprocess import run_preprocessing

    _, X, _ = run_preprocessing(load())
    compare(X)
