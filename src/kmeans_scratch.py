import numpy as np

class KMeansScratch:
    """Minimal K-Means, vectorised with NumPy.

    Mirrors scikit-learn's public attributes (cluster_centers_, labels_,
    inertia_, n_iter_) so the two implementations can be compared directly.
    """

    def __init__(self, n_clusters, n_init=10, max_iter=300,
                 tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    @staticmethod
    def _squared_distances(X, centers):
        """Squared Euclidean distance from every point to every centroid (n, k).

        Uses ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 (one matrix product) and
        clips floating-point negatives to zero.
        """
        xx = np.sum(X * X, axis=1)[:, np.newaxis]              # (n, 1)
        cc = np.sum(centers * centers, axis=1)[np.newaxis, :]  # (1, k)
        cross = X @ centers.T                                  # (n, k)
        d2 = xx - 2.0 * cross + cc
        np.maximum(d2, 0, out=d2)
        return d2

    def _kmeanspp_init(self, X, rng):
        """Choose k initial centroids with k-means++ (D^2-weighted sampling)."""
        n, d = X.shape
        k = self.n_clusters
        centers = np.empty((k, d), dtype=X.dtype)

        first = rng.integers(n)
        centers[0] = X[first]

        # Squared distance of every point to its nearest chosen centroid.
        closest_sq = np.sum((X - centers[0]) ** 2, axis=1)

        for i in range(1, k):
            total = closest_sq.sum()
            if total == 0:
                probs = np.full(n, 1.0 / n)
            else:
                probs = closest_sq / total
            idx = rng.choice(n, p=probs)
            centers[i] = X[idx]
            new_sq = np.sum((X - centers[i]) ** 2, axis=1)
            closest_sq = np.minimum(closest_sq, new_sq)   # keep running nearest

        return centers

    def _single_run(self, X, rng):
        """Run Lloyd's algorithm once; return (centers, labels, inertia, n_iter)."""
        n = X.shape[0]
        k = self.n_clusters
        centers = self._kmeanspp_init(X, rng)
        labels = np.full(n, -1)

        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            # Assignment step.
            d2 = self._squared_distances(X, centers)
            new_labels = np.argmin(d2, axis=1)

            # Update step.
            new_centers = np.empty_like(centers)
            counts = np.bincount(new_labels, minlength=k)
            for j in range(k):
                if counts[j] > 0:
                    new_centers[j] = X[new_labels == j].mean(axis=0)
            # Re-seed empty clusters onto the farthest (worst-served) point.
            empties = np.where(counts == 0)[0]
            if empties.size > 0:
                dist_to_own = d2[np.arange(n), new_labels].copy()
                for j in empties:
                    far = np.argmax(dist_to_own)
                    new_centers[j] = X[far]
                    dist_to_own[far] = -1.0

            # Convergence: centroids barely moved, or assignments unchanged.
            shift = np.sqrt(((new_centers - centers) ** 2).sum(axis=1)).max()
            labels_unchanged = np.array_equal(new_labels, labels)

            centers = new_centers
            labels = new_labels

            if shift <= self.tol or labels_unchanged:
                break

        d2 = self._squared_distances(X, centers)
        labels = np.argmin(d2, axis=1)
        inertia = d2[np.arange(n), labels].sum()
        return centers, labels, float(inertia), n_iter

    def fit(self, X):
        """Run n_init restarts and keep the lowest-inertia solution."""
        X = np.asarray(X, dtype=float)
        if self.n_clusters > X.shape[0]:
            raise ValueError("n_clusters cannot exceed the number of samples.")

        # One Generator for the whole fit so restarts differ but stay reproducible.
        rng = np.random.default_rng(self.random_state)

        best = None
        for _ in range(self.n_init):
            centers, labels, inertia, n_iter = self._single_run(X, rng)
            if best is None or inertia < best[0]:
                best = (inertia, centers, labels, n_iter)

        self.inertia_, self.cluster_centers_, self.labels_, self.n_iter_ = best
        return self

    def predict(self, X):
        """Assign new points to the nearest fitted centroid."""
        X = np.asarray(X, dtype=float)
        d2 = self._squared_distances(X, self.cluster_centers_)
        return np.argmin(d2, axis=1)

    def fit_predict(self, X):
        """Fit then return the training labels."""
        return self.fit(X).labels_


if __name__ == "__main__":
    # Self-test on synthetic blobs (no dataset needed).
    rng = np.random.default_rng(0)
    blob = np.vstack([
        rng.normal(loc, 0.25, size=(50, 2))
        for loc in ([0, 0], [6, 6], [0, 6])
    ])
    km = KMeansScratch(n_clusters=3, random_state=42).fit(blob)
    print("Self-test on 3 synthetic blobs:")
    print("  inertia:", round(km.inertia_, 3))
    print("  iterations (best run):", km.n_iter_)
    print("  cluster sizes:", np.bincount(km.labels_).tolist())
    print("  centroids:\n", np.round(km.cluster_centers_, 2))
