###
## cluster_maker
## Agglomerative (hierarchical) clustering
## MA52109 practical - Task 5
###

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering

VALID_LINKAGES = ("ward", "complete", "average", "single")


def radial_features(
    X: np.ndarray,
    centre: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Transform points into their Euclidean distance from a centre.

    Useful for data made of concentric rings or shells, where cluster
    membership depends on the distance from the centre rather than on the
    position itself. Clustering this single feature separates the rings,
    which distance-based methods on the raw coordinates cannot do.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    centre : ndarray of shape (n_features,) or None, default None
        Centre to measure distances from. If None, the mean of X is used.

    Returns
    -------
    radii : ndarray of shape (n_samples, 1)
        Distance of each sample from the centre, as a single-column array
        ready to pass to any clustering function in cluster_maker.
    """
    X = _validate_array(X)
    if centre is None:
        centre = X.mean(axis=0)
    centre = np.asarray(centre, dtype=float)
    if centre.shape != (X.shape[1],):
        raise ValueError(
            f"centre must have shape ({X.shape[1]},), got {centre.shape}."
        )
    return np.linalg.norm(X - centre, axis=1).reshape(-1, 1)


def agglomerative_clustering(
    X: np.ndarray,
    k: int,
    linkage: str = "ward",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hierarchical clustering using scikit-learn's AgglomerativeClustering.

    Each sample starts as its own cluster, and the closest pair of clusters
    (according to `linkage`) is merged repeatedly until `k` clusters remain.
    The result is deterministic, so no random_state is needed.

    The return values match `kmeans` and `sklearn_kmeans`, so the output can
    be passed directly to `compute_inertia`, `silhouette_score_sklearn` and
    `plot_clusters_2d`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k : int
        Number of clusters; must be between 1 and n_samples.
    linkage : {"ward", "complete", "average", "single"}, default "ward"
        Rule for measuring the distance between two clusters.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
        Mean of the points in each cluster (agglomerative clustering has no
        centroids of its own; these are computed for consistency with k-means).
    """
    X = _validate_array(X)
    n_samples = X.shape[0]
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise TypeError("k must be an integer.")
    if not 1 <= k <= n_samples:
        raise ValueError(
            f"k must be between 1 and the number of samples ({n_samples}), got {k}."
        )
    if linkage not in VALID_LINKAGES:
        raise ValueError(
            f"Unknown linkage '{linkage}'. Use one of: {', '.join(VALID_LINKAGES)}."
        )

    if k == 1:
        # scikit-learn needs at least 2 samples; one cluster is trivial anyway.
        labels = np.zeros(n_samples, dtype=int)
    else:
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = model.fit_predict(X)

    centroids = np.array([X[labels == cluster_id].mean(axis=0) for cluster_id in range(k)])
    return labels, centroids


def _validate_array(X: np.ndarray) -> np.ndarray:
    """Check that X is a finite, non-empty 2D NumPy array and return it as floats."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2D array of shape (n_samples, n_features).")
    X = X.astype(float)
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains missing or infinite values.")
    return X
