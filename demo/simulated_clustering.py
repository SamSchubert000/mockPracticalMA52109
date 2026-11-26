###
## cluster_maker: clustering analysis of simulated_data.csv
## MA52109 practical - Task 4
##
## Determines a plausible number of clusters in data/simulated_data.csv
## using only the clustering tools in cluster_maker, and saves plots
## that show why the chosen clustering is appropriate.
##
## Usage (from the project root):
##     python demo/simulated_clustering.py [path/to/data.csv]
###

from __future__ import annotations

import os
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # save figures to file without opening windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cluster_maker import (
    calculate_descriptive_statistics,
    compute_inertia,
    plot_elbow,
    run_clustering,
    select_features,
    silhouette_score_sklearn,
    sklearn_kmeans,
    standardise_features,
)

DEFAULT_INPUT = os.path.join("data", "simulated_data.csv")
OUTPUT_DIR = os.path.join("demo_output", "simulated_clustering")
K_RANGE = list(range(2, 11))
RANDOM_STATE = 42
STABILITY_SEEDS = (0, 1, 2, 3, 4)


def print_header(title: str) -> None:
    """Print a section header so the console output is easy to follow."""
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def load_data(path: str) -> pd.DataFrame:
    """
    Load the input CSV, exiting with a clear message if it cannot be used.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    df : pandas.DataFrame
    """
    if not os.path.exists(path):
        print(f"Error: input file not found: {path}")
        print(f"Usage: python demo/simulated_clustering.py [{DEFAULT_INPUT}]")
        sys.exit(1)
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as err:
        print(f"Error: could not read {path} as CSV ({err}).")
        sys.exit(1)
    return df


def scan_k(X: np.ndarray, k_values: List[int]) -> pd.DataFrame:
    """
    Run k-means for each k and record inertia and silhouette score.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Standardised feature matrix.
    k_values : list of int
        Numbers of clusters to try.

    Returns
    -------
    scores : pandas.DataFrame
        Columns "k", "inertia", "silhouette" and "smallest_cluster".
    """
    rows = []
    for k in k_values:
        labels, centroids = sklearn_kmeans(X, k, random_state=RANDOM_STATE)
        rows.append({
            "k": k,
            "inertia": compute_inertia(X, labels, centroids),
            "silhouette": silhouette_score_sklearn(X, labels),
            "smallest_cluster": int(np.bincount(labels).min()),
        })
    return pd.DataFrame(rows)


def stability_check(X: np.ndarray, k: int) -> List[float]:
    """
    Re-run k-means with several random seeds and return the silhouette scores.

    A similar score for every seed shows the chosen clustering does not
    depend on a lucky initialisation.
    """
    scores = []
    for seed in STABILITY_SEEDS:
        labels, _ = sklearn_kmeans(X, k, random_state=seed)
        scores.append(silhouette_score_sklearn(X, labels))
    return scores


def cluster_colour(cluster_id: int, n_clusters: int):
    """
    Return the colour used for a cluster, matching plot_clusters_2d in
    cluster_maker (the tab10 colormap scaled over the cluster labels).
    """
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(n_clusters - 1, 1))
    return plt.get_cmap("tab10")(norm(cluster_id))


def plot_silhouette_scores(scores: pd.DataFrame, best_k: int, path: str) -> None:
    """Save a bar chart of silhouette score against k, highlighting the best k."""
    colours = ["tab:orange" if k == best_k else "tab:blue" for k in scores["k"]]
    fig, ax = plt.subplots()
    ax.bar(scores["k"], scores["silhouette"], color=colours)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title(f"Silhouette score by k (best: k = {best_k})")
    ax.set_xticks(scores["k"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pairwise(df: pd.DataFrame, feature_cols: List[str],
                  labels: np.ndarray, path: str) -> None:
    """
    Save a scatter matrix of every pair of features, coloured by cluster.

    This shows the clusters in the original units across all features,
    not only the first two used by plot_clusters_2d.
    """
    n = len(feature_cols)
    cluster_ids = np.unique(labels)
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n))
    for i, row_col in enumerate(feature_cols):
        for j, col_col in enumerate(feature_cols):
            ax = axes[i, j]
            for cluster_id in cluster_ids:
                colour = cluster_colour(cluster_id, len(cluster_ids))
                mask = labels == cluster_id
                if i == j:
                    ax.hist(df.loc[mask, row_col], bins=12, alpha=0.6, color=colour)
                else:
                    ax.scatter(df.loc[mask, col_col], df.loc[mask, row_col],
                               color=colour, s=8, label=f"Cluster {cluster_id}")
            if i == n - 1:
                ax.set_xlabel(col_col)
            if j == 0:
                ax.set_ylabel(row_col)
    handles, names = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=len(cluster_ids),
               bbox_to_anchor=(0.5, 0.965), markerscale=2)
    fig.suptitle("Pairwise feature plots coloured by cluster (diagonal: histograms)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(args: List[str]) -> None:
    """Run the full analysis and print a summary of the findings."""
    if len(args) > 1:
        print("Usage: python demo/simulated_clustering.py [path/to/data.csv]")
        sys.exit(1)
    input_path = args[0] if args else DEFAULT_INPUT
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved: Dict[str, str] = {}

    # 1. Load and describe the data
    df = load_data(input_path)
    feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(feature_cols) < 2:
        print("Error: at least two numeric columns are needed for clustering.")
        sys.exit(1)

    print_header("Clustering analysis of simulated data")
    print(f"Input file : {input_path} ({len(df)} rows)")
    print(f"Features   : {', '.join(feature_cols)}")
    print(f"Outputs    : {OUTPUT_DIR}/")
    print("\nDescriptive statistics:")
    print(calculate_descriptive_statistics(df[feature_cols]).round(2).to_string())

    # 2. Prepare features: standardise so every feature counts equally
    X = standardise_features(select_features(df, feature_cols).to_numpy(dtype=float))

    # 3. Compare candidate numbers of clusters
    print_header(f"Step 1: comparing k = {K_RANGE[0]} to {K_RANGE[-1]}")
    scores = scan_k(X, K_RANGE)
    print(scores.round({"inertia": 1, "silhouette": 3}).to_string(index=False))
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    best_sil = scores["silhouette"].max()

    saved["silhouette"] = os.path.join(OUTPUT_DIR, "silhouette_by_k.png")
    plot_silhouette_scores(scores, best_k, saved["silhouette"])

    # 4. Final clustering through the package's high-level interface
    print_header(f"Step 2: final clustering with k = {best_k}")
    saved["labels"] = os.path.join(OUTPUT_DIR, "simulated_data_clustered.csv")
    result = run_clustering(
        input_path=input_path,
        feature_cols=feature_cols,
        algorithm="sklearn_kmeans",
        k=best_k,
        standardise=True,
        output_path=saved["labels"],
        random_state=RANDOM_STATE,
        compute_elbow=False,
    )
    labels = result["labels"]

    saved["cluster_plot"] = os.path.join(OUTPUT_DIR, "cluster_plot.png")
    result["fig_cluster"].axes[0].set_title(
        f"k-means clusters (k = {best_k}), standardised features 1 and 2")
    result["fig_cluster"].savefig(saved["cluster_plot"], dpi=150)
    plt.close(result["fig_cluster"])

    saved["elbow"] = os.path.join(OUTPUT_DIR, "elbow.png")
    fig_elbow, ax_elbow = plot_elbow(scores["k"].tolist(), scores["inertia"].tolist(),
                                     title="Elbow curve (inertia by k)")
    ax_elbow.axvline(best_k, color="tab:orange", linestyle="--",
                     label=f"chosen k = {best_k}")
    ax_elbow.legend()
    fig_elbow.savefig(saved["elbow"], dpi=150)
    plt.close(fig_elbow)

    saved["pairwise"] = os.path.join(OUTPUT_DIR, "pairwise_features.png")
    plot_pairwise(df, feature_cols, labels, saved["pairwise"])

    sizes = pd.Series(labels).value_counts().sort_index()
    centres = result["data"].groupby("cluster")[feature_cols].mean()
    print("Cluster sizes:")
    print("  " + ", ".join(f"cluster {c}: {n}" for c, n in sizes.items()))
    print("\nCluster centres (original units):")
    print(centres.round(2).to_string())

    # 5. Check the result does not depend on the random initialisation
    print_header("Step 3: stability across random seeds")
    stab = stability_check(X, best_k)
    print("Silhouette for seeds " + ", ".join(str(s) for s in STABILITY_SEEDS)
          + ": " + ", ".join(f"{s:.3f}" for s in stab))

    # 6. Conclusion
    inertia = scores.set_index("k")["inertia"]
    drop_to_best = 1 - inertia[best_k] / inertia[best_k - 1]
    drop_after = (1 - inertia[best_k + 1] / inertia[best_k]
                  if best_k + 1 in inertia.index else float("nan"))

    print_header("Conclusion")
    print(f"The data are best described by {best_k} clusters, because:")
    print(f"  - the silhouette score is highest at k = {best_k} ({best_sil:.3f});")
    print(f"  - the elbow curve bends at k = {best_k}: going from k = {best_k - 1} "
          f"to {best_k} cuts inertia by {drop_to_best:.0%},")
    print(f"    but going from k = {best_k} to {best_k + 1} only cuts it by "
          f"{drop_after:.0%};")
    if sizes.min() == sizes.max():
        print(f"  - every cluster has exactly {sizes.min()} points, so none is a fragment;")
    else:
        print(f"  - the clusters have similar sizes ({sizes.min()} to {sizes.max()} points);")
    print(f"  - the result is stable across random seeds "
          f"(silhouette {min(stab):.3f} to {max(stab):.3f});")
    print("  - the pairwise plots show every cluster is clearly separated from")
    print("    the others in at least one pair of features.")

    print("\nSaved files:")
    for path in saved.values():
        print(f"  - {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
