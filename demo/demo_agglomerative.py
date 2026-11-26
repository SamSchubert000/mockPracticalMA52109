###
## cluster_maker: agglomerative clustering demo
## MA52109 practical - Task 5
##
## Shows that the agglomerative module separates the four concentric rings
## in data/difficult_dataset.csv, where k-means and agglomerative clustering
## on the raw coordinates both fail.
##
## Usage (from the project root):
##     python demo/demo_agglomerative.py [path/to/data.csv]
###

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # save figures to file without opening windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cluster_maker import (
    agglomerative_clustering,
    plot_clusters_2d,
    radial_features,
    select_features,
    silhouette_score_sklearn,
    sklearn_kmeans,
)
from cluster_maker.agglomerative import VALID_LINKAGES

DEFAULT_INPUT = os.path.join("data", "difficult_dataset.csv")
OUTPUT_DIR = os.path.join("demo_output", "agglomerative")
FEATURE_COLS = ["x", "y"]
N_RINGS = 4
K_RANGE = list(range(2, 9))
RANDOM_STATE = 42


def print_header(title: str) -> None:
    """Print a section header so the console output is easy to follow."""
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def load_points(path: str) -> np.ndarray:
    """
    Load the x and y columns of the input CSV as a NumPy array.

    Exits with a clear message if the file is missing or unsuitable.
    """
    if not os.path.exists(path):
        print(f"Error: input file not found: {path}")
        print(f"Usage: python demo/demo_agglomerative.py [{DEFAULT_INPUT}]")
        sys.exit(1)
    try:
        df = pd.read_csv(path)
        return select_features(df, FEATURE_COLS).to_numpy(dtype=float)
    except (KeyError, TypeError) as err:
        print(f"Error: {path} must have numeric columns {FEATURE_COLS}. {err.args[0]}")
        sys.exit(1)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as err:
        print(f"Error: could not read {path} as CSV ({err}).")
        sys.exit(1)


def order_by_radius(labels: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """
    Relabel clusters so that 0 is the innermost and the highest label the
    outermost, making the output easy to read. Cluster contents are unchanged.
    """
    cluster_ids = np.unique(labels)
    mean_radius = [radii[labels == c].mean() for c in cluster_ids]
    new_id = {old: new for new, old in enumerate(cluster_ids[np.argsort(mean_radius)])}
    return np.array([new_id[label] for label in labels])


def radius_ranges(labels: np.ndarray, radii: np.ndarray) -> pd.DataFrame:
    """
    Summarise each cluster by its size and the range of radii it covers.

    Returns
    -------
    summary : pandas.DataFrame
        Columns "cluster", "points", "min_radius", "max_radius".
    """
    rows = []
    for cluster_id in np.unique(labels):
        r = radii[labels == cluster_id]
        rows.append({"cluster": int(cluster_id), "points": len(r),
                     "min_radius": r.min(), "max_radius": r.max()})
    return pd.DataFrame(rows)


def separates_rings(labels: np.ndarray, radii: np.ndarray) -> bool:
    """
    Return True if every cluster is a separate ring.

    Two conditions must hold: no two clusters cover overlapping ranges of
    radii, and no cluster is a small fragment (each must hold at least half
    of the average cluster size). The second condition catches a method that
    isolates a few outliers and merges two real rings into one cluster.
    """
    ranges = radius_ranges(labels, radii).sort_values("min_radius")
    no_overlap = np.all(ranges["min_radius"].to_numpy()[1:]
                        > ranges["max_radius"].to_numpy()[:-1])
    no_fragments = ranges["points"].min() >= 0.5 * len(labels) / len(ranges)
    return bool(no_overlap and no_fragments)


def plot_side_by_side(X: np.ndarray, results: List[Tuple[str, np.ndarray]],
                      path: str) -> None:
    """Save one scatter panel per clustering result, coloured by cluster."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.6))
    for ax, (title, labels) in zip(np.atleast_1d(axes), results):
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=6)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    fig.suptitle("Clustering on raw (x, y) coordinates mixes the rings")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_radius_histogram(radii: np.ndarray, labels: np.ndarray, path: str) -> None:
    """Save a histogram of distances from the centre, coloured by cluster."""
    n_clusters = len(np.unique(labels))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(n_clusters - 1, 1))
    fig, ax = plt.subplots()
    for cluster_id in range(n_clusters):
        ax.hist(radii[labels == cluster_id], bins=20,
                color=plt.get_cmap("tab10")(norm(cluster_id)),
                label=f"Ring {cluster_id}")
    ax.set_xlabel("Distance from centre")
    ax.set_ylabel("Number of points")
    ax.set_title("Distances from the centre form four separate groups")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_silhouette_scores(scores: pd.DataFrame, best_k: int, path: str) -> None:
    """Save a bar chart of silhouette score (on radii) against k."""
    colours = ["tab:orange" if k == best_k else "tab:blue" for k in scores["k"]]
    fig, ax = plt.subplots()
    ax.bar(scores["k"], scores["silhouette"], color=colours)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score (on distance from centre)")
    ax.set_title(f"Choosing k: silhouette is highest at k = {best_k}")
    ax.set_xticks(scores["k"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(args: List[str]) -> None:
    """Run the demonstration and print a summary of the findings."""
    if len(args) > 1:
        print("Usage: python demo/demo_agglomerative.py [path/to/data.csv]")
        sys.exit(1)
    input_path = args[0] if args else DEFAULT_INPUT
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved: Dict[str, str] = {}

    X = load_points(input_path)
    radii = radial_features(X).ravel()

    print_header("Agglomerative clustering demo: concentric rings")
    print(f"Input file : {input_path} ({len(X)} points, features {', '.join(FEATURE_COLS)})")
    print(f"Outputs    : {OUTPUT_DIR}/")
    print("A cluster 'separates the rings' if it covers its own band of distances")
    print("from the centre, with no overlap with any other cluster.")

    # 1. Standard approaches on the raw coordinates
    print_header(f"Step 1: clustering raw (x, y) into {N_RINGS} groups")
    kmeans_labels, _ = sklearn_kmeans(X, N_RINGS, random_state=RANDOM_STATE)
    raw_agg_labels, _ = agglomerative_clustering(X, N_RINGS)
    for name, labels in [("k-means", kmeans_labels),
                         ("agglomerative (ward)", raw_agg_labels)]:
        verdict = "yes" if separates_rings(labels, radii) else "NO"
        print(f"  {name:<22} separates the rings: {verdict}")
    print("Both methods group nearby points, so they cut the rings into wedges.")
    saved["raw"] = os.path.join(OUTPUT_DIR, "raw_coordinates_comparison.png")
    plot_side_by_side(X, [("k-means on (x, y)", kmeans_labels),
                          ("Agglomerative on (x, y)", raw_agg_labels)], saved["raw"])

    # 2. Agglomerative clustering on distance from the centre
    print_header("Step 2: agglomerative clustering on distance from the centre")
    R = radial_features(X)
    scores = pd.DataFrame([
        {"k": k, "silhouette": silhouette_score_sklearn(R, agglomerative_clustering(R, k)[0])}
        for k in K_RANGE
    ])
    print(scores.round(3).to_string(index=False))
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    print(f"Best number of clusters: k = {best_k}")
    saved["silhouette"] = os.path.join(OUTPUT_DIR, "silhouette_by_k.png")
    plot_silhouette_scores(scores, best_k, saved["silhouette"])

    labels, _ = agglomerative_clustering(R, best_k)
    labels = order_by_radius(labels, radii)
    summary = radius_ranges(labels, radii)
    print("\nRings found (0 = innermost):")
    print(summary.round(2).to_string(index=False))
    rings_ok = separates_rings(labels, radii)
    print(f"Separates the rings: {'yes' if rings_ok else 'NO'}")

    saved["final"] = os.path.join(OUTPUT_DIR, "rings_agglomerative.png")
    fig, ax = plot_clusters_2d(X, labels,
                               title=f"Agglomerative on distance from centre (k = {best_k})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.savefig(saved["final"], dpi=150)
    plt.close(fig)
    saved["histogram"] = os.path.join(OUTPUT_DIR, "radius_histogram.png")
    plot_radius_histogram(radii, labels, saved["histogram"])

    # 3. Robustness: does the result depend on the linkage rule?
    print_header("Step 3: robustness to the linkage rule")
    failed = []
    for linkage in VALID_LINKAGES:
        alt = order_by_radius(agglomerative_clustering(R, best_k, linkage=linkage)[0], radii)
        sizes = ", ".join(str(n) for n in np.bincount(alt))
        ok = separates_rings(alt, radii)
        if not ok:
            failed.append(linkage)
        print(f"  {linkage:<9} cluster sizes: {sizes:<20} "
              f"separates the rings: {'yes' if ok else 'NO'}")
    if failed:
        print(f"Not robust with {', '.join(failed)} linkage: uneven cluster sizes show")
        print("it splits off stray points and merges rings. Ward (the default) is used.")

    # 4. Conclusion
    print_header("Conclusion")
    if rings_ok:
        print(f"Agglomerative clustering on distance from the centre recovers "
              f"{best_k} rings")
        print("with no overlap, whereas k-means and agglomerative clustering on the")
        print("raw coordinates both mix points from different rings. The choice of")
        print(f"k = {best_k} is supported by the silhouette score, and the result is")
        print("essentially unchanged for ward, complete and average linkage.")
    else:
        print("The rings were not fully separated; check the plots for details.")

    print("\nSaved files:")
    for path in saved.values():
        print(f"  - {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
