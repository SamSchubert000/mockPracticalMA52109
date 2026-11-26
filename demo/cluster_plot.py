###
## cluster_maker: demo for cluster analysis
## James Foadi - University of Bath
## November 2025
##
## This script produces clustering for a group of points in 2D,
## using k-means for k = 2, 3, 4, 5. The input file is the csv
## file 'demo_data.csv' in folder 'data/'.
###

from __future__ import annotations

import os
import sys
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import run_clustering

OUTPUT_DIR = "demo_output"
K_VALUES = (2, 3, 4, 5)


def main(args: List[str]) -> None:
    """
    Run k-means on a CSV file for several values of k, save the outputs
    and print a summary of the clustering metrics.

    Parameters
    ----------
    args : list of str
        Command-line arguments; expects a single path to an input CSV file.
    """
    if len(args) != 1:
        print("Usage: python demo/cluster_plot.py <input_csv>")
        print("Example: python demo/cluster_plot.py data/demo_data.csv")
        sys.exit(1)

    input_path = args[0]
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # The CSV is assumed to have two or more numeric columns
    df = pd.read_csv(input_path)
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(numeric_cols) < 2:
        print("Error: The input CSV must have at least two numeric columns.")
        sys.exit(1)
    feature_cols = numeric_cols[:2]  # Use the first two numeric columns

    print("=" * 60)
    print("cluster_maker demo: k-means for k = 2, 3, 4, 5")
    print("=" * 60)
    print(f"Input file : {input_path} ({len(df)} rows)")
    print(f"Features   : {', '.join(feature_cols)} (standardised)")
    print(f"Outputs    : {OUTPUT_DIR}/")

    base = os.path.splitext(os.path.basename(input_path))[0]
    metrics_summary = []

    for k in K_VALUES:
        print(f"\n--- Running k-means with k = {k} ---")

        result = run_clustering(
            input_path=input_path,
            feature_cols=feature_cols,
            algorithm="kmeans",
            k=k,
            standardise=True,
            output_path=os.path.join(OUTPUT_DIR, f"{base}_clustered_k{k}.csv"),
            random_state=42,
            compute_elbow=False,
        )

        plot_path = os.path.join(OUTPUT_DIR, f"{base}_k{k}.png")
        result["fig_cluster"].savefig(plot_path, dpi=150)
        plt.close(result["fig_cluster"])

        metrics = {"k": k}
        metrics.update(result.get("metrics", {}))
        metrics_summary.append(metrics)

        sil = metrics.get("silhouette")
        sil_text = f"{sil:.3f}" if sil is not None else "n/a"
        print(f"  Inertia    : {metrics['inertia']:.2f}")
        print(f"  Silhouette : {sil_text}")
        print(f"  Saved      : {plot_path}")

    metrics_df = pd.DataFrame(metrics_summary)
    metrics_csv = os.path.join(OUTPUT_DIR, f"{base}_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    # Bar chart of silhouette score against k
    if "silhouette" in metrics_df.columns:
        fig, ax = plt.subplots()
        ax.bar(metrics_df["k"], metrics_df["silhouette"])
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Silhouette score")
        ax.set_title("Silhouette score for different k")
        ax.set_xticks(metrics_df["k"])
        stats_path = os.path.join(OUTPUT_DIR, f"{base}_silhouette.png")
        fig.savefig(stats_path, dpi=150)
        plt.close(fig)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(metrics_df.round({"inertia": 2, "silhouette": 3}).to_string(index=False))
    best_k = int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])
    print(f"\nHighest silhouette score at k = {best_k} "
          "(higher means better-separated clusters).")
    print(f"\nAll outputs saved in: {OUTPUT_DIR}/")
    print(f"  - cluster plots and labelled CSVs for each k")
    print(f"  - {base}_metrics.csv (metrics table)")
    print(f"  - {base}_silhouette.png (silhouette bar chart)")


if __name__ == "__main__":
    main(sys.argv[1:])