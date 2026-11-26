# Explanation of Task 2

## What was wrong with the original demo
The script `demo/cluster_plot.py` ran without errors but did not do what it was designed to do.

1. **k was capped at 3.** The call to `run_clustering` passed `k=min(k, 3)`, so the runs
   labelled k = 4 and k = 5 actually used 3 clusters. This showed up in the output: the
   inertia and silhouette scores for k = 3, 4 and 5 were identical, even though inertia
   should always fall as k increases. **Fix:** pass `k=k`.
2. **The silhouette plot was never produced.** The script checked for a column called
   `"silhouette_score"`, but `run_clustering` returns the metric under the key
   `"silhouette"`. The condition was always false, so the bar chart was silently skipped.
   **Fix:** use `"silhouette"` in the condition and when plotting.

I also improved the output for the user: clear headings, rounded metrics, a summary table,
the k with the highest silhouette score, and a list of the files saved.

## What the corrected demo does
For an input CSV (for example `data/demo_data.csv`), the script:
- uses the first two numeric columns as features and standardises them;
- runs k-means for k = 2, 3, 4 and 5;
- saves, for each k, a cluster plot and a CSV of the data with cluster labels;
- saves a table of inertia and silhouette for every k, and a bar chart of silhouette against k;
- prints a summary and identifies the k with the highest silhouette score
  (k = 3 for `demo_data.csv`).

All outputs are saved in `demo_output/`.

## Overview of the cluster_maker package
`cluster_maker` is an educational package for generating, clustering, evaluating and
visualising data. Its main components are:

- **dataframe_builder**: defines cluster centres and simulates clustered data around them.
- **data_analyser**: descriptive statistics and correlation matrices.
- **data_exporter**: saves DataFrames to CSV or as formatted text tables.
- **preprocessing**: selects numeric feature columns (with checks for missing or
  non-numeric columns) and standardises features to zero mean and unit variance.
- **algorithms**: a manual k-means implementation (initialise centroids, assign points to
  the nearest centroid, update centroids) and a wrapper around scikit-learn's KMeans.
- **evaluation**: inertia, the silhouette score and the elbow curve.
- **plotting_clustered**: 2D cluster plots with centroids, and elbow plots.
- **interface**: `run_clustering`, which runs the whole workflow (load, select features,
  standardise, cluster, evaluate, plot and export) in a single call.