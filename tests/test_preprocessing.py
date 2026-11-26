###
## cluster_maker - unit tests for preprocessing.py
## MA52109 practical
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.preprocessing import select_features, standardise_features


class TestPreprocessing(unittest.TestCase):
    """Tests for feature selection and standardisation in the clustering pipeline."""

    # Detects standardisation applied to the whole array instead of per column
    # (or not applied at all). K-means uses Euclidean distance, so a feature with
    # a much larger scale would dominate the clustering if columns were not
    # each scaled to mean 0 and standard deviation 1.
    def test_standardise_features_scales_each_column_independently(self):
        X = np.array([
            [1.0, 1000.0],
            [2.0, 2000.0],
            [3.0, 3000.0],
            [4.0, 4000.0],
        ])
        X_scaled = standardise_features(X)

        self.assertEqual(X_scaled.shape, X.shape)
        np.testing.assert_allclose(X_scaled.mean(axis=0), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(X_scaled.std(axis=0), [1.0, 1.0], atol=1e-12)
        # Both columns are the same pattern on different scales, so after
        # correct per-column scaling they must be identical.
        np.testing.assert_allclose(X_scaled[:, 0], X_scaled[:, 1])

    # Detects non-numeric columns slipping through feature selection. A text
    # column (e.g. an ID or category) would otherwise fail later with an
    # unclear error when converted to floats, or be clustered meaninglessly.
    # The error should be a TypeError that names the offending column.
    def test_select_features_rejects_non_numeric_columns(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "label": ["a", "b", "c"],
        })
        with self.assertRaises(TypeError) as ctx:
            select_features(df, ["x", "label"])
        self.assertIn("label", str(ctx.exception))

    # Detects selection that returns the wrong columns, the wrong order, or a
    # view of the original data. Column order decides which feature is plotted
    # on each axis, and returning a view would let later preprocessing
    # silently modify the user's original DataFrame.
    def test_select_features_returns_requested_columns_as_copy(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
            "c": [7.0, 8.0, 9.0],
        })
        X_df = select_features(df, ["c", "a"])

        self.assertEqual(list(X_df.columns), ["c", "a"])
        self.assertEqual(len(X_df), len(df))

        X_df.loc[0, "c"] = -100.0
        self.assertEqual(df.loc[0, "c"], 7.0)


if __name__ == "__main__":
    unittest.main()
