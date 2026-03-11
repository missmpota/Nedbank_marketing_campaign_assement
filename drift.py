import numpy as np
import pandas as pd


def population_stability_index(expected, actual, buckets: int = 10):
    """
    Calculate PSI between expected (training) and actual/recent distributions.
    """
    expected = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna()
    actual = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna()

    if expected.empty or actual.empty:
        return np.nan

    # Use quantile-based bins from expected data
    quantiles = np.linspace(0, 1, buckets + 1)
    breaks = np.unique(np.quantile(expected, quantiles))

    # If there are too few unique breakpoints, PSI is not meaningful
    if len(breaks) < 3:
        return np.nan

    expected_bins = pd.cut(expected, bins=breaks, include_lowest=True)
    actual_bins = pd.cut(actual, bins=breaks, include_lowest=True)

    expected_dist = expected_bins.value_counts(normalize=True, sort=False).replace(0, 1e-6)
    actual_dist = actual_bins.value_counts(normalize=True, sort=False).replace(0, 1e-6)

    psi = ((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)).sum()
    return psi


def classify_drift(psi_value):
    """
    Standard PSI interpretation.
    """
    if pd.isna(psi_value):
        return "Insufficient data"
    if psi_value < 0.10:
        return "Low drift"
    if psi_value < 0.25:
        return "Moderate drift"
    return "High drift"


def compute_drift(train_df: pd.DataFrame, recent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean shift and PSI for numeric features shared by train and recent data.
    Excludes target.
    """
    common_cols = [
        c for c in train_df.columns
        if c in recent_df.columns
        and c != "target"
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    rows = []
    for col in common_cols:
        train_mean = train_df[col].mean()
        recent_mean = recent_df[col].mean()
        psi_value = population_stability_index(train_df[col], recent_df[col])

        rows.append({
            "feature": col,
            "train_mean": train_mean,
            "recent_mean": recent_mean,
            "mean_shift": recent_mean - train_mean,
            "psi": psi_value,
            "drift_flag": classify_drift(psi_value),
        })

    drift_df = pd.DataFrame(rows)

    if drift_df.empty:
        return pd.DataFrame(
            columns=["feature", "train_mean", "recent_mean", "mean_shift", "psi", "drift_flag"]
        )

    return drift_df.sort_values("psi", ascending=False).reset_index(drop=True)