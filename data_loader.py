import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data with a safe delimiter fallback and clean column names."""
    try:
        df = pd.read_csv(path, sep=";")
        if len(df.columns) == 1:
            df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data in line with the final notebook:
    - drop unnamed index column
    - fill selected categorical missing values with 'unknown'
    - fill balance missing values with median
    - drop duplicates
    - drop duration to avoid leakage
    """
    df = df.copy()

    # Drop unnamed index-like columns
    unnamed_cols = [col for col in df.columns if col.lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Fill categorical columns used in notebook
    categorical_fill_unknown = ["job", "education", "contact", "poutcome"]
    for col in categorical_fill_unknown:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # Fill numeric missing values
    if "balance" in df.columns:
        df["balance"] = df["balance"].fillna(df["balance"].median())

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop leakage column if present
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    return df