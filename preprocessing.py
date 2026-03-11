import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


TARGET_COL = "target"


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode features after feature engineering.

    Notes:
    - target: yes/no -> 1/0
    - housing/loan/default: yes/no -> 1/0 if still object
    - age_group and balance_segment are NOT encoded here because
      feature_engineering.py already converts them to numeric codes.
    """
    df = df.copy()

    # ----------------------------------
    # Encode Target
    # ----------------------------------
    if TARGET_COL in df.columns and df[TARGET_COL].dtype == "object":
        df[TARGET_COL] = (
            df[TARGET_COL]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0, "1": 1, "0": 0})
        )

    # ----------------------------------
    # Binary Encoding
    # ----------------------------------
    binary_cols = ["housing", "loan", "default"]

    for col in binary_cols:
        if col in df.columns and df[col].dtype == "object":
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": 1, "no": 0})
            )

    # ----------------------------------
    # Ordinal Encoding
    # Only keep columns that are still categorical and truly ordinal
    # ----------------------------------
    ordinal_cols = ["education", "month_phase"]
    ordinal_cols = [c for c in ordinal_cols if c in df.columns]

    if ordinal_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[ordinal_cols] = oe.fit_transform(df[ordinal_cols])

    # ----------------------------------
    # One Hot Encoding
    # ----------------------------------
    nominal_cols = [
        "job",
        "marital",
        "contact",
        "month",
        "poutcome",
        "salary_cycle",
        "job_stability",
        "life_stage",
        "season",
        "contact_recency",
    ]
    nominal_cols = [c for c in nominal_cols if c in df.columns]

    if nominal_cols:
        df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    # Convert bool columns from get_dummies into int
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    return df


def prepare_xy(df: pd.DataFrame, target_col: str = TARGET_COL):
    """
    Split dataframe into features and target.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train/test split.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def impute_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Impute missing values using median fit on training data only.
    This avoids leakage and fixes LogisticRegression NaN errors.
    """
    imputer = SimpleImputer(strategy="median")

    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_imputed, X_test_imputed, imputer


def scale_for_logistic(X_train, X_test):
    """
    Standardize data for Logistic Regression only.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler