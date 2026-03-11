import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


RANDOM_STATE = 42


def get_stratified_kfold(n_splits: int = 5):
    """
    Stratified K-Fold for imbalanced classification.
    """
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )


def build_models(y_train):
    """
    Build the final model set:
    - Logistic Regression
    - Random Forest
    - XGBoost (if installed)
    """
    log_model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    models = {
        "Logistic Regression": log_model,
        "Random Forest": rf_model,
    }

    if XGBOOST_AVAILABLE:
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE
        )

        models["XGBoost"] = xgb_model

    return models


def run_cross_validation(models, X_train, y_train, X_train_scaled):
    """
    Run stratified cross-validation using ROC-AUC.
    Logistic Regression uses scaled data.
    Tree models use unscaled data.
    """
    skf = get_stratified_kfold()

    rows = []

    for name, model in models.items():
        X_cv = X_train_scaled if name == "Logistic Regression" else X_train

        scores = cross_val_score(
            model,
            X_cv,
            y_train,
            cv=skf,
            scoring="roc_auc",
            n_jobs=-1,
            error_score="raise"
        )

        rows.append({
            "Model": name,
            "CV_ROC_AUC_Mean": scores.mean(),
            "CV_ROC_AUC_Std": scores.std()
        })

    return pd.DataFrame(rows).sort_values("CV_ROC_AUC_Mean", ascending=False)


def fit_models(models, X_train, y_train, X_train_scaled):
    """
    Fit all models on the training data.
    Logistic Regression uses scaled data.
    Tree models use unscaled data.
    """
    fitted_models = {}

    for name, model in models.items():
        X_fit = X_train_scaled if name == "Logistic Regression" else X_train
        model.fit(X_fit, y_train)
        fitted_models[name] = model

    return fitted_models


def compare_cv_results(cv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a simplified comparison table for dashboard/reporting.
    """
    return cv_df[["Model", "CV_ROC_AUC_Mean"]].sort_values(
        "CV_ROC_AUC_Mean",
        ascending=False
    ).reset_index(drop=True)