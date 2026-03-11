import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(name, model, X_test, y_test):
    """
    Evaluate a single fitted model on test data.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Model": name,
        "Confusion_Matrix": confusion_matrix(y_test, y_pred),
        "Classification_Report": classification_report(y_test, y_pred, output_dict=False),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def evaluate_all_models(fitted_models, X_test, y_test, X_test_scaled):
    """
    Evaluate all fitted models.
    Logistic Regression uses scaled test data.
    Tree models use unscaled test data.
    """
    rows = []
    outputs = {}

    for name, model in fitted_models.items():
        X_eval = X_test_scaled if name == "Logistic Regression" else X_test
        out = evaluate_model(name, model, X_eval, y_test)
        outputs[name] = out

        rows.append({
            "Model": name,
            "Accuracy": out["Accuracy"],
            "Precision": out["Precision"],
            "Recall": out["Recall"],
            "F1": out["F1"],
            "ROC_AUC": out["ROC_AUC"],
        })

    perf_df = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    return perf_df, outputs


def feature_importance(model, feature_names):
    """
    Generic feature importance:
    - Logistic Regression -> coefficients
    - Tree models -> feature_importances_
    """
    if hasattr(model, "coef_"):
        imp = pd.DataFrame({
            "feature": feature_names,
            "importance": model.coef_[0],
        })
        imp["abs_importance"] = imp["importance"].abs()
        return imp.sort_values("abs_importance", ascending=False).reset_index(drop=True)

    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        })
        imp["abs_importance"] = imp["importance"].abs()
        return imp.sort_values("abs_importance", ascending=False).reset_index(drop=True)

    return pd.DataFrame(columns=["feature", "importance", "abs_importance"])


def logistic_coefficients(model, feature_names):
    """
    Return logistic regression coefficients sorted descending.
    """
    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])

    coefficients = pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_[0]
    })

    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()

    return coefficients.sort_values("coefficient", ascending=False).reset_index(drop=True)


def logistic_driver_summary(model, feature_names, top_n=10):
    """
    Return most positive and most negative logistic regression drivers.
    """
    coefficients = logistic_coefficients(model, feature_names)

    if coefficients.empty:
        empty = pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])
        return empty, empty

    positive_drivers = coefficients.head(top_n).reset_index(drop=True)
    negative_drivers = coefficients.tail(top_n).sort_values("coefficient", ascending=True).reset_index(drop=True)

    return positive_drivers, negative_drivers


def logistic_feature_plot_df(model, feature_names, top_n=8):
    """
    Return a dataframe ready for plotting the strongest positive and negative
    logistic regression coefficients.
    """
    coefficients = logistic_coefficients(model, feature_names)

    if coefficients.empty:
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])

    top_features = pd.concat([
        coefficients.head(top_n),
        coefficients.tail(top_n)
    ])

    return top_features.reset_index(drop=True)


def build_decile_analysis(model, X_test, y_test):
    """
    Sort customers by predicted probability and split into deciles.
    Decile 9 = highest predicted responders, 0 = lowest.
    """
    results = pd.DataFrame({
        "probability": model.predict_proba(X_test)[:, 1],
        "actual": y_test.reset_index(drop=True) if hasattr(y_test, "reset_index") else y_test,
    })

    # sort highest probability first
    results = results.sort_values("probability", ascending=False).reset_index(drop=True)

    # create deciles from ranked rows
    results["decile"] = pd.qcut(results.index, 10, labels=False, duplicates="drop")

    # reverse labels so 9 = highest responders, 0 = lowest
    if results["decile"].nunique() == 10:
        results["decile"] = 9 - results["decile"]

    decile_response = (
        results.groupby("decile")["actual"]
        .mean()
        .sort_index()
    )

    decile_table = (
        results.groupby("decile")
        .agg(
            customers=("actual", "count"),
            responders=("actual", "sum"),
            uptake_rate=("actual", "mean"),
            avg_probability=("probability", "mean"),
        )
        .reset_index()
        .sort_values("decile")
        .reset_index(drop=True)
    )

    return results, decile_response, decile_table


def uptake_rate(results: pd.DataFrame, percent: float) -> float:
    """
    Uptake rate when targeting the top X% customers ranked by model probability.
    """
    cutoff = max(1, int(len(results) * percent))
    subset = results.head(cutoff)
    return subset["actual"].mean()


def baseline_uptake_rate(y):
    """
    Baseline uptake rate before using the model.
    """
    if hasattr(y, "mean"):
        return float(y.mean())
    return sum(y) / len(y)


def uptake_summary(results: pd.DataFrame, y_test):
    """
    Compare baseline uptake vs targeted uptake after model ranking.
    """
    baseline = baseline_uptake_rate(y_test)

    return pd.DataFrame({
        "Scenario": [
            "Baseline (random targeting)",
            "Top 10% targeted customers",
            "Top 20% targeted customers",
            "Top 30% targeted customers",
        ],
        "Uptake_Rate": [
            baseline,
            uptake_rate(results, 0.10),
            uptake_rate(results, 0.20),
            uptake_rate(results, 0.30),
        ]
    })


def model_business_summary(model, X_test, y_test):
    """
    End-to-end business summary for a chosen final model:
    - decile response
    - decile table
    - uptake summary
    """
    results, decile_response, decile_table = build_decile_analysis(model, X_test, y_test)
    uptake_df = uptake_summary(results, y_test)

    return {
        "ranked_results": results,
        "decile_response": decile_response,
        "decile_table": decile_table,
        "uptake_summary": uptake_df,
    }