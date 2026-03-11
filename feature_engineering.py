import numpy as np
import pandas as pd


MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

EDUCATION_MAP = {
    "unknown": 0,
    "primary": 1,
    "secondary": 2,
    "tertiary": 3,
}


def month_phase(day: int) -> str:
    if day <= 10:
        return "early_month"
    elif day <= 20:
        return "mid_month"
    else:
        return "late_month"


def season(month_num: int) -> str:
    if month_num in [12, 1, 2]:
        return "summer"
    elif month_num in [3, 4, 5]:
        return "autumn"
    elif month_num in [6, 7, 8]:
        return "winter"
    else:
        return "spring"


def recency_bucket(x: int) -> str:
    if x == -1:
        return "never_contacted"
    elif x <= 30:
        return "recent"
    elif x <= 180:
        return "mid_term"
    else:
        return "long_ago"


def salary_cycle(day: int) -> str:
    if day <= 5:
        return "month_start"
    elif day <= 15:
        return "mid_month"
    elif day <= 25:
        return "pre_payday"
    else:
        return "post_payday"


def job_stability(job: str) -> str:
    if job in ["management", "technician", "admin.", "retired"]:
        return "stable"
    elif job in ["services", "blue-collar", "self-employed", "entrepreneur", "housemaid"]:
        return "variable"
    else:
        return "unstable"


def life_stage(row: pd.Series) -> str:
    if row["age"] < 30:
        return "early_career"
    elif row["age"] < 45 and row["marital"] == "married":
        return "family_builder"
    elif row["age"] >= 45:
        return "wealth_accumulation"
    else:
        return "mid_career"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering aligned to the final notebook logic.
    """
    df = df.copy()

    # -----------------------------
    # Time features
    # -----------------------------
    df["month_phase"] = df["day"].apply(month_phase)

    df["month_num"] = df["month"].map(MONTH_MAP)

    df["season"] = df["month_num"].apply(season)

    df["week_of_month"] = (df["day"] // 7) + 1

    # -----------------------------
    # Campaign history features
    # -----------------------------
    # Notebook logic: based on pdays, not previous
    df["previous_contact"] = df["pdays"].apply(lambda x: 0 if x == -1 else 1)

    df["contact_recency"] = df["pdays"].apply(recency_bucket)

    # Notebook logic: previous > 0 and poutcome success
    df["previous_success"] = ((df["previous"] > 0) & (df["poutcome"] == "success")).astype(int)

    df["total_contacts"] = df["campaign"] + df["previous"]

    # -----------------------------
    # Financial behavior features
    # -----------------------------
    # Notebook logic: first 7 days
    df["salary_window"] = df["day"].apply(lambda x: 1 if x <= 7 else 0)

    df["financial_pressure"] = (
        (df["housing"] == "yes").astype(int) +
        (df["loan"] == "yes").astype(int)
    )

    df["salary_cycle"] = df["day"].apply(salary_cycle)

    # -----------------------------
    # Contact/channel features
    # -----------------------------
    df["mobile_contact"] = (df["contact"] == "cellular").astype(int)

    contact_map = {
        "cellular": 2,
        "telephone": 1,
        "unknown": 0
    }
    df["contact_quality"] = df["contact"].map(contact_map)

    df["cellular_high_campaign"] = (
        ((df["contact"] == "cellular") & (df["campaign"] > 3)).astype(int)
    )

    # Notebook logic: same as cellular contact
    df["digital_customer"] = (df["contact"] == "cellular").astype(int)

    # -----------------------------
    # Life stage features
    # -----------------------------
    df["age_group"] = pd.cut(
        df["age"],
        bins=[18, 30, 40, 50, 60, 100],
        labels=["young", "early_career", "mid_career", "late_career", "senior"]
    )

    # First version exists in notebook, but gets overwritten later.
    # Keep final notebook behavior by applying the final version below.
    df["financial_maturity"] = (df["age"] >= 35).astype(int)

    df["job_stability"] = df["job"].apply(job_stability)

    stability_map = {
        "stable": 2,
        "variable": 1,
        "unstable": 0
    }
    df["job_stability_score"] = df["job_stability"].map(stability_map)

    df["education_level"] = df["education"].map(EDUCATION_MAP)

    df["family_commitment"] = df["marital"].map({
        "single": 0,
        "divorced": 1,
        "married": 2
    })

    # Final notebook version overwrites financial_maturity
    df["financial_maturity"] = (
        (df["education_level"] >= 2) &
        (df["marital"] == "married")
    ).astype(int)

    df["life_stage_maturity"] = (
        (df["age"] >= 35) &
        (df["education"].isin(["secondary", "tertiary"])) &
        (df["marital"] == "married")
    ).astype(int)

    df["life_stage"] = df.apply(life_stage, axis=1)

    # -----------------------------
    # Balance and liquidity features
    # -----------------------------
    df["balance_shifted"] = df["balance"] - df["balance"].min() + 1
    df["log_balance"] = np.log(df["balance_shifted"])

    # Notebook logic: qcut with quartiles
    df["balance_segment"] = pd.qcut(
        df["balance"],
        q=4,
        labels=["very_low", "low", "medium", "high"],
        duplicates="drop"
    )

    df["liquidity_pressure"] = (
        (df["balance"] < 500) &
        ((df["housing"] == "yes") | (df["loan"] == "yes"))
    ).astype(int)

    df["high_balance"] = (df["balance"] > df["balance"].quantile(0.75)).astype(int)

    df["negative_balance"] = (df["balance"] < 0).astype(int)

    df["balance_per_contact"] = df["balance"] / (df["campaign"] + 1)

    # -----------------------------
    # Customer stability features
    # -----------------------------
    df["is_married"] = (df["marital"] == "married").astype(int)

    df["stability_score"] = (
        df["job_stability_score"] +
        df["education_level"] +
        df["is_married"]
    )

    # -----------------------------
    # Payday timing
    # -----------------------------
    df["payday_call"] = (
        ((df["day"] >= 23) | (df["day"] <= 3)).astype(int)
    )

    # -----------------------------
    # Credit features
    # -----------------------------
    df["housing_bin"] = df["housing"].map({"yes": 1, "no": 0})
    df["loan_bin"] = df["loan"].map({"yes": 1, "no": 0})

    # Make default robust whether still yes/no or already encoded
    default_num = df["default"].replace({"yes": 1, "no": 0})

    df["credit_risk"] = (
        (default_num == 1) |
        (df["negative_balance"] == 1) |
        (df["financial_pressure"] == 1)
    ).astype(int)

    df["credit_stress"] = (
        df["housing_bin"] +
        df["loan_bin"] +
        df["financial_pressure"]
    )

    # -----------------------------
    # Convert categorical engineered vars to codes
    # -----------------------------
    df["age_group"] = df["age_group"].cat.codes
    df["balance_segment"] = df["balance_segment"].cat.codes

    return df