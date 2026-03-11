import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------
# BRAND COLORS (Nedbank)
# --------------------------------------------------
NEDBANK_GREEN = "#78BE20"
NEDBANK_DARK = "#064F10"
NEDBANK_LIGHT = "#DBEDC6"
NEDBANK_MID = "#4C8621"
NEDBANK_ANTHRACITE = "#0A100C"
LIGHT_GREY = "#CCCCCC"

# Custom Nedbank-style diverging colormap for correlations
NEDBANK_CMAP = LinearSegmentedColormap.from_list(
    "nedbank_corr",
    [NEDBANK_DARK, "white", NEDBANK_GREEN]
)

st.set_page_config(page_title="Nedbank Marketing EDA Dashboard", layout="wide")

# --------------------------------------------------
# CSS Branding
# --------------------------------------------------
st.markdown(f"""
<style>

/* Main app */
.stApp {{
    background-color: white;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {NEDBANK_DARK};
}}

/* Sidebar text */
[data-testid="stSidebar"] * {{
    color: white !important;
}}

/* Metric labels */
[data-testid="stMetricLabel"] {{
    color: {NEDBANK_DARK} !important;
    font-weight: bold !important;
}}

/* Metric values */
[data-testid="stMetricValue"] {{
    color: {NEDBANK_ANTHRACITE} !important;
}}

/* Headers */
h1, h2, h3 {{
    color: {NEDBANK_DARK};
}}

/* Buttons */
.stButton > button {{
    background-color: {NEDBANK_GREEN};
    color: white;
    border-radius: 6px;
    border: none;
    font-weight: 600;
}}

.stButton > button:hover {{
    background-color: {NEDBANK_DARK};
    color: white;
}}

/* -----------------------------------
CHECKBOXES & RADIO BUTTONS
------------------------------------ */
input[type="checkbox"] {{
    accent-color: {NEDBANK_GREEN} !important;
}}

input[type="radio"] {{
    accent-color: {NEDBANK_GREEN} !important;
}}

/* -----------------------------------
TABS
------------------------------------ */
button[role="tab"][aria-selected="true"] {{
    color: {NEDBANK_DARK} !important;
    border-bottom: 3px solid {NEDBANK_GREEN} !important;
    box-shadow: inset 0 -3px 0 {NEDBANK_GREEN};
}}

button[role="tab"] {{
    color: {NEDBANK_ANTHRACITE} !important;
    font-weight: 500 !important;
}}

/* -----------------------------------
MULTISELECT / SELECT
------------------------------------ */

/* Main select box border */
[data-baseweb="select"] > div {{
    border-color: {NEDBANK_GREEN} !important;
}}

/* Selected tags in multiselect */
[data-baseweb="tag"] {{
    background-color: {NEDBANK_GREEN} !important;
    border: 1px solid {NEDBANK_DARK} !important;
    border-radius: 6px !important;
}}

/* Tag text */
[data-baseweb="tag"] span {{
    color: white !important;
    font-weight: 500 !important;
}}

/* Tag remove icon */
[data-baseweb="tag"] svg {{
    fill: white !important;
    color: white !important;
}}

/* -----------------------------------
SLIDER
------------------------------------ */

/* Slider handle */
div[data-baseweb="slider"] div[role="slider"] {{
    background-color: {NEDBANK_GREEN} !important;
    border-color: {NEDBANK_DARK} !important;
}}

/* Slider active track */
div[data-baseweb="slider"] > div > div > div {{
    background: {NEDBANK_GREEN} !important;
}}

/* Slider text / value */
.stSlider div {{
    color: {NEDBANK_DARK} !important;
}}

/* -----------------------------------
DATAFRAME / TABLE FEEL
------------------------------------ */
thead tr th {{
    background-color: {NEDBANK_LIGHT} !important;
    color: {NEDBANK_DARK} !important;
}}

/* -----------------------------------
INFO / SUCCESS BOXES
------------------------------------ */
[data-testid="stAlertContainer"] {{
    border-radius: 8px;
}}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_data(uploaded_file=None, path=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, sep=";")
    if path:
        return pd.read_csv(path, sep=";")
    return None


def style_axis(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, color=NEDBANK_DARK, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=NEDBANK_ANTHRACITE)
    if ylabel:
        ax.set_ylabel(ylabel, color=NEDBANK_ANTHRACITE)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(axis="x", colors=NEDBANK_ANTHRACITE)
    ax.tick_params(axis="y", colors=NEDBANK_ANTHRACITE)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percent": (df.isnull().mean() * 100).round(2),
        "dtype": df.dtypes.astype(str),
        "nunique": df.nunique(dropna=False)
    }).reset_index().rename(columns={"index": "column"})
    return out.sort_values("missing_percent", ascending=False)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    for col in ["job", "education", "contact", "poutcome"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    if "balance" in df.columns:
        df["balance"] = df["balance"].fillna(df["balance"].median())
        df["balance_shifted"] = df["balance"] - df["balance"].min() + 1
        df["log_balance"] = np.log(df["balance_shifted"])
        try:
            df["balance_segment"] = pd.qcut(
                df["balance"],
                q=4,
                labels=["very_low", "low", "medium", "high"],
                duplicates="drop"
            )
        except Exception:
            pass

    if "day" in df.columns:
        def month_phase(day):
            if day <= 10:
                return "early_month"
            elif day <= 20:
                return "mid_month"
            return "late_month"

        df["month_phase"] = df["day"].apply(month_phase)
        df["week_of_month"] = ((df["day"] - 1) // 7) + 1
        df["payday_window"] = df["day"].apply(lambda x: 1 if x >= 23 or x <= 2 else 0)

    if "month" in df.columns:
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        df["month_num"] = df["month"].map(month_map)

        def season(m):
            if pd.isna(m):
                return "unknown"
            if m in [12, 1, 2]:
                return "summer"
            if m in [3, 4, 5]:
                return "autumn"
            if m in [6, 7, 8]:
                return "winter"
            return "spring"

        df["season"] = df["month_num"].apply(season)

    if "pdays" in df.columns:
        df["previous_contact"] = (df["pdays"] != -1).astype(int)

    if {"previous", "poutcome"}.issubset(df.columns):
        df["previous_success"] = (
            (df["previous"] > 0) & (df["poutcome"] == "success")
        ).astype(int)

    if {"campaign", "previous"}.issubset(df.columns):
        df["total_contacts"] = df["campaign"] + df["previous"]

    if {"housing", "loan"}.issubset(df.columns):
        df["financial_pressure"] = (
            (df["housing"] == "yes").astype(int) +
            (df["loan"] == "yes").astype(int)
        )

    if "job" in df.columns:
        def job_stability(job):
            if job in ["management", "technician", "admin.", "retired"]:
                return "stable"
            if job in ["services", "blue-collar", "self-employed", "entrepreneur", "housemaid"]:
                return "variable"
            return "unstable"

        df["job_stability"] = df["job"].apply(job_stability)

    if "education" in df.columns:
        edu_map = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
        df["education_level"] = df["education"].map(edu_map).fillna(0)
        df["high_education"] = (df["education"] == "tertiary").astype(int)

    if "marital" in df.columns:
        df["is_married"] = (df["marital"] == "married").astype(int)

    if {"age", "education", "marital"}.issubset(df.columns):
        df["life_stage_maturity"] = (
            (df["age"] >= 35) &
            (df["education"].isin(["secondary", "tertiary"])) &
            (df["marital"] == "married")
        ).astype(int)

    if "marital" in df.columns:
        marital_map = {
            "single": 0,
            "divorced": 1,
            "married": 2
        }
        df["marital_score"] = df["marital"].map(marital_map).fillna(0)

    if "job_stability" in df.columns:
        stability_map = {
            "unstable": 0,
            "variable": 1,
            "stable": 2
        }
        df["job_stability_score"] = df["job_stability"].map(stability_map).fillna(0)

    if {"job_stability_score", "education_level", "marital_score"}.issubset(df.columns):
        df["stability_score"] = (
            df["job_stability_score"] +
            df["education_level"] +
            df["marital_score"]
        )

    if "balance" in df.columns:
        df["high_balance"] = (df["balance"] > df["balance"].quantile(0.75)).astype(int)
        df["negative_balance"] = (df["balance"] < 0).astype(int)

    if {"balance", "campaign"}.issubset(df.columns):
        df["balance_per_contact"] = df["balance"] / (df["campaign"] + 1)

    if {"campaign", "previous"}.issubset(df.columns):
        df["contact_intensity"] = df["campaign"] / (df["previous"] + 1)
        df["campaign_pressure"] = (df["campaign"] >= 5).astype(int)

    if {"duration", "campaign"}.issubset(df.columns):
        df["duration_per_contact"] = df["duration"] / (df["campaign"] + 1)
        df["marketing_efficiency"] = df["duration"] / (df["campaign"] + 1)
        df["long_call"] = (df["duration"] > df["duration"].quantile(0.75)).astype(int)

    if "contact" in df.columns:
        df["contactable_customer"] = (df["contact"] != "unknown").astype(int)
        df["is_cellular"] = (df["contact"] == "cellular").astype(int)

    if "pdays" in df.columns:
        df["recent_contact"] = ((df["pdays"] > 0) & (df["pdays"] <= 30)).astype(int)
        df["stale_contact"] = (df["pdays"] > 180).astype(int)

    if {"default", "loan"}.issubset(df.columns):
        df["credit_risk"] = (
            ((df["default"] == "yes") | (df["loan"] == "yes"))
        ).astype(int)

    return df


def plot_bar(series, title, normalize=False):
    counts = series.value_counts(normalize=normalize, dropna=False).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax, color=NEDBANK_GREEN, edgecolor=NEDBANK_DARK)
    style_axis(ax, title=title, ylabel="Proportion" if normalize else "Count", xlabel=series.name if series.name else "")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)


def plot_target_rate(df, feature, target="target"):
    temp = pd.crosstab(df[feature], df[target], normalize="index")
    if "yes" in temp.columns:
        vals = temp["yes"].sort_values(ascending=False)
    else:
        vals = temp.iloc[:, -1].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    vals.plot(kind="bar", ax=ax, color=NEDBANK_GREEN, edgecolor=NEDBANK_DARK)
    style_axis(ax, title=f"Campaign uptake rate by {feature}", ylabel="Uptake rate", xlabel=feature)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
    st.dataframe((temp * 100).round(2), use_container_width=True)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Data source")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
default_path = st.sidebar.text_input(
    "Or use local CSV path",
    value=r"C:\Users\LENOVO\Documents\Nedbankassessment\bank_marketing_data.csv"
)

use_engineered = st.sidebar.checkbox("Add engineered features", value=True)
show_raw = st.sidebar.checkbox("Show raw data preview", value=False)

# -----------------------------
# Load data
# -----------------------------
df = load_data(uploaded_file=uploaded_file, path=default_path if not uploaded_file else None)

st.title("Nedbank Marketing EDA Dashboard")
st.caption("Built for exploratory data analysis of the bank marketing campaign dataset.")

if df is None:
    st.info("Upload a CSV file or provide a valid local path in the sidebar.")
    st.stop()

raw_df = df.copy()

if use_engineered:
    df = add_engineered_features(df)

if "target" in df.columns:
    df["target_encoded"] = df["target"].map({"no": 0, "yes": 1})

if show_raw:
    st.subheader("Data preview")
    st.dataframe(df.head(20), use_container_width=True)

# -----------------------------
# KPI Cards
# -----------------------------
customers = len(df)
uptake_rate = df["target"].eq("yes").mean() * 100 if "target" in df.columns else 0
avg_balance = df["balance"].mean() if "balance" in df.columns else 0
previously_contacted = (df["pdays"] != -1).mean() * 100 if "pdays" in df.columns else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", f"{customers:,}")
col2.metric("Uptake Rate", f"{uptake_rate:.2f}%")
col3.metric("Avg Balance", f"{avg_balance:,.0f}")
col4.metric("Previously Contacted", f"{previously_contacted:.2f}%")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Data Quality",
    "Feature Distributions",
    "Campaign Uptake",
    "Relationships"
])

with tab1:
    st.subheader("Column summary")
    summary = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "non_null": df.notnull().sum().values,
        "null_percent": (df.isnull().mean() * 100).round(2).values,
        "nunique": df.nunique(dropna=False).values,
    })
    st.dataframe(summary, use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.subheader("Numeric summary")
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

with tab2:
    st.subheader("Missing value analysis")
    miss = missing_summary(raw_df)
    st.dataframe(miss, use_container_width=True)

    miss_nonzero = miss[miss["missing_count"] > 0]
    if not miss_nonzero.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(
            miss_nonzero["column"],
            miss_nonzero["missing_percent"],
            color=NEDBANK_GREEN,
            edgecolor=NEDBANK_DARK
        )
        style_axis(ax, title="Missing percentage by column", ylabel="Missing %")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.success("No missing values found in the raw dataset.")

with tab3:
    st.subheader("Feature distributions")
    all_cols = df.columns.tolist()
    feature = st.selectbox("Select a feature", all_cols, index=0)

    if pd.api.types.is_numeric_dtype(df[feature]):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[feature].dropna(), bins=30, color=NEDBANK_GREEN, edgecolor=NEDBANK_DARK)
        style_axis(ax, title=f"Distribution of {feature}", xlabel=feature, ylabel="Frequency")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(8, 2.5))
        sns.boxplot(x=df[feature].dropna(), ax=ax2, color=NEDBANK_LIGHT)
        style_axis(ax2, title=f"Boxplot of {feature}")
        st.pyplot(fig2)
    else:
        plot_bar(df[feature], f"Distribution of {feature}")
        value_counts_df = (
            df[feature]
            .value_counts(dropna=False)
            .rename_axis("category")
            .reset_index(name="count")
        )
        st.dataframe(value_counts_df, use_container_width=True)

with tab4:
    st.subheader("Campaign uptake analysis")
    if "target" not in df.columns:
        st.warning("Target column not found.")
    else:
        candidate_features = [c for c in df.columns if c != "target"]
        feature = st.selectbox(
            "Select feature for uptake view",
            candidate_features,
            index=min(1, len(candidate_features) - 1),
            key="uptake_feature"
        )

        if pd.api.types.is_numeric_dtype(df[feature]):
            if df[feature].nunique() > 12:
                temp = df[[feature, "target"]].copy()
                temp["bin"] = pd.qcut(
                    temp[feature],
                    q=min(5, temp[feature].nunique()),
                    duplicates="drop"
                )
                plot_target_rate(temp, "bin", target="target")
            else:
                plot_target_rate(df, feature, target="target")
        else:
            plot_target_rate(df, feature, target="target")

        st.subheader("Recommended business cuts")
        useful = [c for c in [
            "month", "month_phase", "payday_window", "salary_cycle",
            "job_stability", "stability_score", "education", "education_level",
            "marital", "marital_score", "contact", "is_cellular",
            "poutcome", "previous_contact", "previous_success",
            "financial_pressure", "liquidity_pressure", "credit_risk",
            "high_balance", "negative_balance", "campaign_pressure",
            "recent_contact", "stale_contact", "long_call"
        ] if c in df.columns]

        for feat in useful[:6]:
            with st.expander(f"View uptake by {feat}"):
                plot_target_rate(df, feat, target="target")

with tab5:
    st.subheader("Relationships")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) >= 2:
        st.markdown("### Correlation heatmap for numeric features")

        heatmap_cols = st.multiselect(
            "Select numeric features for heatmap",
            numeric_cols,
            default=numeric_cols[:12] if len(numeric_cols) > 12 else numeric_cols,
            key="heatmap_cols"
        )

        show_annotations = st.checkbox("Show correlation values", value=True)
        heatmap_size = st.slider("Heatmap size", min_value=10, max_value=24, value=16)

        if len(heatmap_cols) >= 2:
            corr = df[heatmap_cols].corr(numeric_only=True)
            mask = np.triu(np.ones_like(corr, dtype=bool))

            fig, ax = plt.subplots(figsize=(heatmap_size, heatmap_size * 0.65))
            sns.heatmap(
                corr,
                mask=mask,
                annot=show_annotations,
                fmt=".2f",
                cmap=NEDBANK_CMAP,
                center=0,
                square=False,
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                annot_kws={"size": 9},
                ax=ax
            )

            ax.set_title("Correlation Heatmap", fontsize=16, pad=16, color=NEDBANK_DARK, fontweight="bold")
            ax.tick_params(axis="x", labelrotation=45, labelsize=10, colors=NEDBANK_ANTHRACITE)
            ax.tick_params(axis="y", labelrotation=0, labelsize=10, colors=NEDBANK_ANTHRACITE)
            plt.tight_layout()

            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Please select at least 2 numeric features.")

        if "target_encoded" in df.columns:
            full_corr = df[numeric_cols].corr(numeric_only=True)

            if "target_encoded" in full_corr.columns:
                st.markdown("### Correlation with target")
                target_corr = full_corr["target_encoded"].drop("target_encoded").sort_values(ascending=False)
                target_corr_df = target_corr.reset_index()
                target_corr_df.columns = ["feature", "correlation_with_target"]
                st.dataframe(target_corr_df, use_container_width=True)

                fig_target, ax_target = plt.subplots(figsize=(9, 6))
                target_corr.sort_values().plot(
                    kind="barh",
                    ax=ax_target,
                    color=NEDBANK_GREEN,
                    edgecolor=NEDBANK_DARK
                )
                style_axis(ax_target, title="Feature correlation with target", xlabel="Correlation")
                plt.tight_layout()
                st.pyplot(fig_target)

        st.markdown("### Outlier analysis")

        outlier_candidates = [
            col for col in numeric_cols
            if col not in [
                "target_encoded", "previous_contact", "previous_success",
                "is_married", "high_education", "life_stage_maturity",
                "payday_window"
            ]
        ]

        if outlier_candidates:
            outlier_feature = st.selectbox(
                "Select numeric feature for outlier view",
                outlier_candidates,
                key="outlier_feature"
            )

            fig2, ax2 = plt.subplots(figsize=(10, 2.8))
            sns.boxplot(x=df[outlier_feature], ax=ax2, color=NEDBANK_LIGHT)
            style_axis(ax2, title=f"Outlier Boxplot: {outlier_feature}")
            st.pyplot(fig2)

            fig3, ax3 = plt.subplots(figsize=(10, 4))
            sns.histplot(
                df[outlier_feature].dropna(),
                bins=30,
                kde=True,
                ax=ax3,
                color=NEDBANK_GREEN
            )
            style_axis(ax3, title=f"Distribution: {outlier_feature}")
            st.pyplot(fig3)

            q1 = df[outlier_feature].quantile(0.25)
            q3 = df[outlier_feature].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_count = ((df[outlier_feature] < lower_bound) | (df[outlier_feature] > upper_bound)).sum()
            outlier_pct = (outlier_count / len(df)) * 100

            m1, m2, m3 = st.columns(3)
            m1.metric("Q1", f"{q1:,.2f}")
            m2.metric("Q3", f"{q3:,.2f}")
            m3.metric("Outliers", f"{outlier_count:,} ({outlier_pct:.2f}%)")

        st.markdown("### Scatterplot")
        selected_x = st.selectbox("Scatterplot X", numeric_cols, index=0, key="scatter_x")
        selected_y = st.selectbox("Scatterplot Y", numeric_cols, index=min(1, len(numeric_cols) - 1), key="scatter_y")

        fig4, ax4 = plt.subplots(figsize=(7, 5))
        ax4.scatter(df[selected_x], df[selected_y], alpha=0.4, color=NEDBANK_GREEN, edgecolors=NEDBANK_DARK)
        style_axis(ax4, title=f"{selected_x} vs {selected_y}", xlabel=selected_x, ylabel=selected_y)
        st.pyplot(fig4)

    else:
        st.info("Not enough numeric columns for relationship analysis.")