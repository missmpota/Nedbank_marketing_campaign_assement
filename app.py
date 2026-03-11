# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import roc_curve, auc

# IMPORT PIPELINE MODULES
from data_loader import load_data, clean_data
from feature_engineering import engineer_features
from preprocessing import (
    encode_features,
    prepare_xy,
    split_data,
    impute_data,
    scale_for_logistic,
)
from modeling import build_models, run_cross_validation, fit_models
from evaluation import (
    evaluate_all_models,
    feature_importance,
    logistic_coefficients,
    logistic_driver_summary,
    logistic_feature_plot_df,
    model_business_summary,
)
from drift import compute_drift


# --------------------------------------------------
# BRAND COLORS (Nedbank)
# --------------------------------------------------
NEDBANK_GREEN = "#78BE20"       # Primary Lima green
NEDBANK_DARK = "#064F10"        # Dark green (sidebar/login)
NEDBANK_LIGHT = "#DBEDC6"       # Pale green
NEDBANK_ANTHRACITE = "#0A100C"  # Professional dark text
NEDBANK_MID = "#4C8621"         # Mid green for secondary charts
LIGHT_GREY = "#CCCCCC"          # Neutral grey for guide lines


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Nedbank Campaign Targeting Model", layout="wide")

st.markdown(f"""
<style>

/* Main app */
.stApp {{
    background-color: white;
}}

/* LEFT SIDEBAR PANEL */
[data-testid="stSidebar"] {{
    background-color: {NEDBANK_DARK};
}}

/* Sidebar text white */
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

/* Buttons */
.stButton>button {{
    background-color: {NEDBANK_GREEN};
    color: white;
    border-radius: 6px;
    border: none;
    font-weight: 600;
}}

.stButton>button:hover {{
    background-color: {NEDBANK_DARK};
    color: white;
}}

/* Headers */
h1, h2, h3 {{
    color: {NEDBANK_DARK};
}}

</style>
""", unsafe_allow_html=True)

st.title("Nedbank Campaign Targeting Model")
st.caption("Target the customers most likely to respond, compare model performance, and monitor stability over time.")


# --------------------------------------------------
# HELPER FOR CONSISTENT MATPLOTLIB STYLING
# --------------------------------------------------
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


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Inputs")
train_file = st.sidebar.file_uploader("Upload training CSV", type=["csv"])
recent_file = st.sidebar.file_uploader(
    "Upload recent CSV for drift monitoring (optional)",
    type=["csv"]
)
final_model_name = st.sidebar.selectbox(
    "Final model for deep-dive",
    ["Logistic Regression", "Random Forest", "XGBoost"]
)
use_simulated_recent = st.sidebar.checkbox(
    "Use simulated recent sample if no recent file is uploaded",
    value=True
)

if not train_file:
    st.info(
        "Upload your modeling dataset to begin. The app expects a target column named "
        "'target' with yes/no or 1/0 values."
    )
    st.stop()


# --------------------------------------------------
# LOAD + PREPARE TRAINING DATA
# --------------------------------------------------
df = load_data(train_file)
df = clean_data(df)
df = engineer_features(df)
df = encode_features(df)

X, y = prepare_xy(df)

X_train, X_test, y_train, y_test = split_data(X, y)

# impute missing values before modeling
X_train, X_test, imputer = impute_data(X_train, X_test)

# scale only for logistic regression
X_train_scaled, X_test_scaled, scaler = scale_for_logistic(X_train, X_test)


# --------------------------------------------------
# MODELING
# --------------------------------------------------
models = build_models(y_train)
cv_df = run_cross_validation(models, X_train, y_train, X_train_scaled)
fitted_models = fit_models(models, X_train, y_train, X_train_scaled)

perf_df, outputs = evaluate_all_models(
    fitted_models,
    X_test,
    y_test,
    X_test_scaled
)


# --------------------------------------------------
# FINAL MODEL SELECTION
# --------------------------------------------------
available_models = list(fitted_models.keys())
if final_model_name not in available_models:
    final_model_name = available_models[0]

final_model = fitted_models[final_model_name]
final_X_test = X_test_scaled if final_model_name == "Logistic Regression" else X_test


# --------------------------------------------------
# LOGISTIC REGRESSION DEEP DIVE
# --------------------------------------------------
logistic_available = "Logistic Regression" in fitted_models
if logistic_available:
    log_model = fitted_models["Logistic Regression"]
    coefficients = logistic_coefficients(log_model, X_train.columns)
    positive_drivers, negative_drivers = logistic_driver_summary(
        log_model,
        X_train.columns,
        top_n=10
    )
    logistic_plot_df = logistic_feature_plot_df(
        log_model,
        X_train.columns,
        top_n=8
    )


# --------------------------------------------------
# BUSINESS ANALYSIS FOR FINAL MODEL
# --------------------------------------------------
business = model_business_summary(final_model, final_X_test, y_test)
decile_response = business["decile_response"]
decile_table = business["decile_table"]
uptake_summary_df = business["uptake_summary"]


# --------------------------------------------------
# UPTAKE BEFORE VS AFTER MODEL
# --------------------------------------------------
st.subheader("Uptake Before vs After Model")

baseline_rate = uptake_summary_df.loc[
    uptake_summary_df["Scenario"] == "Baseline (random targeting)",
    "Uptake_Rate"
].iloc[0]

top10 = uptake_summary_df.loc[
    uptake_summary_df["Scenario"] == "Top 10% targeted customers",
    "Uptake_Rate"
].iloc[0]

top20 = uptake_summary_df.loc[
    uptake_summary_df["Scenario"] == "Top 20% targeted customers",
    "Uptake_Rate"
].iloc[0]

top30 = uptake_summary_df.loc[
    uptake_summary_df["Scenario"] == "Top 30% targeted customers",
    "Uptake_Rate"
].iloc[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Baseline uptake", f"{baseline_rate:.2%}")
m2.metric("Top 10% uptake", f"{top10:.2%}")
m3.metric("Top 20% uptake", f"{top20:.2%}")
m4.metric("Top 30% uptake", f"{top30:.2%}")

fig_uptake, ax_uptake = plt.subplots(figsize=(8, 5))
ax_uptake.bar(
    uptake_summary_df["Scenario"],
    uptake_summary_df["Uptake_Rate"],
    color=NEDBANK_GREEN,
    edgecolor=NEDBANK_DARK,
    linewidth=1
)
style_axis(
    ax_uptake,
    title="Uptake Rate Before vs After Model",
    ylabel="Uptake Rate"
)
ax_uptake.tick_params(axis="x", rotation=20)
st.pyplot(fig_uptake)

st.dataframe(uptake_summary_df, use_container_width=True)


# --------------------------------------------------
# FEATURES USED
# --------------------------------------------------
st.subheader("Final Features Used")
st.write(f"Total features used: {X.shape[1]}")
st.dataframe(
    pd.DataFrame({"feature": X.columns.tolist()}),
    use_container_width=True,
    height=250
)


# --------------------------------------------------
# CORRELATION WITH TARGET
# --------------------------------------------------
st.subheader("Correlation with Target")
corr_df = df.corr(numeric_only=True)[["target"]].drop(index="target").reset_index()
corr_df.columns = ["feature", "correlation_with_target"]
corr_df["abs_correlation"] = corr_df["correlation_with_target"].abs()
corr_df = corr_df.sort_values("abs_correlation", ascending=False)

st.dataframe(corr_df.head(20), use_container_width=True)

fig1, ax1 = plt.subplots(figsize=(10, 6))
corr_plot_df = corr_df.head(15).sort_values("correlation_with_target")
ax1.barh(
    corr_plot_df["feature"],
    corr_plot_df["correlation_with_target"],
    color=NEDBANK_GREEN,
    edgecolor=NEDBANK_DARK
)
style_axis(ax1, title="Top Correlations with Target", xlabel="Correlation")
st.pyplot(fig1)


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
st.subheader("Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Cross-validation**")
    st.dataframe(cv_df, use_container_width=True)

with col2:
    st.markdown("**Test performance**")
    st.dataframe(perf_df, use_container_width=True)


# --------------------------------------------------
# ROC CURVE COMPARISON
# --------------------------------------------------
st.subheader("ROC Curve Comparison")

fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

model_colors = {
    "Logistic Regression": NEDBANK_GREEN,
    "Random Forest": NEDBANK_DARK,
    "XGBoost": NEDBANK_MID,
}

for model_name, model in fitted_models.items():
    X_eval = X_test_scaled if model_name == "Logistic Regression" else X_test
    y_prob = model.predict_proba(X_eval)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    ax_roc.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {roc_auc:.3f})",
        linewidth=3,
        color=model_colors.get(model_name, NEDBANK_GREEN)
    )

ax_roc.plot([0, 1], [0, 1], linestyle="--", color=LIGHT_GREY, linewidth=1.5)
style_axis(
    ax_roc,
    title="ROC Curve Comparison",
    xlabel="False Positive Rate",
    ylabel="True Positive Rate"
)
ax_roc.legend(loc="lower right", frameon=False)
st.pyplot(fig_roc)


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------
st.subheader(f"Feature Importance — {final_model_name}")
fi_df = feature_importance(final_model, X.columns)
st.dataframe(fi_df.head(20), use_container_width=True)

fig2, ax2 = plt.subplots(figsize=(10, 6))
plot_fi = fi_df.head(15).sort_values("abs_importance")
ax2.barh(
    plot_fi["feature"],
    plot_fi["importance"],
    color=NEDBANK_GREEN,
    edgecolor=NEDBANK_DARK
)
ax2.axvline(0, color=NEDBANK_DARK, linewidth=1)
style_axis(ax2, title=f"Top Feature Importance — {final_model_name}")
st.pyplot(fig2)


# --------------------------------------------------
# LOGISTIC REGRESSION COEFFICIENTS
# --------------------------------------------------
if logistic_available:
    st.subheader("Logistic Regression Drivers")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top Positive Drivers**")
        st.dataframe(positive_drivers, use_container_width=True)

    with c2:
        st.markdown("**Top Negative Drivers**")
        st.dataframe(negative_drivers, use_container_width=True)

    fig_log, ax_log = plt.subplots(figsize=(10, 6))
    ax_log.barh(
        logistic_plot_df["feature"],
        logistic_plot_df["coefficient"],
        color=NEDBANK_GREEN,
        edgecolor=NEDBANK_DARK
    )
    ax_log.axvline(0, color=NEDBANK_DARK, linewidth=1)
    style_axis(ax_log, title="Logistic Regression Coefficients", xlabel="Coefficient Value")
    st.pyplot(fig_log)


# --------------------------------------------------
# DECILE ANALYSIS
# --------------------------------------------------
st.subheader("Decile Analysis")
st.dataframe(decile_table, use_container_width=True)

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.bar(
    decile_response.index.astype(str),
    decile_response.values,
    color=NEDBANK_GREEN,
    edgecolor=NEDBANK_DARK
)
style_axis(
    ax3,
    title="Response Rate by Decile",
    xlabel="Decile (0 = lowest, 9 = highest)",
    ylabel="Response Rate"
)
st.pyplot(fig3)


# --------------------------------------------------
# DRIFT MONITORING
# --------------------------------------------------
st.subheader("Model Drift Monitoring")

if recent_file:
    recent_df = load_data(recent_file)
    recent_df = clean_data(recent_df)
    recent_df = engineer_features(recent_df)
    recent_df = encode_features(recent_df)
    drift_note = "Using uploaded recent dataset for drift monitoring."

elif use_simulated_recent:
    recent_df = df.sample(frac=0.30, random_state=42).copy()
    drift_note = (
        "Using a simulated recent sample (30% random sample from the same dataset) "
        "for dashboard demonstration."
    )
else:
    recent_df = None
    drift_note = None

if recent_df is not None:
    st.info(drift_note)

    for col in df.columns:
        if col not in recent_df.columns:
            recent_df[col] = 0

    extra_cols = [col for col in recent_df.columns if col not in df.columns]
    if extra_cols:
        recent_df = recent_df.drop(columns=extra_cols)

    recent_df = recent_df[df.columns]

    drift_df = compute_drift(df, recent_df)
    st.dataframe(drift_df.head(20), use_container_width=True)

    if not drift_df.empty:
        fig_drift, ax_drift = plt.subplots(figsize=(10, 6))
        drift_plot_df = drift_df.head(15).sort_values("psi")
        ax_drift.barh(
            drift_plot_df["feature"],
            drift_plot_df["psi"],
            color=NEDBANK_GREEN,
            edgecolor=NEDBANK_DARK
        )
        style_axis(ax_drift, title="Top Drifted Features by PSI", xlabel="PSI")
        st.pyplot(fig_drift)

        st.markdown(
            """
            **PSI interpretation**  
            - **< 0.10** → Stable  
            - **0.10 – 0.25** → Moderate drift  
            - **> 0.25** → Significant drift
            """
        )
else:
    st.info("Upload a recent dataset or enable simulated sampling to activate drift monitoring.")