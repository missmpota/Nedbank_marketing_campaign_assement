# Nedbank_marketing_campaign_assement
## Project Overview
This project was developed to solve a marketing campaign targeting problem for Nedbank.

The marketing team previously used a **shotgun approach**, contacting customers at random. Since each campaign required individual calls, this was inefficient and time-consuming.

The objective of this project is to build a machine learning solution that can **identify customers most likely to respond to a marketing campaign**, allowing the bank to prioritize high-probability customers and improve campaign efficiency.

---

## Business Problem
The Nedbank marketing team ran a campaign and recorded whether customers successfully responded to the offer.

Success means the customer **took up the product**.

Instead of calling customers randomly, the goal is to:
- predict which customers are most likely to respond
- rank customers by probability of response
- reduce unnecessary calls
- improve campaign uptake rate
---

## Project Objectives
- Perform data cleaning and preprocessing
- Engineer relevant banking and campaign features
- Train classification models ( three models we used with logistic regresion as baseline model)
- Compare model performance using stratified cross-validation
- Evaluate campaign uplift using decile analysis
- Build a Streamlit dashboard for exploratory data analysis and another for model interpretation and business insights
- Include model drift monitoring logic

---

## Dataset
The dataset contains customer demographic, financial, and campaign-related variables such as:
- Unnamed: 0
- age
- job
- marital
- education
- default
- balance
- housing
- loan
- contact
- day
- month
- duration
- campaign
- pdays
- previous
- poutcome
- target
- post_campaign_action

  ---
  ## Preprocessing data
In order to pre preprocess data the following data cleaning steps were taken:
- drop columns that are not needed like 'unnamed' and 'duration'. Unnamed does not contain any value while duration introduces data leakage in the dataset in real time we would not know duration of call until the call was completed
- filling in na values for both categorial and nmeric data
- drop duplicate rows in the data set
- determining target variable disribution to assess if data is balanced or not
---
 ## Feature Engineering
Several features were engineered to improve predictive performance, including:

### Time-based features
- `month_phase`
- `month_num`
- `season`
- `week_of_month`
- `salary_window`
- `salary_cycle`
- `payday_call`
  
### Campaign history features
- `previous_contact`
- `contact_recency`
- `previous_success`
- `total_contacts`
  
### Financial behaviour features
- `financial_pressure`
- `balance_shifted`
- `log_balance`
- `balance_segment`
- `liquidity_pressure`
- `high_balance`
- `negative_balance`
- `balance_per_contact`
  
### Customer profiling features
- `age_group`
- `financial_maturity`
- `job_stability`
- `job_stability_score`
- `education_level`
- `family_commitment`
- `life_stage_maturity`
- `life_stage`
- `stability_score`
  
### Credit features
- `credit_risk`
- `credit_stress`

---

## Models Used
The following models were trained and compared:

1. **Logistic Regression**
   - baseline and interpretable model
   - class weighted to handle imbalance

2. **Random Forest**
   - non-linear ensemble model
   - class weighted

3. **XGBoost**
   - boosting-based model for tabular data
   - imbalance handled using `scale_pos_weight`

---
## Model Validation and Strategy
Because the dataset is imbalanced, the project used:

- **Stratified train/test split**
- **Stratified 5-fold cross-validation**
- **ROC-AUC** as the main comparison metric

This ensures class proportions are preserved during training and evaluation.

---
## Evaluation Metrics
The following metrics were used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

In addition, the project used:
- **Feature importance**
- **Logistic regression coefficients**
- **Decile analysis**
- **Uptake improvement analysis**
- **Population Stability Index (PSI)** for drift monitoring

---

## Key Business Insight
The baseline campaign uptake rate was approximately **11.55%** under random targeting.

Using the model to target the top-ranked customers increased uptake significantly:
- **Top 10% targeted customers:** ~47.78%
- **Top 20% targeted customers:** ~32.04%
- **Top 30% targeted customers:** ~25.37%

This shows that the model can substantially improve campaign efficiency by prioritizing customers most likely to respond.

---

## Dashboard Features
Two Streamlit dashboards were built. To run streamlit dashboard, open termina and open directoty where files are save, run "streamlit run app.py" 
### Exploratoy Data Analysis
- number of customers 
- current uptake rate
- average customer balance
- previous contacted customer rate
- data quality statistics
- correlation heatmap
- feature distributions plots
- feature outlier box plots 

### Model Results and Drift Monitoring
- final features used
- correlation with target
- model performance comparison
- ROC curve comparison
- feature importance
- logistic regression drivers
- decile analysis
- uptake rate before vs after model targeting
- model drift monitoring using PSI

