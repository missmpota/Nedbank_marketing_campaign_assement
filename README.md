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

