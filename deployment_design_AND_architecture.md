# Deployment, Monitoring, and Ethical Considerations
This document outlines the strategic approach for moving the model from development into a production environment, focusing on operational efficiency and ethical responsibility.
## Model Deployment strategies
### Batch Data
Currently model runs on full data where training is performend on historical data. As data increases, batch deployment would be appropriate when campaign customer targeting is done on scheduled time basis such as daily, weekly, monhtly etc. 
- data is extracted from warehouse using data extraction pipelines. Pipeline will run sql/hive query that specifies tables, dates etc. Exctraction pipeline will extract data from warehousing using query, export extracted data file into parquet/csv format and save data either in azure cloud platform data storage or on prem server. Data will be exctracted in chunks that will fit file forma type.
- docker containers will be built to package different ml stages such as data loading, preprocessing, feature engineering, saved trained model that has best performance. Reuslts of model t be saved in data warehouse table. during pipeline orchestration, will handle to proper sequence of when each container will run and point to server where result will be saved.
- advantages of using batch are  low latency, provides immediate value to the end-user, good for pre planned campaigns, easier to audit and reproduce predictions

### Real-Time Data
Real-time deployment is appropriate when the business wants to make an immediate decision during a live interaction, such as when a customer logs into an app, calls a contact center, visits online banking, or triggers a transactional event. When a customer event occurs, the application sends the required feature inputs to the service, receives a prediction in milliseconds or seconds, and can decide whether to run campaign
- Customer event such as call to customer, customer logging on to banking app is triggered.
- docker containers deployed as api run on customer event data, api return probability of response  and campaign run to customer based on probablity for product uptake.
- advanatges of using real time data are Low latency, provides immediate value to the end-user, uses most recent live data, improves uptake where time is a factor.
- 
### Recommendation

For this problem, a batch-first deployment is the strongest recommendation unless the business explicitly needs live decisioning due to time constraints

## Performance Monitoring
Once deployed, model monitoring is necessary because customer behavior, product uptake patterns, and campaign strategies can change over time.
*   **Metrics:** Track key performance indicators (KPIs) like **AUC-ROC for ranking quality, Precision, Recall, and F1-Score** in real-time.
*   **Feedback Loops:** Compare model predictions against actual outcomes as they become available to calculate "ground truth" accuracy.

### Drift Detection
*   **Data Drift:** Use statistical tests (e.g., **Kolmogorov-Smirnov test**) to detect shifts in the distribution of input features compared to the training set.
*   **Concept Drift:** Monitor if the relationship between features and the target variable changes over time (e.g., consumer behavior changing during an economic downturn).
*   **Tooling:** streamlit model montoring dashboard to send autommated email to data science once there is a change in drift shift is detect from change in features or model performance going below certian threshold e.g accuracy on 75%

---
# Ethical Considerations 
Even when a model improves campaign efficiency, it can create fairness concerns if some customer groups are consistently excluded.

*   **Algorithmic Bias:** If the training data contains historical biases, the model may systematically exclude specific protected groups (e.g., based on race, gender, or age).
*   **Fairness Metrics:** Regularly audit the model using metrics  to ensure fair treatment across all client segments.
*   **Human and machine interaction** Ensure that campaign teams and model risk stakeholders review model outcomes rather than relying blindly on the model.
*   **Inclusion:** If certain groups are consistently excluded, we must investigate if the model is penalizing them for specific reasin such as  lack of data and consider alternative data sources to promote financial inclusion
