# Product Requirements Document (PRD)

## 1. Objective
The primary objective of this project is to develop a robust quantitative model to accurately estimate the probability of default for customers within a specific book of loans. By improving default prediction, the organization can better manage credit risk and optimize lending strategies.

## 2. Background
Managing credit risk is critical for financial institutions. The ability to accurately predict which loans are likely to default directly impacts profitability and regulatory compliance. This project focuses on analyzing an existing book of loans to identify underlying patterns and risk factors, leveraging advanced quantitative research methods to enhance the current risk assessment framework.

## 3. Scope
This project encompasses:
- **Data Analysis**: Comprehensive exploratory data analysis of the provided loan book to identify key features and trends related to customer defaults.
- **Feature Engineering**: Implementation of a novel dynamic programming approach to convert continuous FICO scores into optimal categorical buckets, maximizing predictive power.
- **Model Development**: Building and evaluating statistical or machine learning models to estimate the probability of default for individual customers.

## 4. Requirements & Features
### 4.1 Data Ingestion and Preprocessing
- Securely ingest historical loan data.
- Handle missing values, outliers, and data inconsistencies.
- Normalize and scale numerical features as necessary for model stability.

### 4.2 FICO Score Categorization (Dynamic Programming)
- Implement a dynamic programming algorithm to segment continuous FICO scores into categorical bins.
- Ensure the algorithm optimizes for a specific objective function (e.g., maximizing information value or minimizing intra-bin variance with respect to default rates).
- Output the optimal bin boundaries and assign categorical labels to the dataset.

### 4.3 Probability of Default Estimation
- Develop a predictive model using the categorized FICO scores along with other relevant customer and loan attributes.
- Train the model to output a calibrated probability of default (a value between 0 and 1) for each customer.

## 5. Success Metrics
The success of the project will be evaluated based on the following criteria:
- **Model Accuracy**: Achieving a high Area Under the Receiver Operating Characteristic Curve (ROC-AUC) or other relevant classification metrics.
- **Calibration**: Ensuring the predicted probabilities closely align with actual observed default rates.
- **Interpretability**: The categorization of FICO scores should be intuitive and provide actionable insights into risk tiers.
