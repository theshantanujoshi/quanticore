# Project Overview: Quantitative Credit Risk & Default Prediction

## Project Summary
During this internship, I focused on applying quantitative research methods to the domain of credit risk modeling. The core of the project involved analyzing a comprehensive book of loans to estimate the probability of default for individual customers. A key innovation in this analysis was the use of dynamic programming to optimally transform continuous FICO scores into categorical data, significantly improving the predictive capability of our models.

## Methodology

### Quantitative Research & Analysis
The project began with a deep dive into the loan book data. This involved extensive exploratory data analysis to understand the distribution of variables, identify correlations between customer attributes and default events, and clean the data for modeling. The quantitative approach ensured that all feature engineering and modeling decisions were driven by statistical evidence rather than intuition alone.

### Dynamic Programming for FICO Score Categorization
FICO scores are a critical indicator of creditworthiness. However, using them purely as a continuous variable can sometimes mask non-linear relationships with default risk. To address this, I implemented a dynamic programming algorithm.
- **Objective**: The goal was to find the optimal set of cut-off points to group FICO scores into distinct categorical buckets.
- **Approach**: The dynamic programming algorithm evaluated various partitionings of the FICO score range, seeking to maximize the distinction in default rates between the buckets while minimizing variance within each bucket. 
- **Outcome**: This resulted in a statistically sound categorization that grouped customers with similar risk profiles more effectively than arbitrary, manual bucketing.

### Probability of Default Modeling
Using the optimized FICO categories alongside other relevant financial and demographic features, I developed a model to estimate the Probability of Default (PD) for each customer. This model provides a granular risk assessment, allowing for more nuanced decision-making regarding credit limits, interest rates, and loan approvals.

## Results & Impact
- **Enhanced Risk Segmentation**: The dynamic programming approach to categorizing FICO scores provided clearer, more actionable risk tiers.
- **Improved Predictive Power**: By converting FICO scores into optimal categorical data, the overall model's ability to accurately predict defaults was improved compared to baseline models.
- **Data-Driven Insights**: The quantitative analysis of the loan book revealed key drivers of default, providing valuable insights for future risk management strategies.

## Future Work
- Incorporate alternative data sources (e.g., macroeconomic indicators) to further refine the probability of default estimates.
- Explore more advanced machine learning algorithms (e.g., Gradient Boosting, Neural Networks) to capture complex, non-linear interactions between variables.
