# Executive Summary: Credit Limit Recommendation System

## Overview
This report outlines the findings and business recommendations from our new credit limit recommendation system. By leveraging machine learning models on customer demographics and historical repayment data, we accurately predict the probability of default to optimize credit limits and balance financial profitability with risk mitigation.

## Key Findings
- **Predictive Power of Payment History**: Late payments and payment ratios are the strongest indicators of future default. Customers with a history of missing payments (`PAY_0`, `avg_delay`) represent a significantly higher financial risk.
- **Credit Utilization**: Customers with high credit utilization ratios exhibit elevated default probabilities, signaling potential credit dependency.
- **Model Validation**: The final XGBoost ensemble model outperforms the baseline Logistic Regression model, successfully capturing non-linear relationships. Stratified K-Fold cross-validation indicates robust performance that generalizable to new customers. The model's predictions are highly interpretable using SHAP values.

## Business Impact Simulation
Implementing this model allows our institution to strategically adjust credit exposure:
- **Low-Risk Segment (Default Prob < 20%)**: We recommend safely increasing credit limits by up to 50% for these customers to encourage spending, drive revenue, and improve customer satisfaction.
- **Medium-Risk Segment (Default Prob 20%-50%)**: Maintain current limits, continuously monitoring their repayment behavior month-to-month.
- **High-Risk Segment (Default Prob > 50%)**: Proactively reduce credit limits by 30% to mitigate financial exposure and prevent profound subsequent losses.

## Strategic Recommendations
1. **Deploy the Dashboard**: Provide the interactive Streamlit dashboard to credit analysts. Analysts can use it to quickly reference automated risk assessments, explainability metrics, and recommended limits during human-in-the-loop application reviews.
2. **Automate Limit Adjustments**: Gradually roll out the dynamic credit limit adjustment policy on existing accounts based on the segmented risk categories defined by the model.
3. **Continuous Monitoring**: Re-evaluate and re-train the XGBoost model bi-annually with new customer default data to ensure the model's calibration and predictive edge remain sharp.
