# Credit Limit Recommendation System

This project is a data science assignment focused on building an explainable machine learning system to recommend optimal credit limits for customers. It aims to demonstrate a complete data pipeline from preprocessing and feature engineering to predictive modeling, explainability, and business dashboarding.

## Project Structure

The project is organized into the following directories:

* data/
  * default of credit card clients.xls: The original raw dataset provided for the project.
  * cleaned_data.csv: Output of the data cleaning script.
  * featured_data.csv: Output of the feature engineering script.
* src/
  * data_prep.py: Script that handles data ingestion, cleaning, and formatting.
  * features.py: Script that creates new behavioral features (such as credit utilization and payment ratios).
  * train.py: Script that trains the machine learning models (Logistic Regression and XGBoost) and performs cross-validation testing.
* models/
  * xgb_model.pkl: The saved XGBoost model file.
  * feature_names.json: The list of exact feature names required by the model.
* notebooks/
  * credit_card_analysis.ipynb: The original Jupyter notebook containing exploratory data analysis (EDA), statistical tests, and SHAP model explainability plots.
* dashboard/
  * app.py: The Streamlit application that provides an interactive interface to test the model.
* reports/
  * executive_summary.md: A formal report summarizing the findings and suggesting a new credit policy based on risk segmentation.

## How to Run the Project

Follow these simple steps to replicate the results and run the dashboard locally.

### 1. Install Required Libraries
Open your terminal or command prompt and install the dependencies needed to run the scripts:

```bash
pip install -r requirements.txt
```

### 2. Execute the Data Pipeline
Run the Python scripts in order to build the dataset and train the model from scratch:

```bash
python src/data_prep.py
python src/features.py
python src/train.py
```

### 3. Launch the Web App Dashboard
After the model finishes training, you can start the user interface to explore predictions:

```bash
streamlit run dashboard/app.py
```
This command will host the dashboard at http://localhost:8502 and should automatically open in your web browser.

## Methodology and Findings

**Methodology**:
We followed an end-to-end data science lifecycle. We ingested the generic customer credit card dataset, mitigated missing/inconsistent values, and developed 7 new behavioral features encompassing *credit utilization*, *average payment delays*, and *spending volatility*. The predictive system compares a Logistic Regression baseline against an advanced XGBoost model. Model interpretability is achieved globally and locally using SHAP (Shapley Additive exPlanations) values.

**Key Findings**:
* **Predictive Indicators**: Payment history (specifically `PAY_0`), credit utilization, and the payment ratio strongly predict default probabilities.
* **Model Superiority**: The XGBoost ensemble captured non-linear relationships much more effectively than standard logistic regression, achieving a cross-validated accuracy of ~82% with excellent interpretability.
* **Risk Strategy**: Customers can be reliably segmented into Low, Medium, and High-risk tiers. We proposed a data-driven policy to actively expand limits (+50%) for low-risk groups, and mitigate exposure (-30%) for high-risk customers, dynamically maximizing return-on-risk.

---
*Note: This README and the pipeline restructuring were generated with the assistance of an AI.*