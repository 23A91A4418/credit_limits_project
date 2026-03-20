import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, f1_score
from xgboost import XGBClassifier
import joblib
import json
import os

def train_and_evaluate():
    input_path = "data/featured_data.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Run features.py first.")
        
    df = pd.read_csv(input_path)
    X = df.drop('default_next_month', axis=1)
    y = df['default_next_month']
    
    # Stratified split to handle imbalanced defaults
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\\n--- Baseline: Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print(classification_report(y_test, y_pred_lr))
    
    print("\\n--- Advanced Model: XGBoost ---")
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    print(classification_report(y_test, y_pred_xgb))
    
    print("\\n--- Cross-Validation XGBoost ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(xgb, X, y, cv=skf, scoring='accuracy')
    print("Mean CV Accuracy:", scores.mean())
    
    # Save the model and feature names for the dashboard
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb, "models/xgb_model.pkl")
    with open("models/feature_names.json", "w") as f:
        json.dump(list(X.columns), f)
        
    print("\\nSaved XGBoost model to models/xgb_model.pkl")
    print("Saved feature names to models/feature_names.json")
    
if __name__ == "__main__":
    train_and_evaluate()
