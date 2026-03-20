import pandas as pd
import numpy as np
import os

def load_and_clean_data(filepath="data/default of credit card clients.xls"):
    """
    Ingest, clean, and preprocess the raw credit card dataset.
    Handles missing values/outliers by replacing inconsistent values.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found.")
        
    df = pd.read_excel(filepath, header=1)
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Drop ID column if it exists
    if 'ID' in df.columns:
        df.drop("ID", axis=1, inplace=True)
        
    # Clean EDUCATION (0, 5, 6 are undocumented -> replace with 4 'others')
    df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)
    
    # Clean MARRIAGE (0 is undocumented -> replace with 3 'others')
    df['MARRIAGE'] = df['MARRIAGE'].replace(0, 3)
    
    # Rename long target column to standard name
    if 'default payment next month' in df.columns:
        df.rename(columns={"default payment next month": "default_next_month"}, inplace=True)
        
    return df

if __name__ == "__main__":
    print("Loading and cleaning data...")
    df_clean = load_and_clean_data()
    os.makedirs("data", exist_ok=True)
    output_path = "data/cleaned_data.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"Data cleaning complete. Cleaned data saved to {output_path}")
