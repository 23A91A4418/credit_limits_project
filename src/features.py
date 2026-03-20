import pandas as pd
import numpy as np
import os

def engineer_features(df):
    """
    Generate at least five new insightful features capturing behavioral patterns.
    """
    df = df.copy()
    
    # 1. Average Bill Amount: Mean of last 6 months billing
    df['avg_bill_amt'] = (df['BILL_AMT1'] + df['BILL_AMT2'] + df['BILL_AMT3'] +
                          df['BILL_AMT4'] + df['BILL_AMT5'] + df['BILL_AMT6']) / 6
    
    # 2. Credit Utilization: Avg bill relative to credit limit
    df['credit_utilization'] = df['avg_bill_amt'] / df['LIMIT_BAL'].replace(0, np.nan)
    
    # 3. Average Payment: Mean of last 6 months payment
    df['avg_payment'] = (df['PAY_AMT1'] + df['PAY_AMT2'] + df['PAY_AMT3'] +
                         df['PAY_AMT4'] + df['PAY_AMT5'] + df['PAY_AMT6']) / 6
                         
    # 4. Payment Ratio: How much of the bill is being paid off
    df['payment_ratio'] = df['avg_payment'] / df['avg_bill_amt'].replace(0, np.nan)
    
    # 5. Average Delay: Mean payment delay over 6 months
    df['avg_delay'] = (df['PAY_0'] + df['PAY_2'] + df['PAY_3'] +
                       df['PAY_4'] + df['PAY_5'] + df['PAY_6']) / 6
                       
    # 6. Spending Volatility (Std Dev): Variance in bill amount
    df['spending_std'] = df[['BILL_AMT1','BILL_AMT2','BILL_AMT3',
                             'BILL_AMT4','BILL_AMT5','BILL_AMT6']].std(axis=1)
                             
    # 7. Bill Growth: Comparing most recent bill to oldest bill
    df['bill_growth'] = (df['BILL_AMT1'] - df['BILL_AMT6']) / (df['BILL_AMT6'] + 1)
    
    # Handle infinite or NaN values resulting from division by zero
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

if __name__ == "__main__":
    print("Loading cleaned data...")
    input_path = "data/cleaned_data.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Please run data_prep.py first.")
        
    df = pd.read_csv(input_path)
    print("Engineering features...")
    df_featured = engineer_features(df)
    
    output_path = "data/featured_data.csv"
    df_featured.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Data saved to {output_path}")
