import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

def load_and_harmonize_lending_club():
    print("Harmonizing Lending Club data...")
    df = pd.read_csv('data/loan_ready.csv', usecols=['loan_amnt', 'emp_length', 'purpose', 'loan_status'])
    df = df.rename(columns={'loan_amnt': 'loan_amount', 'loan_status': 'target'})
    
    # Map emp_length
    def map_emp(x):
        if pd.isna(x): return 'unknown'
        x = str(x)
        if '< 1' in x: return '< 1 year'
        if '10+' in x or '7' in x or '8' in x or '9' in x: return '7+ years'
        if '4' in x or '5' in x or '6' in x: return '4-7 years'
        if '1' in x or '2' in x or '3' in x: return '1-4 years'
        return 'unknown'
        
    df['emp_length'] = df['emp_length'].apply(map_emp)
    return df

def load_and_harmonize_credit_risk():
    print("Harmonizing Generic Credit Risk data...")
    df = pd.read_csv('data/credit_risk_dataset.csv', usecols=['loan_amnt', 'person_emp_length', 'loan_intent', 'loan_status'])
    df = df.rename(columns={'loan_amnt': 'loan_amount', 'loan_intent': 'purpose', 'loan_status': 'target'})
    
    def map_emp(x):
        if pd.isna(x): return 'unknown'
        try:
            val = float(x)
            if val < 1: return '< 1 year'
            if val < 4: return '1-4 years'
            if val < 7: return '4-7 years'
            return '7+ years'
        except:
            return 'unknown'
            
    df['emp_length'] = df['person_emp_length'].apply(map_emp)
    df = df.drop(columns=['person_emp_length'])
    return df

def load_and_harmonize_german():
    print("Harmonizing German Credit data...")
    df = pd.read_csv('data/german_ready.csv', usecols=['hoehe', 'beszeit', 'verw', 'kredit'])
    df = df.rename(columns={'hoehe': 'loan_amount', 'verw': 'purpose', 'kredit': 'target'})
    
    def map_emp(x):
        if pd.isna(x): return 'unknown'
        val = str(x)
        if val == '1': return 'unemployed'
        if val == '2': return '< 1 year'
        if val == '3': return '1-4 years'
        if val == '4': return '4-7 years'
        if val == '5': return '7+ years'
        return 'unknown'
        
    df['emp_length'] = df['beszeit'].apply(map_emp)
    df['purpose'] = df['purpose'].astype(str)
    df = df.drop(columns=['beszeit'])
    return df

def run_incremental_training():
    df_lc = load_and_harmonize_lending_club()
    df_cr = load_and_harmonize_credit_risk()
    df_ger = load_and_harmonize_german()
    
    print("\nDatasets loaded. Shape comparison:")
    print(f"Lending Club: {df_lc.shape}")
    print(f"Credit Risk:  {df_cr.shape}")
    print(f"German Data:  {df_ger.shape}")
    
    # 1. Fit Preprocessor on combined data so it knows all possible categories
    print("\nFitting preprocessor on combined dataset to map all category dimensions...")
    df_combined = pd.concat([df_lc, df_cr, df_ger], ignore_index=True)
    X_combined = df_combined.drop(columns=['target'])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['loan_amount']),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ]), ['emp_length', 'purpose'])
        ]
    )
    
    preprocessor.fit(X_combined)
    
    # 2. Initialize XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    # 3. Train on Dataset 1 (Lending Club)
    print("\n--- Training Step 1: Lending Club Data ---")
    X1 = preprocessor.transform(df_lc.drop(columns=['target']))
    y1 = df_lc['target']
    xgb_clf.fit(X1, y1)
    
    # 4. Update on Dataset 2 (Credit Risk)
    print("\n--- Training Step 2: Updating with Generic Credit Risk Data ---")
    X2 = preprocessor.transform(df_cr.drop(columns=['target']))
    y2 = df_cr['target']
    # xgb_model allows continuing training from the previous booster
    xgb_clf.fit(X2, y2, xgb_model=xgb_clf.get_booster())
    
    # 5. Update on Dataset 3 (German Credit)
    print("\n--- Training Step 3: Updating with German Credit Data ---")
    X3 = preprocessor.transform(df_ger.drop(columns=['target']))
    y3 = df_ger['target']
    xgb_clf.fit(X3, y3, xgb_model=xgb_clf.get_booster())
    
    # Combine everything into a pipeline wrapper for easy usage
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb_clf)
    ])
    
    out_file = 'incremental_master_model.joblib'
    joblib.dump(final_pipeline, out_file)
    print(f"\nIncremental learning complete! Master model saved to {out_file}")

if __name__ == "__main__":
    run_incremental_training()
