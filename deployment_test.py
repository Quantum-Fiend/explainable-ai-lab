import joblib
import pandas as pd
import numpy as np
import sys
import os

MODEL_PATH = 'model_pipeline.joblib'

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Please run the notebook to generate it first.")
        sys.exit(1)
    
    print("Loading model pipeline...")
    pipeline = joblib.load(MODEL_PATH)
    return pipeline

def predict_single(pipeline, data):
    print("\n--- Production Inference Request ---")
    print(f"Input Data:\n{data}")
    
    try:
        # Convert dict to DataFrame
        df = pd.DataFrame([data])
        
        # Predict Probability
        prob = pipeline.predict_proba(df)[0][1]
        
        # Predict Class
        pred = pipeline.predict(df)[0]
        
        status = "HIGH RISK" if pred == 1 else "LOW RISK"
        
        print(f"\nResult: {status}")
        print(f"Risk Probability: {prob:.4f}")
        return status, prob
    except Exception as e:
        print(f"Inference Error: {e}")
        return None

if __name__ == "__main__":
    # Test Payload mimicking a single user request
    test_payload = {
        'checking_status': '<0',
        'duration': 12,
        'credit_history': 'critical/other existing credit',
        'purpose': 'radio/tv',
        'credit_amount': 2000,
        'savings_status': 'no known savings',
        'employment': '>=7',
        'installment_commitment': 4,
        'personal_status': 'male single',
        'other_parties': 'none',
        'residence_since': 4,
        'property': 'real estate',
        'age': 30,
        'other_payment_plans': 'none',
        'housing': 'own',
        'existing_credits': 1,
        'job': 'skilled',
        'num_dependents': 1,
        'own_telephone': 'yes',
        'foreign_worker': 'yes'
    }
    
    # Load and Predict
    model = load_model()
    predict_single(model, test_payload)
