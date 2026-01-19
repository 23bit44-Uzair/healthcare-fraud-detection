import pandas as pd
import joblib

# Load trained model
model = joblib.load('model/fraud_model.pkl')

# Sample input claims (change or add as needed)
sample_input_df = pd.DataFrame(
    [[30, 250000, 5], [50, 100000, 2], [38, 350000, 6]],
    columns=['age', 'claim_amount', 'hospital_days']
)

# Predict
predictions = model.predict(sample_input_df)

# Output results
for i, pred in enumerate(predictions):
    if pred == 1:
        print(f"Claim {i+1}: Fraud Claim ❌")
    else:
        print(f"Claim {i+1}: Genuine Claim ✅")
