import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Create model folder if it doesn't exist
if not os.path.exists("model"):
    os.makedirs("model")

# Read dataset
data = pd.read_csv("dataset/insurance_claims.csv", encoding='utf-8')

# Features and target
X = data[['age', 'claim_amount', 'hospital_days']]
y = data['fraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model/fraud_model.pkl')
print("Model trained and saved successfully in 'model/fraud_model.pkl'!")
