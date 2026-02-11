import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# --- 1. Load and Clean ---
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df.drop('customerID', axis=1, inplace=True)

# --- 2. Preprocessing ---
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn', 'gender']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

df = pd.get_dummies(df)

# --- 3. Split (Stratified 80/20) ---
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- 5. Train Models ---
# Baseline
baseline = LogisticRegression(random_state=42)
baseline.fit(X_train_scaled, y_train)

# Improved
improved = GradientBoostingClassifier(n_estimators=100, random_state=42)
improved.fit(X_train, y_train)

# --- 6. Save Assets ---
joblib.dump(baseline, 'baseline_model.pkl')
joblib.dump(improved, 'improved_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Training Complete. Models and test data saved.")
