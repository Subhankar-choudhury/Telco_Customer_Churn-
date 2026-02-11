import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 1. Load Assets ---
baseline = joblib.load('baseline_model.pkl')
improved = joblib.load('improved_model.pkl')
scaler = joblib.load('scaler.pkl')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

# --- 2. Scale for Baseline ---
X_test_scaled = scaler.transform(X_test)

# --- 3. Run Evaluation ---
def run_eval(model, data, y_true, name):
    preds = model.predict(data)
    print(f"\n=== {name} Report ===")
    print(classification_report(y_true, preds))
    return confusion_matrix(y_true, preds)

cm_b = run_eval(baseline, X_test_scaled, y_test, "Baseline (Logistic Regression)")
cm_i = run_eval(improved, X_test, y_test, "Improved (Gradient Boosting)")

# --- 4. Error Analysis Visual ---
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Baseline CM')
sns.heatmap(cm_i, annot=True, fmt='d', cmap='Greens', ax=ax[1])
ax[1].set_title('Improved CM')
plt.savefig('error_analysis.png')
print("\nEvaluation complete. 'error_analysis.png' generated.")
