import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import os

# Correct the absolute path to the Parquet file
DATA_PATH = os.path.abspath('data/curated.parquet')

# Load the processed dataset
df = pd.read_parquet(DATA_PATH)

# Separate features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# F1-score
f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save classification report to outputs folder
REPORT_PATH = os.path.abspath('outputs/classification_report.txt')
with open(REPORT_PATH, 'w') as f:
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
print(f'Classification report saved to {REPORT_PATH}')

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
plt.figure()
plt.plot(recall, precision, marker='.', label='Logistic Regression')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()

# Save Precision-Recall curve to outputs folder
OUTPUT_PATH = os.path.abspath('outputs/precision_recall_curve.png')
plt.savefig(OUTPUT_PATH)
print(f'Precision-Recall curve saved to {OUTPUT_PATH}')

plt.show()
print('done')