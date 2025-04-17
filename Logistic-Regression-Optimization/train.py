import numpy as np
import pandas as pd
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, recall_score, f1_score, precision_score)
from logisticRegressionModel import LogisticRegression
import matplotlib.pyplot as plt

# Download and Load the Loan Approval Dataset
print("Downloading dataset...")
path = kagglehub.dataset_download("taweilo/loan-approval-classification-data")

# Find the CSV file
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
full_path = os.path.join(path, csv_files[0])
df = pd.read_csv(full_path)
df.columns = df.columns.str.strip()  # Clean column names

# Encode Categorical Features
print("\nPreprocessing data...")
categorical_cols = df.select_dtypes(include='object').columns
df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate Features and Target
target_col = 'loan_status'
X = df_processed.drop(target_col, axis=1).to_numpy()
y = df_processed[target_col].to_numpy()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Compare all optimizer modes
modes = ['batch', 'sgd', 'mini-batch']

plt.figure(figsize=(10, 6))

for mode in modes:
    print(f"\n--- Training with optimizer: {mode.upper()} ---")
    clf = LogisticRegression(lr=0.01, epochs=500, mode=mode, batch_size=32)
    clf.fit(X_train, y_train)

    class_predictions, probabilities = clf.predict(X_test)

    accuracy = accuracy_score(y_test, class_predictions)
    precision = precision_score(y_test, class_predictions)
    recall = recall_score(y_test, class_predictions)
    f1 = f1_score(y_test, class_predictions)
    cm = confusion_matrix(y_test, class_predictions)

    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot loss curve
    plt.plot(range(1, clf.epochs + 1), clf.losses, label=f'{mode}')

# Final plot
plt.title("Loss Curve per Optimizer")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()