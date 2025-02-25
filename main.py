import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Load Dataset (Example using NASA MC1 dataset)
df = pd.read_csv("path_to_dataset.csv")  # Replace with actual dataset path

# Splitting Features and Target
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target (Defect-prone or not)

# Standardizing Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA
pca = PCA(n_components=25)  # Keeping 25 principal components
X_pca = pca.fit_transform(X_scaled)

# Splitting Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# XGBRF Model (XGBoost + Random Forest)
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Training XGBoost Model
xgb_model.fit(X_train, y_train)
# Training Random Forest Model
rf_model.fit(X_train, y_train)

# Predicting
xgb_preds = xgb_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# Combining Predictions (Weighted Averaging)
final_preds = np.round((xgb_preds + rf_preds) / 2)

# Evaluation Metrics
accuracy = accuracy_score(y_test, final_preds)
precision = precision_score(y_test, final_preds)
recall = recall_score(y_test, final_preds)
f1 = f1_score(y_test, final_preds)
mcc = matthews_corrcoef(y_test, final_preds)

# Display Results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"MCC: {mcc:.4f}")