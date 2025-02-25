import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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

# Initialize Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5),
    "MLP": MLPClassifier(hidden_layer_sizes=(50,50), max_iter=500, random_state=42),
    "SVM": SVC(kernel='linear', probability=True),
    "Logistic Regression": LogisticRegression(max_iter=500)
}

# Train and Evaluate Models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    results[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1 Score": f1_score(y_test, preds),
        "MCC": matthews_corrcoef(y_test, preds)
    }

# Display Results
for name, metrics in results.items():
    print(f"{name} Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()
