# Software Defect Prediction using XGBRF and Comparative Analysis

## Introduction
Software defect prediction (SDP) is an essential task in software engineering to enhance software reliability and quality. The aim is to predict defect-prone modules early in the development phase using machine learning techniques. In this project, we propose a hybrid approach using **Extreme Gradient Boosting with Random Forest (XGBRF)** along with **Principal Component Analysis (PCA)** for dimensionality reduction. The performance of XGBRF is compared with other machine learning models including **Random Forest (RF), Multi-Layer Perceptron (MLP), XGBoost, Gradient Boosting, Support Vector Machine (SVM), and Logistic Regression**.

## Methodology
The proposed methodology consists of the following steps:
1. **Data Preprocessing:**
   - Standardization using **StandardScaler**.
   - Dimensionality reduction using **Principal Component Analysis (PCA)** to retain the most significant features.

2. **Model Training & Evaluation:**
   - Various machine learning classifiers were trained and evaluated on different datasets.
   - The hybrid **XGBRF** model combines **XGBoost** and **Random Forest** for enhanced prediction stability and performance.

3. **Performance Metrics Used:**
   - **Accuracy**
   - **Precision**
   - **Recall**
   - **F1 Score**
   - **Matthews Correlation Coefficient (MCC)**

## Datasets Used
The evaluation was conducted using publicly available **NASA datasets** including:
- **CM1** (327 instances, 39 attributes)
- **MC1** (1988 instances, 40 attributes)
- **PC1** (705 instances, 39 attributes)
- **PC2** (745 instances, 38 attributes)
- **MW1** (253 instances, 39 attributes)

## Results & Comparison
Below is the performance comparison of different models tested on the **MC1 dataset**:

| Classifier              | Accuracy | Precision | Recall | F1 Score | MCC  |
|-------------------------|----------|------------|--------|---------|------|
| Logistic Regression     | 92.60%   | 98.98%     | 42.92% | 96.37%  | 19.24|
| Support Vector Machine  | 94.31%   | 99.19%     | 57.14% | 96.57%  | 26.78|
| Multi-Layer Perceptron  | 95.77%   | 99.38%     | 57.34% | 96.63%  | 26.41|
| Random Forest          | 97.63%   | 98.64%     | 48.57% | 96.04%  | 11.58|
| Gradient Boosting       | 98.07%   | 99.43%     | 71.03% | 97.16%  | 36.95|
| **XGBRF (Proposed Model)** | **98.65%** | **99.49%** | **88.91%** | **98.87%** | **55.87** |

The **XGBRF model outperformed all other algorithms**, achieving the highest accuracy of **98.65%** on the MC1 dataset while maintaining robust precision, recall, and MCC scores.

## Installation & Usage
### Prerequisites
Ensure you have Python installed along with the following dependencies:
```sh
pip install numpy pandas matplotlib scikit-learn xgboost
```

### Running the Code
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/software-defect-prediction.git
   cd software-defect-prediction
   ```
2. Place the dataset in the project directory.
3. Run the script:
   ```sh
   python software_defect_prediction.py