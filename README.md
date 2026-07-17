# 💳 Credit Card Fraud Detection
### Machine Learning Project — University of Management and Technology (UMT)

**Student:** Muhammad Anwar (F20232661139)  
**Institution:** University of Management and Technology, Lahore, Pakistan  
**Tools:** Python · Scikit-learn · TensorFlow/Keras · Google Colab  
**Dataset:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/gzdekzlkaya/credit-card-fraud-detection-dataset)

---

## 📌 Project Overview

This project builds and compares **7 machine learning models** to detect fraudulent credit card transactions from a highly imbalanced real-world dataset of **284,807 transactions** containing only **0.17% fraud cases**.

The core challenge is the extreme class imbalance — a naive model predicting "legitimate" for every transaction achieves 99.83% accuracy while catching **zero fraud**. This project addresses that problem using SMOTE, proper evaluation metrics, and threshold tuning.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Kaggle — Credit Card Fraud Detection |
| Total Transactions | 284,807 |
| Fraud Cases | 492 (0.17%) |
| Legitimate Cases | 284,315 (99.83%) |
| Total Features | 31 |
| Missing Values | None |

### Features
- **Time** — Seconds elapsed since the first transaction
- **V1 to V28** — Anonymized PCA-transformed features (privacy protected)
- **Amount** — Transaction amount in euros
- **Class** — Target variable (0 = Legitimate, 1 = Fraud)

---

## 🗂️ Project Structure

```
Credit-Card-Fraud-Detection/
│
├── Machine_Learning.ipynb        ← Complete Jupyter Notebook
├── creditcard_fraud_detection.csv← Dataset (download from Kaggle)
├── README.md                     ← This file
│
└── charts/                       ← Generated visualizations
    ├── class_distribution.png
    ├── amount_distribution.png
    ├── time_distribution.png
    ├── correlation_heatmap.png
    ├── feature_importance.png
    ├── smote_comparison.png
    ├── confusion_matrix_*.png
    ├── random_forest_feature_importance.png
    ├── models_comparison.png
    ├── roc_curves_all_models.png
    ├── ann_training_history.png
    ├── final_comparison_chart.png
    ├── final_roc_curves.png
    └── precision_recall_curves.png
```

---

## 🔬 Methodology

### Step 1 — Exploratory Data Analysis (EDA)
- Analyzed class distribution, feature distributions, and correlations
- Identified extreme class imbalance (492 fraud vs 284,315 legitimate)
- Found V17, V14, V12 as strongest fraud indicators via correlation analysis

### Step 2 — Preprocessing
- Scaled `Amount` and `Time` using `StandardScaler` (V1–V28 already PCA-scaled)
- Stratified 80/20 train/test split — preserving 0.17% fraud ratio in both sets
- Applied **SMOTE** (Synthetic Minority Oversampling) on training data only
- Result: 227,451 fraud + 227,451 legitimate = perfectly balanced training set

### Step 3 — Model Training (7 Models)
1. Logistic Regression
2. Decision Tree
3. Random Forest (100 trees)
4. KNN (K=5, 20% subset for speed)
5. SVM (LinearSVC + CalibratedClassifierCV)
6. Naive Bayes (GaussianNB)
7. ANN — 3 hidden layers with BatchNorm, Dropout, EarlyStopping

### Step 4 — Evaluation
- Metrics: **F1 Score, Precision, Recall, AUC-ROC** (not Accuracy — misleading on imbalanced data)
- ROC Curve and Precision-Recall Curve for all 7 models
- Threshold tuning on ANN to maximize F1 Score

---

## 🏗️ ANN Architecture

```
Input(30)
    → Dense(64, ReLU) → BatchNormalization → Dropout(0.3)
    → Dense(32, ReLU) → BatchNormalization → Dropout(0.2)
    → Dense(16, ReLU) → BatchNormalization → Dropout(0.1)
    → Dense(1, Sigmoid)

Optimizer  : Adam (lr=0.001)
Loss       : Binary Crossentropy
Callbacks  : EarlyStopping(patience=10) + ReduceLROnPlateau(patience=5)
```

---

## 📈 Results

| Model | Precision | Recall | F1 Score | AUC-ROC |
|---|---|---|---|---|
| Random Forest | 0.4279 | 0.8776 | 0.5753 | **0.9836** |
| SVM | 0.0543 | 0.9082 | 0.1024 | 0.9739 |
| Logistic Regression | 0.0581 | 0.9184 | 0.1094 | 0.9698 |
| KNN | 0.1679 | 0.9082 | 0.2834 | 0.9523 |
| ANN (optimal threshold) | **0.8851** | 0.7857 | **0.8324** | 0.9484 |
| Naive Bayes | — | — | — | — |
| Decision Tree | 0.0792 | 0.8061 | 0.1442 | 0.8950 |

> Naive Bayes results populate after running the notebook end-to-end.

### Key Findings

- **Random Forest** — Best AUC-ROC (0.9836): strongest overall discrimination ability
- **ANN with optimal threshold** — Best F1 (0.8324) and Precision (0.8851): best real-world balance
- **Logistic Regression** — Best Recall (0.9184): catches the most fraud but high false alarm rate
- Accuracy was excluded as a metric — every model scores ~99.7–99.9% due to class imbalance, making it meaningless
- Threshold tuning on ANN improved Precision from 0.69 → 0.89 and F1 from 0.76 → 0.83

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)

1. Upload `creditcard_fraud_detection.csv` to your Google Drive root folder
2. Open `Machine_Learning.ipynb` in Google Colab
3. Run all cells top to bottom (`Runtime → Run All`)
4. When prompted, authorize Google Drive access

### Option 2 — Local Jupyter

```bash
# Clone the repository
git clone https://github.com/M-Anwar921/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow

# Launch Jupyter
jupyter notebook Machine_Learning.ipynb
```

> **Note:** If running locally, remove the Google Drive mounting cells (Cells 3) and replace with:
> ```python
> df = pd.read_csv('creditcard_fraud_detection.csv')
> ```

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
tensorflow
```

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow
```

---

## 📓 Notebook Structure

| Day | Cells | Topic |
|---|---|---|
| Day 1 | 1 – 11 | Setup, EDA, Visualizations |
| Day 2 | 12 – 17 | Preprocessing, Scaling, SMOTE |
| Day 3 | 18 – 24 | Logistic Regression, Decision Tree, Random Forest |
| Day 4 | 25 – 30 | KNN, SVM, Naive Bayes, Classical Model Comparison |
| Day 5 | 31 – 37 | ANN Architecture, Training, Evaluation |
| Day 6 | 38 – 43 | Final Comparison, ROC Curves, Conclusions |

---

## 💡 Why Not Use Accuracy?

A model that predicts **"Legitimate"** for **every single transaction** achieves:

```
Accuracy = 284,315 / 284,807 = 99.83%
```

While catching **zero fraud**. This is the accuracy paradox on imbalanced datasets. We used:

- **Precision** — Of all predicted fraud, how many were actually fraud?
- **Recall** — Of all actual fraud, how many did we catch?
- **F1 Score** — Harmonic mean of Precision and Recall
- **AUC-ROC** — Overall model discrimination ability across all thresholds

---

## 🔮 Future Work

- [ ] Add XGBoost and LightGBM — industry standard for tabular fraud detection
- [ ] Implement Autoencoder — unsupervised anomaly detection approach
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] SHAP values for model explainability
- [ ] Deploy best model as FastAPI endpoint
- [ ] Build an agentic fraud detection system using LangChain

---

## 📚 Skills Learned

- Exploratory Data Analysis (EDA)
- Handling extreme class imbalance with SMOTE
- Feature scaling with StandardScaler
- Stratified train/test splitting
- 6 classical ML models from scratch
- ANN with Keras — BatchNorm, Dropout, Callbacks
- Proper evaluation — F1, AUC-ROC, Precision, Recall
- Threshold tuning for optimal real-world performance
- ROC and Precision-Recall curve interpretation
- Professional notebook structure and documentation

---

## 📄 License

This project is for academic purposes — University of Management and Technology (UMT), Lahore, Pakistan.

---

## 🙋 Author

**Muhammad Anwar**  
Computer Science Student — UMT Lahore  
GitHub: [@M-Anwar921](https://github.com/M-Anwar921)  
LinkedIn: [Muhammad Anwar](https://linkedin.com/in/muhammad-anwar)

---

*Built as part of the Machine Learning course project — UMT Lahore, 2025*
