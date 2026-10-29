# 📉 Predicting Customer Churn Using Machine Learning

Customer churn prediction is a critical challenge for subscription-based companies, as customer retention is directly correlated with long-term profitability. This project focuses on developing and evaluating machine learning models to predict customer churn using the **Telco Customer Churn** dataset. The problem is formulated as a binary classification task, leveraging demographic, behavioral, and service-related features.

* **University:** Ontario Tech University — Canada
* **Professor:** Prof. Li Yang
* **Authors:** Adyan Iqbal, Gabriele Martorano, Alexander Younis and Mohamed Yusuf

---

## 📊 Dataset Overview

The dataset used is the publicly available [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle. It simulates a real-world environment of customer behavior in the telecommunications industry.

* **Size:** 7,043 customer records
* **Features:** 21 attributes (demographics, service subscriptions, and billing information)
* **Objective:** Predict the `Churn` target variable (Yes/No) based on historical customer data.

---

## ⚙️ Methodology & ML Pipeline

The architecture follows a robust, modular pipeline designed to systematically handle class imbalances and thoroughly evaluate model performance:

1. **Exploratory Data Analysis (EDA):** Identified severe class imbalance (73.5% Retained vs. 26.5% Churned) and uncovered key churn drivers (e.g., short tenure, month-to-month contracts, high monthly charges).
2. **Data Preprocessing:** 
   * Handled missing/blank values systematically (e.g., in `TotalCharges`).
   * Dropped non-predictive identifiers (`customerID`).
   * Applied one-hot encoding to categorical variables.
   * Standardized numerical features via `StandardScaler` to ensure optimal gradient descent convergence.
3. **Baseline Models:** Evaluated Logistic Regression, Decision Trees, and K-Nearest Neighbors (KNN). Logistic Regression established a strong baseline with a ROC-AUC of 84.14% and a high recall of 78.61%.
4. **Advanced Ensemble Models:** Implemented and tuned Random Forest, CatBoost, and a Multi-Layer Perceptron (MLP). Hyperparameters were optimized via `RandomizedSearchCV` paired with `StratifiedKFold` cross-validation to maintain the underlying class distribution across data splits.
5. **Explainability & Feature Importance:** Leveraged **SHAP** (SHapley Additive Explanations) and Mean Decrease in Impurity (MDI) to interpret non-linear interactions and validate the models' internal logic.

---

## 🏆 Key Results & Insights

Ensemble methods significantly outperformed baseline statistical approaches in capturing complex, non-linear feature relationships.

| Model | Key Achievement | Metric Score |
|---|---|---|
| **CatBoost** | Best Overall & Highest Recall | **84.72%** (ROC-AUC) / **80.75%** (Recall) |
| **Random Forest** | Highest F1-Score | **63.29%** (F1) |

* **Key Predictive Features:** `Tenure`, `TotalCharges`, `Contract_Two year`, and `MonthlyCharges` emerged as the most critical drivers of churn. 
* **Business Insight:** Model explainability (SHAP) confirmed that long-term contractual commitments and high tenure significantly reduce churn probability, suggesting that business retention strategies should focus heavily on early-stage engagement and contract upgrades.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python installed along with the following data science dependencies. Using a virtual environment is highly recommended.
```bash
pip install pandas numpy matplotlib seaborn scikit-learn catboost shap
```

### Running the Code
You can explore the project in two different ways, depending on your preference:

**Option A: Command Line Script**

Execute the main Python script to run the full pipeline end-to-end, from data preprocessing to model evaluation.
```bash
python predicting_customer_churn.py
```
**Option B: Interactive Notebook (Recommended)**

Run the provided Jupyter Notebook step-by-step to view interactive visualizations, performance metric tables, ROC curves, and detailed SHAP explainability plots.

```bash
jupyter notebook Predicting_Customer_Churn.ipynb
```
