# Predicting Customer Churn Using Machine Learning

## 📌 Project Overview
Customer churn prediction is a critical challenge for subscription-based companies, as customer retention is directly correlated with long-term profitability. This project focuses on developing and evaluating machine learning models to predict customer churn using the **Telco Customer Churn** dataset. The problem is formulated as a binary classification task, leveraging demographic, behavioral, and service-related features. 

**Group 6 Members:**
* Alexander Younis
* Yusuf Mohamed
* Gabriele Martorano
* Adyan Iqbal

## 📊 Dataset
The dataset used is the publicly available [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), consisting of 7,043 customer records and 21 features. It simulates a real-world environment of customer behavior in the telecommunications industry, tracking demographic data, service subscriptions, and billing information.

## ⚙️ Methodology & Pipeline
The architecture follows a robust, modular pipeline designed to systematically handle class imbalances and evaluate performance:

1. **Exploratory Data Analysis (EDA):** Identified severe class imbalance (73.5% Retained vs. 26.5% Churned) and uncovered key churn drivers (e.g., short tenure, month-to-month contracts, high monthly charges).
2. **Data Preprocessing:** 
   * Handled missing/blank values systematically (`TotalCharges`).
   * Dropped non-predictive identifiers (`customerID`).
   * Applied one-hot encoding to categorical variables.
   * Standardized numerical features via `StandardScaler` to ensure optimal gradient descent convergence.
3. **Baseline Models:** Evaluated Logistic Regression, Decision Trees, and K-Nearest Neighbors (KNN). Logistic Regression established a strong baseline with a ROC-AUC of 84.14% and a high recall of 78.61%.
4. **Advanced Ensemble Models:** Implemented and tuned Random Forest, CatBoost, and a Multi-Layer Perceptron (MLP). Hyperparameters were optimized via `RandomizedSearchCV` paired with `StratifiedKFold` cross-validation to maintain the underlying class distribution across splits.
5. **Explainability & Feature Importance:** Leveraged SHAP (SHapley Additive Explanations) and Mean Decrease in Impurity (MDI) to interpret non-linear interactions and model predictions.

## 🏆 Key Results
Ensemble methods significantly outperformed baseline statistical approaches in capturing complex, non-linear feature relationships.

* **Best Overall Model (CatBoost):** Achieved the highest ROC-AUC (**84.72%**) and the highest Recall (**80.75%**), making it the most reliable model for identifying at-risk customers (minimizing False Negatives).
* **Random Forest:** Achieved the highest F1-Score (**63.29%**) among all tested algorithms.
* **Key Predictive Features:** `Tenure`, `TotalCharges`, `Contract_Two year`, and `MonthlyCharges` emerged as the most critical drivers of churn. Model explainability (SHAP) confirmed that long-term contractual commitments and high tenure significantly reduce churn probability.

## 🚀 Installation & Execution

### Prerequisites
Ensure you have Python installed along with the following data science dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn catboost shap
```
### Running the Code
You can execute the main script to run the full pipeline, from data preprocessing to model evaluation:
```bash
python predicting_customer_churn.py
```
Alternatively, run the provided Jupyter Notebook (`Predicting_Customer_Churn.ipynb`) step-by-step to view the interactive visualizations, performance metric tables, ROC curves, and SHAP explainability plots.
