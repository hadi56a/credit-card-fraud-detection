# Credit Card Fraud Detection

Detecting fraudulent credit card transactions on a very imbalanced dataset, and estimating how much money each model would save a bank.

Dataset: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), 284,807 transactions, only 0.17% frauds.

## What I did
- Data exploration: class imbalance, amounts, time of day, correlations, boxplots
- Removed duplicates, stratified train/test split, scaled `Time` and `Amount`
- Showed why accuracy is misleading (always predicting "Non-Fraud" gives 99.8%)
- Compared 5 models: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM
- Evaluated them with precision, recall, F1, PR-AUC and the cost for the bank in €

Cost assumption: a missed fraud costs the transaction amount, a false alert costs 5 €.

## Results (test set, threshold 0.5)

| Model | Precision | Recall | F1 | PR-AUC | Bank losses |
|---|---|---|---|---|---|
| No model | - | 0 | - | - | 14,766 € |
| Logistic Regression | 0.848 | 0.589 | 0.696 | 0.692 | 8,806 € |
| Decision Tree | 0.883 | 0.716 | 0.791 | 0.629 | 5,676 € |
| **Random Forest** | **0.972** | **0.726** | **0.831** | 0.788 | **4,335 €** |
| XGBoost | 0.944 | 0.716 | 0.814 | **0.809** | 5,673 € |
| LightGBM | 0.315 | 0.674 | 0.430 | 0.498 | 7,309 € |

Random Forest catches 69 of the 95 frauds with 97% precision and avoids **70.6% of the losses**.

## Next steps
- tune the hyperparameters
- try class weights or SMOTE
- choose a better decision threshold than 0.5

## Run it
```bash
pip install -r requirements.txt
```
Download `creditcard.csv` from Kaggle, put it in a `data/` folder, then open `credit-card-fraud-detection.ipynb`.
