# Suspicicous listing detector
This project builds a machine learning model for identifying potentially suspicious marketplace listings and prioritizing them for manual review.

## 1. How to run the project
### Requirements
Install required packages:

```bash
pip install -r requirements.txt
```

### Project structure
```text
data/
├── historical_data.csv
└── new_data.csv

functions.py
main.ipynb
README.md
```

### Run steps

1. Place `historical_data.csv` and `new_data.csv` in the `Data/` folder.
2. Open `main.ipynb`
3. Run all notebook cells from top to bottom.
4. The notebook will:
   - perform EDA
   - preprocess the data
   - train and compare models
   - tune the final model
   - evaluate performance
   - create predictions and a prioritization list for new data

**Note:**

In case of errors. Change **Step 3** for the following after opening `main.ipynb`

1. Clear All Outputs
2. Restart
3. Run All

## 2. Requirement card

**Stakeholder:** Elin (Trust & Safety / Marketplace moderation)

The stakeholder wants suspicious listings to be identified early while minimizing unnecessary impact on legitimate users. False positives should be limited because incorrectly removing normal listings can negatively affect user trust and experience.

Priority was therefore placed on balancing detection ability with explainability and supporting manual review instead of automatic removal.

Traceability and explainability was prioritized. Furthermore, a clear policy as to why something is flagged as suspicious that can be explained with ease.

## 3. Strategy

The project used a structured machine learning workflow beginning with exploratory data analysis and missing-value investigation.

A train/test split with stratification was applied to preserve class distribution and avoid leakage. Preprocessing was performed inside a pipeline using imputation, scaling, and one-hot encoding.

Three models were compared: Dummy baseline, Logistic Regression, and Random Forest. Since the dataset was imbalanced, PR-AUC was selected as the primary evaluation metric instead of accuracy.

Logistic Regression achieved the strongest overall performance and was selected as the final model. Hyperparameter tuning was performed using GridSearchCV.

A threshold-based decision strategy was used where listings with probability ≥ 0.60 were flagged for manual review and prioritized by risk score.

## 4. Responsibility distribution

This was originally a group assignment. However, it was done individually instead.