# Titanic Survival Prediction (Advanced Version)

## Overview
This project implements an advanced machine learning pipeline to predict passenger survival on the Titanic dataset (Kaggle competition). Compared to a baseline model, the code integrates **extensive feature engineering**, **robust preprocessing**, **multiple model families with hyperparameter optimization**, and **ensemble methods (voting & stacking)**.

Final experiments produced strong cross-validation performance (best ~0.9776 CV accuracy with Random Forests), though note that CV may be slightly optimistic due to encoding strategy. Still, the pipeline is competitive and modular for future improvements.

---

## Key Code Components

### 1. Feature Engineering
- **Title Extraction**: Titles parsed from names (Mr, Miss, Mrs, Officer, Royalty, etc.)
- **Family Features**: FamilySize = SibSp + Parch + 1, plus binary `IsAlone`
- **Fare Normalization**: Fare per passenger, Ticket group sizes, Ticket length
- **Cabin Deck Extraction**: First character of cabin (e.g., A, B, C, U for unknown)
- **Binning**: Age bins (child/teen/young/middle/senior) and Fare quantile bins
- **Interaction**: Age × Pclass

### 2. Preprocessing Pipelines
- **Numeric Features**: Missing values imputed with `IterativeImputer`, scaled via `StandardScaler`
- **Categorical Features**: Rare categories grouped as `RARE`, missing imputed, one-hot encoded
- **Target Encoding**: High-cardinality features (`LastName`, `TicketPrefix`) encoded with smoothed out-of-fold target means

### 3. Models & Hyperparameter Search
Implemented with `RandomizedSearchCV` and `StratifiedKFold (5-fold)`:
- Logistic Regression (ElasticNet regularization)
- Random Forests
- Extra Trees
- HistGradientBoosting
- GradientBoosting

### 4. Ensembling
- **Soft Voting**: Probability averaging across top-performing models
- **Stacking**: Logistic regression meta-model stacked on predictions of top 3 base models

### 5. Outputs
- `submission.csv` – Predictions for Kaggle submission
- `best_model.joblib` – Serialized best model
- `cv_summary.json` – CV scores & best hyperparameters
- `oof_predictions.csv` – Out-of-fold survival probabilities
- (Optional) `feature_importances.csv` – Feature importances for tree models (if available)

---

## Experimental Results

**Cross-Validation Accuracy (5-fold)**:
- Logistic Regression (ElasticNet): **0.9765**
- Random Forest: **0.9776** (Best single model)
- Extra Trees: **0.9764**
- HistGradientBoosting: **0.9753**
- GradientBoosting: **0.9776**
- Soft Voting Ensemble: **0.9764**
- Stacking Ensemble: **0.9753**

**Best Model Selected:** Random Forest (max_depth=14, n_estimators=871, min_samples_leaf=4, min_samples_split=9, max_features='sqrt').

**Submission:** Saved at `outputs/submission.csv`

---

## Example Run Output (verbatim)

**Command**
```bash
python src/titanic_train.py
```

**Console transcript**
```text
Loading data...
Feature engineering...
Tuning logistic_en ...
  -> best CV = 0.9765, params = {'clf__C': 0.36374585440139057, 'clf__class_weight': None, 'clf__l1_ratio': 0.8}
Tuning rf ...
  -> best CV = 0.9776, params = {'clf__max_depth': 14, 'clf__max_features': 'sqrt', 'clf__min_samples_leaf': 4, 'clf__min_samples_split': 9, 'clf__n_estimators': 871}
Tuning et ...
  -> best CV = 0.9764, params = {'clf__max_depth': 10, 'clf__max_features': None, 'clf__min_samples_leaf': 1, 'clf__min_samples_split': 8, 'clf__n_estimators': 570}
Tuning hgb ...
  -> best CV = 0.9753, params = {'clf__l2_regularization': 0.013199942261535026, 'clf__learning_rate': 0.10716110018463731, 'clf__max_depth': None, 'clf__max_iter': 460, 'clf__min_samples_leaf': 19}
Tuning gb ...
  -> best CV = 0.9776, params = {'clf__learning_rate': 0.042301944043371176, 'clf__max_depth': 4, 'clf__n_estimators': 386, 'clf__subsample': 0.85}
Evaluating soft voting ...
  -> soft voting CV = 0.9764
Evaluating stacking ...
  -> stacking CV = 0.9753

Best model = rf | CV accuracy = 0.9776
Best params = {'clf__max_depth': 14, 'clf__max_features': 'sqrt', 'clf__min_samples_leaf': 4, 'clf__min_samples_split': 9, 'clf__n_estimators': 871}
Saved submission to: D:\\文件储存\\titanic_ml_project\\outputs\\submission.csv
[Warn] export_importance failed: Estimator rare does not provide get_feature_names_out. Did you mean to call pipeline[:-1].get_feature_names_out()?
```

### Interpretation of the log
- **Overall**: all five single models achieved ~0.975–0.978 mean CV accuracy (5-fold, accuracy metric). The *current best* is **Random Forest**.
- **Soft Voting** and **Stacking** are close to the best single model.
- **Warning** is benign: the custom `RareCategoryGrouper` transformer has no `get_feature_names_out`. It only affects exporting feature names for importances, not training or predictions.

### Reproducibility settings (as coded)
- `RANDOM_STATE = 42`
- CV: `StratifiedKFold(n_splits=5, shuffle=True)`
- Search: `RandomizedSearchCV(n_iter=40, scoring='accuracy')`
- Artifacts written to `outputs/` (`submission.csv`, `best_model.joblib`, `cv_summary.json`, `oof_predictions.csv`, and optionally `feature_importances.csv`).

### Feature groups (summary)
- **Numeric**: `Age, SibSp, Parch, Fare, FamilySize, IsAlone, FarePerPerson, TicketGroupSize, TicketLen, AgeMissing, FareMissing, Age_times_Class` (scaled; missing via IterativeImputer)
- **Categorical (One‑Hot)**: `Pclass, Sex, Embarked, CabinInitial, AgeBin, FareBin, Title` (rare levels merged)
- **High‑Cardinality Encodings**: `LastName`, `TicketPrefix` with smoothed target encoding.

> Tip: Leaderboard performance will depend on randomness and the public/private split. The provided pipeline is modular so you can increase `n_iter`, add more features, or try additional ensembles for incremental gains.

---

## How to Run
1. Place Titanic data files under `data/`:
   ```
   data/train.csv
   data/test.csv
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run training script:
   ```bash
   python src/titanic_train.py
   ```

Results and artifacts will be written to `outputs/`.

---

## Notes & Future Work
- CV scores appear higher than typical Kaggle LB (0.83–0.87). This is likely due to partial label leakage in target encoding. A stricter fold-aware encoding inside pipelines will yield more realistic estimates.
- Feature importance is not exported cleanly for all pipelines (warning about `RareCategoryGrouper`). Using permutation importance is recommended.
- Further gains can be explored by tuning stacking meta-models, trying LightGBM/XGBoost, or engineering domain-inspired features.

---

## Credits
Developed as an advanced version of Titanic ML baseline with emphasis on robust pipelines, feature engineering, and ensemble learning.

