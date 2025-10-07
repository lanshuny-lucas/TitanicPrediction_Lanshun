# Titanic — Machine Learning from Disaster (Enhanced Version)

> A **Kaggle-ready** solution for the Titanic competition.  
> Features: rich feature engineering, multiple models with hyperparameter tuning, and a stacking ensemble.  
> Running `python src/titanic_train.py` will output a Kaggle submission file in `outputs/submission.csv`.

---

## 1. Project Structure
```
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv   # optional
├── outputs/                    # contains submission.csv / best_model.joblib / cv_summary.json
├── src/
│   └── titanic_train.py        # main training script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 2. Quick Start
```bash
# (1) optional: create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# (2) install dependencies
pip install -r requirements.txt

# (3) run training and prediction
python src/titanic_train.py

# (4) find the Kaggle submission file at:
outputs/submission.csv
```

---

## 3. Workflow Overview

1. **Feature Engineering**
   - Extract `Title` from names, normalize rare titles.
   - Create `FamilySize`, `IsAlone`, `FarePerPerson`.
   - Add `TicketGroupSize`, `CabinInitial`, `TicketLen`.
   - Add missing indicators (`AgeMissing`, `FareMissing`).
   - Create interaction feature `Age_times_Class`.
   - Bin `Age` and `Fare` into discrete intervals.

2. **Preprocessing**
   - Numerical: median imputation + standard scaling.
   - Categorical: mode imputation + one-hot encoding.
   - Combined via `ColumnTransformer`.

3. **Models & Tuning**
   - Logistic Regression (C, class_weight).
   - Random Forest (n_estimators, depth, splits).
   - HistGradientBoosting (learning rate, depth, iterations, regularization).
   - 5-fold Stratified CV with `GridSearchCV`.

4. **Stacking Ensemble**
   - Combine the best LR, RF, and HGB models.
   - Logistic Regression as meta-learner.
   - Evaluate with 5-fold CV.
   - Select the single best performer (CV accuracy).

---

## 4. Results (Example Run)

### Cross-validation scores
| Model                | Best CV Accuracy | Best Parameters |
|-----------------------|------------------|-----------------|
| Logistic Regression   | 0.8034           | {'C': 1.0, 'class_weight': None} |
| Random Forest         | 0.8426           | {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 2, 'min_samples_leaf': 2} |
| HistGradientBoosting  | 0.8539           | {'learning_rate': 0.06, 'max_depth': 5, 'max_iter': 400, 'l2_regularization': 0.001} |
| **Stacking Ensemble** | **0.8615**       | lr + rf + hgb → logistic |

> ✅ Final chosen model: **Stacking Ensemble** with CV accuracy ≈ **0.8615**

---

### Sample of generated `submission.csv`
```
PassengerId,Survived
892,0
893,1
894,0
895,0
896,1
897,0
898,0
899,1
900,0
```

The complete file is saved under:
```
outputs/submission.csv
```

---

## 5. Design Choices & Extensions
- **Balanced between speed and performance**: parameter grids are moderate for fast runs (<10 mins).  
- **Robustness**: all preprocessing encapsulated in pipelines; missing indicators improve stability.  
- **Interpretability**: coefficients (LogReg) and feature importances (RF/HGB) can be exported for analysis.  
- **Future improvements**: try LightGBM/XGBoost/CatBoost, add target encoding, expand hyperparameter search.

---

## 6. License & Data
- Data comes from [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic).  
- You may release this repository under MIT or another open-source license of your choice.
