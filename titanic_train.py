
import os, re, warnings
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from joblib import dump

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

def extract_title(name):
    m = re.search(r",\s*([^\.]+)\.", str(name))
    return m.group(1).strip() if m else "Unknown"

def add_features(df):
    df = df.copy()
    df["Title"] = df["Name"].apply(extract_title).replace({
        "Mlle":"Miss","Ms":"Miss","Mme":"Mrs",
        "Lady":"Royalty","Countess":"Royalty","Sir":"Royalty","Dona":"Royalty","Don":"Royalty","Jonkheer":"Royalty",
        "Capt":"Officer","Col":"Officer","Major":"Officer","Dr":"Officer","Rev":"Officer"
    })
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"]==1).astype(int)
    df["CabinInitial"] = df["Cabin"].fillna("U").astype(str).str[0]
    df["TicketLen"] = df["Ticket"].astype(str).str.len()
    if "Embarked" in df.columns and df["Embarked"].isna().any():
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    return df

def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    train = add_features(train)
    test = add_features(test)

    features = ["Pclass","Sex","Age","Fare","Embarked","SibSp","Parch","FamilySize","IsAlone","CabinInitial","Title","TicketLen"]
    target = "Survived"

    X = train[features]
    y = train[target]
    X_test = test[features]

    cat = ["Pclass","Sex","Embarked","CabinInitial","Title"]
    num = ["Age","Fare","SibSp","Parch","FamilySize","IsAlone","TicketLen"]

    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])

    pipe = Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))])

    grids = [
        {"clf":[LogisticRegression(max_iter=1000, solver='liblinear')],
         "clf__C":[0.5,1.0,2.0],
         "clf__class_weight":[None,"balanced"]},
        {"clf":[RandomForestClassifier(random_state=42)],
         "clf__n_estimators":[200,400],
         "clf__max_depth":[None,5,8],
         "clf__min_samples_split":[2,5]}
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, grids, scoring="accuracy", cv=cv, n_jobs=-1)
    gs.fit(X, y)

    print(f"Best CV: {gs.best_score_:.4f}")
    print(f"Best params: {gs.best_params_}")

    best = gs.best_estimator_
    best.fit(X, y)

    preds = best.predict(X_test).astype(int)
    sub = pd.DataFrame({ "PassengerId": test["PassengerId"], "Survived": preds })
    sub_path = os.path.join(OUT_DIR, "submission.csv")
    sub.to_csv(sub_path, index=False)
    print("Saved:", sub_path)

    dump(best, os.path.join(OUT_DIR, "best_model.joblib"))

if __name__ == "__main__":
    main()
