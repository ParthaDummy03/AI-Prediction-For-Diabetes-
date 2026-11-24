import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

CSV = "bangladesh_diabetes_synthetic_2019_2025.csv"
CLINICAL = ["Age", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction"]
FEATURES = ["Female"] + CLINICAL
MUST_BE_POSITIVE = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df = pd.read_csv(CSV, encoding="utf-8-sig")
duplicates = int(df.duplicated().sum())
invalid = {}
for column in MUST_BE_POSITIVE:
    mask = df[column] <= 0
    invalid[column] = int(mask.sum())
    df.loc[mask, column] = np.nan
df["Female"] = (df["Gender"] == "Female").astype(int)

missing = []
for column in CLINICAL:
    count = int(df[column].isna().sum())
    missing.append({"feature": column, "count": count, "percent": round(100 * count / len(df), 1)})

X = df[FEATURES]
y = df["DiabetesOutcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def build(imputer, model):
    return Pipeline([("impute", clone(imputer)), ("scale", StandardScaler()), ("model", model)])

imputers = {
    "mean": SimpleImputer(strategy="mean"),
    "median": SimpleImputer(strategy="median"),
    "knn": KNNImputer(n_neighbors=5),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
imputer_cv = {}
for name, imputer in imputers.items():
    pipe = build(imputer, LogisticRegression(max_iter=1000, class_weight="balanced"))
    score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    imputer_cv[name] = round(float(score), 4)
chosen = max(imputer_cv, key=imputer_cv.get)

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=42),
}
fitted = {}
results = {}
for name, model in candidates.items():
    pipe = build(imputers[chosen], model)
    pipe.fit(X_train, y_train)
    probability = pipe.predict_proba(X_test)[:, 1]
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    fitted[name] = pipe
    results[name] = {
        "accuracy": round(accuracy_score(y_test, predicted), 4),
        "precision": round(precision_score(y_test, predicted, zero_division=0), 4),
        "recall": round(recall_score(y_test, predicted), 4),
        "f1": round(f1_score(y_test, predicted), 4),
        "roc_auc": round(roc_auc_score(y_test, probability), 4),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

best = max(results, key=lambda key: results[key]["roc_auc"])
forest = fitted["Random Forest"].named_steps["model"]
importance = []
for feature, value in zip(FEATURES, forest.feature_importances_):
    importance.append({"feature": feature, "value": round(float(value), 4)})
importance.sort(key=lambda item: item["value"], reverse=True)

positives = int(y.sum())
report = {
    "rows": int(len(df)),
    "duplicates": duplicates,
    "invalid_set_to_missing": invalid,
    "class_balance": {"negative": int(len(df) - positives), "positive": positives, "positive_percent": round(100 * positives / len(df), 2)},
    "missing": missing,
    "imputer_cv_roc_auc": imputer_cv,
    "chosen_imputer": chosen,
    "models": results,
    "best_model": best,
    "importance": importance,
    "test_size": int(len(y_test)),
}
medians = {}
for feature in FEATURES:
    medians[feature] = float(X_train[feature].median())
joblib.dump({"models": fitted, "best": best, "features": FEATURES, "medians": medians}, "model/model.joblib")
with open("model/metrics.json", "w") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps({"imputer": imputer_cv, "chosen": chosen, "best": best, "models": results, "class": report["class_balance"], "invalid": invalid}))
