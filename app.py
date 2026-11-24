import json
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

bundle = joblib.load("model/model.joblib")
FEATURES = bundle["features"]
MEDIANS = bundle["medians"]
MODEL = bundle["models"][bundle["best"]]
BEST_NAME = bundle["best"]

with open("model/metrics.json") as handle:
    METRICS = json.load(handle)

DATA = pd.read_csv("bangladesh_diabetes_synthetic_2019_2025.csv", encoding="utf-8-sig")

RANGES = {
    "Age": (1, 120),
    "Glucose": (40, 400),
    "BloodPressure": (30, 200),
    "SkinThickness": (1, 100),
    "Insulin": (1, 900),
    "BMI": (10, 70),
    "DiabetesPedigreeFunction": (0.01, 3.5),
}
REQUIRED = ["Age", "Glucose"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def group_summary(frame, column):
    grouped = frame.groupby(column)["DiabetesOutcome"].agg(["count", "sum"]).reset_index()
    rows = []
    for _, item in grouped.iterrows():
        rows.append({"label": str(item[column]), "records": int(item["count"]), "positives": int(item["sum"])})
    return rows


@app.route("/")
def index():
    return render_template("index.html", rows=METRICS["rows"], districts=DATA["District"].nunique())


@app.route("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "The request body was not valid JSON. Send the form fields as JSON."}), 400
    row = {}
    if payload.get("Gender") == "Female":
        row["Female"] = 1.0
    else:
        row["Female"] = 0.0
    for name in RANGES:
        raw = payload.get(name)
        if raw is None or raw == "":
            if name in REQUIRED:
                return jsonify({"error": name + " is required. Enter a value and try again."}), 400
            row[name] = float("nan")
        else:
            try:
                value = float(raw)
            except ValueError:
                return jsonify({"error": name + " must be a number."}), 400
            low, high = RANGES[name]
            if value < low or value > high:
                return jsonify({"error": name + " must be between " + str(low) + " and " + str(high) + "."}), 400
            row[name] = value
    frame = pd.DataFrame([row], columns=FEATURES)
    probability = float(MODEL.predict_proba(frame)[0][1])
    contributions = []
    for name in FEATURES:
        altered = frame.copy()
        altered[name] = MEDIANS[name]
        shifted = float(MODEL.predict_proba(altered)[0][1])
        contributions.append({"feature": name, "effect": round(probability - shifted, 4)})
    contributions.sort(key=lambda item: abs(item["effect"]), reverse=True)
    if probability >= 0.6:
        category = "Higher"
    elif probability >= 0.3:
        category = "Moderate"
    else:
        category = "Lower"
    filled = []
    for name in RANGES:
        if row[name] != row[name]:
            filled.append(name)
    return jsonify({"score": round(probability, 3), "category": category, "contributions": contributions[:5], "filled_with_median": filled, "model": BEST_NAME})


@app.route("/api/explore")
def explore():
    frame = DATA
    district = request.args.get("district", "")
    year = request.args.get("year", "")
    month = request.args.get("month", "")
    if district != "":
        frame = frame[frame["District"] == district]
    if year != "":
        frame = frame[frame["Year"] == int(year)]
    if month != "":
        frame = frame[frame["Month"] == month]
    positives = int(frame["DiabetesOutcome"].sum())
    total = int(len(frame))
    means = {}
    for column in ["Age", "Glucose", "BMI"]:
        if total > 0:
            means[column] = round(float(frame[column].mean()), 1)
        else:
            means[column] = None
    by_month = group_summary(frame, "Month")
    by_month.sort(key=lambda item: MONTHS.index(item["label"]))
    by_district = group_summary(frame, "District")
    by_district.sort(key=lambda item: item["positives"], reverse=True)
    return jsonify({
        "records": total,
        "positives": positives,
        "means": means,
        "by_year": group_summary(frame, "Year"),
        "by_month": by_month,
        "by_district": by_district[:8],
        "districts": sorted(DATA["District"].unique().tolist()),
        "years": sorted(int(item) for item in DATA["Year"].unique()),
        "months": MONTHS,
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
