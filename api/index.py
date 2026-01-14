import os
import pickle
import joblib
from datetime import datetime
from flask import Flask, request, render_template_string
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "visa_processing_model.pkl")
PREPROCESS_PATH = os.path.join(BASE_DIR, "preprocessing_info.pkl")

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'), static_folder=os.path.join(BASE_DIR, 'static'))


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESS_PATH):
        raise FileNotFoundError("Model or preprocessing info not found.")
    model = joblib.load(MODEL_PATH)
    with open(PREPROCESS_PATH, "rb") as f:
        prep = pickle.load(f)
    return model, prep


def build_feature_vector(prep, country, visa_type, application_date_str, processing_office=None):
    feature_names = prep["feature_names"]
    row = {c: 0 for c in feature_names}

    try:
        app_date = pd.to_datetime(application_date_str)
    except Exception:
        app_date = pd.to_datetime("today")

    application_month = int(app_date.month)
    season = "Peak" if application_month in [1, 2, 12] else "Off-Peak"

    if "application_month" in row:
        row["application_month"] = application_month
    if "country_avg" in row:
        row["country_avg"] = float(prep.get("country_avg", {}).get(country, prep.get("mean_processing_days", 0)))
    if "visa_avg" in row:
        row["visa_avg"] = float(prep.get("visa_avg", {}).get(visa_type, prep.get("mean_processing_days", 0)))

    season_col = f"season_{season}"
    if season_col in row:
        row[season_col] = 1

    office_map = prep.get("office_map", {})
    mapped_office = processing_office or office_map.get(country, "Unknown")
    office_col = f"processing_office_{mapped_office}"
    if office_col in row:
        row[office_col] = 1

    df = pd.DataFrame([row], columns=feature_names).fillna(0)
    return df


def predict(model, prep, country, visa_type, application_date_str, processing_office=None):
    X = build_feature_vector(prep, country, visa_type, application_date_str, processing_office)
    pred = model.predict(X)[0]
    pred = max(0.0, float(pred))
    return round(pred, 1)


@app.route("/", methods=["GET"])
def index():
    try:
        html_path = os.path.join(BASE_DIR, 'templates', 'index.html')
        with open(html_path, 'r', encoding='utf-8') as f:
            return render_template_string(f.read())
    except Exception as e:
        return f"<h1>Error loading page: {e}</h1>", 500


@app.route("/predict", methods=["POST"])
def predict_route():
    country = request.form.get("country", "Unknown")
    visa_type = request.form.get("visa_type", "Unknown")
    application_date = request.form.get("application_date", datetime.today().strftime("%Y-%m-%d"))
    processing_office = request.form.get("processing_office", None)
    try:
        model, prep = load_artifacts()
        days = predict(model, prep, country, visa_type, application_date, processing_office)
    except Exception as e:
        return f"Error during prediction: {e}", 500
    return render_template_string(f"<html><body style='font-family:Inter, Poppins, sans-serif;background:#07104a;color:#eaf0ff;display:flex;align-items:center;justify-content:center;height:100vh'><div style='background:rgba(255,255,255,0.02);padding:24px;border-radius:12px;box-shadow:0 20px 40px rgba(0,0,0,0.6)'><h2>Estimated processing days: {days}</h2><p><a href='/'>Back</a></p></div></body></html>")
