from flask import Flask, request, render_template, redirect, url_for, session
from flask_cors import CORS
import mysql.connector
import pandas as pd
import joblib
import os
import traceback

app = Flask(__name__)
app.secret_key = "secret123"
CORS(app)

MODEL_PATH = "churn_best_model.pkl"

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Tony@2404",
    "database": "churn_db",
}

model = None

# ---------------- LOAD MODEL ----------------
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Run train_model.py first")
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded")

# ---------------- DB ---------------- 
def get_db_conn():
    return mysql.connector.connect(**DB_CONFIG)

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "admin123":
            session["user"] = "admin"
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid Credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ---------------- HOME ----------------
@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    try:
        data = request.form.to_dict()
        df = pd.DataFrame([data])
        model_input = df.drop(columns=["customerID"])

        model_input["SeniorCitizen"] = model_input["SeniorCitizen"].astype(int)
        model_input["tenure"] = model_input["tenure"].astype(int)
        model_input["MonthlyCharges"] = model_input["MonthlyCharges"].astype(float)
        model_input["TotalCharges"] = pd.to_numeric(
            model_input["TotalCharges"], errors="coerce"
        ).fillna(0.0)

        prob = float(model.predict_proba(model_input)[0][1])
        pred = int(model.predict(model_input)[0])

        # Risk Levels
        if prob >= 0.7:
            risk = "High Risk"
        elif prob >= 0.4:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        # Save to DB
        conn = get_db_conn()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO churn_predictions
            (customerID, gender, senior_citizen, partner, dependents,
            tenure, phone_service, multiple_lines, internet_service,
            online_security, online_backup, device_protection, tech_support,
            streaming_tv, streaming_movies, contract_type, paperless_billing,
            payment_method, monthly_charges, total_charges, churn_pred, churn_prob)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            data["customerID"],
            data["gender"],
            int(data["SeniorCitizen"]),
            data["Partner"],
            data["Dependents"],
            int(data["tenure"]),
            data["PhoneService"],
            data["MultipleLines"],
            data["InternetService"],
            data["OnlineSecurity"],
            data["OnlineBackup"],
            data["DeviceProtection"],
            data["TechSupport"],
            data["StreamingTV"],
            data["StreamingMovies"],
            data["Contract"],
            data["PaperlessBilling"],
            data["PaymentMethod"],
            float(data["MonthlyCharges"]),
            float(data["TotalCharges"]),
            pred,
            prob
        ))

        conn.commit()
        cur.close()
        conn.close()

        return render_template(
            "index.html",
            result=risk,
            probability=round(prob * 100, 2)
        )

    except Exception as e:
        traceback.print_exc()
        return render_template("index.html", result=f"Error: {e}")

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db_conn()
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT * FROM churn_predictions ORDER BY id DESC LIMIT 10")
    data = cur.fetchall()

    cur.close()
    conn.close()

    return render_template("dashboard.html", data=data)

# ---------------- RUN ----------------
if __name__ == "__main__":
    load_model()
    app.run(debug=True)