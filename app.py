from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load("loan_best_model-2.pkl")   # make sure file exists

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form   # ✅ use form (not JSON)

    features = np.array([[
        int(data['no_of_dependencys']),
        int(data['education']),
        int(data['self_employed']),
        float(data['income_annum']),
        float(data['loan_amount']),
        int(data['loan_term']),
        int(data['cibil_score']),
        float(data['residential_assets_value']),
        float(data['commercial_assets_value']),
        float(data['luxury_assets_value']),
        float(data['bank_asset_value'])
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    confidence = round(max(probability) * 100, 2)

    # EMI
    loan_amount = float(data['loan_amount'])
    term = int(data['loan_term']) * 12
    rate = 0.08 / 12

    emi = int((loan_amount * rate * (1 + rate)**term) / ((1 + rate)**term - 1))

    # Score
    score = int((int(data['cibil_score']) / 900) * 100)

    reasons = []
    suggestions = []

    if int(data['cibil_score']) < 650:
        reasons.append("Low CIBIL Score")
        suggestions.append("Improve credit score")

    if float(data['income_annum']) < 300000:
        reasons.append("Low Income")
        suggestions.append("Increase income")

    if prediction == 1:
        result = "Approved"
    else:
        result = "Rejected"

    return render_template("result.html",
                           prediction=result,
                           confidence=confidence,
                           score=score,
                           emi=emi,
                           reasons=reasons,
                           suggestions=suggestions)

if __name__ == "__main__":
    app.run(debug=True)