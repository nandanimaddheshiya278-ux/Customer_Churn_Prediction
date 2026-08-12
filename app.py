from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Model file ka path
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "customer_churn_model.pkl"
)

# Model load
model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Form se values lena
    tenure = float(request.form["tenure"])
    monthly = float(request.form["MonthlyCharges"])
    total = float(request.form["TotalCharges"])

    contract = request.form["Contract"]
    payment = request.form["PaymentMethod"]
    internet = request.form["InternetService"]
    security = request.form["OnlineSecurity"]
    support = request.form["TechSupport"]

    # DataFrame banana
    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total],
        "Contract": [contract],
        "PaymentMethod": [payment],
        "InternetService": [internet],
        "OnlineSecurity": [security],
        "TechSupport": [support]
    })

    # Prediction
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "⚠️ Customer is likely to Churn"
    else:
        result = "✅ Customer is likely to Stay"

    return render_template(
        "index.html",
        prediction_text=result
    )



    
        
