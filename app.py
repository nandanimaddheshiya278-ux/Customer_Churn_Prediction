from flask import Flask, render_template
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "customer_churn_model.pkl"
)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/test-model")
def test_model():
    try:
        model = joblib.load(MODEL_PATH)
        return "MODEL LOADED SUCCESSFULLY ✅"
    except Exception as e:
        return f"MODEL ERROR ❌: {str(e)}"
