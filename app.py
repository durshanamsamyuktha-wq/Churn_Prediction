from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and scaler
with open("churn.pkl", "rb") as f:
    model = pickle.load(f)

with open("standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Collect form inputs
            features = [
                float(request.form["Gender"]),
                float(request.form["SeniorCitizen"]),
                float(request.form["Partner"]),
                float(request.form["Dependents"]),
                float(request.form["Tenure"]),
                float(request.form["Phone_Service"]),
                float(request.form["MultipleLines"]),
                float(request.form["InternetService"]),
                float(request.form["OnlineSecurity"]),
                float(request.form["OnlineBackup"]),
                float(request.form["DeviceProtection"]),
                float(request.form["StreamingTV"]),
                float(request.form["Streaming_Movies"]),
                float(request.form["Contract"]),
                float(request.form["Paperless_Billing"]),
                float(request.form["Payment_Method"]),
                float(request.form["MonthlyCharges"]),
                float(request.form["TotalCharges"])

            ]

            # Convert to numpy array
            features_array = np.array([features])

            # Scale features
            features_scaled = scaler.transform(features_array)

            # Make prediction
            prediction = model.predict(features_scaled)[0]
            if prediction == 0:
                return render_template("index.html", prediction="Customer will leave")
            else:
                return render_template("index.html", prediction="Customer will stay")

        except Exception as e:
            # If an error occurs during POST, render the template with the error message
            prediction = f"Error: {str(e)}"
            return render_template("index.html", prediction=prediction) # <--- Added return here for error handling

    # This handles the initial GET request (when the page is loaded)
    # and any scenario where the POST block didn't execute or encountered an error.
    return render_template("index.html", prediction=prediction) # <--- ADDED REQUIRED RETURN STATEMENT

if __name__ == "__main__":
    app.run(debug=True)