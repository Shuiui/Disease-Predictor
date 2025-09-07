from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load saved model
model = pickle.load(open("disease_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)[0]

    result = "Disease Detected" if prediction == 1 else "No Disease Detected"
    return render_template("index.html", prediction_text=f"Result: {result}")

if __name__ == "__main__":
    app.run(debug=True)
