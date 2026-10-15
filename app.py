import os

import pandas as pd
from flask import Flask, request, render_template

from src.fraud_detection.pipelines.prediction_pipeline import PredictPipeline

application = Flask(__name__)
app = application
MODEL_PATH = os.path.join("Artifacts", "fraud_pipeline.pkl")
SAMPLE_PATH = os.path.join("notebook", "data", "creditcard_sample.csv")
MAX_ROWS_SHOWN = 100


def build_result(df):
    labels, probs = PredictPipeline().predict(df)
    table = pd.DataFrame({
        "Time": df["Time"].values,
        "Amount": df["Amount"].values,
        "Fraud probability": probs.astype(float).round(4),
        "Prediction": ["FRAUD" if l == 1 else "Not fraud" for l in labels],
    })
    if "Class" in df.columns:
        table["Actual"] = ["Fraud" if c == 1 else "Not fraud" for c in df["Class"]]
    return {
        "total": len(table),
        "flagged": int(labels.sum()),
        "shown": min(len(table), MAX_ROWS_SHOWN),
        "columns": list(table.columns),
        "rows": table.head(MAX_ROWS_SHOWN).values.tolist(),
    }


## Route for the home page
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    model_ready = os.path.exists(MODEL_PATH)

    if request.method == "GET":
        return render_template("home.html", model_ready=model_ready)

    try:
        mode = request.form.get("mode")

        if mode == "csv":
            file = request.files.get("file")
            if not file or not file.filename.lower().endswith(".csv"):
                raise ValueError("Please upload a .csv file.")
            df = pd.read_csv(file)
        else:  # "fraud" or "normal" sample button
            data = pd.read_csv(SAMPLE_PATH)
            subset = data[data["Class"] == (1 if mode == "fraud" else 0)]
            df = subset.sample(1)

        return render_template("home.html", result=build_result(df), model_ready=model_ready)

    except Exception as e:
        return render_template("home.html", error=str(e), model_ready=model_ready)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)