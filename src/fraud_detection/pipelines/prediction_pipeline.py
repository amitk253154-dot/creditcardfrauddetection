import os
import sys

import pandas as pd

from src.fraud_detection.exception import CustomException
from src.fraud_detection.logger import logging
from src.fraud_detection.utils import load_object

# Column order used in training
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class PredictPipeline:
    def __init__(self, model_path=os.path.join("Artifacts", "fraud_pipeline.pkl")):
        self.model_path = model_path

    def predict(self, features: pd.DataFrame):
        """Returns (labels, fraud_probabilities) for one or many transactions."""
        try:
            missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            artifact = load_object(file_path=self.model_path)
            pipeline = artifact["pipeline"]      # preprocessing + XGBoost
            threshold = artifact["threshold"]    # chosen during training

            probability = pipeline.predict_proba(features[FEATURE_COLUMNS])[:, 1]
            labels = (probability >= threshold).astype(int)

            logging.info(f"Predicted {len(labels)} transaction(s)")
            return labels, probability

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """Builds the model input from a dict of values (for the web app)."""

    def __init__(self, values: dict):
        self.values = values

    def get_data_as_dataframe(self):
        try:
            data = {col: [float(self.values[col])] for col in FEATURE_COLUMNS}
            return pd.DataFrame(data)
        except KeyError as e:
            raise CustomException(f"Missing value for {e}", sys)
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Quick test with real rows from the sample dataset
    sample = pd.read_csv(os.path.join("notebook", "data", "creditcard_sample.csv"))
    rows = pd.concat([sample[sample.Class == 1].head(3), sample[sample.Class == 0].head(3)])

    labels, probs = PredictPipeline().predict(rows.drop(columns=["Class"]))
    for actual, pred, p in zip(rows["Class"], labels, probs):
        print(f"Actual: {actual} | Predicted: {pred} | Fraud probability: {p:.4f}")