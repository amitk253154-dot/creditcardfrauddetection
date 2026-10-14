import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import optuna
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.fraud_detection.exception import CustomException
from src.fraud_detection.logger import logging
from src.fraud_detection.utils import save_object
from src.fraud_detection.components.data_transformation import DataTransformation

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("Artifacts", "fraud_pipeline.pkl")


def make_pipeline(clf):
    """Preprocessing + classifier as ONE estimator (fresh preprocessor each call)."""
    prep = DataTransformation().get_data_transformer_object()
    prep.set_output(transform="pandas")
    return Pipeline([("prep", prep), ("clf", clf)])


def evaluate_model(model, X_test, y_test, thresh=0.5):
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= thresh).astype(int)
    return {
        "ap": average_precision_score(y_test, y_scores),
        "roc": roc_auc_score(y_test, y_scores),
        "cm": confusion_matrix(y_test, y_pred),
        "scores": y_scores,
        "y_pred": y_pred,
    }


class ModelTrainer:
    def __init__(self, n_trials=30):
        self.model_trainer_config = ModelTrainerConfig()
        self.n_trials = n_trials

    def initiate_model_trainer(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            X_train = train_df.drop(columns=["Class"])
            y_train = train_df["Class"]
            X_test = test_df.drop(columns=["Class"])
            y_test = test_df["Class"]
            logging.info("Read train and test data")

            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # ---------- Baseline models (same settings as the notebook) ----------
            rf = make_pipeline(RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1))
            lgbm = make_pipeline(LGBMClassifier(
                n_estimators=500, learning_rate=0.05, class_weight="balanced",
                random_state=42, n_jobs=-1, verbose=-1))
            xgb = make_pipeline(XGBClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
                scale_pos_weight=scale_pos_weight, eval_metric="aucpr"))

            baseline_report = {}
            for name, model in {"Random Forest": rf, "LightGBM": lgbm, "XGBoost": xgb}.items():
                model.fit(X_train, y_train)
                res = evaluate_model(model, X_test, y_test)
                baseline_report[name] = {"ap": res["ap"], "roc": res["roc"]}
                logging.info(f"Baseline {name}: {baseline_report[name]}")

            # ---------- Optuna: CV on average precision, whole pipeline ----------
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 300, 800),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
                    "random_state": 42,
                    "n_jobs": -1,
                    "scale_pos_weight": scale_pos_weight,
                    "eval_metric": "aucpr",
                }
                model = make_pipeline(XGBClassifier(**params))
                scores = cross_val_score(model, X_train, y_train, cv=skf,
                                         scoring="average_precision", n_jobs=-1)
                return scores.mean()

            logging.info(f"Optuna tuning started ({self.n_trials} trials)")
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_trials)
            logging.info(f"Best CV AP: {study.best_value:.4f}, params: {study.best_params}")

            best_params = study.best_params
            best_params.update({
                "random_state": 42,
                "n_jobs": -1,
                "scale_pos_weight": scale_pos_weight,
                "eval_metric": "aucpr",
            })
            xgb_tuned = make_pipeline(XGBClassifier(**best_params))

            # ---------- Threshold from CV predictions on TRAIN only ----------
            cv_proba = cross_val_predict(xgb_tuned, X_train, y_train, cv=skf,
                                         method="predict_proba", n_jobs=-1)[:, 1]
            precisions, recalls, thresholds = precision_recall_curve(y_train, cv_proba)
            f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)
            best_idx = np.nanargmax(f1_scores[:-1])
            best_threshold = float(thresholds[best_idx])
            logging.info(f"Chosen threshold: {best_threshold:.4f}")

            # ---------- Final fit and test evaluation ----------
            start = time.time()
            xgb_tuned.fit(X_train, y_train)
            train_time = time.time() - start

            start = time.time()
            res_tuned = evaluate_model(xgb_tuned, X_test, y_test, thresh=best_threshold)
            pred_time = time.time() - start
            y_pred = res_tuned["y_pred"]

            metrics = {
                "threshold": best_threshold,
                "pr_auc": res_tuned["ap"],
                "roc_auc": res_tuned["roc"],
                "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
                "recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
                "f1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
                "train_time_sec": round(train_time, 2),
                "pred_time_sec": round(pred_time, 4),
            }
            logging.info(f"Final test metrics: {metrics}")
            print(classification_report(y_test, y_pred, digits=3))

            # Save pipeline + threshold (same idea as the notebook's joblib export)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj={"pipeline": xgb_tuned, "threshold": best_threshold},
            )
            logging.info("Saved pipeline and threshold")

            return {"baselines": baseline_report, "tuned": metrics}

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.fraud_detection.components.data_ingestion import DataIngestion

    train_path, test_path = DataIngestion().initiate_data_ingestion()
    result = ModelTrainer(n_trials=30).initiate_model_trainer(train_path, test_path)
    print(result)