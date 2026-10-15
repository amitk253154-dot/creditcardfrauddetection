import sys

from src.fraud_detection.exception import CustomException
from src.fraud_detection.logger import logging
from src.fraud_detection.components.data_ingestion import DataIngestion
from src.fraud_detection.components.model_tranier import ModelTrainer


class TrainingPipeline:
    def __init__(self, n_trials=30):
        self.n_trials = n_trials

    def run(self):
        try:
            logging.info("Training pipeline started")

            train_path, test_path = DataIngestion().initiate_data_ingestion()
            result = ModelTrainer(n_trials=self.n_trials).initiate_model_trainer(
                train_path, test_path
            )

            logging.info("Training pipeline completed")
            return result

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    print(TrainingPipeline(n_trials=30).run())