import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer

from src.fraud_detection.exception import  CustomException
from src.fraud_detection.logger import logging
from src.fraud_detection.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("Artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """Responsible for data transformation."""
        try:
            amount_pipeline = Pipeline(
                steps=[
                    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                    ("scaler", RobustScaler()),
                ]
            )

            logging.info("Amount: log1p + RobustScaler, Time: Rob" \
            "ustScaler, V1-V28: unchanged")

            preprocessor = ColumnTransformer(
                [
                    ("amount_pipeline", amount_pipeline, ["Amount"]),
                    ("time_scaler", RobustScaler(), ["Time"]),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            )
            return preprocessor

        except Exception as e:
            raise  CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Class"

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )
            logging.info("Saved preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise  CustomException(e, sys)
if __name__ == "__main__":
    from src.fraud_detection.components.data_ingestion import DataIngestion

    train_path, test_path = DataIngestion().initiate_data_ingestion()
    train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(train_path, test_path)
    print(train_arr.shape, test_arr.shape)