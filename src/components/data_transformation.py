import sys
import pandas as pd
import numpy as np

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact
)
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_numpy_array_data


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Stock dataset transformation:
        - Create Next_Close target
        - Select features
        - Save arrays
        """
        try:
            logging.info("Data Transformation Started")

            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            logging.info("Train and test data loaded")

            # Convert date
            train_df["Date"] = pd.to_datetime(train_df["Date"])
            test_df["Date"] = pd.to_datetime(test_df["Date"])

            # Sort
            train_df = train_df.sort_values(by=["Index_Name", "Date"])
            test_df = test_df.sort_values(by=["Index_Name", "Date"])

            # Create target
            train_df["Next_Close"] = train_df.groupby("Index_Name")["Close"].shift(-1)
            test_df["Next_Close"] = test_df.groupby("Index_Name")["Close"].shift(-1)

            # Drop last rows with no target
            train_df = train_df.dropna()
            test_df = test_df.dropna()

            # Features
            feature_cols = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Daily_Change_Percent"
            ]

            target_col = "Next_Close"

            X_train = train_df[feature_cols]
            y_train = train_df[target_col]

            X_test = test_df[feature_cols]
            y_test = test_df[target_col]

            # Convert to arrays
            train_arr = np.c_[X_train.values, y_train.values]
            test_arr = np.c_[X_test.values, y_test.values]

            # Save arrays
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr
            )

            logging.info("Data transformation completed")

            return DataTransformationArtifact(
                transformed_object_file_path="",
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e
