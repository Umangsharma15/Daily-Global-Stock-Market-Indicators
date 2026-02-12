import sys
import os
import numpy as np
import pandas as pd

from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
)

from src.utils.main_utils import save_numpy_array_data


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Data Transformation Started")

        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info("Train and test data loaded")

            train_df = train_df.sort_values(["Index_Name", "Date"])
            test_df = test_df.sort_values(["Index_Name", "Date"])

            # Next day close
            train_df["Next_Close"] = train_df.groupby("Index_Name")["Close"].shift(-1)
            test_df["Next_Close"] = test_df.groupby("Index_Name")["Close"].shift(-1)

            # Target = return
            train_df["Return"] = (train_df["Next_Close"] - train_df["Close"]) / train_df["Close"]
            test_df["Return"] = (test_df["Next_Close"] - test_df["Close"]) / test_df["Close"]

            # Lag features
            for lag in [1, 2, 3]:
                train_df[f"Prev_Close_{lag}"] = train_df.groupby("Index_Name")["Close"].shift(lag)
                test_df[f"Prev_Close_{lag}"] = test_df.groupby("Index_Name")["Close"].shift(lag)

            # Moving averages
            train_df["MA_5"] = (
                train_df.groupby("Index_Name")["Close"]
                .rolling(5)
                .mean()
                .reset_index(0, drop=True)
            )
            train_df["MA_10"] = (
                train_df.groupby("Index_Name")["Close"]
                .rolling(10)
                .mean()
                .reset_index(0, drop=True)
            )

            test_df["MA_5"] = (
                test_df.groupby("Index_Name")["Close"]
                .rolling(5)
                .mean()
                .reset_index(0, drop=True)
            )
            test_df["MA_10"] = (
                test_df.groupby("Index_Name")["Close"]
                .rolling(10)
                .mean()
                .reset_index(0, drop=True)
            )

            # Encode index
            train_df["Index_Code"] = train_df["Index_Name"].astype("category").cat.codes
            test_df["Index_Code"] = test_df["Index_Name"].astype("category").cat.codes

            train_df = train_df.dropna()
            test_df = test_df.dropna()

            feature_cols = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Daily_Change_Percent",
                "Index_Code",
                "Prev_Close_1",
                "Prev_Close_2",
                "Prev_Close_3",
                "MA_5",
                "MA_10",
            ]

            target_col = "Return"

            x_train = train_df[feature_cols]
            y_train = train_df[target_col]

            x_test = test_df[feature_cols]
            y_test = test_df[target_col]

            train_arr = np.c_[x_train.values, y_train.values]
            test_arr = np.c_[x_test.values, y_test.values]

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr,
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr,
            )

            logging.info("Data transformation completed")

            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
            )

        except Exception as e:
            raise MyException(e, sys)
