import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.proj1_data import Proj1Data


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Export data from MongoDB to CSV
        """
        try:
            logging.info("Exporting data from MongoDB")

            my_data = Proj1Data()
            dataframe = my_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )

            # -----------------------------
            # FIX 1: Clean column names
            # -----------------------------
            dataframe.columns = dataframe.columns.str.strip()

            # -----------------------------
            # FIX 2: Remove MongoDB _id column
            # -----------------------------
            if "_id" in dataframe.columns:
                dataframe.drop(columns=["_id"], inplace=True)

            logging.info(f"Shape of dataframe after cleaning: {dataframe.shape}")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            return dataframe

        except Exception as e:
            raise MyException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Split dataframe into train and test sets
        """
        logging.info("Entered split_data_as_train_test method")

        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            # Clean column names again (safety)
            train_set.columns = train_set.columns.str.strip()
            test_set.columns = test_set.columns.str.strip()

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,
                header=True
            )

            logging.info("Train and test files exported successfully")

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Main ingestion method
        """
        logging.info("Entered initiate_data_ingestion method")

        try:
            dataframe = self.export_data_into_feature_store()
            logging.info("Data exported from MongoDB")

            self.split_data_as_train_test(dataframe)
            logging.info("Train-test split completed")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys) from e
