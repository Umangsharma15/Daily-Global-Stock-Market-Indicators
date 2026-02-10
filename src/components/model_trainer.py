import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training...")

            # Load transformed arrays
            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )

            # Split features and target
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Create model
            model = RandomForestRegressor(
                n_estimators=self.model_trainer_config._n_estimators,
                max_depth=self.model_trainer_config._max_depth,
                min_samples_split=self.model_trainer_config._min_samples_split,
                min_samples_leaf=self.model_trainer_config._min_samples_leaf,
                random_state=self.model_trainer_config._random_state
            )

            logging.info("Training RandomForestRegressor...")
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            logging.info(f"Model R2 score: {r2}")
            logging.info(f"Model MAE: {mae}")

            # Check performance
            if r2 < self.model_trainer_config.expected_accuracy:
                raise Exception("Model performance is below expected threshold")

            # Save model
            save_object(
                self.model_trainer_config.trained_model_file_path,
                model
            )

            logging.info("Model saved successfully")

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=None
            )

        except Exception as e:
            raise MyException(e, sys) from e
