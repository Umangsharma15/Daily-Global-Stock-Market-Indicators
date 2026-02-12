from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import r2_score
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import Proj1Estimator
from dataclasses import dataclass


TARGET_COLUMN = "Return"


@dataclass
class EvaluateModelResponse:
    trained_model_score: float
    best_model_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Get production model from S3 if exists
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path

            proj1_estimator = Proj1Estimator(
                bucket_name=bucket_name,
                model_path=model_path
            )

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator

            return None

        except Exception as e:
            raise MyException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Evaluate regression model using R2 score
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Sort by index and date
            test_df = test_df.sort_values(by=["Index_Name", "Date"])

            # Create next day close
            test_df["Next_Close"] = test_df.groupby("Index_Name")["Close"].shift(-1)

            # Create return target
            test_df["Return"] = (
                test_df["Next_Close"] - test_df["Close"]
            ) / test_df["Close"]

            # Lag features
            for lag in [1, 2, 3]:
                test_df[f"Prev_Close_{lag}"] = (
                    test_df.groupby("Index_Name")["Close"].shift(lag)
                )

            # Moving averages
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

            # Encode index name
            test_df["Index_Code"] = (
                test_df["Index_Name"].astype("category").cat.codes
            )

            # Drop NaN rows
            test_df = test_df.dropna()

            # Feature columns
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
                "MA_10"
            ]

            x = test_df[feature_cols]
            y = test_df[TARGET_COLUMN]

            logging.info("Test data prepared for regression evaluation")

            trained_model = load_object(
                file_path=self.model_trainer_artifact.trained_model_file_path
            )

            y_hat_trained = trained_model.predict(x)
            trained_model_score = r2_score(y, y_hat_trained)

            logging.info(f"R2 score (new model): {trained_model_score}")

            best_model_score = None
            best_model = self.get_best_model()

            if best_model is not None:
                y_hat_best_model = best_model.predict(x)
                best_model_score = r2_score(y, y_hat_best_model)
                logging.info(
                    f"R2 production: {best_model_score}, R2 new: {trained_model_score}"
                )

            tmp_best_model_score = 0 if best_model_score is None else best_model_score

            # tolerance logic (without modifying dataclass)
            tolerance = 0.02
            is_accepted = trained_model_score >= (tmp_best_model_score - tolerance)

            result = EvaluateModelResponse(
                trained_model_score=trained_model_score,
                best_model_score=best_model_score,
                is_model_accepted=is_accepted,
                difference=trained_model_score - tmp_best_model_score
            )

            logging.info(f"Evaluation Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Start model evaluation
        """
        try:
            logging.info("Initialized Model Evaluation Component.")

            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys) from e
