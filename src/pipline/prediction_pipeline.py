import sys
import pandas as pd
from pandas import DataFrame

from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging


class StockData:
    def __init__(
        self,
        Open,
        High,
        Low,
        Close,
        Volume,
        Daily_Change_Percent,
        Index_Code,
        Prev_Close_1,
        Prev_Close_2,
        Prev_Close_3,
        MA_5,
        MA_10,
    ):
        try:
            self.Open = Open
            self.High = High
            self.Low = Low
            self.Close = Close
            self.Volume = Volume
            self.Daily_Change_Percent = Daily_Change_Percent
            self.Index_Code = Index_Code
            self.Prev_Close_1 = Prev_Close_1
            self.Prev_Close_2 = Prev_Close_2
            self.Prev_Close_3 = Prev_Close_3
            self.MA_5 = MA_5
            self.MA_10 = MA_10

        except Exception as e:
            raise MyException(e, sys) from e

    def get_stock_input_data_frame(self) -> DataFrame:
        try:
            stock_input_dict = {
                "Open": [self.Open],
                "High": [self.High],
                "Low": [self.Low],
                "Close": [self.Close],
                "Volume": [self.Volume],
                "Daily_Change_Percent": [self.Daily_Change_Percent],
                "Index_Code": [self.Index_Code],
                "Prev_Close_1": [self.Prev_Close_1],
                "Prev_Close_2": [self.Prev_Close_2],
                "Prev_Close_3": [self.Prev_Close_3],
                "MA_5": [self.MA_5],
                "MA_10": [self.MA_10],
            }

            return DataFrame(stock_input_dict)

        except Exception as e:
            raise MyException(e, sys) from e


class StockPredictor:
    def __init__(
        self,
        prediction_pipeline_config: VehiclePredictorConfig = VehiclePredictorConfig(),
    ) -> None:
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe):
        try:
            logging.info("Entered predict method of StockPredictor class")

            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )

            result = model.predict(dataframe)
            return result

        except Exception as e:
            raise MyException(e, sys)
