from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.entity.estimator import MyModel
import sys
from pandas import DataFrame

class Proj1Estimator:

 def __init__(self, bucket_name, model_path):
    """
    :param bucket_name: Name of your model bucket
    :param model_path: Location of your model in bucket
    """
    self.bucket_name = bucket_name
    self.s3 = SimpleStorageService()
    self.model_path = model_path
    self.loaded_model: MyModel = None

 def is_model_present(self, model_path):
    """
    Check if model exists in S3
    """
    try:
        return self.s3.s3_key_path_available(
            bucket_name=self.bucket_name,
            s3_key=model_path
        )
    except MyException as e:
        print(e)
        return False

 def load_model(self) -> MyModel:
    """
    Load the model from S3
    """
    try:
        self.loaded_model = self.s3.load_model(
            self.model_path,
            bucket_name=self.bucket_name
        )
        return self.loaded_model
    except Exception as e:
        raise MyException(e, sys)

 def save_model(self, from_file, remove: bool = False) -> None:
    """
    Save the model to S3
    """
    try:
        self.s3.upload_file(
            from_file,
            to_filename=self.model_path,
            bucket_name=self.bucket_name,
            remove=remove
        )
    except Exception as e:
        raise MyException(e, sys)

 def predict(self, dataframe: DataFrame):
    """
    Make prediction using loaded model
    """
    try:
        if self.loaded_model is None:
            self.load_model()

        return self.loaded_model.predict(dataframe)
    except Exception as e:
        raise MyException(e, sys)
