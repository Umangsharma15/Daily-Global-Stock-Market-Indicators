import os
import pickle
import glob
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Stock Next-Day Prediction API")

# -----------------------------
# Find latest model automatically
# -----------------------------
artifact_folders = glob.glob("artifact/*")
latest_folder = max(artifact_folders, key=os.path.getctime)

model_path = os.path.join(
    latest_folder,
    "model_trainer",
    "trained_model",
    "model.pkl"
)

with open(model_path, "rb") as f:
    model = pickle.load(f)


# -----------------------------
# Input schema
# -----------------------------
class StockInput(BaseModel):
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float
    Daily_Change_Percent: float
    Index_Code: int


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "Stock Prediction API is running"}


@app.post("/predict")
def predict(data: StockInput):
    features = np.array([[
        data.Open,
        data.High,
        data.Low,
        data.Close,
        data.Volume,
        data.Daily_Change_Percent,
        data.Index_Code
    ]])

    prediction = model.predict(features)[0]

    return {
        "predicted_next_close": round(float(prediction), 2)
    }
