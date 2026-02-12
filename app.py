from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from typing import Optional
import pandas as pd
import os
import glob
import pickle

from src.constants import APP_HOST, APP_PORT
from src.pipline.training_pipeline import TrainPipeline

# -----------------------------
# Load latest trained model
# -----------------------------
def load_latest_model():
    artifact_folders = glob.glob("artifact/*")
    if not artifact_folders:
        raise Exception("No trained model found. Run /train first.")

    latest_folder = max(artifact_folders, key=os.path.getctime)
    model_path = os.path.join(
        latest_folder,
        "model_trainer",
        "trained_model",
        "model.pkl"
    )

    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_latest_model()

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Stock Data Form (12 features)
# -----------------------------
class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.Open = None
        self.High = None
        self.Low = None
        self.Close = None
        self.Volume = None
        self.Daily_Change_Percent = None
        self.Index_Code = None
        self.Open_Close_Diff = None
        self.High_Low_Diff = None
        self.Close_Lag_1 = None
        self.Close_Lag_2 = None
        self.Volume_Change = None

    async def get_stock_data(self):
        form = await self.request.form()
        self.Open = float(form.get("Open"))
        self.High = float(form.get("High"))
        self.Low = float(form.get("Low"))
        self.Close = float(form.get("Close"))
        self.Volume = float(form.get("Volume"))
        self.Daily_Change_Percent = float(form.get("Daily_Change_Percent"))
        self.Index_Code = int(form.get("Index_Code"))

        # engineered features
        self.Open_Close_Diff = self.Open - self.Close
        self.High_Low_Diff = self.High - self.Low
        self.Close_Lag_1 = self.Close
        self.Close_Lag_2 = self.Close
        self.Volume_Change = 0.0


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        "stock.html",
        {"request": request, "context": "Enter stock values"}
    )

@app.get("/train")
async def trainRouteClient():
    global model
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        model = load_latest_model()
        return Response("Training successful!!!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_stock_data()

        input_data = pd.DataFrame([{
            "Open": form.Open,
            "High": form.High,
            "Low": form.Low,
            "Close": form.Close,
            "Volume": form.Volume,
            "Daily_Change_Percent": form.Daily_Change_Percent,
            "Index_Code": form.Index_Code,
            "Open_Close_Diff": form.Open_Close_Diff,
            "High_Low_Diff": form.High_Low_Diff,
            "Close_Lag_1": form.Close_Lag_1,
            "Close_Lag_2": form.Close_Lag_2,
            "Volume_Change": form.Volume_Change
        }])

        prediction = model.predict(input_data)[0]

        return templates.TemplateResponse(
            "stock.html",
            {
                "request": request,
                "context": f"Predicted Next Close: {round(float(prediction), 2)}"
            },
        )

    except Exception as e:
        return {"status": False, "error": f"{e}"}

# -----------------------------
# Run server
# -----------------------------
# -----------------------------
# Run server directly
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=5000,
        reload=True
    )
