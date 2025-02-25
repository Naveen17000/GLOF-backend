from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import BaseModel

# Load Model
with open("glof_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

feature_names = [
    "air_temp_C", "air_humidity_%", "water_temp_C", "altitude_change_m",
    "tilt_x_deg", "tilt_y_deg", "tilt_z_deg", "ground_temp_C",
    "seismic_activity_Hz", "flow_velocity_mps"
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputFeatures(BaseModel):
    features: list[float]

@app.post("/predict")
async def predict(input_data: InputFeatures):
    try:
        if len(input_data.features) != len(feature_names):
            return {"error": f"Expected {len(feature_names)} features, but got {len(input_data.features)}"}

        input_df = pd.DataFrame([input_data.features], columns=feature_names)
        dmatrix_input = xgb.DMatrix(input_df, feature_names=feature_names)
        probabilities = model.predict(dmatrix_input)

        if len(probabilities) == 0:
            return {"error": "Prediction failed, empty result"}

        predicted_class = int(np.argmax(probabilities))

        return {
            "predicted_class": predicted_class,
            "probabilities": probabilities[0].tolist()
        }

    except Exception as e:
        return {"error": str(e)}
