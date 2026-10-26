import modal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Modal Setup
app = modal.App("heartcare-ai-api")
image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]", 
    "pandas", 
    "scikit-learn", 
    "joblib", 
    "pydantic"
)
volume = modal.Volume.from_name("heartcare-models", create_if_missing=True)

# 2. FastAPI Setup
web_app = FastAPI(title="CardioAI Heart Disease Prediction API")

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    Age: int
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    MaxHR: int
    Oldpeak: float
    Sex: str
    ChestPainType: str
    RestingECG: str
    ExerciseAngina: str
    ST_Slope: str

# Lazy loading taaki container build time par error na aaye
model = None
scaler = None
columns = None

def load_models():
    global model, scaler, columns
    if model is None:
        try:
            model = joblib.load("/data/best_model.pkl")
            scaler = joblib.load("/data/scaler.pkl")
            columns = joblib.load("/data/columns.pkl")
            logger.info("ML assets loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load assets: {e}")
            raise

@web_app.post("/predict")
async def predict_heart_disease(data: PatientData):
    load_models()
    try:
        input_data = data.dict()
        input_data["Sex"] = "M" if data.Sex == "Male" else "F"
        input_data["ExerciseAngina"] = "Y" if data.ExerciseAngina == "Yes" else "N"

        df = pd.DataFrame([input_data])
        categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

        df = pd.get_dummies(df, columns=categorical_cols)
        df = df.reindex(columns=columns, fill_value=0)

        scaled_data = scaler.transform(df)

        prediction = int(model.predict(scaled_data)[0])
        probability = float(model.predict_proba(scaled_data)[0][1])

        return {
            "prediction": prediction,
            "probability": round(probability * 100, 2),
        }

    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# 3. Modal Deployment
@app.function(image=image, volumes={"/data": volume}, cpu=2.0)
@modal.asgi_app()
def serve():
    return web_app