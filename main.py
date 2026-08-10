from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
import logging
import os

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CardioAI Heart Disease Prediction API")

# CORS Middleware (Fully enabled for GitHub Pages and all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Machine Learning Assets
model = None
scaler = None
columns = None

try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('columns.pkl')
    logger.info("ML assets loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load assets at startup: {e}")

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

@app.get("/")
def read_root():
    return {"message": "CardioAI API is running. Send POST request to /predict"}

@app.post("/predict")
async def predict_heart_disease(data: PatientData):
    if not model or not scaler or not columns:
        logger.error("Prediction requested but ML assets are missing.")
        raise HTTPException(
            status_code=500, 
            detail="Model files (pkl) failed to load on server."
        )

    try:
        logger.info(f"Received prediction request for Patient Data.")
        
        # 1. Input Mapping
        input_dict = data.dict()
        input_dict['Sex'] = 'M' if data.Sex.strip().lower() == 'male' else 'F'
        input_dict['ExerciseAngina'] = 'Y' if data.ExerciseAngina.strip().lower() == 'yes' else 'N'
        
        # 2. DataFrame Creation & Categorical One-Hot Encoding
        df = pd.DataFrame([input_dict])
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        df = pd.get_dummies(df, columns=categorical_cols)
        
        # 3. Align DataFrame columns with trained model columns
        df = df.reindex(columns=columns, fill_value=0)
        
        # 4. Scaling & Prediction
        scaled_data = scaler.transform(df)
        
        prediction = int(model.predict(scaled_data)[0])
        probability = float(model.predict_proba(scaled_data)[0][1])
        
        return {
            "prediction": prediction,
            "probability": round(probability * 100, 2)
        }
        
    except Exception as e:
        logger.error(f"Prediction Execution Error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Model Prediction Error: {str(e)}"
        )

if __name__ == "__main__":
    # Binding host to 0.0.0.0 and dynamically picking Railway PORT env variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
