from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CardioAI Heart Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    logger.info("ML assets loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load assets: {e}")
    raise


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
    return FileResponse("index.html")


@app.post("/predict")
async def predict_heart_disease(data: PatientData):
    try:
        logger.info(f"Received request for age: {data.Age}")

        input_data = data.dict()

        input_data["Sex"] = "M" if data.Sex == "Male" else "F"
        input_data["ExerciseAngina"] = (
            "Y" if data.ExerciseAngina == "Yes" else "N"
        )

        df = pd.DataFrame([input_data])

        categorical_cols = [
            "Sex",
            "ChestPainType",
            "RestingECG",
            "ExerciseAngina",
            "ST_Slope",
        ]

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
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
