from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from starlette.responses import JSONResponse
import os

# Get the current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained model from the same directory
model_path = os.path.join(current_dir, "trained_model.joblib")
model = joblib.load(model_path)

# Define the FastAPI app
app = FastAPI()

# Define the input data model with integer fields
class PredictData(BaseModel):
    CSC101_total: int
    CSC201_total: int
    CSC203_total: int
    CSC205_total: int
    CSC102_total: int
    MAT202_total: int
    MAT203_total: int
    MAT103_total: int
    CSC206_total: int
    MAN101_total: int
    SWE201_total: int
    SWE301_total: int
    SWE303_total: int
    CNE202_total: int
    CNE203_total: int
    CNE304_total: int
    CSC301_total: int
    CNE302_total: int
    CSC309_total: int
    CSC302_total: int
    CSC303_total: int
    CNE308_total: int

# Define the prediction endpoint
@app.post("/predict")
def predict(data: PredictData):
    try:
        # Convert the input data to a numpy array
        input_data = np.array([[data.CSC101_total, data.CSC201_total, data.CSC203_total,
                                data.CSC205_total, data.CSC102_total, data.MAT202_total,
                                data.MAT203_total, data.MAT103_total, data.CSC206_total,
                                data.MAN101_total, data.SWE201_total, data.SWE301_total,
                                data.SWE303_total, data.CNE202_total, data.CNE203_total,
                                data.CNE304_total, data.CSC301_total, data.CNE302_total,
                                data.CSC309_total, data.CSC302_total, data.CSC303_total,
                                data.CNE308_total]])

        # Make the prediction
        prediction = model.predict(input_data)

        department_mapping = {0: 'Swe', 1: 'Cs', 2: 'Cne', 3: 'Ai'}
        predicted_department = department_mapping[prediction[0]]

        return JSONResponse({"predicted_department": predicted_department})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


