from typing import List

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
logged_model = "runs:/c0e2d8d695244f76889ae18bd349b5a2/model_disparate_impact_remover"

# Load the MLflow model using mlflow.pyfunc.load_model()
loaded_model = mlflow.pyfunc.load_model(logged_model)

app = FastAPI(title="HolisticAIDeploy")  # Initialize a FastAPI application

# Define a Pydantic model for input data validation
class InputData(BaseModel):
    data: List[List[float]]
    columns: List[str]


# Define a route and function for making predictions
@app.post("/predict")  # Define a route '/predict' that accepts HTTP POST requests
async def predict(input_data: InputData):
    try:
        input_df = pd.DataFrame(
            jsonable_encoder(input_data.data), columns=input_data.columns
        )

        # Make predictions using the loaded model
        predictions = loaded_model.predict(
            input_df
        )  # Use the model to make predictions on input data

        # Return predictions as a JSON response
        return {
            "predictions": predictions.tolist()
        }  # Convert predictions to a JSON response

    except Exception as e:
        # Handle exceptions (e.g., invalid input or model errors) and return an error message as a JSON response
        raise HTTPException(status_code=400, detail=str(e))
