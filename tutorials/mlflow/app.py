from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List

import pandas as pd
import mlflow.pyfunc
import mlflow

# set the tracking server to be localhost
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# define the path to the MLflow model
logged_model = 'runs:/dede439c576a4d079f562e3d9ad7bb52/model_learning_fair_representation'

# load the MLflow model using mlflow.pyfunc.load_model()
loaded_model = mlflow.pyfunc.load_model(logged_model)

app = FastAPI()  # Initialize a FastAPI application

# define a Pydantic model for input data validation
class InputData(BaseModel):
    data: List[List[float]]
    columns: List[str]

# define a route and function for making predictions
@app.post('/predict')  # Define a route '/predict' that accepts HTTP POST requests
async def predict(input_data: InputData):
    try:
        input_df = pd.DataFrame(jsonable_encoder(input_data.data), columns=input_data.columns)

        # make predictions using the loaded model
        predictions = loaded_model.predict(input_df)  # Use the model to make predictions on input data

        # return predictions as a JSON response
        return {"predictions": predictions.tolist()}  # Convert predictions to a JSON response

    except Exception as e:
        # Handle exceptions (e.g., invalid input or model errors) and return an error message as a JSON response
        raise HTTPException(status_code=400, detail=str(e))