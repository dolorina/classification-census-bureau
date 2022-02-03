'''
Script to build the FastAPI App

Author: Marina Dolokov
Date: February 2022
'''
from model import load_model, inference
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

model = load_model()
encoder = load_model("./mlp_encoder.sav")

class Data(BaseModel):
    age: int = Field(..., example = 50)
    workclass: str = Field(..., example = "Self-emp-not-inc")
    education: str = Field(..., example = "Bachelors")
    marital_status: str = Field(..., example = "Married-civ-spouse")
    occupation: str = Field(..., example = "Exec-managerial")
    relationship: str = Field(..., example = "Husband")
    race: str = Field(..., example = "White")
    sex: str = Field(..., example = "Male")
    capital_gain: int = Field(..., example = 0)
    capital_loss: int = Field(..., example = 0)
    hours_per_week: int = Field(..., example = 13)
    native_country: str = Field(..., example = "United-States")
        
    
def calculate_inference(data):
    X_cat = np.array([
        data.workclass, 
        data.education, 
        data.marital_status, 
        data.occupation,
        data.relationship, 
        data.race,
        data.sex, 
        data.native_country])
    X_cat = np.reshape(X_cat, (1,-1))
    columns = ["workclass", "education", "marital-status", "ocupation", "relationship", "race", "sex", "native-country"]
    X_cat = pd.DataFrame(data=X_cat, columns=columns, index = [1])
    X_cat = encoder.transform(X_cat)
    
    X_cnt = np.array([
        data.age, 
        data.capital_gain,
        data.capital_loss, 
        data.hours_per_week])
    X_cnt = np.reshape(X_cnt, (1,-1))
    
    X = np.concatenate([X_cnt, X_cat], axis=1)

    preds = inference(model, X)
    print(preds)
    if preds == 0 :
        salary = "<=50k"
    else:
        salary = ">50k"
    return{"Predicted salary": salary}


app = FastAPI()

@app.get("/")
def get_root():
    return{"message": "Welcome to the census classification API"}

@app.post("/items/")
async def model_inference(data: Data):
    return calculate_inference(data)