'''
Unit tests for API App

Author: Marina Dolokov
Date: February 2022 

'''
from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_get():
    r = client.get("/items/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the census classification API"}

def test_post():
    r = client.post("/items/")
    assert r.status_code == 422

def test_post_prediction():
    r = client.post("/items/", 
                    json = {"age": 42,
                            "workclass": "Self-emp-not-inc",
                            "education": "HS-grad",
                            "marital_status": "Married-civ-spouse",
                            "occupation": "Farming-fishing",
                            "relationship": "Husband",
                            "race": "Asian-Pac-Islander",
                            "sex": "Male",
                            "capital_gain": 0,
                            "capital_loss": 0,
                            "hours_per_week": 40,
                            "native_country": "Cambodia"
                            }
                    )
    assert r.json() == {"Predicted salary": ">50k"}
    r = client.post("/items/", 
                    json = {"age": 50,
                            "workclass": "Self-emp-not-inc",
                            "education": "Bachelors",
                            "marital_status": "Married-civ-spouse",
                            "occupation": "Exec-managerial",
                            "relationship": "Husband",
                            "race": "White",
                            "sex": "Male",
                            "capital_gain": 0,
                            "capital_loss": 0,
                            "hours_per_week": 13,
                            "native_country": "United-States"
                            }
                    )
    assert r.json() == {"Predicted salary": "<=50k"}
