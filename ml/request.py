import requests
import json

data = {"age": 42,
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
# Making a POST request
response = requests.post('https://census-bureau-app.herokuapp.com/', auth=('usr', 'pass'), data=json.dumps(data))

print("Check status code for response received:", response.status_code)
print("Print content of request:", response.json())