import requests
# import json

response = requests.post('https://census-bureau-app.herokuapp.com/')

# data = {"age": 42,
#         "workclass": "Self-emp-not-inc",
#         "education": "HS-grad",
#         "marital_status": "Married-civ-spouse",
#         "occupation": "Farming-fishing",
#         "relationship": "Husband",
#         "race": "Asian-Pac-Islander",
#         "sex": "Male",
#         "capital_gain": 0,
#         "capital_loss": 0,
#         "hours_per_week": 40,
#         "native_country": "Cambodia"
#         }
# response = requests.post('http://127.0.0.1:8000/items/', auth=('usr', 'pass'), data=json.dumps(data))

print(response.status_code)
print(response.json())