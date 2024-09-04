
import requests
import os
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'

# Verify that the environment variable is set
print(os.getenv('MLFLOW_TRACKING_URI'))

url = "http://localhost:8000/online_inference_model2?version=1"


data={
  "age": 0,
  "gender": 1,
  "annual_income": 0,
  "purchase_amount": 0.7,
  "year": 0,
  "month": 5,
  "purchase_amount_group": 3
}
response = requests.post(
        url,
        json=data
    )

print("respone",response.json())