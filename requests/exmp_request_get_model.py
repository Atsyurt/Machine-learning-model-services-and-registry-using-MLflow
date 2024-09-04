import requests
import os
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'

# Verify that the environment variable is set
print(os.getenv('MLFLOW_TRACKING_URI'))

url = "http://localhost:8000/get_model/"

data={"name":"model2_extra_tree_classifier","version":"1"}
response = requests.post(
        url,
        data=data
    )

print("response",response.json())