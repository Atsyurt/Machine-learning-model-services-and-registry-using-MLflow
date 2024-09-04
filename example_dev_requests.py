# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:43:28 2024

@author: ayhan
"""

#upload model example post request
import requests


# import os
# os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'

# # Verify that the environment variable is set
# print(os.getenv('MLFLOW_TRACKING_URI'))

url = "http://localhost:8000/upload/"
file_path = "finalized_model_multi_target_classifier.pkl"


params2=log_params='{"log1":"log_test","log2":"log_test"}'

#example  str signature
# '{\'inputs\': \'[{"type": "integer", "name": "age", "required": true}, {"type": "integer", "name": "gender", "required": true}, {"type": "integer", "name": "annual_income", "required": true}, {"type": "double", "name": "purchase_amount", "required": true}, {"type": "integer", "name": "year", "required": true}, {"type": "integer", "name": "month", "required": true}, {"type": "integer", "name": "purchase_amount_group", "required": true}]\', \'outputs\': \'[{"type": "tensor", "tensor-spec": {"dtype": "float64", "shape": [-1, 10]}}]\', \'params\': None}'

metadata = {
    "user_name":"anonymous",
    "name": "multi_target_classifier2",
    "version": "1.0",
    "description": "This is an example model.",
    "model_path":"temp_models",
    "model_signature":' none',
    "params":log_params
    
}

#tt=signature.to_dict()


with open(file_path, "rb") as file:
    response = requests.post(
        url,
        files={"file": file},
        data=metadata
    )
    print(response.json)

#get model

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

print("respone",response.json())



# make online inference for model2

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





 