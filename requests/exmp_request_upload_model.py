import requests


# import os
# os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'

# # Verify that the environment variable is set
# print(os.getenv('MLFLOW_TRACKING_URI'))

url = "http://localhost:8000/upload/"
file_path = "finalized_model_multi_target_classifier.pkl"


params2=log_params='{"log1":"log_test","log2":"log_test"}'

metadata = {
    "user_name":"anonymous",
    "name": "user_uploaded_model",
    "version": "1.0",
    "description": "This is an example user model.",
    "model_path":"temp_models",
    "model_signature":' none',
    "params":log_params
    
}

with open(file_path, "rb") as file:
    response = requests.post(
        url,
        files={"file": file},
        data=metadata
    )
    print(response.json)