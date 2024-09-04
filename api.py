# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:14:45 2024

@author: ayhan
"""

from fastapi import FastAPI, UploadFile, File,Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import traceback
import sys
from typing import Dict
import mlflow
import mlflow.sklearn
from mlflow.models.signature import ModelSignature
from mlflow.models import infer_signature
from mlflow.types import Schema, ColSpec
import mlflow.pyfunc
import json
import pandas as pd

#in order to communicate with mlflow server we should configure this env varaiable for api server
import os
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'

# Verify that the environment variable is set
print(os.getenv('MLFLOW_TRACKING_URI'))



class MLmodel_metadata(BaseModel):
    name: str
    description: str
    params: dict


def get_exception_info():
    exception_type, exception_value, exception_traceback = sys.exc_info()
    file_name, line_number, procedure_name, line_code = traceback.extract_tb(exception_traceback)[-1]
    exception_info=" [File Name]:" + str(file_name)+"\n"+" [Procedure Name]:"+str(procedure_name) +"\n [Error Message]:"+ str(exception_value)+"\n [Error Type]:"+str(exception_type)+"\n [Line Number]:"+str(line_number)+"\n [Line Code]:" + str(line_code)
    return exception_info


#MLmodel_metadata={"name":"test","description":"","params":{}}

app = FastAPI()

# @app.post("/upload_model/")
# async def upload_model(file: UploadFile = File(...),name:str = Form(...)):
#     # # Load the model from path
#     try:
#         file_path = f"temp/{file.filename}"
#         with open(file_path, "wb+") as file_object:
#             file_object.write(file.file.read())
#             print("model is saved.")
        
#         with open(file_path, 'rb') as pickle_file:
#             loaded_model = pickle.load(pickle_file)
#             print("model is loaded")
#             print("METADATA",name)
        
#         # meta=metadata.dict()
#         print("METADATA",name)
#         # with open(filename, 'rb') as file:
#         #      loaded_model = pickle.load(file)
#         data = {"message": "Successful,model is loaded"}
#         return JSONResponse(content=data, status_code=200)
    
#     except:
#         exception_information= get_exception_info()
#         print(exception_information)
#         data = {"message": "Request Failed Reason:"+exception_information}
#         return JSONResponse(content=data, status_code=500)



        

class Metadata(BaseModel):
    name: str
    version: str
    description: str
    model_path:str
    params:str




@app.post("/upload/")
async def upload_model(
    file: UploadFile = File(...),
    user_name:str = Form(...),
    name: str = Form(...),
    version: str = Form(...),
    description: str = Form(...),
    model_path:str=Form(...),
    model_signature: str = Form(...),
    params:str=Form(...)):
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = user_name
    
    #convert  model_signature str to mlflow model metadata model signature type
    if model_signature!="":
        try:
            signature_str=model_signature
            signature_dict = json.loads(signature_str)
            print("model signature is set")
            input_schema = Schema([ColSpec(**col) for col in signature_dict['inputs']])
            output_schema = Schema([ColSpec(**col) for col in signature_dict['outputs']])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)
            print("signature created succesfully")
        except:
            exception_information= get_exception_info()
            print(exception_information)
            
    metadata = Metadata(name=name, version=version, description=description,model_path=model_path,params=params)
    print("params",params)
    print(metadata)
    
    try:
    # Process the file and metadata as needed
        print("filename", file.filename)
        file_path = f"temp/{file.filename}"
        with open(file_path, "wb+") as file_object:
            file_object.write(file.file.read())
            print("model is saved.")
    except:
        exception_information= get_exception_info()
        print(exception_information)
        data = {"message": "Request Failed Reason:"+exception_information}
        return JSONResponse(content=data, status_code=500)
    
    

        
    with mlflow.start_run() as run:
        params_object = json.loads(metadata.params)
        with open(file_path, 'rb') as pickle_file:
            loaded_model = pickle.load(pickle_file)
            print("model is loaded")
            print("METADATA",name)
            
        mlflow.log_params(params_object)
        mlflow.sklearn.log_model(
            sk_model=loaded_model,
            registered_model_name=metadata.name,
            artifact_path=metadata.model_path
        )
       

        
    data = {"message": "Successful,model is loaded"}
    return JSONResponse(content=data, status_code=200)


@app.post("/get_model/")
async def get_model(name: str = Form(...),version: str = Form(...),):
    model_version=version
    model_name=name
    model_uri = f"models:/{model_name}/{model_version}"
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        data={"mesage": "model is found ","model_uri":model_uri,"model_signature":str(loaded_model.metadata.signature)}
        return JSONResponse(content=data, status_code=200)
    except:
        exception_information= get_exception_info()
        data={"mesage": "model is not found ","info":exception_information}
        return JSONResponse(content=data, status_code=500)

@app.get("/")
async def root():
    return {"mesage": "Hello you can use upload_model post method to upload a model and its metadata "}


@app.get("/healthcheck")
async def healthcheck():
    return {"mesage": "Mlflow is running.Mlflow version: "+mlflow.__version__}




class data_Item(BaseModel):
    age: float
    gender: float
    annual_income: float
    purchase_amount: float
    year: float
    month: float
    purchase_amount_group: float
    

@app.post("/online_inference_model2")
async def online_inference_model2(version: str ,data: data_Item ):
    model_uri="models:/model2_extra_tree_classifier/"+str(version)
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print(data.age)
        print(data)
        data_dict={"age":data.age,"gender":data.gender,"annual_income":data.annual_income,"purchase_amount":data.purchase_amount,"year":data.year,"month":data.month,"purchase_amount_group":data.purchase_amount_group}
        pd_data= pd.DataFrame.from_dict(data_dict,orient='index')
        pd_data=pd_data.T
        print(str(pd_data.info()))
        pred=loaded_model.predict(pd_data)
        result={"mesage": "online infer is successfull","data_info":str(pd_data.info()),"inference_result":str(pred)}
        return JSONResponse(content=result,status_code=200)
    except:
        exception_information= get_exception_info()
        result={"mesage": "unsucessfull inference","info":exception_information}
        return JSONResponse(content=result, status_code=500)
    
