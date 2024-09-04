# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:53:15 2024

@author: ayhan
"""

import mlflow
import pandas as pd

logged_model = 'runs:/1c2da6ff7e7245c19d0edcaaffe2a886/sklearn-rf-model-multi-label-classfier'
model_name="finalized_model_regression"
model_version="1"
model_uri = f"models:/{model_name}/{model_version}"
new_model="models:/model2_extra_tree_classifier/1"
loaded_model = mlflow.pyfunc.load_model(new_model)
loaded_model = mlflow.pyfunc.load_model(model_uri)
loaded_model = mlflow.pyfunc.load_model(logged_model)

test=X_test.iloc[:5]
preds=loaded_model.predict(test)
#column_values = ['age', 'gender', 'annual_income','purchase_amount','year','month','purchase_amount_group']
data_dict={"age":int(2),"gender":0,"annual_income":2,"purchase_amount":-0.17654564,"year":1,"month":10,"purchase_amount_group":1}
df = pd.DataFrame.from_dict(data_dict,orient='index')

df=df.T
df.info()
#df["age"].astype("int")

preds=loaded_model.predict(df)




import mlflow.pyfunc



model_name="multi_target_classifier"
model_version="1"
model_uri = f"models:/{model_name}/{model_version}"
# Load the model
model = mlflow.pyfunc.load_model(model_uri)

# Print the model signature
print(model.metadata.signature)



from mlflow.tracking import MlflowClient

# Initialize the MLflow client
client = MlflowClient()

# Get the list of all registered models
registered_models = client.search_registered_models()

for model in registered_models:
    print(model.name)








