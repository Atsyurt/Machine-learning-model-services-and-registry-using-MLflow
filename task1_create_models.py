# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:49:04 2024

@author: ayhan
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle
import numpy as np
pd.set_option('display.precision',4)
import warnings
warnings.filterwarnings('ignore')

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
#for task run name
import uuid
unique_id = uuid.uuid4()
print(unique_id)
#set mllow env tracking url
#make sure that you build and run mlflow_local service for model regsitry

import os
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
# Set the MLflow tracking username who is created this work
os.environ['MLFLOW_TRACKING_USERNAME'] = "ayhant"
# Verify that the environment variable is set
print(os.getenv('MLFLOW_TRACKING_URI'))
print(os.environ['MLFLOW_TRACKING_USERNAME'])

    




#Dataset preparing for amount_of_purchase y values based on previous
#consecutive purchase dates
#currently dataset is given is not ready and is not provide any information for train y values.

#step0-)load daatset given
ds=pd.read_csv('customer_purchases.csv')
print(ds.head)
ds_desc=ds.describe()
columns=ds.columns


##step1-) Handle purchase data object type and convert it to pandas date format
p_d_info=ds["purchase_date"].info()
p_d_desc=ds["purchase_date"].describe()

ds['purchase_date'] = pd.to_datetime(ds['purchase_date'])
p_d_info=ds["purchase_date"].info()
p_d_desc=ds["purchase_date"].describe()
ds.head()

##step2-) Extract usefull infos from new purchase_date pandas Dtype column
#Since we already converted to Dtype format
#I can utilize many usefull features from pandas Dtype
ds['year'] = ds['purchase_date'].dt.year
ds['month'] = ds['purchase_date'].dt.month
ds['day'] = ds['purchase_date'].dt.day
#these features are also nice but i  wont use
#ds['day_of_week'] = ds['purchase_date'].dt.dayofweek
#ds['is_weekend'] = df['day_of_week'] >= 5



#create a copy of ds
ds2=ds.copy()
#sort ds2 by customer_id and purchase_date
ds2 = ds2.sort_values(by=["customer_id","purchase_date"])
ds2.reset_index(drop=True, inplace=True)
ds2["next_month_purchase_amount"]=-9999.0
#remove Nan values other than next_month_purchase_amount column
ds2=ds2.dropna()

##step3-)In the Dataset there are  multiple entries
# with  same month and same year for same customer
# we should accumalate purchase_amount values to single value

temp_year=None
temp_index=None
temp_month=None
temp_id=None
accumulated_purchase_amount=0
dlist=[]

#ds2.loc[0]={"next_month_purchase_amount":52}
for index, row in ds2.iterrows():
    if temp_year and temp_month and  temp_id ==None:
        temp_year=row['year']
        temp_month=row['month']
        temp_index=index
        temp_id=row['customer_id']
        accumulated_purchase_amount=row["purchase_amount"]
        continue
#find the same customer_id year month values with previous record      
    if row['customer_id']==temp_id:
        if row['year']==temp_year:
            if row['month']==temp_month:
                accumulated_purchase_amount=accumulated_purchase_amount+row["purchase_amount"]
                #i will drop these same month values in order have single accumulated value
                #so i put a mark on these values pd.NA
                ds2.at[index,'purchase_amount']=pd.NA
                continue
    #assign first index newly accumuluated purchase value
    if accumulated_purchase_amount>0:
        data={"customer_id":temp_id,"index":temp_index,
              "purchase_amount":accumulated_purchase_amount,
              "year":temp_year,"month":temp_month}
        dlist.append(data)
        ds2.at[temp_index,'purchase_amount']=accumulated_purchase_amount
        
     
    #reset temp vallue holders
    accumulated_purchase_amount=row["purchase_amount"]  
    temp_year=row['year']
    temp_month=row['month']
    temp_index=index
    temp_id=row['customer_id']

#in order remove duplicate same month same year value groups i will drop
ds2=ds2.dropna()

##step4-) now we are ready to crate next month purchase values
# for our dataset's  next_month_purchase_amount column so we can prepare y values for our model
#Right now they are all have -9999 value and sorted by date
#but using newly created year and month columns
#I can fill these columns by filtering consecutive 2 month on same year data entries

ds2["next_month_purchase_amount"]=pd.NA

for index, row in ds2.iterrows():
    if temp_year and temp_month and  temp_id ==None:
        temp_year=row['year']
        temp_month=row['month']
        temp_index=index
        temp_id=row['customer_id']
        continue
    if row['customer_id']==temp_id:
        if row['year']==temp_year:
            #find consecutive month  in the same year for same id
            if row['month']==temp_month+1:
                #print(row['month'],temp_month,"id",temp_id)
                ds2.at[temp_index,'next_month_purchase_amount']=row["purchase_amount"]
                #ds2.at[index,'next_month_purchase_amount']=pd.NA
                #ds2.at[index,'row["purchase_amount"]']=pd.NA
        # find consecutive months for the December and january
        elif row['year']==temp_year+1 and row['month']==temp_month-11:
            ds2.at[temp_index,'next_month_purchase_amount']=row["purchase_amount"]
        

    temp_year=row['year']
    temp_month=row['month']
    temp_index=index
    temp_id=row['customer_id']
    
#remove entries which have not next month purchase info

ds2=ds2.dropna()
print("dataset shape:",ds2.shape)

##step5-)now our ds2 dataset is have next month purchase y values for our model
#now we can preprocess data before the model train
#first remove day column
#since i will not plan to use day,dayof the week or is_weekend info however it may be nice to use 

ds2=ds2.drop(["day"],axis=1)

#i will also remove purchase _date dtype data column
# because i already have month and year column im my dataset ds2

ds2=ds2.drop(["purchase_date"],axis=1)
print("after removing features dataset shape:",ds2.shape)


#now i will apply some bins and labes for big number features such as;


# bins for annual_income,age,year,purchase_amount,next_month_purchase_amount
#this params variable logged as metadata for model in registry
params={}

bin_number=10

params["bin_number"]=bin_number

ain_data=pd.cut(ds2["annual_income"],3,labels=["low","medium","high"],retbins=True)
ds2["annual_income"]=ain_data[0]
registry_df_annual_income_bins_param=ain_data[1]
params["registry_df_annual_income_bins_param"]=registry_df_annual_income_bins_param
ds2["annual_income"]=ds2["annual_income"].astype('object')


age_data=pd.cut(ds2["age"],3,labels=["young","middle","old"],retbins=True)
ds2["age"]=age_data[0]
registry_df_age_bins_param=age_data[1]
params["registry_df_age_bins_param"]=registry_df_age_bins_param
ds2["age"]=ds2["age"].astype('object')


year_data=pd.cut(ds2["year"],2,labels=["old","new"],retbins=True)
ds2["year"]=year_data[0]
registry_df_year_bins_param=year_data[1]
params["registry_df_year_bins_param"]=registry_df_year_bins_param
ds2["year"]=ds2["year"].astype('object')


pag_data=pd.cut(ds2["purchase_amount"],bin_number,labels=[0,1,2,3,4,5,6,7,8,9],retbins=True)
ds2["purchase_amount_group"]=pag_data[0]
registry_df_purchase_amount_bins_param=pag_data[1]
params["registry_df_purchase_amount_bins_param"]=registry_df_purchase_amount_bins_param
ds2["purchase_amount_group"]=ds2["purchase_amount_group"].astype('object')

nmpa_data=pd.cut(ds2["next_month_purchase_amount"],bins=params["registry_df_purchase_amount_bins_param"],labels=[0,1,2,3,4,5,6,7,8,9],retbins=True)
ds2["next_month_purchase_amount_group"]=nmpa_data[0]
registry_df_next_month_purchase_amount_amount_bins_param=nmpa_data[1]
params["registry_df_next_month_purchase_amount_amount_bins_param"]=registry_df_next_month_purchase_amount_amount_bins_param
ds2["next_month_purchase_amount_group"]=ds2["next_month_purchase_amount_group"].astype('object')




#now i will convert categrocial type for newly created columns such as gender,age,year
# some algos dont need to converted ones  but i will do
#i will use famous LabelEncoder lib

from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for column in ds2.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    ds2[column] = le.fit_transform(ds2[column])
    label_encoders[column] = le

#i also standardize the purchase_amount and next_month_purchase_amount columns
registry_purchase_amount_mean_param=ds2['purchase_amount'].mean()
registry_purchase_amount_std_param=ds2['purchase_amount'].std()
ds2['purchase_amount'] = (ds2['purchase_amount'] - registry_purchase_amount_mean_param) / registry_purchase_amount_std_param

registry_nextm_purchase_amount_mean_param=ds2['next_month_purchase_amount'].mean()
registry_nextm_purchase_amount_std_param=ds2['next_month_purchase_amount'].std()
ds2['next_month_purchase_amount'] = (ds2['next_month_purchase_amount'] - registry_nextm_purchase_amount_mean_param) / registry_nextm_purchase_amount_std_param
###-=model registry add params for ds2['purchase_amount'].mean(),ds2['purchase_amount'].std() into model metadata
###-=model registry add params for ds2['next_month_purchase_amount'].mean()),ds2['next_month_purchase_amount'].std() into model metadata



# Now our dataset ready to train iwll use 2 different y valeus for 2 type ml model
#model1 will use regression to predict next_month_purchase_amount
#model2 will use classfier to predict next_month_purchase_amount_group which is binned before
#but our x train values are same for 2 model

##step6-) prepare train and test sets for train
ds2.reset_index(drop=True, inplace=True)

X = ds2.drop(['next_month_purchase_amount','next_month_purchase_amount_group','customer_id'], axis=1)
X=X.astype("float")
y1 = ds2['next_month_purchase_amount']
y2 = ds2['next_month_purchase_amount_group']


##step7-) train model1 evaluate and save
#model1 - regression uses next_month_purchase_amount column as a target variable

with mlflow.start_run() as run:
    mlflow.set_tag('mlflow.runName', '_Model1_AyhanT_Hiring_Task_'+str(unique_id))

    X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=42)
    
    regmodel = RandomForestRegressor(n_estimators=100, random_state=42)
    regmodel.fit(X_train, y_train)
    y_pred = regmodel.predict(X_test)
    signature = infer_signature(X_test, y_pred)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    mlflow.log_metrics({"mse": mse})
    
    R2 = r2_score(y_test, y_pred)
    print(f'R2 score: {R2}')
    mlflow.log_metrics({"R2 score": R2})
    
    filename = 'finalized_model_regression.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(regmodel, file)
    
    mlflow.log_params(params)
    #log the model and register
    #you can serve this model with this
    #mlflow models serve -m sklearn-model-regression\ --env-manager local
    
    mlflow.sklearn.log_model(
        sk_model=regmodel,
        artifact_path="sklearn-model-regression",
        signature=signature,
        registered_model_name="model1_random_forest_regression",
    )
        
    
##step8-) train model2 evaluate and save
#model2 - classifier uses next_month_purchase_amount_group column as a target variable
 
with mlflow.start_run() as run:
    mlflow.set_tag('mlflow.runName', '_Model2_AyhanT_Hiring_Task_'+str(unique_id))
    y2 = np.eye(bin_number)[y2]
    X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(max_depth=1000, random_state=0)
    # multi_target_rf = MultiOutputClassifier(rf, n_jobs=-1)
    # multi_target_rf.fit(X_train, y_train)
    clf_extra_tree = ExtraTreesClassifier(n_estimators=16)
    clf_extra_tree.fit(X_train, y_train)

    y_pred = clf_extra_tree.predict(X_test)
    print(classification_report(y_test, y_pred))
    signature = infer_signature(X_test, y_pred)
    mlflow.log_params(params)
        
    mcm=multilabel_confusion_matrix(y_test, y_pred)
    #mlflow.log_metrics({"multilabel_confusion_matrix": mcm})
    # for i, cm in enumerate(mcm):
    #     print(f"Confusion matrix for label {i}:")
    #     mlflow.log_metrics({f"Confusion matrix for label {i}": str(cm)})
    #     print(cm)
    
    
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    print("precision:",precision)
    print("recall:",recall)
    print("f1:",f1)
    mlflow.log_metrics({"precision": precision.mean(),"recall":recall.mean(),"f1":f1.mean()})

    filename = 'finalized_model_multi_target_classifier.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(clf_extra_tree, file)
    
    
    mlflow.log_params(params)
    #log the model and register
    #you can serve this model with this
    #mlflow models serve -m multi-label-finalized_model_regression\ --env-manager local
    
    mlflow.sklearn.log_model(
        sk_model=clf_extra_tree,
        artifact_path="sklearn-extra-tree-model-multi-label-classfier",
        signature=signature,
        registered_model_name="model2_extra_tree_classifier",
    )
    

# # Load the model from disk
# filename = 'finalized_model_multi_target_classifier.pkl'
# with open(filename, 'rb') as file:
#      loaded_model = pickle.load(file)
    
# y_pred = mrf.predict(X_test)


