# import libraries
import boto3, re, sys, math, json, os, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import display
from time import gmtime, strftime
from sagemaker.predictor import csv_serializer

# Define IAM Role
role = get_execution_role()

prefix = 'sagemaker/BankCd-xgboost-dm'
my_region = boto3.session.Session().region_name
bucket_name = 'tcb-bankcd'


# try:
#     model_data = pd.read_csv('./bank_clean.csv',index_col=0)
#     print('Success: Data loaded into dataframe.')
# except Exception as e:
#     print('Data load error: ',e)

# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
xgboost_container = sagemaker.image_uris.retrieve("xgboost", my_region, "latest")

print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + xgboost_container + " container for your SageMaker endpoint.")

# Upload training data into S3 bucket
# boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('/opt/ml/processing/train/train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')


sess = sagemaker.Session()
xgb = sagemaker.estimator.Estimator(xgboost_container,role, instance_count=1, instance_type='ml.m4.xlarge',output_path='s3://{}/{}/output'.format(bucket_name, prefix),sagemaker_session=sess)
xgb.set_hyperparameters(max_depth=5,eta=0.2,gamma=4,min_child_weight=6,subsample=0.8,silent=0,objective='binary:logistic',num_round=100)

# Fit model on training data and deploy model
xgb.fit({'train': s3_input_train})
xgb_predictor = xgb.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')

# Save model as a pickle file to be used in evaluation.py
filename = 'xgb_pred.pkl'
pickle.dump(xgb_predictor, open(filename, 'wb'))
print('TCB XGboost demo model training completed')

