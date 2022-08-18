import boto3, re, sys, math, json, os, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import display
from time import gmtime, strftime
from sagemaker.predictor import csv_serializer
import logging

# Define IAM role
role = get_execution_role()
# prefix = 'sagemaker/DEMO-xgboost-dm'
prefix = "/opt/ml/processing"
my_region = boto3.session.Session().region_name # set the region of the instance

# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
# xgboost_container = sagemaker.image_uris.retrieve("xgboost", my_region, "latest")
# print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + xgboost_container + " container for your SageMaker endpoint.")


bucket_name = 'tcb-bankcd' # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
s3 = boto3.resource('s3')
try:
    if  my_region == 'us-east-1':
      s3.create_bucket(Bucket=bucket_name)
    else: 
      s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={ 'LocationConstraint': my_region })
    logging.info('S3 bucket created successfully')
except Exception as e:
    logging.error('S3 error: ',e)

# Download the bank dataset as a csv file.
try:
    urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
    logging.info('Success: Data loaded into dataframe.')
except Exception as e:
  logging.error('Data load error: ',e)

# Read downloaded csv as a dataframe.
try:
    model_data = pd.read_csv('./bank_clean.csv',index_col=0)
    logging.info('Success: Data loaded into dataframe.')
except Exception as e:
    logging.error('Data load error ', e)

# Create train and test data sets and upload data sets to AWS directory
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)
pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv(f"{prefix}/train/train.csv", index=False, header=False)
pd.concat([test_data['y_yes'], test_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv(f"{prefix}/test/test.csv", index=False, header=False)

logging.info('Success: pre-processed bank_clean.csv.')

# Upload training data into S3 bucket
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('/opt/ml/processing/train/train.csv')
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/test.csv')).upload_file('/opt/ml/processing/test/test.csv')
