# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:28:00 2021

@author: laced
"""


import boto3
import os
import numpy as np
import pandas as pd


# USA and CA data

# path = 'share/FinML/usa/preprocessedData/selected/'
# path = 'share/FinML/canada/preprocessedData/newData/'

# client =  boto3.client('s3',aws_access_key_id="AKIASU6LMKM3SGCWBRND", aws_secret_access_key="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")
# resource =  boto3.resource('s3',aws_access_key_id="AKIASU6LMKM3SGCWBRND", aws_secret_access_key="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")
# assets = client.list_objects(Bucket='evovest', Prefix=path,Delimiter='/')
# assets = [os.path.basename(i['Key']) for i in assets['Contents']]
# assets = assets[1:]

# for filename in assets:
#     obj = client.get_object(Bucket='evovest', Key=path + filename)
#     df = pd.read_csv(obj['Body'])
#     try:
#         name = df['gticker'][10]
#         eid = df['eid'][10]
#         print(name + ' ... ' + str(eid) + ' ... ' + str(np.shape(df)))
#         df = df[['date','E_D_Tech_price_chg_1d','return_close']]
#         df.columns = ['date', 'x_close', 'y_close']
#         df.to_csv('data/can/' + name + '.csv',sep=',', index=False)
#     except:
#         print('not saved')
        
        

# COVID data

path = 'share/FinML/canada/preprocessedData/tsx/'
client =  boto3.client('s3',aws_access_key_id="AKIASU6LMKM3SGCWBRND", aws_secret_access_key="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")
resource =  boto3.resource('s3',aws_access_key_id="AKIASU6LMKM3SGCWBRND", aws_secret_access_key="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")
assets = client.list_objects(Bucket='evovest', Prefix=path,Delimiter='/')
assets = [os.path.basename(i['Key']) for i in assets['Contents']]
assets = assets[1:]

for filename in assets:
    obj = client.get_object(Bucket='evovest', Key=path + filename)
    df = pd.read_csv(obj['Body'])
    try:
        name = filename.split('.')[0]
        print(name + ' ... ' + str(np.shape(df)))
        df = df[['Date','Open','High','Low','Close']]
        df.columns = ['date','x_open','x_high','x_low','x_close']
        
        # Compute returns from prices
        df['x_open'] = df['x_open'].shift(-1)/df['x_close'] - 1
        df['x_high'] = df['x_high'].shift(-1)/df['x_close'] - 1
        df['x_low'] = df['x_low'].shift(-1)/df['x_close'] - 1
        df['x_close'] = df['x_close'].shift(-1)/df['x_close'] - 1
        
        # # Compute y values
        df['y_close'] = df['x_close'].shift(-2) + 1
        
        df.to_csv('data/covid/' + name + '.csv',sep=',', index=False)
    except:
        print('not saved')