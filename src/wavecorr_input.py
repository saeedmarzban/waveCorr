# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:26:26 2021

@author: ---
"""

import os
import numpy as np
import pandas as pd
from wavecorr_enum import normalization_mode

class loadInput:
    def __init__(self, dataset_name, data_normalization, number_of_stocks):
        
        # Read the file names
        filenames = os.listdir('data/' + dataset_name + '/')
        filenames = ['data/' + dataset_name + '/' + i for i in filenames]
        
        # Load data from csv files
        sample = pd.read_csv(filenames[0])
        d1 = np.shape(sample)[0]
        d2 = np.shape(sample)[1] - 2
        self.x_dates = sample['date']
        self.x = np.empty([d1,0,d2])
        self.y = np.empty([d1,0])
        for filename in filenames[0:number_of_stocks]:
            df = pd.read_csv(filename)
            df_array = np.expand_dims(df,1)
            new_d = np.minimum(self.x.shape[0],df_array.shape[0])
            self.x = np.append(self.x[0:new_d,:,:], df_array[0:new_d,:,1:-1],axis=1)
            self.y = np.append(self.y[0:new_d,:], df_array[0:new_d,:,-1],axis=1)
            
        self.x_dates = self.x_dates[0:new_d]        
        
        if dataset_name == 'can' or dataset_name == 'us':
            self.x = self.x[0:4220,:,:]
            self.y = self.y[0:4220,:]
            self.x_dates = self.x_dates[0:4220]
        else:            
            self.x = self.x[0:-10,:,:]
            self.y = self.y[0:-10,:]
            self.x_dates = self.x_dates[0:-10]
        
        # Normalize the input data
        if data_normalization == normalization_mode.standard_norm:
            x_mean = np.mean(self.x, axis=1, keepdims=True)
            x_std = np.std(self.x.astype(float), axis=1, keepdims=True)
            self.x = np.divide((self.x - x_mean), (x_std) + 1e-8)/10
        elif data_normalization == normalization_mode.capped_norm:
            x_min = np.min(self.x, axis=1, keepdims=True)
            x_max = np.max(self.x, axis=1, keepdims=True)
            self.x = (2*np.divide((self.x - x_min), (x_max - x_min) + 1e-8)-1)/10