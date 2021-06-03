# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:44:24 2021

@author: Saeed Marzban
"""

# Loading required packages and dependecies
from wavecorr_input import loadInput
from wavecorr_enum import normalization_mode
from wavecorr_agent import agent_manager
import tensorflow as tf
import numpy as np

# Parameters of the portfolio management problem

###############################################################################

dataset_name = 'can' # possible values: can, us, covid
trainSetLen = 2766 # For can and us data sets: 2766, For covid data set: 1967
number_of_stocks = 30 # The number of stocks to be loaded from the assigned data set
tradeFee = 0 # The commission rate
constrain_action = 0 # 0: No constraint on the maximum weight allocation, 1: With constraint
maxStockWeight = 0.25 # The maximum weight allocated to each individual asset in case of constrain_action = 1
network_model = 'waveCorr'# The model to use. Possible values: waveCorr, cs, eiie
restoreSavedModel = False # True: restores a previously saved model
number_of_experiments = 10
train_mode = 1 # 1: Experiments with different seeds
               # 2: Experiments with different permutation of stocks

###############################################################################


# Hyperparameters of the Neural Networks

###############################################################################

epochs = 5000 # The number of epochs
learningRate = 5e-5 # The learning rate of the algorithm
decayRate = 0.9999 # The exponential decay rate
minimmumRate = 1e-5 # The lower bound of the learning rate in case of decayRate < 1
minibatchSize = 1 # The size of each minibatch. The case of minibatchSize = 1 is explained in the paper and provides the highest efficiency of training
lookback_window = 32 # The length of historical window of prices used for training the model
planning_horizon = 32 # The horizon on which the SR is computed
seed = 1111 # Seed value used for initialization of all random values throughout the code
net_depth = 8 # The number of channels in the waveCorr architecture
regularizer_multiplier=1e-6 # Regularization coefficient to penalize the objective function to not overfit
RNN=True # If True, the actions on each period are sent back and used as part of the sate for the next action
keep_prob_value=.5 # The probability ratio in the dropout layer of the model

###############################################################################


# load the data from the assigned data set
data = loadInput(dataset_name,normalization_mode.no_norm,number_of_stocks)


# Defining the functions that run different experiments mentioned in the paper

###############################################################################

# Running a single experiment
def runSingleExperiment(seed):
    agent = agent_manager(data.x,data.y,learningRate,decayRate,minimmumRate,
                         minibatchSize,epochs,number_of_stocks,lookback_window,planning_horizon,
                         tradeFee,seed,constrain_action=False,maxStockWeight=maxStockWeight,network_model=network_model,
                         net_depth=net_depth,regularizer_multiplier=regularizer_multiplier,RNN=RNN,keep_prob_value=keep_prob_value)
            
    with tf.compat.v1.Session() as sess:
        agent.train_test(sess, data.x, data.y, data.x_dates, trainSetLen, saveModel=False,
                           planning_horizon=planning_horizon, number_of_stocks=number_of_stocks,
                           restoreSavedModel=restoreSavedModel,off_policy=False)
    

# Running 10 experiments with different permutation of stocks
def stockPermutationExperiments(seed, number_of_experiments):
    for i in range(number_of_experiments):
        xx = np.zeros(data.x.shape)
        yy = np.zeros(data.y.shape)
    
        rnd_num = np.arange(number_of_stocks)
        np.random.seed(seed + i)
        np.random.shuffle(rnd_num)
        for i in range(number_of_stocks):
            xx[:, i, :] = data.x[:, rnd_num[i], :]
            yy[:, i, :] = data.y[:, rnd_num[i], :]
    
        data.x = xx
        data.y = yy
        
        runSingleExperiment(seed)
        
# Running 10 experiments using different seeds
def differentSeedExperiments(seed, number_of_experiments):
    for i in range(number_of_experiments):
        seed = seed + 1
        runSingleExperiment(seed)
        
###############################################################################


# Running the experiment
if train_mode == 1:
    differentSeedExperiments(seed, number_of_experiments)
elif train_mode == 2:
    stockPermutationExperiments(seed, number_of_experiments)
    
    
    
    
    
    
    
    
    
    
    
    
    
    



