# -*- coding: utf-8 -*-
"""
Created on Mon May 31 13:28:29 2021

@author: Saeed Marzban
"""

import numpy as np
import tensorflow as tf
from random import randint
from keras import regularizers
from wavecorr_enum import activation_functions


class neuralNetwork:
    
    def __init__(self,seed,minibatchSize,lookback_window,planning_horizon,number_of_stocks,net_depth,RNN,tradeFee,scope,regularizer_multiplier):
        self.seed = seed
        self.regularizer_multiplier = regularizer_multiplier        
        self.minibatchSize = minibatchSize
        self.lookback_window = lookback_window
        self.planning_horizon = planning_horizon 
        self.number_of_stocks = number_of_stocks
        self.RNN = RNN
        self.scope = scope
        self.net_depth = net_depth
        self.name_counter = 0
        self.tradeFee = tradeFee
    
    def initializeNN(self,filter_shape,name='',trainable=True,reuse=False):
        with tf.name_scope(self.scope):
            with tf.compat.v1.variable_scope(self.scope, reuse=reuse):
                
                # Assign name to the NN parameters
                self.name_counter += 1
                if name == '':
                    name = 'v_' + str(self.name_counter)

                # Weight initialization method of Xavier
                stddev = np.sqrt(1 / (filter_shape[0] * filter_shape[1] * (filter_shape[2] + filter_shape[3])))

                
                # Initialization according to assigned seed
                if self.seed != 0:
                    seed = self.seed
                else:
                    seed = np.random.randint(0,1e8)
                
                # Define the parameter
                par = tf.compat.v1.get_variable(shape=filter_shape,
                                                 initializer=tf.compat.v1.truncated_normal_initializer(mean=0, stddev=stddev,seed=seed),
                                                 regularizer=tf.keras.regularizers.l2(self.regularizer_multiplier),
                                                 trainable=trainable,
                                                 name=name)

                return par

   

    
    def causalConv(self,inputTensor,filter_shape,name='',trainable=True,reuse=False,padding='VALID',activation=activation_functions.relu):
        with tf.name_scope(self.scope):
            
            W = self.initializeNN(filter_shape,name=name,trainable=trainable,reuse=reuse)
            b = tf.Variable(tf.zeros([filter_shape[3]]), trainable=trainable,name='bias')
            out_layer = tf.nn.conv2d(inputTensor, W, [1, 1, 1, 1], padding=padding)
            x = tf.nn.bias_add(out_layer, b)

            if activation==activation_functions.relu:
                x = tf.nn.relu(x)
            elif activation==activation_functions.sigmoid:
                x = tf.nn.sigmoid(x)
            elif activation==activation_functions.tanh:
                x = tf.nn.tanh(x)

            return x
    
    
    def dilatedCausalConv(self,inputTensor,filter_shape,name='',trainable=True,reuse=False,padding='VALID',activation=activation_functions.relu,dilation=1):
        with tf.name_scope(self.scope):
            
            W = self.initializeNN(filter_shape,name=name,trainable=trainable,reuse=reuse)
            b = tf.Variable(tf.zeros([filter_shape[3]]), trainable=trainable,name='bias')
            out_layer = tf.nn.atrous_conv2d(inputTensor, W, dilation, padding=padding)
            x = tf.nn.bias_add(out_layer, b)

            if activation==activation_functions.relu:
                x = tf.nn.relu(x)
            elif activation==activation_functions.sigmoid:
                x = tf.nn.sigmoid(x)
            elif activation==activation_functions.tanh:
                x = tf.nn.tanh(x)

            return x

    def eiie(self, inputTensor, rates):
        with tf.name_scope(self.scope):
            is_train = tf.compat.v1.placeholder(tf.bool, name="is_train")
            keep_prob = tf.compat.v1.placeholder(tf.float32)
            mIn = tf.compat.v1.placeholder(tf.float32,[self.minibatchSize, self.lookback_window + self.planning_horizon - 1, self.number_of_stocks, inputTensor.shape[2]])
            wIn = tf.compat.v1.placeholder(tf.float32,[self.minibatchSize, self.lookback_window + self.planning_horizon - 1, self.number_of_stocks])
            wIn_exp = tf.expand_dims(wIn, 3)
            y = tf.compat.v1.placeholder(tf.float32, [self.minibatchSize, self.planning_horizon, self.number_of_stocks])
            y_exp = tf.expand_dims(y, 3)

            x = mIn
            
            x = self.causalConv(x,filter_shape = (3,1,x._shape_as_list()[3],2))

            x = self.causalConv(x,filter_shape = (self.lookback_window-2,1,x._shape_as_list()[3],20))
         
            x = tf.concat([x, wIn_exp[:, -self.planning_horizon:, :, :]], 3)
            
            x = self.causalConv(x,filter_shape = (1,1,x._shape_as_list()[3],1),activation=activation_functions.linear)
            
            # Decision making module
            ###################################################################
            
            action = tf.nn.softmax(x, axis=2)            
            mu_norm = tf.reduce_sum(tf.abs(tf.subtract(action, wIn_exp[:, -self.planning_horizon:, :, :])), axis=2)
            mu = 1 - mu_norm * self.tradeFee
            ret_per_stock = tf.multiply(action, y_exp)
            cum_ret = tf.reduce_sum(ret_per_stock, axis=2, keepdims=False)
            cum_ret_tradefee = tf.multiply(cum_ret, mu)
            reward = tf.log(cum_ret_tradefee)
            return mIn, wIn, y, tf.squeeze(action, 3), reward, keep_prob, is_train
        
    def corrLayer(self,inputTensor,name='',trainable=True,reuse=False,padding='VALID',activation=activation_functions.relu,keep_prob=.5):
        
        self.initializeNN([1, self.number_of_stocks + 1, inputTensor._shape_as_list()[3], 1],name=name,trainable=trainable,reuse=False)

        for iii in range(self.number_of_stocks):
            if iii == 0:
                xx = tf.concat([tf.expand_dims(inputTensor[:,:,iii,:],2),inputTensor],axis=2)
                out = self.causalConv(xx,filter_shape = (1, self.number_of_stocks+1,xx._shape_as_list()[3],1),name=name,reuse=True)

            else:
                xx = tf.concat([tf.expand_dims(inputTensor[:,:,iii,:],2),inputTensor],axis=2)
                x_corr = self.causalConv(xx,filter_shape = (1, self.number_of_stocks+1,xx._shape_as_list()[3],1),name=name,reuse=True)
                out = tf.concat([out, x_corr], 2)
        
        
        outputTensor = tf.concat([inputTensor, out], 3)
        return outputTensor 
    
    
    def waveCorrBlock(self,x,kernel_size,depth,dilation,keep_prob,block_counter):
        
        x_short = tf.zeros(shape = x._shape_as_list())
        
        if x._shape_as_list()[1] - self.planning_horizon >= 4 * dilation:        
            x_short = x
            names = ['wave_1_'+str(block_counter),'corr_1_'+str(block_counter),'wave_2_'+str(block_counter),'corr_2_'+str(block_counter),'short_'+str(block_counter)]
                
            x = self.dilatedCausalConv(x,filter_shape=(kernel_size,1,x._shape_as_list()[3],depth),name=names[0],dilation=dilation)
            x = tf.nn.dropout(x, rate=1 - keep_prob, seed=self.seed)
            
            # x = self.corrLayer(x,names[1],keep_prob=keep_prob)
            
            x = self.dilatedCausalConv(x,filter_shape=(kernel_size,1,x._shape_as_list()[3],int(self.net_depth * 1)),name=names[2],dilation=dilation)
            x = tf.nn.dropout(x, rate=1 - keep_prob, seed=self.seed)
    
            x = self.corrLayer(x,names[3],keep_prob=keep_prob)
            
            # The 1*1 residual connection
            ###################################################################
            xx = self.causalConv(x_short[:, int(4*dilation):, :, :],filter_shape = (1,1,x_short._shape_as_list()[3],x._shape_as_list()[3]),name=names[4],activation=activation_functions.linear)
            x = x + xx
            x = tf.nn.relu(x)
            
        return x , x_short
    
    
    def waveCorr(self, inputTensor, rates):
        with tf.name_scope(self.scope):
            is_train = tf.compat.v1.placeholder(tf.bool, name="is_train")
            keep_prob = tf.compat.v1.placeholder(tf.float32)
            mIn = tf.compat.v1.placeholder(tf.float32,
                                           [self.minibatchSize, self.lookback_window + self.planning_horizon - 1, self.number_of_stocks, inputTensor.shape[2]])
            wIn = tf.compat.v1.placeholder(tf.float32,
                                           [self.minibatchSize, self.lookback_window + self.planning_horizon - 1, self.number_of_stocks])
            wIn_exp = tf.expand_dims(wIn, 3)
            y = tf.compat.v1.placeholder(tf.float32, [self.minibatchSize, self.planning_horizon, self.number_of_stocks])
            y_exp = tf.expand_dims(y, 3)

            kernel_size = 3

            x = mIn            
            
            # waveCorr blocks (Depending on the lookback window, the number of active blocks change. For a lookback window of 32 days, only 3 blocks are activated)
            ###################################################################
            
            x, x_short1 = self.waveCorrBlock(x,kernel_size,self.net_depth*1,dilation=1,keep_prob=keep_prob,block_counter=1)            
            x, x_short2 = self.waveCorrBlock(x,kernel_size,self.net_depth*2,dilation=2,keep_prob=keep_prob,block_counter=2)            
            x, x_short3 = self.waveCorrBlock(x,kernel_size,self.net_depth*2,dilation=4,keep_prob=keep_prob,block_counter=3)            
            x, x_short4 = self.waveCorrBlock(x,kernel_size,self.net_depth*2,dilation=4,keep_prob=keep_prob,block_counter=4)            
            x, x_short5 = self.waveCorrBlock(x,kernel_size,self.net_depth*2,dilation=4,keep_prob=keep_prob,block_counter=5)
            
            ###################################################################
            
            # The causal convolution layer to adjust the receptive field to the length of the lookback window
            ###################################################################
            if x._shape_as_list()[1] > self.planning_horizon:
                last_conv = x._shape_as_list()[1] - self.planning_horizon + 1
                x = self.causalConv(x,filter_shape = (last_conv,1,x._shape_as_list()[3],int(self.net_depth * 2)),name='conv',activation=activation_functions.relu)
            
            ###################################################################
            
            # Skip connections to prevent gradient vanishing/explosion
            ###################################################################
            x_short1 = self.causalConv(x_short1[:, -x._shape_as_list()[1]:, :, :],filter_shape = (1,1,x_short1._shape_as_list()[3],x._shape_as_list()[3]),activation=activation_functions.linear)
            x_short2 = self.causalConv(x_short2[:, -x._shape_as_list()[1]:, :, :],filter_shape = (1,1,x_short2._shape_as_list()[3],x._shape_as_list()[3]),activation=activation_functions.linear)
            x_short3 = self.causalConv(x_short3[:, -x._shape_as_list()[1]:, :, :],filter_shape = (1,1,x_short3._shape_as_list()[3],x._shape_as_list()[3]),activation=activation_functions.linear)
            x_short4 = self.causalConv(x_short4[:, -x._shape_as_list()[1]:, :, :],filter_shape = (1,1,x_short4._shape_as_list()[3],x._shape_as_list()[3]),activation=activation_functions.linear)
            x_short5 = self.causalConv(x_short5[:, -x._shape_as_list()[1]:, :, :],filter_shape = (1,1,x_short5._shape_as_list()[3],x._shape_as_list()[3]),activation=activation_functions.linear)
            
            x = x + x_short1 + x_short2 + x_short3 + x_short4 + x_short5
            x = tf.nn.relu(x)
            
            
            # Decision making module (if RNN = True the action on each period are considered as part of the state for the next period action)
            ###################################################################
            if self.RNN == False:
                x = tf.concat([x, wIn_exp[:, -self.planning_horizon:, :, :]], 3)
                x_action = self.causalConv(x,filter_shape = (1,1,x._shape_as_list()[3],1),name='',activation=activation_functions.linear)
                action = tf.nn.softmax(x_action, axis=2)
                mu_norm = tf.reduce_sum(tf.abs(tf.subtract(action, wIn_exp[:, -self.planning_horizon:, :, :])), axis=2)
            else:
                self.initializeNN([1, 1, x._shape_as_list()[3] + 1, 1],name='action_var',reuse=False)                
                wIn_exp = wIn_exp[:, -self.planning_horizon:, :, :]
                wIn_RNN = tf.expand_dims(wIn_exp[:, 0, :, :], 1)
                action = 0
                for i in range(self.planning_horizon):
                    output_temp = tf.expand_dims(x[:, i, :, :], 1)
                    output_temp = tf.concat([output_temp, wIn_RNN], 3)
                    output_temp = self.causalConv(output_temp,filter_shape = (1,1,output_temp._shape_as_list()[3],1),name='action_var',activation=activation_functions.linear, reuse=True)                    

                    output_temp = tf.nn.softmax(output_temp, axis=2)
                    if action == 0:
                        action = output_temp
                    else:
                        action = tf.concat([action, output_temp], 1)
                    wIn_RNN = output_temp

                
                wPrime = action * y_exp / tf.reduce_sum(action * y_exp, axis=2, keepdims=True)
                wPrime = tf.concat([wIn_exp[:, -self.planning_horizon:-self.planning_horizon + 1, :, :], wPrime[:, 0:-1, :, :]],axis=1)
                mu_norm = tf.reduce_sum(tf.abs(tf.subtract(action, wPrime)), axis=2)                
                
            mu = 1 - mu_norm * self.tradeFee
            ret_per_stock = tf.multiply(action, y_exp)
            cum_ret = tf.reduce_sum(ret_per_stock, axis=2, keepdims=False)
            cum_ret_tradefee = tf.multiply(cum_ret, mu)
            reward = tf.math.log(cum_ret_tradefee)
            
            ###################################################################

            return mIn, wIn, y, tf.squeeze(action, 3), reward, keep_prob, is_train

    def cs(self, inputTensor, rates):
        with tf.name_scope(self.scope):            
            is_train = tf.compat.v1.placeholder(tf.bool, name="is_train")
            keep_prob = tf.compat.v1.placeholder(tf.float32)
            mIn = tf.compat.v1.placeholder(tf.float32,
                                           [self.minibatchSize, self.lookback_window + self.planning_horizon - 1, self.number_of_stocks, inputTensor.shape[2]])
            wIn = tf.compat.v1.placeholder(tf.float32,
                                           [self.minibatchSize, self.lookback_window + self.planning_horizon - 1, self.number_of_stocks])
            wIn_exp = tf.expand_dims(wIn, 3)
            y = tf.compat.v1.placeholder(tf.float32, [self.minibatchSize, self.planning_horizon, self.number_of_stocks])
            y_exp = tf.expand_dims(y, 3)

            x1 = self.sequential_net(mIn)
            x2 = self.correlation_net(mIn, keep_prob, is_train)
            x = tf.concat([x1, x2], 3)
            
            # Decision making module            
            ###################################################################
            
            x = tf.concat([x, wIn_exp[:, -self.planning_horizon:, :, :]], 3)
            x_action = self.causalConv(x,filter_shape = (1,1,x._shape_as_list()[3],1),name='',activation=activation_functions.linear)
            action = tf.nn.softmax(x_action, axis=2)
            mu_norm = tf.reduce_sum(tf.abs(tf.subtract(action, wIn_exp[:, -self.planning_horizon:, :, :])), axis=2)
            
            mu = 1 - mu_norm * self.tradeFee
            ret_per_stock = tf.multiply(action, y_exp)
            cum_ret = tf.reduce_sum(ret_per_stock, axis=2, keepdims=False)
            cum_ret_tradefee = tf.multiply(cum_ret, mu)
            reward = tf.log(cum_ret_tradefee)

            return mIn, wIn, y, tf.squeeze(action, 3), reward, keep_prob, is_train

    def lstm_cell(self, mIn, unit_num):
        with tf.name_scope(self.scope):
            x = mIn
            unit_num = str(unit_num)
            outputchannels = 16
            
            self.initializeNN([1, 1, x._shape_as_list()[3] + outputchannels, outputchannels],name='forgetGate_' + unit_num,reuse=False)
            
            self.initializeNN([1, 1, x._shape_as_list()[3] + outputchannels, outputchannels],name='inputGate_' + unit_num,reuse=False)
            
            self.initializeNN([1, 1, x._shape_as_list()[3] + outputchannels, outputchannels],name='outputGate_' + unit_num,reuse=False)
            
            self.initializeNN([1, 1, x._shape_as_list()[3] + outputchannels, outputchannels],name='lstmGate_' + unit_num,reuse=False)
            
            H = 0

            for i in range(self.lookback_window + self.planning_horizon - 1):
                input = x[:, i, :, :]
                input = tf.expand_dims(input, axis=1)
                if i ==0:
                    c_prev = tf.Variable(tf.zeros([self.minibatchSize, 1, self.number_of_stocks, outputchannels]),
                                         trainable=False,
                                         name='c_prev' + unit_num)
                    h_prev = tf.Variable(tf.zeros([self.minibatchSize, 1, self.number_of_stocks, outputchannels]),
                                         trainable=False,
                                         name='h_prev' + unit_num)


                input = tf.concat([input, h_prev], 3)
                
                ft = self.causalConv(input,filter_shape = (1, 1,input._shape_as_list()[3],outputchannels),name='forgetGate_'+unit_num,reuse=True)
                
                it = self.causalConv(input,filter_shape = (1, 1,input._shape_as_list()[3],outputchannels),name='inputGate_'+unit_num,reuse=True)
                
                ct = self.causalConv(input,filter_shape = (1, 1,input._shape_as_list()[3],outputchannels),name='outputGate_'+unit_num,reuse=True,activation=activation_functions.linear)
                
                ot = self.causalConv(input,filter_shape = (1, 1,input._shape_as_list()[3],outputchannels),name='lstmGate_'+unit_num,reuse=True)
                
                ct = tf.nn.tanh(ct)

                ft = tf.multiply(ft, c_prev)
                ct = tf.multiply(ct, it)
                c_prev = tf.add(ct, ft)
                ht = tf.nn.tanh(ct)
                h_prev = tf.multiply(ht, ot)

                if H == 0:
                    H = h_prev
                else:
                    H = tf.concat([H, h_prev], 1)

            return H

    def sequential_net(self, mIn):
        with tf.name_scope(self.scope):
            x = mIn

            x = self.lstm_cell(x, 1)

            x = x[:, -self.planning_horizon:, :, :]

            return x

    def correlation_net(self, mIn,keep_prob,is_train):
        with tf.name_scope(self.scope):

            kernel_size = 3

            x = mIn

            x_short1 = x
            
            x = self.dilatedCausalConv(x,filter_shape=(kernel_size,1,x._shape_as_list()[3],int(self.net_depth * 1)),name='wave1',dilation=1)
            x = tf.nn.dropout(x, rate=1 - keep_prob, seed=self.seed)
            
            x = self.dilatedCausalConv(x,filter_shape=(kernel_size,1,x._shape_as_list()[3],int(self.net_depth * 1)),name='wave2',dilation=1)
            x = tf.nn.dropout(x, rate=1 - keep_prob, seed=self.seed)

            x = self.causalConv(x,filter_shape = (1,self.number_of_stocks,x._shape_as_list()[3],x._shape_as_list()[3]),name='',padding='SAME')
            
            xx = self.causalConv(x_short1[:, 4:, :, :],filter_shape = (1,1,x_short1._shape_as_list()[3],x._shape_as_list()[3]),name='short1',activation=activation_functions.linear)
            x = x + xx
            x = tf.nn.relu(x)
                

            x_short3 = x
            x = self.dilatedCausalConv(x,filter_shape=(kernel_size,1,x._shape_as_list()[3],int(self.net_depth * 2)),name='wave3',dilation=2)
            x = tf.nn.dropout(x, rate=1 - keep_prob, seed=self.seed)
            
            x = self.dilatedCausalConv(x,filter_shape=(kernel_size,1,x._shape_as_list()[3],int(self.net_depth * 2)),name='wave4',dilation=2)
            x = tf.nn.dropout(x, rate=1 - keep_prob, seed=self.seed)

            x = self.causalConv(x,filter_shape = (1,self.number_of_stocks,x._shape_as_list()[3],x._shape_as_list()[3]),name='',padding='SAME')
            
            xx = self.causalConv(x_short3[:, 8:, :, :],filter_shape = (1,1,x_short3._shape_as_list()[3],x._shape_as_list()[3]),name='short3',activation=activation_functions.linear)
            x = x + xx
            x = tf.nn.relu(x)

            last_conv = self.lookback_window - 12
            
            if self.lookback_window > 28:
                
                last_conv = self.lookback_window - 28
                
                x_short5 = x
                
                x = self.dilatedCausalConv(x,filter_shape=(kernel_size,1,x._shape_as_list()[3],int(self.net_depth * 2)),name='wave5',dilation=4)
                x = tf.nn.dropout(x, rate=1 - keep_prob, seed=self.seed)
                
                x = self.dilatedCausalConv(x,filter_shape=(kernel_size,1,x._shape_as_list()[3],int(self.net_depth * 2)),name='wave6',dilation=4)
                x = tf.nn.dropout(x, rate=1 - keep_prob, seed=self.seed)
    
                x = self.causalConv(x,filter_shape = (1,self.number_of_stocks,x._shape_as_list()[3],x._shape_as_list()[3]),name='',padding='SAME')
                                
                xx = self.causalConv(x_short5[:, 16:, :, :],filter_shape = (1,1,x_short5._shape_as_list()[3],x._shape_as_list()[3]),name='short5',activation=activation_functions.linear)
                x = x + xx
                x = tf.nn.relu(x)            
            

            
            x = self.causalConv(x,filter_shape = (last_conv,1,x._shape_as_list()[3],int(self.net_depth * 2)),name='conv',activation=activation_functions.relu)
            
            x = x[:, -self.planning_horizon:, :, :]

            return x