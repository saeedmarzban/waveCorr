# -*- coding: utf-8 -*-
"""
Created on Mon May 31 13:19:19 2021

@author: Anonymous
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
import datetime
from wavecorr_network import neuralNetwork


class agent_manager:
    def __init__(self,dataset_name,x,y,learningRate,decayRate,minimmumRate,
                 minibatchSize,epochs,number_of_stocks,lookback_window,planning_horizon,
                 tradeFee,seed,constrain_action=False,maxStockWeight=.25,network_model='eiie',
                 net_depth=4,regularizer_multiplier=1e-6,RNN=True,keep_prob_value=.5):

        self.dataset_name = dataset_name
        self.learningRate = np.array([learningRate])
        self.decayRate = decayRate
        self.minimmumRate = minimmumRate
        self.minibatchSize = minibatchSize
        self.minibatchCount = planning_horizon - 2
        self.epochs = epochs
        self.number_of_stocks = number_of_stocks
        self.lookback_window = lookback_window
        self.planning_horizon = planning_horizon
        self.tradeFee = tradeFee
        self.seed = seed
        self.constrain_action = constrain_action
        self.maxStockWeight = maxStockWeight
        self.network_model = network_model
        self.net_depth = net_depth
        self.regularizer_multiplier = regularizer_multiplier
        self.RNN = RNN       
        self.reset()
        self.APV=np.array([1])
        self.weightsMatrix = np.array([self.weights])
        self.keep_prob_value = keep_prob_value
        tf.compat.v1.reset_default_graph()

        if self.seed != 0:
            tf.compat.v1.random.set_random_seed(self.seed)

        self.initPvm(y)
        self.scope = 'online_actor'
        self.NN = neuralNetwork(seed,minibatchSize,lookback_window,planning_horizon,number_of_stocks,net_depth,RNN,tradeFee,self.scope,regularizer_multiplier)        
        self.onlineActor(x, y)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=4)
        self.init_op = tf.compat.v1.global_variables_initializer()
    

    def actor(self, inputTensor,rates):
        if self.network_model == 'waveCorr':
            mIn, wIn, y, actions_lst, reward, keep_prob, is_train = self.NN.waveCorr(inputTensor, rates)
        elif self.network_model == 'eiie':
            mIn, wIn, y, actions_lst, reward, keep_prob, is_train = self.NN.eiie(inputTensor, rates)
        elif self.network_model == 'cs_LSTM_CNN':
            mIn, wIn, y, actions_lst, reward, keep_prob, is_train = self.NN.cs_LSTM_CNN(inputTensor, rates)
        elif self.network_model == 'cs_CNN':
            mIn, wIn, y, actions_lst, reward, keep_prob, is_train = self.NN.cs_CNN(inputTensor, rates)
            
        return mIn, wIn, y, actions_lst, reward, keep_prob, is_train
            

    def onlineActor(self, inputTensor,rates):
        with tf.name_scope(self.scope):

            self.mIn, self.wIn, self.y, self.actions_lst, self.reward_lst, self.keep_prob, self.is_train = self.actor(inputTensor, rates)

            if self.planning_horizon == 1:
                axis_value = 0
            else:
                axis_value = 1
                

            if self.constrain_action == True:
                M = 1000
                loss_action = M * tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.abs(self.actions_lst) - self.maxStockWeight,0), axis=2),axis=axis_value)
            else:
                loss_action = 0

            lossL2 = tf.reduce_sum(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
            mean_reward = tf.reduce_mean(self.reward_lst, axis=axis_value)
            std_reward = tf.math.reduce_std(self.reward_lst, axis=axis_value)
            profit = tf.divide(mean_reward, std_reward) - lossL2 - loss_action            
            self.loss = -profit
            
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.lr = tf.compat.v1.placeholder(tf.float32)
            self.global_step_value = tf.compat.v1.placeholder(tf.float32)
            optim_adam = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr,beta1=0.9, beta2=0.999)
            gvs = optim_adam.compute_gradients(self.loss)
            self.actorOptimiser_adam = optim_adam.apply_gradients(gvs, global_step=self.global_step) 

    

    def reset(self):
        #the weight in the last period
        self.weights = np.array([1/self.number_of_stocks for i in range(self.number_of_stocks)])
        #the portfolio value so far
        self.value = 1.0

    def initPvm(self, rates):
        #initializing portfolio vector memory for all periods
        self.pvm = np.array([[1/self.number_of_stocks for i in range(self.number_of_stocks)] for j in (rates)])

        #initializing portflio immediate reward
        self.pvm_reward = np.array([1. for j in (rates)])
    

    def updateRateShift(self, prevWeights, rates):
        reward=np.multiply(prevWeights,rates)
        wPrime=np.divide(reward,np.sum(reward,axis=2,keepdims=True))
        return wPrime


    def calculateMu(self, wPrime, w):
        mu_1 = np.zeros(shape=[w.shape[0],w.shape[1],1])
        mu_2 = np.ones(shape=[w.shape[0],w.shape[1],1])
        epsilon = 1e-10
        while np.max(mu_2 - mu_1) > epsilon:
            mu = (mu_1+mu_2)/2
            f = 1 - (self.tradeFee * np.sum(np.maximum(wPrime - mu * w, 0), axis=2)) - self.tradeFee * np.sum(
                np.maximum(mu * w - wPrime, 0), axis=2)
            f = mu - np.expand_dims(f,axis=2)
            mu_1 = np.multiply((f > 0),mu_1) + np.multiply((f <= 0), mu)
            mu_2 = np.multiply(mu_2,(f <= 0)) + np.multiply((f > 0),mu)
        mu = np.squeeze(mu,axis=2)
        return mu

    def trainOnMinibatch_PG(self,sess, index, inTensor,rates,do_backprop,is_train=True,trainSetLen=100,off_policy=False):

        win_batch = self.lookback_window + self.planning_horizon - 1
        
        mIn = np.array([inTensor[index[i]:index[i] + win_batch, :, :] for i in range(len(index))])
        y = np.array([rates[index[i]+self.lookback_window - 1:index[i] + win_batch, :] for i in range(len(index))])

        wIn = np.array([self.pvm[index[i] - 1:index[i] + win_batch-1] for i in range(len(index))])
        wPrime = self.updateRateShift(wIn, np.array([rates[index[i] - 1:index[i] + win_batch-1] for i in range(len(index))]))
        
        if (do_backprop == False):
            [actorAction,reward] = sess.run([self.actions_lst,self.reward_lst],
                                   feed_dict={self.mIn: mIn,
                                              self.y: y,
                                              self.wIn: wPrime,
                                              self.keep_prob: 1,
                                              self.is_train: False})
            
            mu = self.calculateMu(wPrime[:, -self.planning_horizon:, :], actorAction)
            ret_per_stock = np.multiply(actorAction, y)
            cum_ret = np.sum(ret_per_stock, axis=2, keepdims=False)
            cum_ret_tradefee = np.multiply(cum_ret, mu)
            reward = cum_ret_tradefee - 1

            for i in range(len(index)):
                self.pvm[index[i] + self.lookback_window - 1:index[i] + win_batch] = actorAction[i]
                self.pvm_reward[index[i] + self.lookback_window - 1:index[i] + win_batch] = reward[i,:] + 1

        else:
            temp = sess.run(self.global_step)
            if self.learningRate[-1] > self.minimmumRate:
                self.learningRate = np.append(self.learningRate, self.learningRate[-1] * self.decayRate)


            temp = sess.run(self.global_step)
            
            if off_policy==True:
                if self.shortsell==False:
                    wPrime_new = np.maximum(np.random.normal(0.5, 1, np.shape(wPrime)),0)
                else:
                    wPrime_new = np.random.normal(0, 1, np.shape(wPrime))

                wPrime = wPrime_new/np.sum(wPrime_new,2,keepdims=True)

            sess.run(self.actorOptimiser_adam, feed_dict={self.mIn: mIn,
                                                          self.wIn: wPrime,
                                                          self.y: y,
                                                          self.lr: self.learningRate[-1],
                                                          self.keep_prob: self.keep_prob_value,
                                                          self.is_train: True,
                                                          self.global_step_value: temp})

    

    def train_test(self, sess, x_inTensor, rates,x_dates, trainSetLen, saveModel=False,
                   planning_horizon=30, number_of_stocks=10,restoreSavedModel=False,off_policy=False,perm=1):

        
        # Path to save the model and results
        path = 'results/'
        path += str(self.dataset_name)
        path += '_' + str(self.network_model)
        path += '_' + str(self.number_of_stocks) + 'stocks'
        path += '_' + 'seed' + str(self.seed)
        path += '_' + 'permutation' + str(perm)
        
        if restoreSavedModel == True:
            if (os.path.exists(path)):
                self.saver.restore(sess, path + "/model.ckpt")
                self.pvm = np.loadtxt(path + '/pvm.csv', delimiter=',')
                self.epochs = 1
                print('------------------------------------')
                print("Model restored.")
                print('------------------------------------')
            else:
                print('------------------------------------')
                print('The model does not exist!')
                print('------------------------------------')
                return
        else:
            sess.run(self.init_op)
            self.initPvm(rates)        
        
        
        track_test = np.array([])
        track_train = np.array([])

        seed_counter = 0

        for epoch in range(self.epochs):
            step_start = time.time()
            self.reset()

            if self.planning_horizon == 1:
                last_index = rates.shape[0] - self.lookback_window - self.minibatchSize
            else:
                last_index = rates.shape[0] - self.lookback_window - self.planning_horizon
            time_spots = [1+i*self.minibatchCount for i in range(int(last_index/self.minibatchCount)+1)]+[last_index]
            time_spots = time_spots + [last_index]*(self.minibatchSize - len(time_spots)%self.minibatchSize)
            time_spots = np.reshape(np.array(time_spots),[-1,self.minibatchSize])


            for t_ in range(time_spots.shape[0]):

                self.trainOnMinibatch_PG(sess, time_spots[t_], x_inTensor, rates,
                                         do_backprop=False,
                                         is_train=False,
                                         trainSetLen=trainSetLen,
                                         off_policy=off_policy)

            if epoch < self.epochs - 1:
                for s_ in range(30):
                    seed_counter+=1
                    np.random.seed(seed_counter)
                    if self.planning_horizon == 1:
                        random_indexes = np.random.randint(1, trainSetLen - self.lookback_window - self.minibatchSize)
                        random_indexes = np.arange(random_indexes,random_indexes+self.minibatchSize)
                    else:
                        random_indexes = np.random.randint(1, trainSetLen - self.lookback_window - self.planning_horizon, self.minibatchSize)

                    self.trainOnMinibatch_PG(sess, random_indexes, x_inTensor, rates,
                                             do_backprop=True,
                                             is_train=True,
                                             trainSetLen=trainSetLen,
                                             off_policy=off_policy)
            

            train_APV = np.cumprod(self.pvm_reward[self.lookback_window - 1:trainSetLen])
            inSample_value = train_APV[-1]

            print('-----learning rate changed to----- : ' + str(self.learningRate[-1]))
            track_train = np.append(track_train, inSample_value)

            test_APV = np.cumprod(self.pvm_reward[trainSetLen:])
            self.APV = test_APV

            portfolio_turnover = self.print_train_valid_res(epoch, train_APV, test_APV, trainSetLen,rates)
            if portfolio_turnover < 1e-16:
                return 0, epoch, 0

            track_test = np.append(track_test, test_APV[-1])
            step_finish = time.time()
            print('---------time          : ' + str(datetime.timedelta(seconds=int(step_finish - step_start))))    
            
            # Saving the model every 1000 epochs
            if (epoch > 0 and epoch % 1000 == 0) or epoch == self.epochs - 1:
                
                self.equally_weighted(x_inTensor, rates, trainSetLen)
                self.prepare_for_saving(x_dates,trainSetLen)
                track_test_train = np.append(np.expand_dims(track_train,axis=1),np.expand_dims(track_test,axis=1),axis=1)
                self.save_results(track_test_train,sess,path)
                self.compute_performance_measures(rates,trainSetLen)
                
                # Save the model
                if restoreSavedModel == False:
                    self.saver.save(sess, path + '/model.ckpt')
        
        print('------------------------------------')
        print('Training is over')
        print('------------------------------------')

    def compute_performance_measures(self,rates,trainSetLen):
        # comptue different measures
        self.annual_return = self.APV[-1, 1] ** (252 / len(self.APV)) - 1
        self.annual_return_EW = self.APV[-1, 2] ** (252 / len(self.APV)) - 1
        self.annual_vol = np.std(np.diff(np.log(self.APV[:, 1].tolist()), n=1, axis=0)) * np.sqrt(252)
        self.annual_vol_EW = np.std(np.diff(np.log(self.APV[:, 2].tolist()), n=1, axis=0)) * np.sqrt(252)
        self.reward_risk = self.annual_return / self.annual_vol
        self.reward_risk_EW = self.annual_return_EW / self.annual_vol_EW
        dd = [self.APV[i, 1] / max(self.APV[0:i, 1]) - 1 for i in range(1, len(self.APV))]
        self.max_drawdown = min(dd)
        dd_EW = [self.APV[i, 2] / max(self.APV[0:i, 2]) - 1 for i in range(1, len(self.APV))]
        self.max_drawdown_EW = min(dd_EW)
        lst_ret = [self.APV[i, 1] / self.APV[i - 1, 1] - 1 for i in range(1, len(self.APV))]
        lst_ret_EW = [self.APV[i, 2] / self.APV[i - 1, 2] - 1 for i in range(1, len(self.APV))]
        self.beta_vs_benchmark = np.cov(lst_ret, lst_ret_EW)[0, 1] / np.var(lst_ret_EW)
        self.daily_hit_rate = sum([lst_ret[i] > lst_ret_EW[i] for i in range(len(lst_ret))]) / len(lst_ret)
        self.trackingError = np.std(np.diff(np.log(self.APV[:, 1].tolist()), n=1, axis=0)
                                    - np.diff(np.log(self.APV[:, 2].tolist()), n=1, axis=0)) * np.sqrt(252)
        self.portfolio_turnover = self.compute_portfolio_turnover(self.pvm[trainSetLen:,:],rates[trainSetLen:,:])


        res_list =     ['Annual return   :' + str(self.annual_return)]
        res_list.append('Annual return EW:' + str(self.annual_return_EW))
        res_list.append('Annual vol      :' + str(self.annual_vol))
        res_list.append('Annual vol EW   :' + str(self.annual_vol_EW))
        res_list.append('SR              :' + str(self.reward_risk))
        res_list.append('SR              :' + str(self.reward_risk_EW))
        res_list.append('MDD             :' + str(self.max_drawdown))
        res_list.append('MDD EW          :' + str(self.max_drawdown_EW))
        res_list.append('Daily hit rate  :' + str(self.daily_hit_rate))
        res_list.append('Turnover        :' + str(self.portfolio_turnover))
        print(*res_list, sep="\n")

    def prepare_for_saving(self,x_dates,trainSetLen):
        self.APV = np.expand_dims(self.APV, axis=1)
        self.APV_EW = np.expand_dims(self.APV_EW, axis=1)
        self.APV = np.append(self.APV, self.APV_EW, axis=1)
        test_dates = np.expand_dims(x_dates[trainSetLen:], axis=1)
        self.APV = np.append(test_dates, self.APV, axis=1)
        self.weightsMatrix = np.append(test_dates, self.pvm[trainSetLen:], axis=1)
        
    def compute_portfolio_turnover(self,w,rates):        
        w_prime = (rates * w)/np.sum(rates * w,1,keepdims=True)
        mu = self.calculateMu(np.expand_dims(w_prime,0),np.expand_dims(w,0))
        portfolio_turnover = np.sum(np.linalg.norm(np.subtract(w_prime[0:-1, :],
                                                                  w[1:, :] * np.expand_dims(
                                                                      mu[0, 1:], 1)), ord=1, axis=1)) / (2 * (len(w[1:, :]) - 1))
        return portfolio_turnover
                                                                                      
        

    def print_train_valid_res(self,epoch,train_APV,test_APV,trainSetLen,rates):
        print('epoch:' + str(epoch))
        portfolio_turnover = self.compute_portfolio_turnover(self.pvm[trainSetLen:,:],rates[trainSetLen:,:])
        
        for i in range(1, 11):
            print(np.round(np.sort(self.pvm[-i * 50])[-10:], 2))
            
        print('---------train APV     : ' + str(train_APV[-1]))
        print('---------max test APV  : ' + str(np.round(max(test_APV), 8)))
        print('---------test APV      : ' + str(np.round(test_APV[-1], 8)))
        print('---------min test APV  : ' + str(np.round(min(test_APV), 8)))
        print('---------test turnover : ' + str(np.round(portfolio_turnover, 8)))
        
        return portfolio_turnover


    def equally_weighted(self,inTensor, rates,trainSetLen):
        actorAction = [[1/self.number_of_stocks for i in range(self.number_of_stocks)] for j in range(len(rates)-trainSetLen)]
        wPrime = self.updateRateShift(np.expand_dims(actorAction,axis=0), np.expand_dims(rates[trainSetLen:],axis=0))
        mu = self.calculateMu(wPrime, np.expand_dims(actorAction,0))

        reward = np.sum(np.multiply(actorAction, rates[trainSetLen:]), axis=1)
        immediateReward = reward * np.squeeze(mu,0)
        # immediateReward = reward - mu
        self.APV_EW=np.cumprod(immediateReward)

    def save_results(self,track_test_train,sess,path):
        # suffixName = str(self.tradeFee) + '_'  + str(self.learningRate[0]) + '_' + str(self.minibatchSize) +\
        #              '_' + str(self.lookback_window)+ '_' + '_' + str(self.epochs)+'_' + str(self.seed)
        suffixName = ''
        fileTosave_APV = 'APV_' + suffixName + '.csv'
        fileTosave_weights = 'weightsMatrix_' + suffixName + '.csv'
        fileTosave_track_test_train = 'track_test_train_' + suffixName + '.csv'
        fileTosave_pvm = 'pvm.csv'        
        
        try:
            os.mkdir(path)
        except:
            print('directory already exists!')
            
        pd.DataFrame(self.APV).to_csv(path + '/'+fileTosave_APV, index=False,header=False)
        pd.DataFrame(self.pvm).to_csv(path + '/'+fileTosave_pvm, index=False,header=False)
        pd.DataFrame(self.weightsMatrix).to_csv(path + '/'+fileTosave_weights, index=False,header=False)
        pd.DataFrame(track_test_train).to_csv(path + '/'+fileTosave_track_test_train, index=False,header=False)  
        