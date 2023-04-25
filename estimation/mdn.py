"""
@author: maxbaldus

Using the alenn package to estimate the posterior distribution of the abm model by delli gatti.
This class implements the estimation of a gaussian mixture distribution using a mixture density network 
in order to approximate the likelihood function ?of the agent based models
???????? s.t. a MH-algorithm can be used later on to sample the posterior distribution. 
"""

import os
import logging

# Disable Tensorflow Deprecation Warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Libraries
import tensorflow as tf
import numpy as np
from sklearn import preprocessing


class mdn:

    """
    adapted code tailored to the BAM model using:
    
    Approximate Likelihood Estimation using Neural Networks (ALENN)
    Donovan Platt
    Mathematical Institute, University of Oxford
    Institute for New Economic Thinking at the Oxford Martin School
    Copyright (c) 2020, University of Oxford. All rights reserved.
    """

    def __init__(self, data_sim, data_obs,
                 L:int, K:int,
                 neurons:int, layers:int, batch_size:int, epochs:int, eta_y:float, eta_x:float, act_fct:str
                 ):

        """
        Initialise the class by storing simulated and observed data as inputs, which are used approximate the likelihood respectively.
        """
        self.data_sim = data_sim # simulated data: 2-d numpy array (matrix)
        self.data_obs = data_obs # observed data: 1-d numpy array (vector)

        """
        Hyper parameters of the mixture density network and gaussian mixture distribution
        """
        self.L = L # number of lags to be considered, which amounts to the number of input features for the mdn
        self.K = K # number of mixture components

        self.neurons = neurons # number of neurons per hidden layer
        self.layers = layers # number of hidden layers
        self.batch_size = batch_size # size of the (random) sample used to update gradient once
        self.epochs = epochs # final number of rounds for the entire training set used once for updating the gradient

        self.eta_y = eta_y # standard deviation of the gaussian noise distribution to avoid overfitting applied onto y training vector
        self.eta_x = eta_x # standard deviation of the gaussian noise distribution to avoid overfitting applied onto feature matrix X

        self.act_fct = act_fct # activation function in the hidden layers

        """act_func:str, eta_x:float, eta_y:float,
                univar_output:bool"""

       
    def approximate_likelihood(self):
        
        """
        This method estimates the parameter of a gaussian mixture distribution with a
        feed forward neural network, called mixture density network, using simulated data as input.
        Output is the likelihood of the certain parameter combination, given the empirical data. 

        The entire procedure is split into 3 parts:

        1) The distribution, here a gaussian mixture, is estimated using simulated data and a feed-foward, artifical neural network (mdn) 
        """

        # construct one large ordered training set with self.L number of features and 1-d target,
        # being the value that follows each window of lagged observations of length self.L
        X_train, y_train = mdn.create_training_data(self)
        
        # standardize training data by subtracting the means and dividing by std
        X_means = X_train.mean(axis = 0)
        X_std = X_train.std(axis = 0)
        y_mean = y_train.mean(axis = 0)
        y_std = y_train.std(axis = 0)

        X_train  = (X_train - X_means) / X_std
        y_train = (y_train - y_mean) / y_std

        # nn COMPUTATIONAL GRAPH ?? erstmal rauslassen
        
        # inputs of the neural net
        y_input = tf.keras.layers.Input(shape=(1,)) # ????? TWO INPUTS ???        
        X = tf.keras.layers.Input(shape = (self.L,)) # NN input tensor with with L dimensions

        # Noise Regularisation: applying additive zero-centered Gaussian noise to avoid overfitting
        y_reg = tf.keras.layers.GaussianNoise(self.eta_y)(y_input) 
        X_reg = tf.keras.layers.GaussianNoise(self.eta_x)(X)

        # WHERE GOES y_reg ???

        # first hidden layer using fully connected layers s.t. output = activation(dot(input, kernel) + bias)
        h = tf.keras.layers.Dense(self.neurons, activation=self.act_fct)(X_reg)

        # additional hidden fully connected layers in the network
        for i in range(1, self.layers):
            h = tf.keras.layers.Dense(self.neurons, activation=self.act_fct)(h)

        # mean output: directly represented by the last hidden activations
        mu = tf.keras.layers.Dense(units = self.K, activation=None, name = "mean_layer")(h) # since univariate target y, the number of means of each component amounts to 1*K
        
        # variance output: should be greater than 0, hence exponetiate values of last output
        sigma_sqr = tf.keras.layers.Dense(units = self.K, activation=None)(h)
        sigma_sqr = tf.keras.layers.Lambda(lambda x: tf.math.exp(x), output_shape=(self.K,), name="variance_layer")(sigma_sqr) # exponentiating is treated as new layer
        #   log_var = Dense(self.num_mix)(h) # not exp. it???
        # mixing coefficients output: with sum(pi) = 1 and range(pi)=[0,1], therefore using softmax activation
        pi = tf.keras.layers.Dense(units=self.K,activation='softmax', name='pi_layer')(h)

        # Instantiate the mdn
        nn = tf.keras.models.Model([X,y_input], [mu,sigma_sqr,pi])

        # add the loss fct (likelihood function) used to estimate the nn  
        test = self.loss_fct(y_reg, pi, mu, sigma_sqr)     


        print("blub")

        # nn.estimate

        # 2) compute density of each observed data point (y_sim) under the estimated mixture distribution
        # i.e. a) PREDICT gaussian mixture parameters using y_sim :  nn.predict (params of gaussian) 
        # b) compute density of each y_emp (numerically evaluating the density function)
        # plot mixture distribution ???
        
        # 3a) Likelihood = product of the densities 
        # 3b) ll = sum of log(densities)




        # return likelihood for the parameter combination

    
    def evaluate_posterior(self, ll):

        """
        The computed likelihood of theta is now used to finally compute the posterior probability of theta, given its prior 
        """

        # compute prior probability of theta
        # ll + log(prior) = posterior
        

        # return posterior

 
# --------------------------------------    
    def create_training_data(self):

        """
        This function orders the simulated data (2d array) and prepares mixture density network inputs
        """

        # order the simulated data with by splitting the observations into a train and test set 
        # each lag is one regressor/feature
        MC_replications = self.data_sim.shape[1] 
        T = self.data_sim.shape[0]
 
        data_sim_trans = np.transpose(self.data_sim) # transport the simulated df
        
        # training features
        X_train = np.array([data_sim_trans[i, j : j + self.L] for i in range(MC_replications) for j in range(T - self.L)])
        
        """X_train_rows = MC_replications*(T-self.L) # number of rows
        X_train = np.zeros((X_train_rows, self.L)) # initialize regressor matrix 
        # fill X_train according to rolling window of size self.L
        for i in range(MC_replications):

            for j in range(T - self.L):

                for k in range(self.L):

                    X_train[i, j : j + self.L] = data_sim_trans[i,j:j + self.L]  # ???
          
        # fill current mc into X_train"""

        # training targets are observations that appear after each self.L
        y_train = np.array([data_sim_trans[i, j] for i in range(MC_replications) for j in range(self.L, T)])
        
        return X_train, y_train


    def loss_fct(self, y, pi, mu, sigma_sqr): 

        # univariat gaussian likelihood of each y_reg (element-wise for each training observation)
        diff = tf.subtract(y, mu)**2 # subtracting the mean of each component from each observation (nx1 array - mu of each component): output is nxK matrix
        univ_gauss_likeli = (1/tf.math.sqrt(2*np.pi*sigma_sqr)) * tf.math.exp((-1/2)*(diff/sigma_sqr)**2)

        # HIER WEITER
        # gaussian mixture values
        out = tf.multiply(univ_gauss_likeli, pi) # multiply with each component (Returns an element-wise x * y)
        out = tf.keras.backend.sum(out) # sum K component outputs

        out = tf.math.reduce_sum(out, 1, keepdims=True)  # computing the sum of all elements across dimensions of a tensor

        
        # likelihood part
        out = -tf.keras.backend.sum(tf.keras.backend.log(out))

        return out






        #





        
