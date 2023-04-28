"""
@author: maxbaldus

"""

import os
import logging

# Disable Tensorflow Deprecation Warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Libraries
import tensorflow as tf 

from keras.layers import Input, Dense, GaussianNoise , Lambda # import the different layers used inside the nn 
from keras.models import Model  # group the layers into a model that can be estimated

# TEST LOSS
#from keras import backend as K

# scaling the data
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import random as rn
from scipy import stats


class mdn:

    """
    Using the alenn package to estimate the posterior distribution of the abm model by delli gatti.
    This class implements the estimation of a gaussian mixture distribution using a mixture density network 
    in order to approximate the likelihood function ?of the agent based models
    ???????? s.t. a MH-algorithm can be used later on to sample the posterior distribution. 
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
        self.T = len(data_obs)

        self.neurons = neurons # number of neurons per hidden layer
        self.layers = layers # number of hidden layers
        self.batch_size = batch_size # size of the (random) sample used to update gradient once
        self.epochs = epochs # final number of rounds for the entire training set used once for updating the gradient

        self.eta_y = eta_y # standard deviation of the gaussian noise distribution to avoid overfitting applied onto y training vector
        self.eta_x = eta_x # standard deviation of the gaussian noise distribution to avoid overfitting applied onto feature matrix X

        self.act_fct = act_fct # activation function in the hidden layers

        """act_func:str, eta_x:float, eta_y:float,
                univar_output:bool"""

       
    def approximate_posterior(self):
        
        """
        This method estimates the posterior of a parameter combination by, 
        approximating the likelihood of the empirical data via estimating the parameter of a gaussian mixture distribution 
        with a feed forward neural network, called mixture density network, using simulated data as input/training data.
        Multiplying (summing log) with the prior probability returns the (approximated) posterior distribution.

        The entire procedure is split into ? parts:

        1) The distribution of simulated (artificial) data is estimated:
        Distribution is a gaussian mixture, its parameters result from a feed-foward, artifical neural network: mdn 
        """

        # construct one large ordered training set with self.L number of features and 1-d target,
        # being the value that follows each window of lagged observations of length self.L
        X_train, y_train = mdn.order_data(np.transpose(self.data_sim), self.L)

        # standardize training data
        scaler = MinMaxScaler() # Transform features by scaling each feature to a given range.
        X_train = scaler.fit_transform(X_train) 
        y_train= scaler.fit_transform(y_train.reshape(-1,1))
        
        """# standardize training data by subtracting the means and dividing by std
        X_means = X_train.mean(axis = 0)
        X_std = X_train.std(axis = 0)
        y_mean = y_train.mean(axis = 0)
        y_std = y_train.std(axis = 0)

        # (X_std==0).any()
        if 0 not in X_std:
            X_train  = (X_train - X_means) / X_std
        if y_std != 0:
            y_train = (y_train - y_mean) / y_std"""

        # not needed in tf2 ??
        """# Initiate the network graph
        graph = tf.compat.v1.get_default_graph() # get default graph for the current thread (get root default graph tf sets)
        sess = tf.compat.v1.Session(graph) #  set session: the TensorFlow Graph object in which tensors are processed through operations
        tf.compat.v1.keras.backend.set_session(sess) # set the global TensorFlow session """

        # set seeds (such that using same mdn for each parameter combination)
        np.random.seed(12)
        rn.seed(13)
        tf.random.set_seed(14)
   
        # Input layers (including taget data layer: each layer is input of the next layer (graphs))
        X = tf.keras.layers.Input(shape = (self.L,)) # NN input tensor with with L dimensions
        y = Input(shape=(1,)) # taget data layer

        # Noise Regularisation: applying additive zero-centered Gaussian noise to avoid overfitting
        X_reg = GaussianNoise(self.eta_x)(X)
        y_reg = GaussianNoise(self.eta_y)(y) 

        # first hidden layer using fully connected layers s.t. output = activation(dot(input, kernel) + bias)
        h = Dense(self.neurons, activation=self.act_fct)(X_reg)

        # additional hidden fully connected layers in the network
        for i in range(1, self.layers):
            h = Dense(self.neurons, activation=self.act_fct)(h)

        # Output layers:
        # Mean: directly represented by the last hidden activations
        mu = Dense(units = self.K, activation=None, name = "mean_layer")(h) # since univariate target y, the number of means of each component amounts to 1*K
        
        # Mixing coefficients output: with sum(pi) = 1 and range(pi)=[0,1], hence using softmax activation
        pi = Dense(units=self.K,activation='softmax', name='pi_layer')(h) # K mixtue coefficients: one node for each coefficientt
        
        # Variance:
        log_var = Dense(units = self.K, activation=None)(h)
        # exponentiate sigma to make it s.t. variance is always >= 0: the transformation is treated as a new layer
        sigma_sqr = Lambda(lambda x: tf.math.exp(x), output_shape=(self.K,), name="variance_layer")(log_var) # K variances (one for each gaussian component)

        # Instantiate model (the mdn) by using keras.input objects
        nn = Model(inputs = [X,y], outputs = [mu,pi,sigma_sqr])
        #nn.summary()

        # add the loss fct (likelihood function) used to estimate the mdn (minimizing using a gradient descent/back propagation)
        nn.add_loss(self.loss_fct(y_reg, pi, mu, sigma_sqr))    
        # nn.add_loss(self._TEST(y_reg, pi, mu, sigma_sqr))   # LOSS NOCHMAL KONTROLLIEREN 

        # compile model by setting optimizer
        nn.compile(optimizer = 'adam')

        # fit mdn
        nn.fit([X_train, y_train],
                batch_size = self.batch_size,
                epochs = self.epochs,
                verbose = True)
                
        """
        2) Computing density of observed (empirical) data under the estimated mixture distribution (using the fitted mdn)
        """

        # order empirical data
        T_tilde = len(self.data_obs) # length of empirical data 
        X_obs, y_obs = mdn.order_data(self.data_obs.reshape(1, T_tilde), self.L)

        # scaling empirical data 
        scaler = MinMaxScaler() # Transform features by scaling each feature to a given range.
        X_obs = scaler.fit_transform(X_obs) 
        y_obs = scaler.fit_transform(y_obs.reshape(-1,1))

        """# scaling empirical data 
        X_obs_mean = X_obs.mean(axis = 0)
        X_obs_sd = X_obs.std(axis = 0)
        y_obs_mean = y_obs.mean(axis = 0)
        y_obs_sd = y_obs.std(axis = 0)
        
        if 0 not in X_obs_sd:
            X_obs  = (X_obs - X_obs_mean) / X_obs_sd
        if y_obs_sd != 0:
            y_obs = (y_obs - y_obs_mean) / y_obs_sd"""

        # Create Data Structure to Store Likelihood Values
        likelihood = np.zeros(T_tilde)

        # compute likelihood of each empirical observation for the given parameter combination
        # since rolling window on 1 ts, loosing the last L observations => hence T_tilde - L = len(y_obs)
        for i in range(len(y_obs)):

            # 1) prediciting the gauss parameter using the nn and the ordered observed data X (y is zero in this case)
            mu, pi, sigma_sqrd = nn.predict([X_obs[i,:].reshape(1, self.L), np.array([0])], verbose = True) # using - mean / sd

            # 2) using the estimated mixture parameter and the y_obs following each window to finally compute the likelihood
            likelihood[i] = mdn.gmm_likelihood(y_obs[i], mu, pi, sigma_sqrd)
            # scale likelihood by std of training target 
            
            # likelihood[i] = (1 / y_std.prod()) * self.__gmm_univar((self.y_empirical[i] - y_mean) / y_std, self.num_mix, alpha.flatten(), mean.flatten(), log_var.flatten())
            # print("blub")
        
        print("blub")

        # 2) compute density of each observed data point (y_sim) 
        # i.e. a) PREDICT gaussian mixture parameters using y_sim :  nn.predict (params of gaussian) 
        # b) compute density of each y_emp (numerically evaluating the density function)
        # plot mixture distribution ???
        
        """
        3) The computed likelihood of the observed data is now used to finally compute the posterior probability of theta, given the prior probability
        """
        # close everything in order to ensure using same network for each parameter combination s
        # tf.keras.backend.clear_session # reset everything and close session
        # np.random.seed()
        # rn.seed()
        
        # 3a) Likelihood = product of the densities 
        # 3b) ll = sum of log(densities)

        # compute prior probability of theta
        # ??? each theta has prior probabilty => same likelihood for each parameter * prior probability of each parameter => get array of posteriors: one for each param??
        # ll + log(prior) = posterior
        

        # return posterior

 
# -------------------------------------- 
    def order_data(data, L):

        """
        This function orders either the simulated or empirical data (2d array) and prepares mixture density network inputs
        """

        # order the simulated data with by splitting the observations into a train and test set 
        # each lag is one regressor/feature
        """MC_replications = data.shape[1] 
        T = data.shape[0]"""
 
        # data_sim_trans = np.transpose(data) # transport the simulated df
        
        # Input features
        # X = np.array([data[i, j : j + L] for i in range(MC_replications) for j in range(T - L)])
        X = np.array([data[i, j : j + L] for i in range(data.shape[0]) for j in range(data.shape[1] - L)])
        
        # targets are observations that appear after each self.L
        # y = np.array([data[i, j] for i in range(MC_replications) for j in range(L, T)])
        y = np.array([data[i, j] for i in range(data.shape[0]) for j in range(L, data.shape[1])])

        return X, y
    

    def loss_fct(self, y, pi, mu, sigma_sqr): 

        # using eager mode from tf2 
        # tf.executing_eagerly()
        print("blub")
        
        # gaussian pdf
        p = (1/(tf.math.sqrt(2* np.pi *sigma_sqr))) * tf.math.exp( (-1/(2*sigma_sqr))*(tf.subtract(y, mu)**2) ) # probaility of one gaussian component

        # mixture density
        p_mix = tf.reduce_sum(tf.multiply(p,pi), axis = 1, keepdims=True)
        
        # negative (log) likelihood function
        log_likelihood_fct = - tf.reduce_sum(tf.math.log(p_mix), axis = 0)

        return  log_likelihood_fct

        """# univariat gaussian likelihood of each y_reg (element-wise for each training observation)
        diff = tf.subtract(y, mu)**2 # subtracting the mean of each component from each observation (nx1 array - mu of each component): output is nxK matrix
        univ_gauss_likeli = (1/tf.math.sqrt(2*np.pi*sigma_sqr)) * tf.math.exp((-1/2)*(diff/sigma_sqr)**2)

        # HIER WEITER
        # gaussian mixture values
        out = tf.multiply(univ_gauss_likeli, pi) # multiply with each component (Returns an element-wise x * y)
        out = tf.keras.backend.sum(out) # sum K component outputs

        out = tf.math.reduce_sum(out, 1, keepdims=True)  # computing the sum of all elements across dimensions of a tensor

        
        # likelihood part
        out = -tf.keras.backend.sum(tf.keras.backend.log(out))

        return out"""
    
    def gmm_likelihood(y, mu, pi, sigma_sqrd):
        
        print("blub")

        # compute likelihood of each component
        individual_likelihoods = stats.norm.pdf(y[0], loc = mu, scale = sigma_sqrd) # using minmax scaler
        # individual_likelihoods = stats.norm.pdf(y, loc = mu, scale = sigma_sqrd) # using mean / std scaling

        
        # weight the likelihood value from each normal component by its corresponding weight and sum
        mixture_likelihood = np.sum(pi * individual_likelihoods)
        
        # return (np.array([stats.norm.pdf(y, loc = m[i], scale = s[i]) for i in range(k)]) * alpha).sum()

        return mixture_likelihood 

    """def _TEST(y, alpha, m, log_var):
        '''
        Maximum likelihood-based Loss function used to train the MDN.
        '''
    
        # Convert the Log Variance to the Standard Deviation
        s = K.exp(0.5 * log_var)
        
        # Calculate the Coefficient for the Exponential Function
        coeff = (2 * np.pi) ** (-0.5)
        coeff = coeff / s
        
        # Determine the Exponent for the Exponential Function
        exponent = -K.square(y - m) / (2 * K.square(s))
        
        # Return the Loss Function
        return -K.sum(K.log(K.sum(alpha * coeff * K.exp(exponent), axis = 1)), axis = 0)"""






        
