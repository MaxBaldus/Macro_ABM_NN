"""
Mixture density network
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
from keras import backend as K

# scaling the data
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

import numpy as np
import random as rn
from scipy import stats


class mdn:

    """
    Using the alenn package, this class implements the estimation of a gaussian mixture distribution using a mixture density network 
    in order to approximate the likelihood function of the agent based model.
    """

    def __init__(self, data_obs, L:int, K:int,
                 neurons:int, layers:int, batch_size:int, epochs:int, eta_y:float, eta_x:float, act_fct:str
                 ):
        
        # data_obs: observed data - 1-d numpy array (vector)
        self.data_obs = data_obs

        """
        Hyper parameters of the mixture density network 
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
       
    def approximate_likelihood(self, data_sim):
        
        """
        This method approximates the likelihood of the empirical data given a set of parameter values through using their associated simulated time series 
        as input/training data (MC simulations) for estimating the parameter of a gaussian mixture distribution,
        with a feed forward neural network, called mixture density network (mdn).

        Input: simulated data - 2d numpy array (matrix)

        The entire procedure is split into 2 parts:

        1) The distribution of simulated (artificial) data is estimated:
        The distribution is a gaussian mixture, its parameters result from a feed-foward, artifical neural network: mdn 
        """

        # construct one large ordered training set with self.L number of features and 1d target,
        # being the value that follows each window of lagged observations of length self.L
        X_train, y_train = mdn.order_data(np.transpose(data_sim), self.L)
        # sd of y
        y_train_std = y_train.std(axis = 0)

        # deal with -inf values (0 or close to 0 output values)

        """# standardize training data
        X_scaler = MinMaxScaler() 
        y_scaler = MinMaxScaler() 
        # X_train = scaler.fit_transform(X_train) 
        X_scaler.fit(X_train)
        X_train = X_scaler.transform(X_train)
        # y_train= scaler.fit_transform(y_train.reshape(-1,1))
        y_scaler.fit(y_train.reshape(-1,1))
        y_train= y_scaler.transform(y_train.reshape(-1,1))"""
        
        # standardize training data by subtracting the means and dividing by std
        X_means = X_train.mean(axis = 0)
        X_std = X_train.std(axis = 0)
        y_mean = y_train.mean(axis = 0)
        y_std = y_train.std(axis = 0)
        
        X_train  = (X_train - X_means) / X_std
        y_train = (y_train - y_mean) / y_std

        """# (X_std==0).any()
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
        # h = Dense(self.neurons, activation=self.act_fct)(X)

        # additional hidden fully connected layers in the network
        for i in range(1, self.layers):
            h = Dense(self.neurons, activation=self.act_fct)(h)

        # Fully connected output layers:
        # Mean: directly represented by the last hidden activations
        mu = Dense(units = self.K, activation=None, name = "mean_layer")(h) # since univariate target y, the number of means of each component amounts to 1*K
        
        # Mixing coefficients output: with sum(pi) = 1 and range(pi)=[0,1], hence using softmax activation
        pi = Dense(units=self.K,activation='softmax', name='pi_layer')(h) # K mixtue coefficients: one node for each coefficientt
        
        # Variance:
        var_raw = Dense(units = self.K, activation=None)(h)
        # exponentiate sigma to make it s.t. variance is always >= 0: the transformation is treated as a new layer
        sigma_sqr = Lambda(lambda x: 0.5 * tf.math.exp(x), output_shape=(self.K,), name="variance_layer")(var_raw) # K variances (one for each gaussian component)

        # Instantiate model (the mdn) by using keras.input objects
        nn = Model(inputs = [X,y], outputs = [mu,pi,sigma_sqr])  # NOT var, constrainted input !! 
        #nn.summary()

        # add the loss fct (likelihood function) used to estimate the mdn (minimizing using a gradient descent/back propagation)
        nn.add_loss(self.loss_fct(y_reg, pi, mu, sigma_sqr))  
        #nn.add_loss(self.loss_fct(y, pi, mu, sigma_sqr))    
  
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

        """# scaling empirical data 
        # X_obs = scaler.fit_transform(X_obs) 
        X_obs = X_scaler.transform(X_obs) 
        # y_obs = scaler.fit_transform(y_obs.reshape(-1,1))
        y_obs = y_scaler.transform(y_obs.reshape(-1,1))"""
        
        """# scaling empirical data 
        X_obs_mean = X_obs.mean(axis = 0)
        X_obs_sd = X_obs.std(axis = 0)
        y_obs_mean = y_obs.mean(axis = 0)
        y_obs_sd = y_obs.std(axis = 0)
        
        if 0 not in X_obs_sd:
            X_obs  = (X_obs - X_obs_mean) / X_obs_sd
        if y_obs_sd != 0:
            y_obs = (y_obs - y_obs_mean) / y_obs_sd"""

        # Create Data Structure to store the likelihood values for each observation
        likelihoods = np.zeros(len(y_obs))

        # compute likelihood of each empirical observation for the given parameter combination
        # since rolling window on 1 ts, loosing the last L observations => hence T_tilde - L = len(y_obs)
        for i in range(len(y_obs)):

            # 1) predicting the gauss parameter using the nn and the ordered observed data X (y is zero in this case)
            # mu, pi, var = nn.predict([X_obs[i,:].reshape(1, self.L), np.array([0])], verbose = False) # using - mean / sd

            # 2) using the estimated mixture parameter and the y_obs following each window to finally compute the likelihood 
            # of each observed value, scaled by std of empirical data
            #likelihoods[i] = (self.gmm_density(y_obs[i], mu, pi, var))
            
            # scale by dividing by var ??
            # likelihoods[i] = (self.gmm_density(y_obs[i], mu, pi, log_var)) * 1/ y_train_std if y_train_std > 0 else self.gmm_density(y_obs[i], mu, pi, log_var) 
            
            # scale the likelihood values by scaling_factor => L HUGE e^226
            # scaling_factor = 10 
            # likelihoods[i] = (self.gmm_density(y_obs[i], mu, pi, log_var)) * (scaling_factor)
            
            # SCALING
            mu, pi, var = nn.predict([(X_obs[i,:].reshape(1, self.L) - X_means) / X_std, np.array([0])], verbose = False) # using - mean / sd
            #likelihoods[i] = (self.gmm_density((y_obs[i] - y_mean) / y_std , mu, pi, var)) * (1/ y_std) if y_std > 0 else (self.gmm_density((y_obs[i] - y_mean) / y_std , mu, pi, var))
            likelihoods[i] = (self.gmm_density((y_obs[i] - y_mean) / y_std , mu, pi, var)) 
        # reset everything and close session 
        tf.keras.backend.clear_session() 
        np.random.seed()
        rn.seed()
        
        # scale values between 1 and 2
        # densities = preprocessing.minmax_scale(densities, feature_range=(1, 2))
        scaling_factor = 10
        return likelihoods 
        
 
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
        
        """# convert log variances to sd
        s = np.exp(0.5 * log_var)
        # variance
        s_2 = s**2"""
        
        # gaussian pdf
        p = (1/(tf.math.sqrt(2* np.pi *sigma_sqr))) * tf.math.exp( (-1/(2*sigma_sqr))*(tf.subtract(y, mu)**2) ) # probaility of one gaussian component
        # p = (1/(tf.math.sqrt(2* np.pi) * s)) * tf.math.exp( (-1/(2*s_2))*(tf.subtract(y, mu)**2) ) # probaility of one gaussian component
        
        # mixture density
        p_mix = tf.reduce_sum(tf.multiply(p,pi), axis = 1, keepdims=True)
        
        # negative (log) likelihood function (since want to max. likelihood, but minimising gradient descent algorithm in tensorflow)
        log_likelihood_fct = - tf.reduce_sum(tf.math.log(p_mix), axis = 0)

        return  tf.reduce_mean(log_likelihood_fct)

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
    
    def gmm_density(self, y, mu, pi, var):
        
        # apply transformation to ensure var > 0 
        # s = np.exp(0.5 * log_var)
        
        # compute density of each component
        # individual_likelihoods = stats.norm.pdf(y[0], loc = mu, scale = sigma_sqrd) # using minmax scaler
        # individual_densities = stats.norm.pdf(y, loc = mu, scale = s) # using mean / std scaling
        individual_densities = stats.norm.pdf(y, loc = mu, scale = var) # using mean / std scaling

        # weight the likelihood value from each normal component by its corresponding weight and sum
        mixture_likelihood = np.sum(pi * individual_densities)
        
        # his version: result almost identical 
        # mixture_likelihood = (np.array([stats.norm.pdf(y, loc = mu[0][i], scale = sigma_sqrd[0][i]) for i in range(self.K)]) * pi[0]).sum()

        return mixture_likelihood 

    # TEST HIS LOSS FUNCTION !!
    def _TEST(y, alpha, m, log_var):
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
        return -K.sum(K.log(K.sum(alpha * coeff * K.exp(exponent), axis = 1)), axis = 0)






        
