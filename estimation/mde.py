"""
...
"""

# Libraries
import numpy as np 
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dense, GaussianNoise
from keras import backend as K


class mdn:
    
    """
    This class implements the estimation of a mixture density network in order to approximate the likelihood function 
    s.t. a MH-algorithm can be used later on to sample the posterior distribution. 
    ????????
    """

    def __init__(self, data_sim, data_obs):

        """
        Initialise the class by storing simulated and observed data as inputs, which are used for the estimation procedure.
        """
        self.data_sim = data_sim
        self.data_obs = data_obs

    def rolling_window(self):

        """
        Order the data 
        """

       

    def set_mdn(self, num_lags:int, num_mix:int, num_neurons:int, num_layers:int, batch_size:int,
                num_epochs:int, act_func:str, eta_x:float, eta_y:float,
                univar_output:bool):
        
        """
        Here the mixture density network (mdn) is defined: 
        
        ???
        The entire model needs to be python function, 
        with the only input being calibrated parameter values (as a list).
        In the univariat case the output is a 2-D numpy array with TxMC dimensions. 
         
        """
        # return model
    
    def estimate_mixture(self):
        
        """
        This method estimates the above defined mixture density network and therefore the parameters of a
        gaussian mixture distribution using simulated data from an agent based model.
        It then returns the likelihood as well as the log likelihood value of each 
        
        
        """

        # return params (estimated mixture parameters now) ??? 

    
    def sample_mixture(self):

        """
        This method samples from a gaussian mixture density estimated above
        - using MH ?? 
        """




