"""
Here two mixture densities network classes are defined. 
One class for sampling possible theta values and simulating for each theta 
and another class when using an MH algorithm to sample the posterior. 
"""

# Libraries
import numpy as np 
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dense, GaussianNoise
from keras import backend as K


class mdn_MHsampling:
    
    """
    This class implements a mixture density network in order to approximate the likelihood function 
    s.t. a MH-algorithm can be used later on to sample the posterior distribution. 
    The code is mainly taken from: 
    Donovan Platt, Mathematical Institute, 
    University of Oxford Institute for New Economic Thinking at the Oxford Martin School
    ALEEN package
    """

    def __init__(self, num_lags:int, num_mix:int, num_neurons:int, num_layers:int, batch_size:int, 
                 num_epochs:int, act_func:str, eta_x:float, eta_y:float):
        
        """        
        Parameters
        ----------
        num_lags : int
            Number of lags.
        num_mix : int
            Number of mixture components.
        num_neurons : int
            Number of nodes per hidden layer.
        num_layers : int
            Number of hidden layers.
        batch_size : int
            Training batch size.
        num_epochs : int
            Number of training epochs.
        act_func : string
            A Keras activation function string.
        eta_x : float
            Noise regularisation standard deviation (network inputs).
        eta_y : float
            Noise regularisation standard deviation (mixture model output).
        """
        
        # Set MDN Attributes
        self.num_lags = num_lags 
        self.num_mix = num_mix
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.act_func = act_func
        self.eta_x = eta_x
        self.eta_y = eta_y

    def model(self, univar_output:bool):
        """
        Here the model is set: The entire model needs to be python function, 
        with the only input being calibrated parameter values (as a list).
        In the univariat case the output is a 2-D numpy array with TxMC dimensions. 
         
        """




