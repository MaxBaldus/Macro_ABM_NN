# Libraries



class mdn:
    
    """
    This class implements the estimation of a gaussian mixture distribution using a mixture density network 
    in order to approximate the likelihood function ?of the agent based models
    ???????? s.t. a MH-algorithm can be used later on to sample the posterior distribution. 
    """

    def __init__(self, data_sim, data_obs):

        """
        Initialise the class by storing simulated and observed data as inputs, which are used approximate the likelihood respectively.
        """
        self.data_sim = data_sim # simulated data
        self.data_obs = data_obs # observed data

     

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
        # return nn model 
        # inside estimate_mixture!!
    
    def estimate_mixture(self):
        
        """
        This method estimates the parameter of a gaussian mixture distribution with a
        feed forward neural network, called mixture density network,
        using simulated and real data as input and outputs the estimated parameter values. 
        """
        
        # order the data with rolling window function
        # set_mdn : mdn object
        # nn.estimate
        # nn.predict (params of gaussian)

        # return params (estimated mixture parameters now) ??? 

    
    def eval_mixture(self):

        """
        This method computes the likelihood values of the estimated gaussian mixture density estimated,
        using what data ??!!
        """

        # plot mixture distribution ???
        # return ll

    def rolling_window(self):

        """
        Order the data 
        """


