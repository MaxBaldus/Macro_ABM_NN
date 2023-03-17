# Libraries
import tensorflow as tf
import numpy as np


class mdn:
    
    """
    This class implements the estimation of a gaussian mixture distribution using a mixture density network 
    in order to approximate the likelihood function ?of the agent based models
    ???????? s.t. a MH-algorithm can be used later on to sample the posterior distribution. 
    """

    def __init__(self, data_sim, data_obs,
                 L:int, K:int,
                 neurons:int, layers:int, batch_size:int, epochs:int
                 ):

        """
        Initialise the class by storing simulated and observed data as inputs, which are used approximate the likelihood respectively.
        """
        self.data_sim = data_sim # simulated data: 2-d numpy array 
        self.data_obs = data_obs # observed data: 1-d numpy array

        """
        Hyper parameters of the mixture density network and gaussian mixture distribution
        """
        self.L = L # number of lags to be considered, which amounts to the number of input features of the mdn
        self.K = K # number of mixture components

        self.neurons = neurons # number of neurons per hidden layer
        self.layers = layers # number of hidden layers
        self.batch_size = batch_size # size of the (random) sample used to update gradient once
        self.epochs = epochs # final number of rounds for the entire training set used once for updating the gradient

        """act_func:str, eta_x:float, eta_y:float,
                univar_output:bool"""


    def create_training_data(self):

        """
        this function orders the simulated data (2d array) and prepares mixture density network inputs
        """

        # order the simulated data with by splitting the observations into a train and test set 
        # through a rolling window applied onto each mc simulation and then finally concatenated in the end
        MC_replications = self.data_sim.shape[1] 
        T = self.data_sim.shape[0]
        
        X_train_rows = MC_replications*(T-self.L)
        X_train = np.zeros((X_train_rows, self.L)) # initialize regressor matrix 
        
        # fill X_train according to rolling window of shape L
        for i in range(MC_replications):

            for j in range(T - self.L):

                X_train[i, j : j + self.L] = self.data_sim # ??? 
          
            print("blub")

        # copying
        # data_sim_tra = np.transpose(self.data_sim)
        test = np.array([self.data_sim[i, j : j + self.L] for i in range(self.data_sim.shape[0]) for j in range(self.data_sim.shape[1] - self.L)])
        MC_replications*(T-self.L)
        test2 = np.array([np.transpose(self.data_sim)[i, j : j + self.L] for i in range(np.transpose(self.data_sim).shape[0]) for j in range(np.transpose(self.data_sim).shape[1] - self.L)])

        # split the combined large set of MC simulations again into feature data X and target data Y


        return X_train, Y_train
    
           
    def estimate_mixture(self):
        
        """
        This method estimates the parameter of a gaussian mixture distribution with a
        feed forward neural network, called mixture density network using simulated and real data as input 
        - outputs the estimated parameter values. 
        """

        # construct one large ordered training set 
        mdn.create_training_data(self)
        # X_train, Y_train = mdn.create_training_data()
        print("blub")


        # FOR EACH ORDERED MC ??!!

        # set_mdn : mdn object
        # print "keras nn successfully created.. " ??

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

 
    
    def rolling_window_sim(self):

        """
        this function order the simulated data and prepare mixture density network inputs
        1-d array
        """


