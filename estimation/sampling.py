"""
Here two sampling algorithms are implemented which sample the posterior distribution and output the approximated densities.
The first simply uses a pre-defined grid with equally spaced parameter values.
The second one is a metropolis hastings algorithm which explores the parameter space (computationally more evolved).
"""

# libraries
import numpy as np
from tqdm import tqdm


# classes
#from mde import mdn


class sample_posterior:

    def __init__(self, model, bounds):
        
        """
        Initiate the posterior sampling class by inputing the class of the agent based model and
        the respective parameter bounds as a numpy array for the free parameters of the model 
        (parameters to be estimated)
        """

        self.model = model
        self.bounds = bounds

    def grid_search(self, grid_size, path):

        """
        Simulation block
        """

        # get the number of parameter
        number_para = self.bounds.shape[1] 

        # initialize matrix to store the grid as one column for each parameter
        theta = np.zeros((grid_size,number_para))

        # create equally spaced grid of values for each parameter
        for b in range(number_para):
            lower = self.bounds[0,b]
            upper = self.bounds[1,b]
            theta[:,b] = np.linspace(lower, upper, num=grid_size)
        
        # simulate the model MC times and store each TxMC matrix into 3-d array
        simulations = np.zeros((grid_size, self.model.Time, self.model.MC))  # grid_size TxMC matrices
        
        for i in tqdm(range(grid_size)):

            # simulate the model grid_size times, each time with a new parameter combination from the grid
            simulations[i] = self.model.simulation(theta[i])

        np.save(path, simulations)
        # load the data 

        """
        Likelihood ll block 
        """
        # for each TxMC matrix (i.e. for each parameter combination)
        # instantiate mde object by inputting data -> object = mdn(DATA)
        # mixture_params = object.sample_mixture()
        # ll = object.eval_mixture(mixture_params)

        # compute prior proba p(theta)
            
        # return ll + p(theta) and ll

    def MH_sampling(self):

        print("blub")
        # now mh sampling
        # 

    def posterior_plots(self):

        print("blub")
        # output posterior estimates (mean of theta chain)
        # plot the sampled posterior parameter values 


    def prior(self):
        
        # use self.bounds to compute prior probabilities with bounds
        # use uniform priors (sample from uniform distribution for current theta, always with the same boundss)
        print("blub")
         