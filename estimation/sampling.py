"""
Here two sampling algorithms are implemented which sample the posterior distribution and output the approximated densities.
The first simply uses a pre-defined grid with equally spaced parameter values.
The second one is a metropolis hastings algorithm which explores the parameter space (computationally more evolved).
"""

# libraries
import numpy as np
from tqdm import tqdm
from itertools import product


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

        # get the number of parameter to be estimated
        number_para = self.bounds.shape[1] 

        # initialize matrix to store the grid as one column for each parameter
        theta = np.zeros((grid_size,number_para))

        # create equally spaced grid of values for each parameter
        for b in range(number_para):
            lower = self.bounds[0,b]
            upper = self.bounds[1,b]
            theta[:,b] = np.linspace(lower, upper, num=grid_size)
        
        test = np.array(np.meshgrid(theta[:,0],theta[:,1], theta[:,2], theta[:,3])).T.reshape(-1,4)
        unpack = [theta[:,i] for i in range(number_para)]
        test2 = np.array(np.meshgrid(*unpack)).T.reshape(-1,number_para)

        # create all possible theta combinations for the grid_size of each parameter
        num_para_combinations = grid_size**number_para # total number of parameter combination
        theta_combination = np.zeros((num_para_combinations, number_para))

        for comb in range(num_para_combinations):
            for row in range(theta.shape[0]):
                theta_combination[comb,:] = theta[row,:]


        # initialize 3-d array to store MC simulations (TxMC matrices) for each parameter combination
        # uncomment this for creating new training and test samples 
        num_para_combinations = grid_size**number_para # total number of parameter combination
        para_combination = np.zeros((num_para_combinations, number_para))
        # simulations = np.zeros((num_para_combinations, self.model.Time, self.model.MC))  # MC TxMC matrices for each para_combination

        # initialie list of all possible parameter combinations
        current_theta = [0 for x in range(number_para)]
        for combination in product(theta[0]):
            print(combination)
        
        print("blub")

        
        # simulate the model MC times for each parameter combination and store each TxMC matrix into a 3-d array
        """for i in tqdm(range(grid_size)):

            # simulate the model grid_size times, each time with a new parameter combination from the grid
            simulations[i] = self.model.simulation(theta[i])

        # save simulated data 
        np.save(path, simulations)"""

        # load the 
        load_path = path + '.npy'
        simulations = np.load(load_path)

        """
        Likelihood ll block 
        """
        # for each TxMC matrix (i.e. for each parameter combination)
            # instantiate mde object by inputting data -> object = mdn(DATA)
            # mixture_params = object.sample_mixture()
            # ll = object.eval_mixture(mixture_params)

            # compute prior proba p(theta)

            # store for each parameter combination 
            
        # return ll + p(theta) and ll

    def MH_sampling(self):

        print("blub")
        # now mh sampling
        # 
    
    def hypercube_sampling(self):

        print("blub")
        # use scipy.stats.qmc.LatinHypercube instead of of equally spaced parameter values 

    def posterior_plots(self):

        print("blub")
        # output posterior estimates (mean of theta chain)
        # plot the sampled posterior parameter values 


    def prior(self):
        
        # use self.bounds to compute prior probabilities with bounds
        # use uniform priors (sample from uniform distribution for current theta, always with the same boundss)
        print("blub")
         