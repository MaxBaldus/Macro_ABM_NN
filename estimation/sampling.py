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
from estimation.mde import mdn

class sample_posterior:

    def __init__(self, model, bounds, data_obs):
        
        """
        Initiate the posterior sampling class by inputing 
        - observed data
        - the class of the agent based model 
        - the respective parameter bounds as a numpy array for the free parameters of the model (parameters to be estimated)
        """

        self.model = model # agent based model class with a simulation function
        self.bounds = bounds # upper lower parameter bounds (2d numpy array) with two rows for each parameter
        self.data_obs = data_obs # observed data: 1-d numpy array

    def grid_search(self, grid_size, path):

        """
        1) Simulation block
        """
        print("")
        print('--------------------------------------')
        print("Simulation Block (creating grid)")

        # get the number of parameter to be estimated
        number_para = self.bounds.shape[1] 

        # initialize matrix to store the grid as one column for each parameter
        theta = np.zeros((grid_size,number_para))

        # create equally spaced grid of values for each parameter
        for b in range(number_para):
            lower = self.bounds[0,b]
            upper = self.bounds[1,b]
            theta[:,b] = np.linspace(lower, upper, num=grid_size)
        
        # create all possible theta combinations for the grid_size of each parameter
        # test = np.array(np.meshgrid(theta[:,0],theta[:,1], theta[:,2], theta[:,3])).T.reshape(-1,4)
        list_of_theta = [theta[:,i] for i in range(number_para)] # each parameter column (np.array) in list
        theta_combination = np.array(np.meshgrid(*list_of_theta)).T.reshape(-1,number_para) # create meshgrid from the parameter columns
        
        # total number of parameter combination
        num_para_combinations = grid_size**number_para 

        if num_para_combinations == theta_combination.shape[0]:
            print("")
            print("parameter combination array, similar to 'nested for-loops', successfully created")
            print("")
            print("number of parameter combinations: %s" %num_para_combinations)

        # uncomment this section for creating new training and test samples for a different grid and combinations 
        """
        # initialize 3-d array to store MC simulations (TxMC matrices) for each parameter combination
        simulations = np.zeros((num_para_combinations, self.model.Time, self.model.MC))  # TxMC matrix for each para_combination

        # simulate the model MC times for each parameter combination and store each TxMC matrix (of each combination) into a 3-d array
        print("")
        print("Simulate the model MC times for each parameter combination:")
        for i in tqdm(range(num_para_combinations)):

            # simulate the model each time with a new parameter combination from the grid
            simulations[i] = self.model.simulation(theta[i,:])

        # save simulated data 
        # np.save(path, simulations)
        """

        # load the simulated data 
        load_path = path + '.npy'
        simulations = np.load(load_path)

        """
        2) Likelihood ll block 
        """
        print("")
        print('--------------------------------------')
        print("Likelihood block and evaluating the posterior for each parameter")

        # approximate likelihood and evaluate posterior for each parmaeter combination
        for i in range(simulations.shape[0]): # range(num_para_combinations)
            
            # instantiate the likelihood approximation method
            ll_appro = mdn(data_sim = simulations[i], data_obs = self.data_obs,
                           L = 3, K = 16, 
                           neurons = 32, layers = 3, batch_size = 512, epochs = 12
                           ) 
            
            ll_appro.estimate_mixture()
            print("blub")


            # approximate the likelihood
            ll_appro.estimate_mixture()

        # for each TxMC matrix (i.e. for each parameter combination)

            # instantiate mde object by inputting data -> object = mdn(DATA)
            # mixture_params = object.sample_mixture(NN PARAMETER)
            # ll = object.eval_mixture(mixture_params)

            # compute prior proba p(theta)

            # store for each parameter combination 
            
        # return ll + p(theta) and ll

    def MH_sampling(self):

        print("blub")
        # now mh sampling
        # 
#################################################################################################

    def posterior_plots(self):

        print("blub")
        # output posterior estimates (mean of theta chain)
        # plot the sampled posterior parameter values 


    def prior(self):
        
        # use self.bounds to compute prior probabilities with bounds
        # use uniform priors (sample from uniform distribution for current theta, always with the same boundss)
        print("blub")
    

    def hypercube_sampling(self):

        print("blub")
        # use scipy.stats.qmc.LatinHypercube instead of of equally spaced parameter values (i.e. np.linspace)