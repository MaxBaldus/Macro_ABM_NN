"""
Here two sampling algorithms are implemented which sample the posterior distribution and output the approximated densities.
The first method simply employes latin hypercube sampling in order to obtain potential parameter values.
The second one is a metropolis hastings algorithm which explores the parameter space dependent on the parameter value from before (MCMC).
"""

# libraries
import numpy as np
from tqdm import tqdm
from itertools import product
from scipy.stats import qmc

# classes
from estimation.mdn import mdn

class sample_posterior:

    def __init__(self, model, bounds, data_obs):
        
        """
        Initiate the posterior sampling class by inputing 
        - observed data
        - the class of the agent based model 
        - the respective parameter bounds as a 1-d numpy array for the free parameters of the model (parameters to be estimated)
        """

        self.model = model # agent based model class with a simulation function
        self.bounds = bounds # upper lower parameter bounds (2d numpy array) with two rows for each parameter
        self.data_obs = data_obs # observed data: 1-d numpy array

    def grid_search(self, grid_size, path):

        """
        1) Simulation block: simulation and storing the TxMC matrix for each parameter combination
        """
        print("")
        print('--------------------------------------')
        print("Simulation Block")

        # get the number of parameters to be estimated
        number_para = self.bounds.shape[1] 

        # get the lower and upper bounds
        l_bounds = list(self.bounds[0,:])
        u_bounds = list(self.bounds[1,:])

        # create latin hypercube sampler
        latin_sampler = qmc.LatinHypercube(d=number_para)

        # sample theta values between 0 and 1 and scale to the given bounds (ranges)
        theta = latin_sampler.random(n=grid_size)
        qmc.scale(theta, l_bounds, u_bounds)
       
        # ??? NECESSARY ???
        # create all possible theta combinations for the grid_size of each parameter
        # test = np.array(np.meshgrid(theta[:,0],theta[:,1], theta[:,2], theta[:,3])).T.reshape(-1,4)

        """# uncomment this section for creating new training and test samples for a different grid and combinations 
        
        # simulate the model MC times for each parameter combination and save each TxMC matrix
        print("")
        print("Simulate the model MC times for each parameter combination:")
        for i in tqdm(range(grid_size)):
            
            # simulate the model each time with a new parameter combination from the grid
            simulations = self.model.simulation(theta[i,:])

            # current path to save the simulation to
            current_path = path + '_' + str(i)

            # save simulated data 
            np.save(current_path, simulations)
        """
        
        # apply filters later !!
        # if filters=True:
            # load data and apply filter to each column 
            # save again filtered data.. 
            

        """
        2) Likelihood ll block: compute the likelihood and the posterior probability of each parameter combination.
        """
        print("")
        print('--------------------------------------')
        print("Likelihood block and evaluating the posterior for each parameter")

        # approximate likelihood and evaluate posterior for each parameter combination (and corresponding TxMC matrix with simulated data)
        for i in tqdm(range(grid_size)):

            # load the simulated data for the current parameter combination
            load_path = path + '_' + str(i) + '.npy'
            simulation = np.load(load_path)

            # neglect the first simlated values for each mc column to ensure convergence 
            # of the major report variables
          
            # instantiate the likelihood approximation method
            ll_appro = mdn(data_sim = simulation, data_obs = self.data_obs,
                           L = 3, K = 16, 
                           neurons = 32, layers = 3, batch_size = 512, epochs = 12, 
                           eta_x=0.2, eta_y=0.2, act_fct="relu"
                           ) 
            
            # approximate the likelihood of the observed data, given the current parameter combination
            ll = ll_appro.approximate_likelihood()
            print("blub")

            # evaluate the posterior
            posterior_value = ll_appro.estimate_mixture(ll)

        # for each TxMC matrix (i.e. for each parameter combination)

            # instantiate mde object by inputting data -> object = mdn(DATA)
            # mixture_params = object.sample_mixture(NN PARAMETER)
            # ll = object.eval_mixture(mixture_params)

            # compute prior proba p(theta)

            # store posterior for each parameter combination !!!
            
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