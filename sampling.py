"""
@author: maxbaldus

"""

# libraries
import numpy as np
from tqdm import tqdm
from itertools import product
from scipy.stats import qmc
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import multiprocessing

# classes
from estimation.mdn import mdn

class sample_posterior:

    """
    Here a grid-search sampling algorithms is implemented which samples the posterior distribution 
    and outputs the approximated densities for each parameter.
    The entire sampling routine is split into 2 blocks: A simulation and an estimation block.
    """

    def __init__(self, model, bounds, data_obs, filters):
        
        """
        Initiating the posterior sampling class by inputing 
        - observed data
        - the class of the agent based model 
        - the respective parameter bounds as a 1-d numpy array for the free parameters of the model (parameters to be estimated)
        """

        self.model = model # agent based model class with a simulation function
        self.bounds = bounds # upper lower parameter bounds (2d numpy array) with two rows for each parameter
        self.data_obs = data_obs # observed data: 1-d numpy array
        self.filters = filters

    """
    1) Simulation block: simulation and storing the TxMC matrix for each parameter combination
    """
    def simulation_block(self, grid_size, path):

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
        np.random.seed(123)
        theta = latin_sampler.random(n=grid_size)

        # scale parameters to the given bounds
        qmc.scale(theta, l_bounds, u_bounds)
                      
        # simulate the model MC times for each parameter combination and save each TxMC matrix
        print("")
        print("Simulate the model MC times for each parameter combination:")

        # num_cores = (multiprocessing.cpu_count()) - 4 
        num_cores = 4 # lux working station  

        """for i in tqdm(range(grid_size)):

            # current parameter combination
            theta_current = theta[i,:]
            # simulate the model each time with a new parameter combination from the grid
            simulations = self.model.simulation(theta_current)
            # current path to save the simulation to
            current_path = path + '_' + str(i)
            # save simulated data 
            np.save(current_path, simulations)
            # plot the first mc simulation
            plt.clf()
            plt.plot(np.log(simulations[:,0]))
            plt.xlabel("Time")
            plt.ylabel("Log output")
            plt.savefig(current_path + "png")"""


        # parallize the grid search: using joblib
        """def grid_search_parallel(grid_size, theta, model, path, i):
            
            for i in range(grid_size):

                # current parameter combination
                theta_current = theta[i,:]
                # simulate the model each time with a new parameter combination from the grid
                simulations = model.simulation(theta_current)
                
                # current path to save the simulation to
                current_path = path + '_' + str(i)
                # save simulated data 
                np.save(current_path, simulations)
                
                # plot the first mc simulation
                plt.clf()
                plt.plot(np.log(simulations[:,0]))
                plt.xlabel("Time")
                plt.ylabel("Log output")
                plt.savefig(current_path + ".png")

        Parallel(n_jobs=num_cores)(
                delayed(grid_search_parallel)
                (grid_size, theta, self.model, path, i) for i in range(grid_size)
                )"""


        # parallize the grid search: muliprocessing librarys
        args = []
        for i in range(grid_size):
            # args.append([theta[i,:], path + '_' + str(i)])
            args.append({'theta': theta[i,:], 'path': path + '_' + str(i)})
        
        return args



    """
    2) Estimation block: compute the likelihood and the posterior probability of each parameter (combination) of the abm model by delli gatti.
    """
    def approximate_posterior(self, grid_size, path):
            
        print("")
        print('--------------------------------------')
        print("Likelihood block and evaluating the posterior for each parameter")

        # instantiate the likelihood approximation method
        likelihood_appro = mdn(self.data_obs, L = 3, K = 2, 
                           neurons = 32, layers = 3, batch_size = 512, epochs = 12, 
                           eta_x=0.2, eta_y=0.2, act_fct="relu"
                           ) 

        # initiate the vector with likelihood values for each parameter combination
        ll = np.zeros(grid_size)
        
        # initiate the matrix with marginal posterior probablities
        number_parameter = np.shape(self.bounds)[1]
        log_posterior = np.zeros((grid_size, number_parameter))

        # approximate likelihood and evaluate posterior for each parameter combination (and corresponding TxMC matrix with simulated data)
        for i in tqdm(range(grid_size)):
            i = i + 2
            # load the simulated data for the current parameter combination
            load_path = path + '_' + str(i) + '.npy'
            simulation = np.load(load_path)

            # neglect the first simlated values for each mc column to ensure convergence of the major report variables 
            # only use last observations with length of the observed ts
            simulation_short = simulation[simulation.shape[0]-len(self.data_obs) : simulation.shape[0],:]

            # if self.filter:
                # filter
            # LOG FILTER ??!!
            
            # approximate the posterior probability of the given parameter combination
            densities = likelihood_appro.approximate_likelihood(simulation_short)
            
            # compute likelihood of the observed data for the given parameter combination
            # likelihood = np.prod(densities)
            ll = np.sum(np.log(densities))

            # sample the prior probabilities (AGAIN FOR EACH LIKELIHOOD VALUE ?! YES)
            np.random.seed(i)
            marginal_priors = self.sample_prior() 

            # compute marginal log posteriors
            log_posterior[i,:] = ll + np.log(marginal_priors)
            
            # QUESTIONS:
            # one ll value * each prior value for the marginal posterior ???????
            # prior.prod for the joint posterior ??!!  JA ?!
            # if having marcov chain => each theta candidat is vector (one value for each parameter in vector)
            # => computing likelihood * prior.product() to evaluate candidate (JOINTLY for all candidates)
            # -inf ??!! -> - inf to 0 .. : just discard when plotting ??!!
            # likelihood value positive ??!!
            
            print("blub")
            # 12th loss : -96.4340


#################################################################################################

    def sample_prior(self):

        """
        Sample uniform prior probabilities for each theta, according to its bounds
        """

        # use self.bounds to compute prior probabilities with the the respective bounds
        number_parameter = np.shape(self.bounds)[1]
        prior_probabilities = np.zeros(number_parameter)
        
        for i in range(number_parameter):
            prior_probabilities[i] = np.random.uniform(low=self.bounds[0,i], high=self.bounds[1,i])
        
        return prior_probabilities


    def posterior_plots(self):

        print("blub")
        # output posterior estimates (mean of theta chain)
        # plot the sampled posterior parameter values 