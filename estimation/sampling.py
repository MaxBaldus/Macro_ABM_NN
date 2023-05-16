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

import time

# classes
from estimation.mdn import mdn

class sample_posterior:

    """
    Here a grid-search sampling algorithms is implemented which samples the posterior distribution 
    and outputs the approximated densities for each parameter.
    The entire sampling routine is split into 2 blocks: A simulation and an estimation block.
    """

    def __init__(self, model, bounds, data_obs, filter):
        
        """
        Initiating the posterior sampling class by inputing 
        - observed data
        - the class of the agent based model 
        - the respective parameter bounds as a 1-d numpy array for the free parameters of the model (parameters to be estimated)
        """

        self.model = model # agent based model class with a simulation function
        self.bounds = bounds # upper lower parameter bounds (2d numpy array) with two rows for each parameter
        self.data_obs = data_obs # observed data: 1-d numpy array
        self.filter = filter

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
        latin_sampler = qmc.LatinHypercube(d=number_para, seed=123)

        # sample theta values between 0 and 1 and scale to the given bounds (ranges)
        Theta = latin_sampler.random(n=grid_size)

        # scale parameters to the given bounds
        qmc.scale(Theta, l_bounds, u_bounds)
                      
        # simulate the model MC times for each parameter combination and save each TxMC matrix
        print("")
        print("Simulate the model MC times for each parameter combination:")

        # num_cores = (multiprocessing.cpu_count()) - 4 
        num_cores = 56 # lux working station  

        # 1) Simulation block (outsourced to main.py)
        # simulation and storing the TxMC matrix for each parameter combination
        
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
        """def grid_search_parallel(theta, model, path, i):

            # current parameter combination
            # theta_current = theta[i,:]
            
            # simulate the model each time with a new parameter combination from the grid
            simulations = model.simulation(theta[i,:])
            
            # current path to save the simulation to
            # current_path = path + '_' + str(i)
            
            # save simulated data 
            np.save(path + '_' + str(i), simulations)
            
            # plot the first mc simulation
            plt.clf()
            plt.plot(np.log(simulations[:,0]))
            plt.xlabel("Time")
            plt.ylabel("Log output")
            plt.savefig( path + '_' + str(i) + ".png")

            return

        Parallel(n_jobs=num_cores)(
                delayed(grid_search_parallel)
                (grid_size, theta, self.model, path, i) for i in range(grid_size)
                )"""
        
        return Theta



    """
    2) Estimation block: compute the likelihood and the posterior probability of each parameter (combination) of the abm model by delli gatti.
    """
    def approximate_posterior(self, grid_size, path):
            
        print("")
        print('--------------------------------------')
        print("Likelihood block and evaluating the posterior for each parameter")

        # apply filter or transformation to observed time series
        data_obs_log = np.log(self.data_obs)

        # instantiate the likelihood approximation method
        likelihood_appro = mdn(data_obs_log, L = 3, K = 16, 
                           neurons = 32, layers = 3, batch_size = 512, epochs = 12, 
                           eta_x=0.2, eta_y=0.2, act_fct="relu"
                           ) 

        # initiate the vector with likelihood values for each parameter combination
        # ???
        # ll = np.zeros(grid_size)
        
        # initiate the matrix with marginal posterior probablities
        number_parameter = np.shape(self.bounds)[1]
        posterior = np.zeros((grid_size, number_parameter))
        log_posterior = np.zeros((grid_size, number_parameter))

        # save start time
        start_time = time.time()
        
        # approximate likelihood and evaluate posterior for each parameter combination (and corresponding TxMC matrix with simulated data)
        # for i in tqdm(range(grid_size)):
        """for i in [27, 31, 45, 49]:
           
            # load the simulated data for the current parameter combination
            load_path = path + '_' + str(i) + '.npy'
            simulation = np.load(load_path)

            # neglect the first simlated values for each mc column to ensure convergence of the major report variables 
            # only use last observations with length of the observed ts
            simulation_short = simulation[simulation.shape[0]-len(self.data_obs) : simulation.shape[0],:]

            if self.filter:
                # filter
                print("blub")
            else:
                # log transformation
                simulation_short = np.log(simulation_short)

            # TEST!! same data -> apply single column filter !!
            # simulation_short = data_obs_log
            
            # approximate the posterior probability of the given parameter combination
            densities = likelihood_appro.approximate_likelihood(simulation_short)

            # USE HIS SCALING => otherwise ll huge and negative (hence really small probabilities)
            
            # compute likelihood of the observed data for the given parameter combination
            L = np.prod(densities)
            ll = np.sum(np.log(densities))

            # sample the prior probabilities (AGAIN FOR EACH LIKELIHOOD VALUE ?! YES)
            np.random.seed(i)
            marginal_priors = self.sample_prior() 

            # compute marginal (log) posteriors
            posterior[i,:] = L * marginal_priors
            log_posterior[i,:] = ll + np.log(marginal_priors)"""

        # using parallel computing
        def approximate_parallel(path, i):
           
            # load the simulated data for the current parameter combination
            load_path = path + '_' + str(i) + '.npy'
            simulation = np.load(load_path)

            # neglect the first simlated values for each mc column to ensure convergence of the major report variables 
            # only use last observations with length of the observed ts
            simulation_short = simulation[simulation.shape[0]-len(self.data_obs) : simulation.shape[0],:]

            if self.filter:
                # filter
                print("blub")
            else:
                # log transformation
                simulation_short = np.log(simulation_short)
           
            # approximate the posterior probability of the given parameter combination
            densities = likelihood_appro.approximate_likelihood(simulation_short)

            # USE HIS SCALING => otherwise ll huge and negative (hence really small probabilities)
            
            # compute likelihood of the observed data for the given parameter combination
            L = np.prod(densities)
            ll = np.sum(np.log(densities))

            # sample the prior probabilities (AGAIN FOR EACH LIKELIHOOD VALUE ?! YES)
            np.random.seed(i)
            marginal_priors = self.sample_prior() 

            # compute marginal (log) posteriors
            posterior[i,:] = L * marginal_priors
            log_posterior[i,:] = ll + np.log(marginal_priors)

            return posterior, log_posterior 
            
        posterior, log_posterior = Parallel(n_jobs=56)(
        delayed(approximate_parallel)
        (path, i) for i in range(70)
        )

        np.save('estimation/BAM/log_posterior', log_posterior)
        np.save('estimation/BAM/posterior', posterior)

        # test with range = 70
        
        print("")
        print("--- %s minutes ---" % ((time.time() - start_time)/60))

        # parallel: 10 theta with 5 MC => 3.6551249821980796 minutes

        # Problem: WITHOUT using log transformation
        # - densities <= 1 , hence np.log(densities) < 0  => large negative number for ll
        # - also np.log(marginal_priors) < 0 
        # => hence large negative number for ll + np.log(marginal_priors) 

        # using logs => probabilities > 1, hence not really probabilities: but easier to eye-ball distribution??
        # solution?!: return - log posterior and scale values when plotting ?!
        
        # QUESTIONS:
        # one ll value * each prior value for the marginal posterior ???????
        # prior.prod for the joint posterior ??!!  JA ?!
        # if having marcov chain => each theta candidat is vector (one value for each parameter in vector)
        # => computing likelihood * prior.product() to evaluate candidate (JOINTLY for all candidates)
        # -inf ??!! -> - inf to 0 .. : just discard when plotting ??!!
        # likelihood value positive ??!!
        # WHAT IS THE POSTERIOR PARAMETER ESTIMATOR NOW: 
        # since having grid: use value with highest proba as estimator (not mean of the parameter values, since no MC chain)
        
        # ISSUES
        # Influence of prior rather small: array([-2.34817589, -2.12968255, -2.17047113, -2.629976  ]) vs ll = -750.96674469632

        # check losses:
        # 12th loss : -96.4340
        # 500 run: -84.1205

        # good simulations: 27, 31, 45, 49, 
        # CHECK WHETHER log_posterior values large (large negative values?!)

        return log_posterior, posterior


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


    def posterior_plots(self, log_posterior):

        print("blub")
        # output posterior estimates (mean of theta chain)
        # plot the sampled posterior parameter values 

        # save plots  'plots/posterior/BAM/NAME'
        # scale values ?!