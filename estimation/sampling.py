"""
sampling and estimation routine
@author: maxbaldus
"""

# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.interpolate import make_interp_spline

from scipy.stats import gaussian_kde
#import seaborn as sns

from tqdm import tqdm
from itertools import product

from joblib import Parallel, delayed
import multiprocessing

from sklearn import preprocessing

import time

# classes
from estimation.mdn import mdn
from estimation.data_prep import Filters

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
        self.filter = filter # set to True: filter is applied 

    """
    1) Simulation block
    """
    def simulation_block(self, grid_size, path, order_Theta):

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
        Theta = qmc.scale(Theta, l_bounds, u_bounds)
        
        # check Theta values on random basis
        """plt.scatter(Theta[:,0], Theta[:,1])
        first = [i for i in Theta[:,0] < 0.011]
        check = Theta[first]
        # slice = [i for i in Theta[:,0] if i < 0.011]
        # Theta[:,0][slice]"""
               
        # order Theta parameter in ascending order 
        if order_Theta:
            for i in range(number_para):
                Theta[:,i] = np.sort(Theta[:,i]) 
        
        
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
            plt.savefig(current_path + "png")

        # parallel computing using joblib
        def grid_search_parallel(theta, model, path, i):

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
    def approximate_posterior(self, grid_size, path, t_zero, kde, empirical):
            
        # apply filter to observed time series
        if self.filter:
                filters = Filters(self.data_obs, inflation=None, unemployment=None)
                components = filters.HP_filter(empirical = False)

                # use cyclical component of HP filter as observed data
                data_obs = components[0]  # both cyclical and trend component
        else:
            data_obs = self.data_obs
            #data_obs = np.log(self.data_obs) # use log transforms

        # instantiate the likelihood approximation method
        likelihood_appro = mdn(data_obs, L = 3, K = 16, 
                           neurons = 32, layers = 3, batch_size = 512, epochs = 12, 
                           eta_x=0.2, eta_y=0.2, act_fct="relu"
                           ) 

               
        # initiate the matrix with marginal posterior probablities
        number_parameter = np.shape(self.bounds)[1]
        posterior = np.zeros((grid_size, number_parameter))
        log_posterior = np.zeros((grid_size, number_parameter))
        Likelihoods = np.zeros(grid_size)
        log_Likelihoods = np.zeros(grid_size)

        # sample the prior probabilities for each parameter combination
        marginal_priors = np.zeros((grid_size, number_parameter))
        for i in range(grid_size):
            marginal_priors[i,:] = self.sample_prior() 
                
        """
        Approximate likelihood and evaluate posterior for each parameter combination (and corresponding TxMC matrix with simulated data)
        """
        # without parallised computing, mostly used for testing 
        # for i in tqdm(range(grid_size)):
        """for i in range(grid_size):
            
            # choose a good simulation for testing
            i = 11
            
            # load the simulated data for the current parameter combination
            load_path = path + '_' + str(i) + '.npy'
            simulation = np.load(load_path)

            # neglect the first simlated values for each mc column to ensure convergence of the major report variables 
            # only use last observations with length of the observed ts
            # simulation_short = simulation[simulation.shape[0]-len(self.data_obs) : simulation.shape[0],]
            
            #simulation_short = simulation[simulation.shape[0]-len(self.data_obs) : simulation.shape[0],0:12]
            #simulation_short = np.log(simulation[simulation.shape[0]-len(self.data_obs) : simulation.shape[0],])
            
            # use values from t_0 onwards
            if empirical:
                # scale the simulated data for the empirical applicatin
                #simulation_short = simulation[t_zero : simulation.shape[0],:] * 100 # scale by 100 for US GDP 
                simulation_short = simulation[t_zero : simulation.shape[0],:] * 10000
            
            else:
                simulation_short = simulation[t_zero : simulation.shape[0],:]


            # apply filter to simulated time series
            if self.filter:
                simulation_short_filtered = np.zeros(simulation_short.shape)
                
                # apply filter to each column of simulated data 
                for i in range(simulation_short.shape[1]):
                    filters = Filters(simulation_short[:,i], inflation=None, unemployment=None)
                    components = filters.HP_filter(empirical = False) # both cyclical and trend component

                    # save cyclical component of HP filter
                    simulation_short_filtered[:,i] = components[0] 
                    
                # use filtered time series from now on forward
                simulation_short = simulation_short_filtered
            
            # apply log transformation if no filter is used:
            else:
                simulation_log = np.log(simulation_short)
                simulation_short = simulation_log
                
            # test: benutze die gleichen Daten 
            #for i in range(20): 
                #simulation_short[:,i] = data_obs 
                        
           # approximate the posterior probability of the given parameter combination
            if kde:
                likelihoods = likelihood_appro.kde_approximation(simulation_short)
            else:
                # using mdns
                likelihoods = likelihood_appro.approximate_likelihood(simulation_short)

            # compute likelihood of the observed data for the given parameter combination
            L = np.prod(likelihoods)
            
            # replace 0 likelihood values with 0.0001 in order to avoid log(0) = -inf
            # if any(likelihoods == 0) == True:
                #densities[np.where(likelihoods == 0)[0]] = min(likelihoods) 
                       
            ll = np.sum(np.log(likelihoods)) 
            
            # testing: for i=27, with 1/std(y_tilde): ll = -2446.8013911398457, without 1/std(y_tilde): ll = -1529.9878020603692  , 
            # testing: for i=49, with 1/std(y_tilde): ll = -inf , 
            # testing: for i=45 (keine 0's), with 1/std(y_tilde): ll = -inf , without 1/std(y_tilde): ll = -inf  , 
            
            # compute marginal (log) posteriors
            posterior[i,:] = L * marginal_priors[i,:]
            log_posterior[i,:] = ll + np.log(marginal_priors[i,:])
            
            # bei i = 999: L = 1.1623265223664991e-275 => * 275
            print("")"""
            

        # using parallel computing
        def approximate_parallel(path, marginal_priors, i):
           
             # load the simulated data for the current parameter combination
            load_path = path + '_' + str(i) + '.npy'
            simulation = np.load(load_path)
            
            # neglect the first simlated values for each mc column to ensure convergence of the major report variables 
            # use values from t_0 onwards
            if empirical:
                # scale the simulated data for the empirical applicatin
                #simulation_short = simulation[t_zero : simulation.shape[0],:] * 100 # scale by 100 for US GDP 
                simulation_short = simulation[t_zero : simulation.shape[0],:] * 10000 # German GDP
            
            else:
                simulation_short = simulation[t_zero : simulation.shape[0],:]
            
            # apply filter to simulated time series
            if self.filter:
                simulation_short_filtered = np.zeros(simulation_short.shape)
                
                # apply filter to each column of simulated data 
                for i in range(simulation_short.shape[1]):
                    filters = Filters(simulation_short[:,i], inflation=None, unemployment=None)
                    components = filters.HP_filter(empirical = False) # both cyclical and trend component

                    # save cyclical component of HP filter
                    simulation_short_filtered[:,i] = components[0] 
                    
                # use filtered time series from now on forward
                # simulation_short = simulation_short_filtered
                simulation_short = simulation_short_filtered 
            
            # apply log transformation if no filter is used:
            """else:
                simulation_log = np.log(simulation_short)
                simulation_short = simulation_log"""
                
            # approximate the posterior probability of the given parameter combination
            if kde:
                # using kernel density estimation
                likelihoods = likelihood_appro.kde_approximation(simulation_short)
            else:
                # using mdns
                likelihoods = likelihood_appro.approximate_likelihood(simulation_short)
            
            # compute likelihood of the observed data for the given parameter combination
            L = np.prod(likelihoods)
            ll = np.sum(np.log(likelihoods))

            priors = marginal_priors[i,:]

            return L, ll, L * priors, ll + np.log(priors) 
        
        computations = Parallel(n_jobs=32)(
        delayed(approximate_parallel)
        (path, marginal_priors, i) for i in range(grid_size) 
        )

        # unpack and save the parallel computed posterior probabilities
        for i in range(len(computations)):
            Likelihoods[i] = computations[i][0]
            log_Likelihoods[i] = computations[i][1]
            posterior[i,:] = computations[i][2]
            log_posterior[i,:] = computations[i][3]

        return posterior, log_posterior, marginal_priors, Likelihoods, log_Likelihoods


#################################################################################################

    def sample_prior(self):

        """
        Sample uniform prior probabilities for each theta, according to its bounds
        """

        # use self.bounds to compute prior probabilities with the the respective bounds for each parameter to be estimated
        number_parameter = np.shape(self.bounds)[1]
        prior_probabilities = np.zeros(number_parameter)
        
        for i in range(number_parameter):
            prior_probabilities[i] = np.random.uniform(low=self.bounds[0,i], high=self.bounds[1,i])
        
        return prior_probabilities
    
    
    def posterior_plots_identification(self, Theta, posterior, log_posterior, marginal_priors, Likelihoods, log_Likelihoods,
                        para_names, path, plot_name, bounds_BAM, true_values, zoom):

        """
        Plot posterior and log posterior and prior probabilities
        """
        
        # get number of parameter
        number_parameter = np.shape(Theta)[1]
                
        """
        1) log posterior 
        """
        # handle NANs
        nans = np.sum(np.isnan(log_posterior))/bounds_BAM.shape[1]
        print("number of NANs in log posterior: %s" %nans)
        slicer_nan = np.isnan(log_posterior)
        log_posterior_non_NAN = log_posterior[~slicer_nan.any(axis=1)]
        Theta_non_NAN = Theta[~slicer_nan.any(axis=1)]
        prior_non_NAN = marginal_priors[~slicer_nan.any(axis=1)]
        
        # handle -inf values
        infs = np.sum(np.isinf(log_posterior))/bounds_BAM.shape[1]
        print("number of -inf in log posterior: %s" %infs)
        slicer_inf = np.isinf(log_posterior_non_NAN)
        log_posterior_non_NAN_inf = log_posterior_non_NAN[~slicer_inf.any(axis=1)]
        Theta_non_NAN_inf = Theta_non_NAN[~slicer_inf.any(axis=1)]
        prior_non_NAN_inf = prior_non_NAN[~slicer_inf.any(axis=1)]
        
        if zoom:
            # scaling log posterior values btw. 0 and 10
            log_posterior_scaled = preprocessing.minmax_scale(log_posterior_non_NAN_inf, feature_range=(0, 10))
            
        else:
            # scaling log posterior values btw. 0 and 10
            log_posterior_scaled = preprocessing.minmax_scale(log_posterior_non_NAN_inf, feature_range=(0, 10))
        
        # log_posterior_scaled = log_posterior_non_NAN_inf
                   
        for i in range(number_parameter):
                
            # mode of posterior is final parameter estimate
            max_post = Theta_non_NAN_inf[np.argmax(log_posterior_non_NAN_inf[:,i]),i]
            
            plt.clf()
            
            # zoom into certain range when plotting, neglecting other values
            if zoom:
                # log posterior values
                plt.plot(Theta_non_NAN_inf[0:2000,i], log_posterior_scaled[0:2000,i], color='b', linewidth=0.5, label='scaled log Posterior values')
                # prior values
                plt.plot(Theta_non_NAN_inf[0:2000,i], prior_non_NAN_inf[0:2000,i], linewidth=0.5, color = 'orange', label = 'Prior density values')
                
            else:
                # log posterior values
                plt.plot(Theta_non_NAN_inf[:,i], log_posterior_scaled[:,i], color='b', linewidth=0.5, label='scaled log Posterior values')
                
                # prior values
                plt.plot(Theta_non_NAN_inf[:,i], prior_non_NAN_inf[:,i], linewidth=0.5, color = 'orange', label = 'Prior density values')
                
            plt.xlabel(para_names[i])
            plt.ylabel(r'$log$' + ' ' + r'$p($' + para_names[i] + r'$|X)$')
            
            # parameter estimates
            plt.axvline(x = max_post,  color = 'red', linestyle = 'dashed', alpha = 0.75, label = r'$\hat \theta$ =' + str(np.round(max_post, 4)))
            plt.axvline(x = true_values[i], linestyle = 'dashed', alpha = 0.75, label = "true " + r'$\theta$', c = 'k')
            
            # legend
            plt.legend(loc="lower right", fontsize = 8)
            if zoom:
                plt.savefig(path + plot_name + 'ZOOM' +'_log_post_' + 'parameter_' + str(i) + '.png')
            else:
                plt.savefig(path + plot_name + '_log_post_' + 'parameter_' + str(i) + '.png')
        
    def posterior_plots_unordered(self, Theta, posterior, log_posterior, marginal_priors, Likelihoods, log_Likelihoods,
                        para_names, path, plot_name, bounds_BAM, true_values):

        """
        Plot posterior and log posterior and prior probabilities
        """
        
        # get number of parameter
        number_parameter = np.shape(Theta)[1]
                
        """
        1) log posterior 
        """
        # handle NANs
        nans = np.sum(np.isnan(log_posterior))/bounds_BAM.shape[1]
        print("number of NANs in log posterior: %s" %nans)
        slicer_nan = np.isnan(log_posterior)
        log_posterior_non_NAN = log_posterior[~slicer_nan.any(axis=1)]
        Theta_non_NAN = Theta[~slicer_nan.any(axis=1)]
        prior_non_NAN = marginal_priors[~slicer_nan.any(axis=1)]
        
        # handle -inf values
        infs = np.sum(np.isinf(log_posterior))/bounds_BAM.shape[1]
        print("number of -inf in log posterior: %s" %infs)
        slicer_inf = np.isinf(log_posterior_non_NAN)
        log_posterior_non_NAN_inf = log_posterior_non_NAN[~slicer_inf.any(axis=1)]
        Theta_non_NAN_inf = Theta_non_NAN[~slicer_inf.any(axis=1)]
        prior_non_NAN_inf = prior_non_NAN[~slicer_inf.any(axis=1)]
                  
        # scaling log posterior values btw. 0 and 10
        log_posterior_scaled = preprocessing.minmax_scale(log_posterior_non_NAN_inf, feature_range=(0, 10))
        
        # log_posterior_scaled = log_posterior_non_NAN_inf
                   
        for i in range(number_parameter):
            
            # order sampled parameter values (ascending order) and corresponding probabilities
            theta_ordered = np.sort(Theta_non_NAN_inf[:,i])
            theta_index = Theta_non_NAN_inf[:,i].argsort()
            log_posterior_array = log_posterior_scaled[:,i]
            log_posterior_ordered = log_posterior_array[theta_index]
            log_prior_array = prior_non_NAN_inf[:,i]
            log_prior_ordered = log_prior_array[theta_index]
                
            # mode of posterior is final parameter estimate
            max_post = Theta_non_NAN_inf[np.argmax(log_posterior_non_NAN_inf[:,i]),i]
            
            plt.clf()
                            
            # log posterior values
            plt.plot(theta_ordered, log_posterior_ordered, color='b', linewidth=0.5, label='scaled log Posterior values')
            
            # prior values
            plt.plot(theta_ordered, log_prior_ordered, linewidth=0.5, color = 'orange', label = 'Prior density values')
                
            plt.xlabel(para_names[i])
            plt.ylabel(r'$log$' + ' ' + r'$p($' + para_names[i] + r'$|X)$')
            
            # parameter estimates
            plt.axvline(x = max_post,  color = 'red', linestyle = 'dashed', alpha = 0.75, label = r'$\hat \theta$ =' + str(np.round(max_post, 4)))
            plt.axvline(x = true_values[i], linestyle = 'dashed', alpha = 0.75, label = "true " + r'$\theta$', c = 'k')
            
            # legend
            plt.legend(loc="lower right", fontsize = 8)
            
            plt.savefig(path + plot_name + '_log_post_' + 'parameter_' + str(i) + '.png')
    
    def posterior_plots_empirical(self, Theta, posterior, log_posterior, marginal_priors, Likelihoods, log_Likelihoods,
                        para_names, path, plot_name, bounds_BAM):

        """
        Plot posterior and log posterior and prior probabilities
        """
        
        # get number of parameter
        number_parameter = np.shape(Theta)[1]
                
        """
        1) log posterior 
        """
        # handle NANs
        nans = np.sum(np.isnan(log_posterior))/bounds_BAM.shape[1]
        print("number of NANs in log posterior: %s" %nans)
        slicer_nan = np.isnan(log_posterior)
        log_posterior_non_NAN = log_posterior[~slicer_nan.any(axis=1)]
        Theta_non_NAN = Theta[~slicer_nan.any(axis=1)]
        prior_non_NAN = marginal_priors[~slicer_nan.any(axis=1)]
        
        # handle -inf values
        infs = np.sum(np.isinf(log_posterior))/bounds_BAM.shape[1]
        print("number of -inf in log posterior: %s" %infs)
        slicer_inf = np.isinf(log_posterior_non_NAN)
        log_posterior_non_NAN_inf = log_posterior_non_NAN[~slicer_inf.any(axis=1)]
        Theta_non_NAN_inf = Theta_non_NAN[~slicer_inf.any(axis=1)]
        prior_non_NAN_inf = prior_non_NAN[~slicer_inf.any(axis=1)]
                  
        # scaling log posterior values btw. 0 and 10
        log_posterior_scaled = preprocessing.minmax_scale(log_posterior_non_NAN_inf, feature_range=(0, 10))
        
        # array to save estimates
        posterior_modes = np.zeros(4)
                   
        for i in range(number_parameter):
            
            # order sampled parameter values (ascending order) and corresponding probabilities
            theta_ordered = np.sort(Theta_non_NAN_inf[:,i])
            theta_index = Theta_non_NAN_inf[:,i].argsort()
            log_posterior_array = log_posterior_scaled[:,i]
            log_posterior_ordered = log_posterior_array[theta_index]
            log_prior_array = prior_non_NAN_inf[:,i]
            log_prior_ordered = log_prior_array[theta_index]
                
            # mode of posterior is final parameter estimate
            max_post = Theta_non_NAN_inf[np.argmax(log_posterior_non_NAN_inf[:,i]),i]
            
            plt.clf()
                            
            # log posterior values
            plt.plot(theta_ordered, log_posterior_ordered, color='b', linewidth=0.5, label='scaled log Posterior values')
            
            # prior values
            plt.plot(theta_ordered, log_prior_ordered, linewidth=0.5, color = 'orange', label = 'Prior density values')
                
            plt.xlabel(para_names[i])
            plt.ylabel(r'$log$' + ' ' + r'$p($' + para_names[i] + r'$|X)$')
            
            # parameter estimates
            plt.axvline(x = max_post,  color = 'red', linestyle = 'dashed', alpha = 0.75, label = r'$\hat \theta$ =' + str(np.round(max_post, 4)))
            
            # legend
            plt.legend(loc="lower right", fontsize = 8)
            
            plt.savefig(path + plot_name + '_log_post_' + 'parameter_' + str(i) + '.png')

            posterior_modes[i] = max_post
        
        return posterior_modes


            
                
        
        
        
        

        