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
        self.filter = filter # set to True: filter is applied 

    """
    1) Simulation block
    """
    def simulation_block(self, grid_size, path):

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
                      
        # order Theta parameter in ascending order 
        for i in range(number_para):
            Theta[:,i] = np.sort(Theta[:,i]) 
        
        print("bb")
        """
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
    def approximate_posterior(self, grid_size, path, Theta):
            
        # apply filter to observed time series
        if self.filter:
                # apply filter
                print("blub")
        else:
            data_obs = self.data_obs

        # instantiate the likelihood approximation method
        likelihood_appro = mdn(data_obs, L = 3, K = 16, 
                           neurons = 32, layers = 3, batch_size = 512, epochs = 12, 
                           eta_x=0.2, eta_y=0.2, act_fct="relu"
                           ) 

               
        # initiate the matrix with marginal posterior probablities
        number_parameter = np.shape(self.bounds)[1]
        posterior = np.zeros((grid_size, number_parameter))
        log_posterior = np.zeros((grid_size, number_parameter))

        # sample the prior probabilities for each parameter combination
        marginal_priors = np.zeros((grid_size, number_parameter))
        for i in range(grid_size):
            marginal_priors[i,:] = self.sample_prior() 
        
        # save start time
        start_time = time.time()
        
        """
        Approximate likelihood and evaluate posterior for each parameter combination (and corresponding TxMC matrix with simulated data)
        without parallised computing
        """
        # for i in tqdm(range(grid_size)):
        for i in range(grid_size):
            
            # good simulations for testing:[27, 31, 45, 49]
            i = 45
            
            # load the simulated data for the current parameter combination
            load_path = path + '_' + str(i) + '.npy'
            simulation = np.load(load_path)

            # neglect the first simlated values for each mc column to ensure convergence of the major report variables 
            # only use last observations with length of the observed ts
            simulation_short = simulation[simulation.shape[0]-len(self.data_obs) : simulation.shape[0],:]
            
            #simulation_short[:,i for in range(4)] = data_obs

            # apply filter to simulated time series
            if self.filter:
                # filter: apply to each column
                print("blub")
            
            # approximate the posterior probability of the given parameter combination
            densities = likelihood_appro.approximate_likelihood(simulation_short)
            
            # compute likelihood of the observed data for the given parameter combination
            L = np.prod(densities)
            ll = np.sum(np.log(densities)) 

            # testing: for i=27, with 1/std(y_tilde): ll = -2446.8013911398457, without 1/std(y_tilde): ll = -1529.9878020603692  , 
            # testing: for i=49, with 1/std(y_tilde): ll = -inf , 
            # testing: for i=45 (keine 0's), with 1/std(y_tilde): ll = -inf , without 1/std(y_tilde): ll = -inf  , 
            
            # compute marginal (log) posteriors
            posterior[i,:] = L * marginal_priors[i,:]
            log_posterior[i,:] = ll + np.log(marginal_priors[i,:])

        # using parallel computing
        def approximate_parallel(path, marginal_priors, i):
           
             # load the simulated data for the current parameter combination
            load_path = path + '_' + str(i) + '.npy'
            simulation = np.load(load_path)

            # neglect the first simlated values for each mc column to ensure convergence of the major report variables 
            # only use last observations with length of the observed ts
            simulation_short = simulation[simulation.shape[0]-len(self.data_obs) : simulation.shape[0], :]

            # apply filter to simulated time series
            if self.filter:
                # filter: apply to each column
                print("blub")
            
            # approximate the posterior probability of the given parameter combination
            densities = likelihood_appro.approximate_likelihood(simulation_short)
            
            """# compute likelihood of the observed data for the given parameter combination
            L = np.prod(densities)
            ll = np.sum(np.log(densities))

            priors = marginal_priors[i,:]"""

            return  densities
        
        densities = Parallel(n_jobs=32)(
        delayed(approximate_parallel)
        (path, marginal_priors, i) for i in range(40)
        )
        
        """posteriors = Parallel(n_jobs=32)(
        delayed(approximate_parallel)
        (path, marginal_priors, i) for i in range(grid_size)
        )

        # unpack and save the parallel computed posterior probabilities
        for i in range(len(posteriors)):
            posterior[i,:] = posteriors[i][0]
            log_posterior[i,:] = posteriors[i][1]
            # marginal_priors[i,:] = posteriors[i][1]"""

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
        # since having grid: use value with highest proba as estimator, i.e. mode (not mean of the parameter values, since no MC chain)
        
        # ISSUES
        # Influence of prior rather small: array([-2.34817589, -2.12968255, -2.17047113, -2.629976  ]) vs ll = -750.96674469632

        # huge posterior values (positive log values) when using 1/std in mdn 
        # without 1/std => posterior values are 0 now 
        # scale values down?!?!?!?!

        # good simulations: 27, 31, 45, 49, 
        # CHECK WHETHER log_posterior values large (large negative values?!)

        return posterior, log_posterior, marginal_priors


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


    def posterior_plots(self, Theta, posterior, log_posterior, marginal_priors,
                        para_names, path, plot_name):

        """
        Plot posterior and prior probabilities
        """
        
        number_parameter = np.shape(Theta)[1]

        """
        1) plot log posteriors 
        """
        
        # order Theta values
        
        
        # remove NAN's
        slicer_nan = np.isnan(log_posterior)
        log_post_tmp = log_posterior[~slicer_nan.any(axis=1)]

        # determine min (most negative) log posterior values, ignoring -inf values
        slicer_inf = np.isinf(log_post_tmp)
        log_post_tmp_noinf = log_post_tmp[~slicer_inf.any(axis=1)]
        min_log_post = min(log_post_tmp_noinf.min(axis=0))

        # replace -inf values with largest negative value - 1000 from above
        np.place(log_post_tmp, log_post_tmp < -10000, [min_log_post - 1000, min_log_post - 1000,min_log_post - 1000, min_log_post - 1000])

        # delete corresponding Theta values and prior values with nan's
        Theta_new = Theta[~slicer_nan.any(axis=1)]
        prior_new = marginal_priors[~slicer_nan.any(axis=1)]

        # einfach posteriors plotten mit x-achse von lower bis upper bounds
        
        # plotting the marginal posteriors
        for i in range(number_parameter):

            # scatter all marginal posterior values and corresponding parameter values
            plt.clf()
            plt.scatter(Theta_new[:,i], log_post_tmp[:,i])
            plt.xlabel(para_names[i])
            plt.ylabel("log_posterior")
            plt.show()
            
            # only consider positive posterior values (and corresponding parameter and prior values)
            log_post_tmp_positive = log_post_tmp[:,i][log_post_tmp[:,i] > 0]
            slicer_greaterzero = np.where(log_post_tmp[:,i] > 0)[0]
            Theta_positive = Theta_new[:,i][slicer_greaterzero]
            prior_positive = prior_new[:,i][slicer_greaterzero]

            # scatterplot of positive parameter values
            plt.clf()
            plt.scatter(Theta_positive, log_post_tmp_positive)
            plt.savefig(path + plot_name + '_parameter' + str(i) + '.png')
            
            # create pd dataframe and order values according to theta values
            df = pd.DataFrame()
            df['theta'] = Theta_positive
            df['log_post'] = log_post_tmp_positive
            df['prior'] = prior_positive
            df = df.sort_values('theta')
            
            # smooth scatterplot/posterior distribution 
            #gfg = make_interp_spline(df['theta'], df['log_post'], k=3)
            #log_post_smooth = gfg(df['theta'])

            plt.clf()
            # plt.plot(df['theta'], log_post_smooth)
            plt.plot(df['theta'], df['log_post'])
            
            # prior
            plt.plot(df['theta'], np.log(df['prior']))
            
            # posterior mode
            df = df.reset_index()
            max_log_post = df.loc[int(df['log_post'].idxmax()), 'theta']
            plt.axvline(x = max_log_post, c = 'k', linestyle = 'dashed', alpha = 0.75)
            
            # axis and legend
            plt.xlabel(para_names[i])
            plt.ylabel(r'$log$' + ' ' + r'$p($' + para_names[i] + r'$|X)$')
            plt.legend(['log Posterior Density', 'log Prior Density', 'Posterior Mode'], fontsize = 8)
            plt.savefig(path + plot_name + '_smooth_parameter' + str(i) + '.png')

            # prior
            # plt.plot(Theta[:,i], marginal_priors[:,i])
            
            
            # ordering some stuff 
            """
            # order sampled parameter values (ascending order) and corresponding probabilities
            theta_ordered = np.sort(Theta[:,i])
            theta_index = Theta[:,i].argsort()
            log_posterior_array = log_posterior[:,i]
            log_posterior_ordered = log_posterior_array[theta_index]
            
            # convert nan, if any, also to 0 probability
            if any(np.isnan(log_posterior_ordered)) == True:
                print("numbers of NAN values: %s" %sum(np.isnan(log_posterior_ordered)))
                log_posterior_ordered = np.nan_to_num(log_posterior_ordered)
            """


            """plt.clf()
            plt.hist(log_post_tmp_positive, 25, density = True, color = 'b', alpha = 0.5)
            plt.show()

            # apply kde to smooth the distribution
            plt.clf()
            sns.kdeplot(data=df, x = 'log_post')
            plt.show()
            
            
            kde = gaussian_kde(df['log_post'])
            
            plt.plot(kde)
            plt.show()"""

        print('--------------------------------------')
        print("Done")
        
    
    
    
    
    def posterior_plots_new(self, Theta, posterior, log_posterior, marginal_priors,
                        para_names, path, plot_name):

        """
        Plot posterior and log posterior and prior probabilities
        """
        
        # get number of parameter
        number_parameter = np.shape(Theta)[1]
        
        """
        1) posterior 
        """
        
        """
        2) log posterior 
        """
           
        for i in range(number_parameter):
                
            # plot .. 
            plt.clf()
            plt.plot(log_posterior[:,i])
            plt.xlabel(para_names[i])
            plt.ylabel("log_posterior")
            plt.show()
            
            
            print("") 
         
            
                
        
        
        
        

        