"""
Simulating and estimating BAM model from Delli Gatti 2011 (and a toy model from Caiani et al. 2016). 
@author: maxbaldus
"""

# libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

import time

# parallel computing
from joblib import Parallel, delayed
# import multiprocessing
import multiprocess as mp

# import model classes
from models.toymodel import Toymodel
from models.BAM import BAM_mc # BAM with regular mc loop
from models.BAM_parallel import BAM_parallel # BAM model parallized (only for demonstration: parallized computing used later on with BAM_mc)

# import estimation classes
from estimation.data_prep import Filters
from estimation.mdn import mdn
from estimation.sampling import sample_posterior



#################################################################################################
# Simulating a simple macro ABM
#################################################################################################

"""
Simulating a simple macro model with 5 Monte carlo replications. 
The plots are saved into plots/toymodel folder.
"""

"""
# instantiate the toymodel class
toymodel = Toymodel(Time=1000, Ni=100, MC=5,  
                    plots=True, filters=False)

# simulate the toy model and create the plots 
# toy_simulations = toymodel.simulation(gamma=2, pbar=0.01, delta=0.05, rbar=0.075) 
parameter = np.array([2, 0.01, 0.05, 0.075])
toy_simulations = toymodel.simulation(parameter)
"""

#################################################################################################
# Simulating the BAM model(s) by Delli Gatti (2011)
#################################################################################################

""" 
Simulating the base model MC times, with and without parallel computing.
The plots are saved into plots/cut and plots/full respectively.  
"""

# number of MC simulations
MC = 5

# upper bound of price growth rate
H_eta=0.1
# upper bound of quantity growth rate
H_rho=0.1
# upper bound bank costs 
H_phi=0.1 
# upper bound growth rate of wages
h_xi=0.05
# parameter
parameter = np.array([H_eta, H_rho, H_phi, h_xi])

# simulating BAM model MC times without parallising 
BAM_model = BAM_mc(T=1000, MC = MC, Nh=500, Nf=100, Nb=10,
                plots=True, csv=False) 
print("")
print('--------------------------------------')
print("Simulating BAM model without parallising %s times" %MC)
start_time = time.time()

"""
BAM_simulations =  BAM_model.simulation(theta=parameter)

print("")
print("--- %s minutes ---" % ((time.time() - start_time)/60))
np.save('data/simulations/BAM_pseudo_empirical', BAM_simulations) # save the MC simulations
# approximately 2 minutes for one 1 run 
"""

# uncomment for running BAM model MC times in parallel
"""
# simulate the base BAM model MC times using parallel computing
BAM_model_parallel = BAM_parallel(T=1000, MC = 5, Nh=500, Nf=100, Nb=10,
                plots=True, csv=False) 

num_cores = (multiprocessing.cpu_count()) 
start_time = time.time()

print("")
print('--------------------------------------')
print("Simulating BAM model %s times in parallel" %MC)

BAM_simulations_parallel = Parallel(n_jobs=num_cores)(
                        delayed(BAM_model_parallel.simulation)
                        (parameter, mc) for mc in range(MC)
                        )

print("")
print("--- %s minutes ---" % ((time.time() - start_time)/60))
# np.save('data/simulations/BAM_10MC', BAM_simulations_parallel) # save the 10 simulations
# """


#################################################################################################
# Estimating the BAM model
#################################################################################################

"""
General estimation set-up: Grid search:
Generate MC simluations of the model, ech with length T, for each parameter combination.

A)
First the estimation method is tested by using pseudo-empirical data with a-priori specified parameter values.
The first mc simulation with the parameter configuration from above is used as the 'observed' dataset. 
"""
# number of MC simulations per parameter combination
MC = 20

# pseudo empirical data
#BAM_simulations = np.transpose(np.load("data/simulations/BAM_10MC.npy")) # load pesudo random data when used parallizing before (need to transpse data frame)
BAM_simulations = np.load("data/simulations/BAM_pseudo_empirical.npy") # load pesudo random data

# obtain pseudo-empirical data by using only the last 500 iterations of the a priorie simulated ts 
BAM_obs = BAM_simulations[BAM_simulations.shape[0]-500:BAM_simulations.shape[0],0]  

# create new instance of the BAM model (without any plotting)
BAM_model = BAM_mc(T=1000, MC = MC, Nh=500, Nf=100, Nb=10,
                plots=False, csv=False) 

# define the upper and lower bound for each parameter value, packed into a 2x#free parameters dataframe (2d numpy array) 
# with one column for each free parameter and the first (second) row being the lower (upper) bound respectively
# bounds_BAM = np.transpose(np.array([ [0.07,0.13], [0.07,0.13], [0.07,0.13], [0.02,0.08] ]))
bounds_BAM = np.transpose(np.array([ [0,0.5], [0,0.5], [0,0.5], [0,0.25] ]))

# initialize the estimation method: here without applying any filter to observed as simulated time series
BAM_posterior = sample_posterior(model = BAM_model, bounds = bounds_BAM, data_obs=BAM_obs, filter=False)

"""
1) Simulation block: simulating and storing the TxMC matrix for each parameter combination
"""

# number of parameter combinations
grid_size = 5000

# simulate the model MC times for each parameter combination and save each TxMC matrix
print("")
print('--------------------------------------')
print("1) Simulation Block: Simulate the model MC times for each parameter combination and store:")

# save start time
start_time = time.time()

# generate grid with parameter values
np.random.seed(123)
Theta = BAM_posterior.simulation_block(grid_size, path = '')

# save and load theta combinations
"""np.save('estimation/BAM/Theta_ordered', Theta)
Theta = np.load('estimation/BAM/Theta_500.npy') # load test parameter combinations (with large bounds)
Theta = np.load('estimation/BAM/Theta.npy') # load parameter grid with 5000 combinations """

# define path where to store the simulated time series
#path = 'data/simulations/BAM_simulations/latin_hypercube'
#path = 'data/simulations/BAM_simulations/test/latin_hypercube' # test data
#path = 'data/simulations/toymodel_simulations/latin_hypercube' # toymodel data
path = 'data/simulations/BAM_simulations/Theta_ordered/Theta_ordered'


# parallize the grid search: using joblib
def grid_search_parallel(Theta, model, path, i):

    # current parameter combination
    # theta_current = theta[i,:]
    
    # simulate the model each time with a new parameter combination from the grid
    simulations = model.simulation(Theta[i,:])
    
    # current path to save the simulation to
    # current_path = path + '_' + str(i)
    
    # save simulated data 
    np.save(path + '_' + str(i), simulations)
    
    # plot the first mc simulation
    """plt.clf()
    plt.plot(np.log(simulations[:,0]))
    plt.xlabel("Time")
    plt.ylabel("Log output")
    plt.savefig( path + '_' + str(i) + ".png")"""

    return

# set number of cores for multiprocessing 
# num_cores = (multiprocessing.cpu_count()) - 4 
num_cores = 56 

# uncomment for running the 5000 times 20MC simulations (per theta) in parallel and save
"""Parallel(n_jobs=num_cores, verbose=50)(
        delayed(grid_search_parallel)
        (Theta, BAM_model, path, i) for i in range(grid_size)
        )"""


print("")
print("--- %s minutes ---" % ((time.time() - start_time)/60))
# without parallising: 2MC=5 minutes * 5thetas = 25 minutes
# parallel with 4 cores, without plots: 2MC=5 minutes * 5thetas = 25 / 4 = 7 minutes (5.571771335601807)
# parallel with 4 cores, with plots: (5.985158399740855)


"""
2) Estimation block: compute the likelihood and the marginal posterior probability of each parameter of the abm model by delli gatti.
"""
print("")
print('--------------------------------------')
print("2) Estimation block: Approximating Likelihood and evaluating the posterior for each parameter")
start_time = time.time()

# Approximate the posterior distr. of each parameter using the simulated data and given empirical data via mdn's
posterior, log_posterior, prior_probabilities, Likelihoods, log_Likelihoods = BAM_posterior.approximate_posterior(grid_size, path = path, Theta=Theta)

# saving posterior and prior values 
np.save('estimation/BAM/Theta_ordered/final_run/kde/log_posterior_identification', log_posterior)
np.save('estimation/BAM/Theta_ordered/final_run/kde/posterior_identification', posterior)
np.save('estimation/BAM/Theta_ordered/final_run/kde/prior_identification', prior_probabilities)
np.save('estimation/BAM/Theta_ordered/final_run/kde/Likelihoods_identification', Likelihoods)
np.save('estimation/BAM/Theta_ordered/final_run/kde/log_Likelihoods_identification', log_Likelihoods)


print("")
print("--- %s minutes ---" % ((time.time() - start_time)/60))
# --- 290.84607830047605 minutes ---

"""
plotting the posterior, log posterior and prior values (marginal), for each theta in the grid
"""
# load ordered Theta sample (no filter applied)
"""log_posterior = np.load('estimation/BAM/Theta_ordered/log_posterior_identification_Theta_ordered.npy')
posterior = np.load('estimation/BAM/Theta_ordered/posterior_identification_Theta_ordered.npy')
prior_probabilities = np.load('estimation/BAM/Theta_ordered/prior_identification_Theta_ordered.npy')
Likelihoods = np.load('estimation/BAM/Theta_ordered/Likelihoods_Theta_ordered.npy')
log_Likelihoods = np.load('estimation/BAM/Theta_ordered/log_Likelihoods_Theta_ordered.npy')"""

# load ordered Theta sample (FILTER applied)
"""log_posterior = np.load('estimation/BAM/Theta_ordered/log_posterior_identification_Theta_ordered_FILTER.npy')
posterior = np.load('estimation/BAM/Theta_ordered/posterior_identification_Theta_ordered_FILTER.npy')
prior_probabilities = np.load('estimation/BAM/Theta_ordered/prior_identification_Theta_ordered_FILTER.npy')
Likelihoods = np.load('estimation/BAM/Theta_ordered/Likelihoods_Theta_ordered_FILTER.npy')
log_Likelihoods = np.load('estimation/BAM/Theta_ordered/log_Likelihoods_Theta_ordered_FILTER.npy')"""

# load ordered Theta sample (log transformations)
"""log_posterior = np.load('estimation/BAM/Theta_ordered/log_transform/log_posterior_identification.npy')
posterior = np.load('estimation/BAM/Theta_ordered/log_transform/posterior_identification.npy')
prior_probabilities = np.load('estimation/BAM/Theta_ordered/log_transform/prior_identification.npy')
Likelihoods = np.load('estimation/BAM/Theta_ordered/log_transform/Likelihoods_identification.npy')
log_Likelihoods = np.load('estimation/BAM/Theta_ordered/log_transform/log_Likelihoods_identification.npy')"""

# load ordered Theta sample (mdn, raw simulated data)
log_posterior = np.load('estimation/BAM/Theta_ordered/final_run/log_posterior_identification.npy')
posterior = np.load('estimation/BAM/Theta_ordered/final_run/posterior_identification.npy')
prior_probabilities = np.load('estimation/BAM/final_run/log_transform/prior_identification.npy')
Likelihoods = np.load('estimation/BAM/Theta_ordered/final_run/Likelihoods_identification.npy')
log_Likelihoods = np.load('estimation/BAM/Theta_ordered/final_run/log_Likelihoods_identification.npy')

# load ordered Theta sample (kdn, log transform)
log_posterior = np.load('estimation/BAM/Theta_ordered/final_run/kde/log_posterior_identification.npy')
posterior = np.load('estimation/BAM/Theta_ordered/final_run/kde/posterior_identification.npy')
prior_probabilities = np.load('estimation/BAM/final_run/kde/log_transform/prior_identification.npy')
Likelihoods = np.load('estimation/BAM/Theta_ordered/final_run/kde/Likelihoods_identification.npy')
log_Likelihoods = np.load('estimation/BAM/Theta_ordered/final_run/kde/log_Likelihoods_identification.npy')

# parameter names
para_names = [r'$H_{\eta}$', r'$H_{\rho}$', r'$H_{\phi}$', r'$H_{\xi}$']

# names of the plots 
# plot_name= 'Theta_ordered_5000_NO_filter'
# plot_name= 'Theta_ordered_5000_HP_filter'
plot_name= 'Theta_ordered_5000_log_transform'


BAM_posterior.posterior_plots_new(Theta=Theta, posterior=posterior, log_posterior=log_posterior, 
                                Likelihoods = Likelihoods, log_Likelihoods = log_Likelihoods,
                                marginal_priors=prior_probabilities, para_names = para_names, bounds_BAM = bounds_BAM,
                                path = 'plots/posterior/BAM/Theta_ordered/', plot_name= plot_name)

print('--------------------------------------')
print("Done")


# Simulation hyperparameters:
# Gatti 2020
# 5000 combinations
# MC = 20 
# T = 3000, discarding the first 2500

# aleen
# MC = 100 




"""
B) Estimating the BAM model using real data on US GDP ??, using the same artificial data generated above. 
"""

# using un-ordered Theta sample 