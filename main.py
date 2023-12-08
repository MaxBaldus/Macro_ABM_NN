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
                plots=True, csv=False, empirical=False, path_empirical="") 
print("")
print('--------------------------------------')
print("Simulating BAM model without parallising %s times" %MC)
start_time = time.time()

#BAM_simulations =  BAM_model.simulation(theta=parameter)

print("")
print("--- %s minutes ---" % ((time.time() - start_time)/60))
#np.save('data/simulations/BAM_pseudo_empirical', BAM_simulations) # save the MC simulations
# approximately 2 minutes for one 1 run 


# uncomment for running BAM model MC times in parallel
"""
# simulate the base BAM model MC times using parallel computing
BAM_model_parallel = BAM_parallel(T=1000, MC = 5, Nh=500, Nf=100, Nb=10,
                plots=True, csv=False, empirical=False, path_empirical="") 

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
#MC = 100
#MC = 50

# pseudo empirical data
#BAM_simulations = np.transpose(np.load("data/simulations/BAM_10MC.npy")) # load pesudo random data when used parallizing before (need to transpse data frame)
BAM_simulations = np.load("data/simulations/BAM_pseudo_empirical.npy") # load pesudo random data

# obtain pseudo-empirical data by using only the last 500 iterations of the a priorie simulated ts 
BAM_obs = BAM_simulations[BAM_simulations.shape[0]-500:BAM_simulations.shape[0],0]  

# create new instance of the BAM model (without any plotting)
BAM_model = BAM_mc(T=1000, MC = MC, Nh=500, Nf=100, Nb=10, plots=False, csv=False, empirical=False, path_empirical="") 
#BAM_model = BAM_mc(T=1500, MC = MC, Nh=500, Nf=100, Nb=10, plots=False, csv=False) 

# define the upper and lower bound for each parameter value, packed into a 2x#free parameters dataframe (2d numpy array) 
# with one column for each free parameter and the first (second) row being the lower (upper) bound respectively
# bounds_BAM = np.transpose(np.array([ [0.07,0.13], [0.07,0.13], [0.07,0.13], [0.02,0.08] ])) # first test
bounds_BAM = np.transpose(np.array([ [0,0.5], [0,0.5], [0,0.5], [0,0.25] ]))
#bounds_BAM = np.transpose(np.array([ [0.05,0.2], [0.05,0.2], [0.05,0.2], [0.025,0.075] ]))


# initialize the estimation method: here without applying any filter to observed as simulated time series
BAM_posterior = sample_posterior(model = BAM_model, bounds = bounds_BAM, data_obs=BAM_obs, filter=True)

"""
1) Simulation block: simulating and storing the TxMC matrix for each parameter combination
"""

# number of parameter combinations
grid_size = 5000
#grid_size = 1500
#grid_size = 1000

# simulate the model MC times for each parameter combination and save each TxMC matrix
print("")
print('--------------------------------------')
print('A)')
print("1) Simulation Block: Simulate the model MC times for each parameter combination and store:")

# save start time
start_time = time.time()

# generate grid with parameter values
np.random.seed(123)
Theta = BAM_posterior.simulation_block(grid_size, path = '', order_Theta=False)

# save and load Theta combinations
#np.save('estimation/BAM/Theta_ordered', Theta)
#Theta = np.load('estimation/BAM/Theta_500.npy') # load test parameter combinations (with large bounds)
#Theta = np.load('estimation/BAM/Theta.npy') # load parameter grid with 5000 combinations """

# define path where to store the simulated time series, which are then loaded in part 2)
path = 'data/simulations/BAM_simulations/latin_hypercube' # no ordered Theta 
#path = 'data/simulations/BAM_simulations/test/latin_hypercube' # test data
#path = 'data/simulations/toymodel_simulations/latin_hypercube' # toymodel data
#path = 'data/simulations/BAM_simulations/Theta_ordered/Theta_ordered'
#path = 'data/simulations/BAM_simulations/50MC/Theta_NOT_ordered'


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

# Approximate the posterior distr. of each parameter using the simulated data and given empirical data 
# by default, mdns are used. Set kde = True to use kde instead 

#posterior, log_posterior, prior_probabilities, Likelihoods, log_Likelihoods = BAM_posterior.approximate_posterior(grid_size, path = path, t_zero=500, kde=True, empirical = False)


# choose folder to save posterior and prior values: mdn
"""np.save('estimation/BAM/final_run/HP_filter/log_posterior_identification', log_posterior)
np.save('estimation/BAM/final_run/HP_filter/posterior_identification', posterior)
np.save('estimation/BAM/final_run/HP_filter/prior_identification', prior_probabilities)
np.save('estimation/BAM/final_run/HP_filter/Likelihoods_identification', Likelihoods)
np.save('estimation/BAM/final_run/HP_filter/log_Likelihoods_identification', log_Likelihoods)"""

# saving posterior and prior values: kde
"""np.save('estimation/BAM/Theta_ordered/final_run/kde/HP_filter/log_posterior_identification', log_posterior)
np.save('estimation/BAM/Theta_ordered/final_run/kde/HP_filter/posterior_identification', posterior)
np.save('estimation/BAM/Theta_ordered/final_run/kde/HP_filter/prior_identification', prior_probabilities)
np.save('estimation/BAM/Theta_ordered/final_run/kde/HP_filter/Likelihoods_identification', Likelihoods)
np.save('estimation/BAM/Theta_ordered/final_run/kde/HP_filter/log_Likelihoods_identification', log_Likelihoods)
"""
print("")
print("--- %s minutes ---" % ((time.time() - start_time)/60))
# --- 271.5109751145045 minutes ---
# --- 279.17994863589604 minutes ---

"""
plotting the posterior, log posterior and prior values (marginal), for each theta in the grid
"""
# load approximations regarding ordered Theta sample: no filter applied, raw data, not div by std in mdn
"""log_posterior = np.load('estimation/BAM/Theta_ordered/final_run/NOT_div_by_std/log_posterior_identification.npy')
posterior = np.load('estimation/BAM/Theta_ordered/final_run/NOT_div_by_std/posterior_identification.npy')
prior_probabilities = np.load('estimation/BAM/Theta_ordered/final_run/NOT_div_by_std/prior_identification.npy')
Likelihoods = np.load('estimation/BAM/Theta_ordered/final_run/NOT_div_by_std/Likelihoods_identification.npy')
log_Likelihoods = np.load('estimation/BAM/Theta_ordered/final_run/NOT_div_by_std/log_Likelihoods_identification.npy')"""

# load approximations regarding ordered Theta sample: no filter applied, raw data, div by std in mdn
"""log_posterior = np.load('estimation/BAM/Theta_ordered/final_run/div_by_std/log_posterior_identification.npy')
posterior = np.load('estimation/BAM/Theta_ordered/final_run/div_by_std/posterior_identification.npy')
prior_probabilities = np.load('estimation/BAM/Theta_ordered/final_run/div_by_std/prior_identification.npy')
Likelihoods = np.load('estimation/BAM/Theta_ordered/final_run/div_by_std/Likelihoods_identification.npy')
log_Likelihoods = np.load('estimation/BAM/Theta_ordered/final_run/div_by_std/log_Likelihoods_identification.npy')"""

# load approximations regarding ordered Theta sample: HP filter, div by std in mdn
"""log_posterior = np.load('estimation/BAM/Theta_ordered/final_run/hp_filter/log_posterior_identification.npy')
posterior = np.load('estimation/BAM/Theta_ordered/final_run/hp_filter/posterior_identification.npy')
prior_probabilities = np.load('estimation/BAM/Theta_ordered/final_run/hp_filter/prior_identification.npy')
Likelihoods = np.load('estimation/BAM/Theta_ordered/final_run/hp_filter/Likelihoods_identification.npy')
log_Likelihoods = np.load('estimation/BAM/Theta_ordered/final_run/hp_filter/log_Likelihoods_identification.npy')"""

# load approximations regarding ordered Theta sample: log transformations, div by std in mdn
"""log_posterior = np.load('estimation/BAM/Theta_ordered/final_run/log_transform/log_posterior_identification.npy')
posterior = np.load('estimation/BAM/Theta_ordered/final_run/log_transform/posterior_identification.npy')
prior_probabilities = np.load('estimation/BAM/Theta_ordered/final_run/log_transform/prior_identification.npy')
Likelihoods = np.load('estimation/BAM/Theta_ordered/final_run/log_transform/Likelihoods_identification.npy')
log_Likelihoods = np.load('estimation/BAM/Theta_ordered/final_run/log_transform/log_Likelihoods_identification.npy')"""

# load approximations regarding ordered Theta sample: no filter: kde
"""log_posterior = np.load('estimation/BAM/Theta_ordered/final_run/kde/log_posterior_identification.npy')
posterior = np.load('estimation/BAM/Theta_ordered/final_run/kde/posterior_identification.npy')
prior_probabilities = np.load('estimation/BAM/Theta_ordered/final_run/kde/prior_identification.npy')
Likelihoods = np.load('estimation/BAM/Theta_ordered/final_run/kde/Likelihoods_identification.npy')
log_Likelihoods = np.load('estimation/BAM/Theta_ordered/final_run/kde/log_Likelihoods_identification.npy')"""

# load approximations regarding ordered Theta sample: HP filter: kde
"""log_posterior = np.load('estimation/BAM/Theta_ordered/final_run/kde/HP_filter/log_posterior_identification.npy')
posterior = np.load('estimation/BAM/Theta_ordered/final_run/kde/HP_filter/posterior_identification.npy')
prior_probabilities = np.load('estimation/BAM/Theta_ordered/final_run/kde/HP_filter/prior_identification.npy')
Likelihoods = np.load('estimation/BAM/Theta_ordered/final_run/kde/HP_filter/Likelihoods_identification.npy')
log_Likelihoods = np.load('estimation/BAM/Theta_ordered/final_run/kde/HP_filter/log_Likelihoods_identification.npy')"""

# load approximations regarding UNordered Theta sample: MDN, no filter
"""log_posterior = np.load('estimation/BAM/final_run/log_posterior_identification.npy')
posterior = np.load('estimation/BAM/final_run/posterior_identification.npy')
prior_probabilities = np.load('estimation/BAM/final_run/prior_identification.npy')
Likelihoods = np.load('estimation/BAM/final_run/Likelihoods_identification.npy')
log_Likelihoods = np.load('estimation/BAM/final_run/log_Likelihoods_identification.npy')"""

# load approximations regarding UNordered Theta sample: MDN, HP FILTER
"""log_posterior = np.load('estimation/BAM/final_run/HP_filter/log_posterior_identification.npy')
posterior = np.load('estimation/BAM/final_run/HP_filter/posterior_identification.npy')
prior_probabilities = np.load('estimation/BAM/final_run/HP_filter/prior_identification.npy')
Likelihoods = np.load('estimation/BAM/final_run/HP_filter/Likelihoods_identification.npy')
log_Likelihoods = np.load('estimation/BAM/final_run/HP_filter/log_Likelihoods_identification.npy')"""

# parameter names
para_names = [r'$H_{\eta}$', r'$H_{\rho}$', r'$H_{\phi}$', r'$H_{\xi}$']

# names of the plots 
#plot_name= 'Theta_ordered_5000_raw_data_NOT_div_by_std'
#plot_name= 'Theta_ordered_5000_raw_data_div_by_std'
#plot_name= 'Theta_ordered_5000_HP_filter'
#plot_name= 'Theta_ordered_5000_log_transform'
#plot_name= 'Theta_ordered_5000_KDE'
#plot_name = 'Theta_ordered_5000_KDE_HP_filter'
#plot_name = 'Theta_NOT_ordered_5000_MDN'
plot_name = 'Theta_NOT_ordered_5000_MDN_HP_filter'

# path to save the plots
#plot_path = 'plots/posterior/BAM/Theta_ordered/final_run/NOT_div_by_std/'
#plot_path = 'plots/posterior/BAM/Theta_ordered/final_run/div_by_std/'
#plot_path = 'plots/posterior/BAM/Theta_ordered/final_run/hp_filter/'
#plot_path = 'plots/posterior/BAM/Theta_ordered/final_run/log_transform/'
#plot_path = 'plots/posterior/BAM/Theta_ordered/final_run/kde/'
#plot_path = 'plots/posterior/BAM/Theta_ordered/final_run/kde/HP_filter/'
#plot_path = 'plots/posterior/BAM/NOT_ordered/'
plot_path = 'plots/posterior/BAM/NOT_ordered/HP_filter/'


# plot posteriors for ordered Theta
"""BAM_posterior.posterior_plots_identification(Theta=Theta, posterior=posterior, log_posterior=log_posterior, 
                                Likelihoods = Likelihoods, log_Likelihoods = log_Likelihoods,
                                marginal_priors=prior_probabilities, para_names = para_names, bounds_BAM = bounds_BAM,
                                path = plot_path, plot_name= plot_name,
                                true_values = parameter, zoom=False)"""

# plot posteriors for unordered Theta
"""BAM_posterior.posterior_plots_unordered(Theta=Theta, posterior=posterior, log_posterior=log_posterior, 
                                Likelihoods = Likelihoods, log_Likelihoods = log_Likelihoods,
                                marginal_priors=prior_probabilities, para_names = para_names, bounds_BAM = bounds_BAM,
                                path = plot_path, plot_name= plot_name,
                                true_values = parameter)"""


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
B) Estimating the BAM model using real data on US GDP, using the same artificial data generated above. 
"""

# using un-ordered Theta sample 

MC = 20
BAM_model = BAM_mc(T=1000, MC = MC, Nh=500, Nf=100, Nb=10, plots=False, csv=False, empirical=False, path_empirical="") 
bounds_BAM = np.transpose(np.array([ [0,0.5], [0,0.5], [0,0.5], [0,0.25] ]))

# load empirical data and convert np array : US GDP 
GDP_US = pd.read_csv("data/GDPC1.csv")
GDP_US_array = np.array(GDP_US)[:,1]

# load empirical data and convert np array : German GDP 
GDP_Germany = pd.read_csv("data/CLVMNACSCAB1GQDE.csv")
GDP_Germany_array = np.array(GDP_Germany)[:,1]

# initialize the estimation method: apply HP filter 
#BAM_posterior = sample_posterior(model = BAM_model, bounds = bounds_BAM, data_obs=GDP_US_array, filter=True)
BAM_posterior = sample_posterior(model = BAM_model, bounds = bounds_BAM, data_obs=GDP_Germany_array, filter=True)

grid_size = 5000
np.random.seed(123)
Theta = BAM_posterior.simulation_block(grid_size, path = '', order_Theta=False)

print("")
print('--------------------------------------')
print("B) Empirical application Estimation block: Approximating Likelihood and evaluating the posterior for each parameter")
start_time = time.time()
path = 'data/simulations/BAM_simulations/latin_hypercube' # not ordered Theta 
#posterior, log_posterior, prior_probabilities, Likelihoods, log_Likelihoods = BAM_posterior.approximate_posterior(grid_size, path = path, t_zero=500, kde=False, empirical = True)

# choose folder to save posterior and prior values: mdn
"""np.save('estimation/BAM/empirical/Germany/log_posterior_identification', log_posterior)
np.save('estimation/BAM/empirical/Germany/posterior_identification', posterior)
np.save('estimation/BAM/empirical/Germany/prior_identification', prior_probabilities)
np.save('estimation/BAM/empirical/Germany/Likelihoods_identification', Likelihoods)
np.save('estimation/BAM/empirical/Germany/log_Likelihoods_identification', log_Likelihoods)"""

print("")
print("--- %s minutes ---" % ((time.time() - start_time)/60))
print("")

# load approximations regarding UNordered Theta sample: MDN - US
log_posterior = np.load('estimation/BAM/empirical/US/log_posterior_identification.npy')
posterior = np.load('estimation/BAM/empirical/US/posterior_identification.npy')
prior_probabilities = np.load('estimation/BAM/empirical/US/prior_identification.npy')
Likelihoods = np.load('estimation/BAM/empirical/US/Likelihoods_identification.npy')
log_Likelihoods = np.load('estimation/BAM/empirical/US/log_Likelihoods_identification.npy')

# load approximations regarding UNordered Theta sample: MDN - Germany
"""log_posterior = np.load('estimation/BAM/empirical/Germany/log_posterior_identification.npy')
posterior = np.load('estimation/BAM/empirical/Germany/posterior_identification.npy')
prior_probabilities = np.load('estimation/BAM/empirical/Germany/prior_identification.npy')
Likelihoods = np.load('estimation/BAM/empirical/Germany/Likelihoods_identification.npy')
log_Likelihoods = np.load('estimation/BAM/empirical/Germany/log_Likelihoods_identification.npy')"""

# parameter names
para_names = [r'$H_{\eta}$', r'$H_{\rho}$', r'$H_{\phi}$', r'$H_{\xi}$']

# names of the plots 
plot_name = 'Theta_NOT_ordered_5000_MDN_empirical' # US GDP
#plot_name = 'Theta_NOT_ordered_5000_MDN_empirical_German_GDP'


# path to save the plots
plot_path = 'plots/posterior/BAM/empirical/'
#plot_path = 'plots/posterior/BAM/empirical/Germany/'

# plot posteriors for unordered Theta and save respective parameter estimates
us_estimates = BAM_posterior.posterior_plots_empirical(Theta=Theta, posterior=posterior, log_posterior=log_posterior, 
                                Likelihoods = Likelihoods, log_Likelihoods = log_Likelihoods,
                                marginal_priors=prior_probabilities, para_names = para_names, bounds_BAM = bounds_BAM,
                                path = plot_path, plot_name= plot_name)

# plot posteriors for unordered Theta and save respective parameter estimates
"""german_estimates = BAM_posterior.posterior_plots_empirical(Theta=Theta, posterior=posterior, log_posterior=log_posterior, 
                                Likelihoods = Likelihoods, log_Likelihoods = log_Likelihoods,
                                marginal_priors=prior_probabilities, para_names = para_names, bounds_BAM = bounds_BAM,
                                path = plot_path, plot_name= plot_name)"""
                        

# simulate the BAM model again, now using estimated parameter values
MC = 5
print("")
print('--------------------------------------')
print("Simulating estimated BAM model without parallising %s times" %MC)

# path to save simulated time series using estimated parameters, for US and Germany 
path_estimated_simulations = "plots/empirical/US/"
#path_estimated_simulations = "plots/empirical/Germany/"

# simulating BAM model MC times without parallising 
BAM_model_estimated = BAM_mc(T=1000, MC = MC, Nh=500, Nf=100, Nb=10,
                plots=False, csv=False, empirical=True, path_empirical=path_estimated_simulations) 

start_time = time.time()

BAM_simulations_US =  BAM_model_estimated.simulation(theta=us_estimates)
#BAM_simulations_Germany =  BAM_model_estimated.simulation(theta=german_estimates)

print("")
print("--- %s minutes ---" % ((time.time() - start_time)/60))
print("Done")
print("")
