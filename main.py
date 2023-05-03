"""
BAM model from Delli Gatti 2011. 
@author: maxbaldus
"""

# libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

import os
import logging

import time

# parallel computing
from joblib import Parallel, delayed
import multiprocessing


# Disable Tensorflow Deprecation Warnings
#logging.disable(logging.WARNING)
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import model classes
from models.toymodel import Toymodel
from models.BAM import BAM_mc # BAM with regular mc loop
from models.BAM_parallel import BAM_parallel # BAM model parallized


#from models.BAM_numba import BAM_simulation_numba # BAM as a function, exploiting numba

# import estimation classes
from estimation.data_prep import Filters
from estimation.mdn import mdn
from sampling import sample_posterior



#################################################################################################
# Simulating a simple macro ABM
#################################################################################################

"""
Simulating a simple macro model with 10 Monte carlo replications. 
The plots are saved into plots/toymodel folder.
"""

# instantiate the toymodel class
toymodel = Toymodel(Time=1000, Ni=100, MC=5,  
                    plots=True, filters=False)

# simulate the toy model and create the plots 
# toy_simulations = toymodel.simulation(gamma=2, pbar=0.01, delta=0.05, rbar=0.075) 
parameter = np.array([2, 0.01, 0.05, 0.075])
toy_simulations = toymodel.simulation(parameter) 

#################################################################################################
# Simulating the BAM model(s) by Delli Gatti (2011)
#################################################################################################

""" 
Simulating the base model and the plus version MC times, with and without parallel computing.
The plots of both models are saved into plots/BAM and plots/BAM_plus respectively.  
"""

# number of MC simulations
MC = 2

# upper bound of price growth rate
H_eta=0.1
# upper bound of quantity growth rate
H_rho=0.1
# upper bound bank costs 
H_phi=0.01
# upper bound growth rate of wages
h_xi=0.05
# parameter
parameter = np.array([H_eta, H_rho, H_phi, h_xi])

"""# simulate the model MC times using parallel computing
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
# np.save('data/simulations/BAM_10MC', BAM_simulations_parallel) # save the 10 simulations"""


# simulating model 1 time without parallising 
"""BAM_model = BAM_mc(T=1000, MC = MC, Nh=500, Nf=100, Nb=10,
                plots=True, csv=False) 
print("")
print('--------------------------------------')
print("Simulating BAM model without parallising %s times" %MC)
start_time = time.time()

BAM_simulations =  BAM_model.simulation(theta=parameter)

print("")
print("--- %s minutes ---" % ((time.time() - start_time)/60))"""
# approximately 2 minutes for one 1 run


"""
# simulate plus version by setting growth parameters not to 0 !!
# NO: use another function s.t. having specific bounds later .. !!


Analyse:
Idee: extra class mit function für MC plots,
=> MC analysis "nicht" so wie in Buch, sondern analog zu MC analysis in book (pract. applications)
"""

#################################################################################################
# Estimating the BAM model(s)
#################################################################################################

"""
Generate MC simluations of the model, ech with length T, for each parameter combination.

1)
First the estimation method is tested by using pseudo-empirical data with a-priori specified parameter values.
The first mc simulation with the parameter configuration from above is used as the 'observed' dataset. 
"""
# number of MC simulations per parameter combination
MC = 2

# pseudo empirical data
BAM_simulations = np.transpose(np.load("data/simulations/BAM_10MC.npy")) # load pesudo random data
BAM_obs = BAM_simulations[BAM_simulations.shape[0]-500:BAM_simulations.shape[0],0]

# create new instance of the BAM model with 100 MC replications 
BAM_model = BAM_mc(T=1000, MC = MC, Nh=500, Nf=100, Nb=10,
                plots=False, csv=False) 

# define the upper and lower bound for each parameter value, packed into a 2x#free parameters dataframe (2d numpy array) 
# with one column for each free parameter and the first (second) row being the lower (upper) bound respectively
bounds_BAM = np.transpose(np.array([ [0.07,0.13], [0.07,0.13], [0.07,0.13], [0.02,0.08] ]))

# initialize the sampling methods 
BAM_posterior = sample_posterior(model = BAM_model, bounds = bounds_BAM, data_obs=BAM_obs, filter=False)

grid_size = 5

# Use a plain grid to compute MC simulations of length T for each parameter combination
start_time = time.time()

args = BAM_posterior.simulation_block(grid_size, path = 'data/simulations/BAM_simulations/latin_hypercube')

def grid_search_parallel(args):
            
    # current parameter combination
    theta_current = args['theta']
    # simulate the model each time with a new parameter combination from the grid
    simulations = BAM_model.simulation(theta_current)
    
    # current path to save the simulation to
    current_path = args['path']
    # save simulated data 
    np.save(current_path, simulations)
    
    # plot the first mc simulation
    plt.clf()
    plt.plot(np.log(simulations[:,0]))
    plt.xlabel("Time")
    plt.ylabel("Log output")
    plt.savefig(current_path + ".png")

pool = multiprocessing.Pool(processes=4)
pool.map_async(grid_search_parallel, args)
pool.close()

print("")
print("--- %s minutes ---" % ((time.time() - start_time)/60))
# without parallising: 2MC=5 minutes * 5thetas = 25 minutes
# parallel with 4 cores: 2MC=5 minutes * 5thetas = 25 /4 = 7 minutes

print("blub")

# Approximate the posterior distr. of each parameter using the simulated data and given empirical data via mdn.
BAM_posterior.approximate_posterior(grid_size, path = 'data/simulations/toymodel_simulations/latin_hypercube')
# path = 'data/simulations/toymodel_simulations/latin_hypercube'
print("blub")

# Gatti 2020
# 5000 combinations
# MC = 20 
# T = 3000, discarding the first 2500

# aleen
# MC = 100 




"""
2) Estimating the BAM model using real data on US GDP ??, using the same artificial data generated above. 
"""












#################################################################################################
# Estimating the simple macro ABM
#################################################################################################

"""
...
1) First the estimation method is tested by using pseudo-empirical data with a-priori specified parameter values.
The first mc simulation with the parameter configuration from above is used as the 'observed' dataset. 
"""

# pseudo empirical data
toy_data_obs = toy_simulations[:,0] 

# create new instance of the toy model with 100 MC replications 
toymodel_est = Toymodel(Time=1500, Ni=100, MC=100, 
                    plots=False, filters=False)

# define the upper and lower bound for each parameter value, packed into a 2x#free parameters dataframe (2d numpy array) 
# with one column for each free parameter and the first (second) row being the lower (upper) bound respectively
bounds_toy_para = np.transpose(np.array([ [1,3], [0.001, 0.1], [0.001, 0.1], [0.001, 0.1] ]))

# initialize the sampling methods (grid search and MH algorithm)
toy_posterior = sample_posterior(model = toymodel_est, bounds = bounds_toy_para, data_obs=toy_data_obs, filter=False)

# Use the plain grid search to compute the posterior estimates of each free parameter
# the likelihood approximation method used inside the sampling method is set inside the sampling class
toy_posterior.grid_search(grid_size = 3000, path = 'data/simulations/toymodel_simulations/latin_hypercube')
print("blub")

# Gatti 2020
# 5000 combinations
# MC = 20 
# T = 3000, discarding the first 2500

# aleen
# MC = 100 




# Estimating the TOY model using real data on US GDP ??
# loading the data and plotting the data and its components
#german_gdp = pd.read_csv("data/CLVMNACSCAB1GQDE.csv") # gdp 
# Inflation
# Unemployment100   
#filters = Filters(german_gdp['CLVMNACSCAB1GQDE'], inflation=None, unemployment=None)
#gdp_components = filters.HP_filter(empirical=True) # plots are saved in plots ..
# print(component# s)


# erstmal nur mit einer Zeitreihe
# robustness checks a la pape100    
# tuning the lag length? -> for forecasts ..??
# MH sampling or grid-search ??!!



#################################################################################################
# Forecasting
#################################################################################################


