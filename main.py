import numpy as np
from scipy import stats
import pandas as pd

# import classes
from models.toymodel import Toymodel
from models.BAM import BAM 
from estimation.data_prep import Filters
from estimation.mde import mdn
from estimation.sampling import sample_posterior

#################################################################################################
# Simulating a simple macro ABM
#################################################################################################

"""
Simulating a simple macro model with 10 Monte carlo replications. 
The plots are saved into plots/toymodel folder.
"""

# instantiate the toymodel class
toymodel = Toymodel(Time=1000, Ni=100, MC=2,  
                    plots=True, filters=False)

# simulate the toy model and create the plots 
# toy_simulations = toymodel.simulation(gamma=2, pbar=0.01, delta=0.05, rbar=0.075) 
parameter = np.array([2, 0.01, 0.05, 0.075])
toy_simulations = toymodel.simulation(parameter) 

#################################################################################################
# Simulating the BAM model(s) by Delli Gatti (2011)
#################################################################################################

""" 
Simulating the base model and the plus version two times (MC = 2).
Two different kind of plots are saved into plots/BAM and plots/BAM_plus respectively. 
Once the entire simulation (here T = 1000) and once only the last 500 observations (T-500) s.t. 
the model(s) have time to convergence to their statistical equilibria w.r.t. to the major 
aggregate report variables (GDP, unemployment and inflation rate). 
"""
"""BAM = BAM(T=1000, MC = 1, plots=True,
               Nh=500, Nf=100, Nb=10,
               H_eta=0.1, H_rho=0.1, H_phi=0.01, h_xi=0.05) 
BAM_simulation = BAM.simulation()
print(BAM_simulation)"""

# simulate plus version by setting growth parameters not to 0 !!
# NO: use another function s.t. having specific bounds later .. !!

# change parameter input to np.array 
# ...

#################################################################################################
# Estimating the simple macro ABM
#################################################################################################

"""
First the estimation method is tested by using pseudo-empirical data with a-priori specified parameter values.
The first mc simulation with the parameter configuration from above is used as the 'observed' dataset. 
"""

# simulate the toymodel 1 times with 1000 observations, treted as the 'observed' time series in the following.
toy_data_obs = toy_simulations[:,0] # check again from above..  

# create new instance of the with 1000 ??? MC replications now
toymodel = Toymodel(Time=1000, Ni=100, MC=100, 
                    plots=False, filters=False)

# define the upper and lower bound for each parameter value, packed into a 2x#free parameters matrix 
# with one column for each free parameter and the first (second) row being the lower (upper) bound respectively
bounds_toy_para = np.transpose(np.array([ [1,5], [0.001, 0.1], [0.001, 0.1], [0.001, 0.1] ]))

# initialize the sampling methods (grid search and MH algorithm)
toy_posterior = sample_posterior(model = toymodel, bounds = bounds_toy_para)

# initialize the likelihood approximation method used inside the sampling method
# gleich in sampling rein ??!!

# first use the plain grid search to compute the posterior estimates of each free parameter
toy_posterior.grid_search(grid_size = 5, path = 'data/simulations/toymodel_simulations')
print("blub")

# sample possible parameter values 

# initialize mdn

# 
 

#print(toymodel_simulation)


"""
loading the data and plotting the data and its components
"""
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
# Estimating the BAM model(s)
#################################################################################################

""" 
Idee: extra class mit function für MC plots,
=> MC analysis "nicht" so wie in Buch, sondern analog zu MC analysis in book (pract. applications)
"""


#################################################################################################
# Forecasting
#################################################################################################


