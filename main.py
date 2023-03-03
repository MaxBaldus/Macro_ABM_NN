import numpy as np
from scipy import stats
import pandas as pd

import models.toymodel
#from models.toymodel import Toymodel # importing toy model (benchmark?)
from models.BAM import BAM_base # BAM base model
from estimation.data_prep import Filters

#################################################################################################
# Simulating a simple macro ABM
#################################################################################################

"""
Simulating a simple macro model with 3 Monte carlo replications. Plots are saved into plots/toymodel
"""
#toymodel = Toymodel(Time=1000, Ni=100, MC=3, 
#                    gamma=2, pbar=0.01, delta=0.05, rbar=0.075, 
#                    plots=True, filters=False)
#toymodel_simulation = models.toymodel.simple_macro_ABM(
#                    Time=1000, Ni=100, MC=3, 
#                    gamma=2, pbar=0.01, delta=0.05, rbar=0.075, 
#                    plots=True, filters=False)
#print(toymodel_simulation)

#################################################################################################
# Simulating the BAM base model(s) by Delli Gatti (2011)
#################################################################################################

""" 
Simulating the base model and the plus version two times (MC = 2).
Two different kind of plots are saved into plots/BAM and plots/BAM_plus respectively. 
Once the entire simulation (here T = 1000) and once only the last 500 observations (T-500) s.t. 
the model(s) have time to convergence to their statistical equilibria w.r.t. to the major 
aggregate report variables (GDP, unemployment and inflation rate). 
"""
BAM = BAM_base(T=1000, MC = 1, plots=True,
               Nh=500, Nf=100, Nb=10,
               H_eta=0.1, H_rho=0.1, H_phi=0.01, h_xi=0.05) 
BAM_simulation = BAM.simulation()
print(BAM_simulation)

# simulate plus version


#################################################################################################
# Estimating the simple macro ABM
#################################################################################################

"""
First the estimation method is tested by using pseudo-empirical data with a-priori specified parameter values.
The first mc simulation is used as the 'observed' dataset. 
"""

#pseudo_empirical = models.toymodel.simple_macro_ABM(
#                    Time=2000, Ni=100, MC=1, 
#                    gamma=2, pbar=0.01, delta=0.05, rbar=0.075, 
#                   plots=True, filters=False)

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


