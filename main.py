import numpy as np
from scipy import stats
import pandas as pd

from models.toymodel import Toymodel # importing toy model (benchmark?)
from models.BAM import BAM_base # BAM base model
from models.BAM_nonrobust import BAM_base_nonrobust
#from models.BAM_firstcycles import BAM_base # BAM base model
from estimation.data_prep import Filters

#################################################################################################
# Simulating a simple macro ABM
#################################################################################################

#toymodel = Toymodel(Time=1000, Ni=100, MC=1, gamma=2, pbar=0.01, delta=0.05, rbar=0.075, plots=True)
#toymodel_simulation = toymodel.simulation_toy()
#print(toymodel_simulation)

#################################################################################################
# Simulating the BAM base model by Delli Gatti (2011)
#################################################################################################

""" intial Parameters to be estimated later"""
parameters = {'Nh':500, 'Nf':100, 'Nb':10, 'T':100, 
              'Z':2, 'M':4, 'H':2, 'H_eta': 0.1, 'H_rho':0.1, 'H_phi':0.01, 'h_xi':0.05,
              'c_P':1, 'c_R':0.5}
# 'Z':2 im Buch -> bei ihm 3 !!!
#BAM_old = BAM_base_nonrobust(MC = 1, parameters=parameters, plots=True)
#BAM_simulation = BAM_old.simulation()
#print(BAM_simulation)

""" simulate the base model"""
BAM = BAM_base(T=10, MC = 1, plots=True,
               Nh=500, Nf=100, Nb=10,
               H_eta=0.1, H_rho=0.1, H_phi=0.01, h_xi=0.05) 
BAM_simulation = BAM.simulation()
print(BAM_simulation)


#################################################################################################
# Estimating the simple macro ABM
#################################################################################################

"""loading the data and plotting the data and its components"""
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

""" Idee: extra class mit function für MC plots,
=> MC analysis "nicht" so wie in Buch, sondern analog zu MC analysis in book (pract. applications)"""


#################################################################################################
# Forecasting
#################################################################################################


