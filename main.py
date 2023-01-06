from random import seed
from models.toymodel import Toymodel # importing toy model (benchmark?)
from models.BAM import BAM_base # BAM base model

"""Simulating the toy model"""
#toymodel = Toymodel(Time=10, Ni=100, MC=3, gamma=2, phi=0.1, pbar=0.01, delta=0.05, rbar=0.075)
#ts = toymodel.simulation()



#################################################################################################
# Simulating the BAM base model by Delli Gatti (2011)
#################################################################################################

""" intial Parameters to be estimated later"""
parameters = {'Nh':50, 'Nf':10, 'Nb':5, 'T':1000, 
              'Z':2, 'M':4, 'H':2, 'H_eta': 0.1, 'H_rho':0.1, 'H_phi':0.01, 'h_xi':0.05,
              'c_P':1, 'c_R':0.5}

""" simulate the base model"""
BAM = BAM_base(MC = 2, parameters=parameters, plots=False) 
BAM_simulate = BAM.simulation()
print(BAM_simulate)

# structure: 
# Class model# 
# one model is one class => can be called => inputs are simulation parameters and specific model parameters
# output is specific time series (e.g. gdp as numpy array) needed for estimation -> analog to output by DP

# class estimation
# calling neural network with "real and simulated" data and estimate ??
# output are ESTIMATED parameters.. ??

# class forecasting
# using estimated model parameters for forecasting.. 



""" Idee: extra class mit function für MC plots,
=> MC analysis "nicht" so wie in Buch, sondern analog zu MC analysis in book (pract. applications)"""