import numpy as np

"""# main.py rein kopieren und aufbereiten ...
    #	erst initializieren => dann füllen (jeweils für HH, Firms und Banks) => checken mit NWA .. 
    #	orientiren an version_2_2 model 
    
    # alle individual variables mit self. ??
    	# an base orientieren.. ???s"""

class BAM_base:

    def __init__(self, MC:int, parameters:dict, plots:bool):
        
        self.MC = MC # Monte Carlo replications
        self.plots = plots # plot parameter decides whether plots are produced

        #################################################################################################
        # Parameters
        #################################################################################################
        """Parameter to be estimated"""
        self.Nh = parameters['Nh'] # number of HH
        self.Nf = parameters['Nf'] # number of firms - Fc in main_base
        self.Fb = parameters['Nb'] # number of banks
        self.T = parameters['T'] # simulation periods
        self.Z = parameters['Z'] # Number of trials in the goods market (number of firms HH's randomly chooses to buy goods from)
        self.M = parameters['M'] # Number of trials in the labor market (number of firms HH's apply to)
        self.H = parameters['H'] # Number of trials in the credit market (Number of banks a firms select randomly to search for credit)
        self.H_eta = parameters['H_eta'] # Maximum growth rate of prices (max value of price update parameter) - h_eta in main_ase
        self.H_rho = parameters['H_rho'] # Maximum growth rate of quantities (Wage increase parameter)
        self.H_phi = parameters['H_phi'] # Maximum amount of banks’ costs
        self.h_zeta = parameters['h_zeta'] # Maximum growth rate of wages (Wage increase parameter)
        
        # !!Noch nicht mit drin!!
        self.c_P = parameters['c_P'] # Propensity to consume of poorest people
        self.c_R = parameters['c_R'] # Propensity to consume of richest people

        "Parameters set by modeller"
        self.beta = 4 # ???
        self.theta = 8 # Duration of individual contract
        self.r_bar = 0.4 # Base interest rate set by central bank (absent in this model)

    def simulation(self):
        
        #################################################################################################
        # AGGREGATE REPORT VARIABLES
        #################################################################################################
        self.unemp_rate = np.zeros((self.T,self.MC)) # unemployment rate
        self.S_avg = np.zeros((self.T,self.MC)) # ??
        self.P_lvl = np.zeros((self.T,self.MC)) # price level
        self.avg_prod = np.zeros((self.T,self.MC)) # average production level
        self.inflation = np.zeros((self.T,self.MC)) # price inflation
        self.wage_lvl = np.zeros((self.T,self.MC)) # wage level
        self.wage_inflation = np.zeros((self.T,self.MC)) # wage inflation
        self.production = np.zeros((self.T,self.MC)) # YY
        self.production_by_value = np.zeros((self.T,self.MC)) # YY ??
        self.consumption = np.zeros((self.T,self.MC)) # HH extra consumption ??
        self.demand = np.zeros((self.T,self.MC)) # Demand
        self.h_income = np.zeros((self.T,self.MC)) # HH income
        self.hh_savings = np.zeros((self.T,self.MC)) # HH savings
        self.hh_consumption = np.zeros((self.T,self.MC)) # HH consumption ??

        # own
        self.consumption_rate = np.zeros((self.T,self.MC))

        for mc in range(self.MC):
            print("MC run number: %s" %mc)
            
            # set new seed each MC run
            np.random.seed(mc)
            
            #################################################################################################
            # INDIVIDUAL REPORT VARIABLES
            #################################################################################################

            """Initialize all individual report variables:
            structure: a vector with elements, one element for each Agent (1xNf, 1xNh, 1xNb) """

            # HOUSEHOLDS






            """Fill some individual report variables with random number when t = 0: """










            """Plot main aggregate report variables and save them in FOLDER"""
            if self.plots == True and self.MC <= 2:
                "plot"
                









            
    
    
    
    




class BAM_base_estimate:
    
    """same code as above, but here less invidual data is stored along the way to save memory (fast computation, no plots etc.)
    and output is such that it can be used later """

    def __init__(self, MC:int, parameters:dict):
        
        self.MC = MC # Monte Carlo replications



            

    



    


        

    
