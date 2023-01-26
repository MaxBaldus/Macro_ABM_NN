import numpy as np

"""Explaining model in short words..."""

class Toymodel:
    """ Initializing the class and constructer objects (init class) by assign arguments passed to attributes of the class. 
    These attributes from the init class can then be used throughout all methods of the particular class.""" 
    def __init__(self, Time: int, Ni: int, MC: int, gamma: int, phi: int, pbar: int, delta:int, rbar: int):
        """simulation hyperparameter"""
        self.Time = Time # number of simulation runs
        self.Ni = Ni # number of firms
        self.MC = MC # MC runs
        
        """model parameters"""
        self.gamma = gamma # investment accelerator
        self.phi = phi # capital productivity
        self.pbar = pbar # constant price part 
        self.delta = delta # capital depreceation rate
        self.rbar = rbar # constant interest rate part
    
    """ The simulation is run with the simulation method"""
    def simulation(self):
        
        """ Aggregate report variables: dimension = Time x MC (1 time series for each MC run)"""
        YY = np.zeros((self.Time,self.MC)) # aggregate production
        AA = np.zeros((self.Time,self.MC)) # aggregate net worth
        BB = np.zeros((self.Time,self.MC)) # aggregate debt
        LEV = np.zeros((self.Time,self.MC)) # leverage 
        
        for mc in range(self.MC):
            
            """ By setting a different seed for each mc run a different sequence of random numbers is sampled during each simulation."""
            np.random.seed(mc) 

            print("MC run %s" %mc)

            """ Initializing the individual report variables. Data is overwritten with each time period t (i.e. no tensors used here).""" 
            A = np.ones(self.Ni) # net worth
            K = np.ones(self.Ni) # capital
            B = np.zeros(self.Ni) # debt
            I = np.zeros(self.Ni) # investment
            P = np.zeros(self.Ni) # prices (stochastic)
            Y = np.zeros(self.Ni) # production
            Z = 2*np.random.uniform(low=0,high=1, size = self.Ni) + self.pbar # profits are scaled random uniform distributed numbers
            R = np.ones(self.Ni) # interest rates
            LEV = np.zeros(self.Ni) # leverage
            

            for t in range(self.Time):
                print("blub")
                """1) Each firm determines investment level"""
                I = self.gamma * Z
    
        return YY, AA, BB, LEV

            

    



    


        

    