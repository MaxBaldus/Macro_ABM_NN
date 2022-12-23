import numpy as np

class Toymodel:
    # initialize the class / construct objects / instructor -> assign arguments passed to attributes of the class 
    def __init__(self, Time: int, Ni: int, MC: int, gamma: int, phi: int, pbar: int, delta:int, rbar: int):
        # simulation hyperparameter
        self.Time = Time # number of simulation runs
        self.Ni = Ni # number of firms
        self.MC = MC # MC runs
        # model parameters
        self.gamma = gamma # investment accelerator
        self.phi = phi # capital productivity
        self.pbar = pbar # constant price part 
        self.delta = delta # capital depreceation rate
        self.rbar = rbar # constant interest rate part
    
    # run the simulation with the simulation method
    def simulation(self):
        
        # aggregate report variables: dim = MC x Time (1 time series for each MC run)
        YY = np.zeros((self.Time,self.MC)) # aggregate production
        AA = np.zeros((self.Time,self.MC)) # aggregate net worth
        BB = np.zeros((self.Time,self.MC)) # aggregate debt
        LEV = np.zeros((self.Time,self.MC)) # leverage 
        
        for mc in range(self.MC):
            print("MC run %s" %mc)
            # employ different sequence of random numbers during each simulation, i.e. assign to every simuation a different seed
            # np.random.seed(mc) 
            
            # individual variables (for each agent): due to memory stoarge, it is always overwritten with each new mc run 
            A = np.ones((self.Ni, 1)) # net worth
            K = np.ones((self.Ni, 1)) # capital
            B = np.ones((self.Ni, 1)) # debt
            I = np.ones((self.Ni, 1)) # investment
            P = np.ones((self.Ni, 1)) # prices
            Y = np.ones((self.Ni, 1)) # production
            Z = np.ones((self.Ni, 1)) # profits 

            for t in range(2, self.Time + 1):
                print("do computations and fill aggregated variables")
    
        return YY, AA, BB, LEV

            

    



    


        

    