# libraries
import numpy as np
import matplotlib.pyplot as plt
import tqdm

# classes
from estimation.data_prep import Filters

"""
Explaining model in short words...
"""

class Toymodel:
    
    """
    Initializing the class and constructer objects (init class) by assigning arguments passed to attributes of the class. 
    These attributes from the init class can then be used throughout all methods of the particular class.
    """ 
    
    def __init__(self, Time: int, Ni: int, MC: int, plots:bool, filters:bool):
        
        """
        simulation hyperparameters
        """
        self.Time = Time # number of simulation runs
        self.Ni = Ni # number of firms
        self.MC = MC # MC runs
        
        self.plots = plots # plot parameter decides whether plots are produced
        self.filters = filters # decide whether ts filters are applied  

    def simulation(self, theta):

        """
        The simulation of the toy base model is run with the simulation method
        
        Inputs are the simulation hyper parameters and 1-d numpy array with the following model parameters:

        gamma = investment accelerator
        pbar = constant price part 
        delta = capital depreceation rate
        rbar = constant interest rate part
        """
        gamma = theta[0]
        pbar = theta[1]
        delta = theta[2]
        rbar = theta[3] 

        """
        Aggregate report variables: dimension = Time x MC (1 time series for each MC run)
        """
        YY = np.zeros((self.Time,self.MC)) # aggregate production (GDP)
        YY_growth = np.zeros((self.Time,self.MC)) # GDP growth rate
        AA = np.zeros((self.Time,self.MC)) # aggregate net worth
        BB = np.zeros((self.Time,self.MC)) # aggregate debt
        LEV = np.zeros((self.Time,self.MC)) # leverage 

        YY_cycle = np.zeros((self.Time,self.MC))
        YY_trend = np.zeros((self.Time,self.MC))
        
        for mc in tqdm(range(self.MC)):
            
            """
            By setting a different seed for each mc run a different sequence of random numbers is sampled during each simulation.
            """
            np.random.seed(mc) 

            """
            Initializing the individual report variables. Data is overwritten with each time period t (i.e. no tensors used here).
            """ 
            A = np.ones(self.Ni) # net worth
            K = np.ones(self.Ni) # capital
            B = np.zeros(self.Ni) # debt
            I = np.zeros(self.Ni) # investment
            P = np.zeros(self.Ni) # prices (stochastic)
            Y = np.zeros(self.Ni) # production
            Z = 2*np.random.uniform(low=0,high=1, size = self.Ni) + pbar # initial profits are scaled random uniform distributed numbers
            R = np.ones(self.Ni) # interest rates
            
            phi = 0.1 # capital productivity

            for t in range(self.Time):

                """
                1) Each firm determines investment level
                """
                I = gamma * Z # investment of current round are profits of last round times the investment accelerator
                for f in range(self.Ni):
                    I[f] = 0 if I[f] < 0 else I[f] # since investments cannot be negative, they are set to 0 in case profits were negative last round
                
                """
                2) capital accumulation
                """
                K = (1-delta)*K + I # capital of each firm in this round is capital of last round minus depreceation plus investments in this round

                """
                3) Production
                """
                Y = phi * K # productivity of the capital determines production (constant share each round)

                """
                4) Firms determine Debt
                """
                B = K - A # debt depends on capital accumulation and net worth of the firm 
                for f in range(self.Ni):
                    B[f] = 0 if B[f] < 0 else B[f] # firms with negative debt are not considered, hence their debt is set to 0 in case debt is negative (firms only self financed)

                """
                5) Prices
                """
                P = 2 * np.random.uniform(low=0,high=1, size = self.Ni) + pbar # stochastic price changes each period 

                """
                6) Interest rates
                """
                R = rbar + rbar*(B/A)**rbar # firms specific interest rate: base interest rate (set by CB) + adjusted risk premium (i.e. leverage = loans/equity)

                """
                7) Profits
                """
                Z = (P * Y) - (R * K) # profit is price * quantity (assuming supply = demand, i.e. everything sold) - interest paid on kapital 

                """
                8) Net worth
                """
                A = A + Z # new net worth is past net worth + profits of this round

                """
                9) Entry-exit
                """
                for f in range(self.Ni):
                    # if net worth of f firm goes negative, firm goes bankrupt => have a 1:1 replacement with the new firm 
                    if A[f] < 0: 
                        Z[f] = 0 # profits of new firm are 0
                        K[f] = 1 # capital of new firm is 1
                        A[f] = 1 # net worth of new firm is also 1 

                """
                Compute and store aggregated report variables
                """
                YY[t,mc] = sum(Y) # aggregate production 
                YY_growth[t,mc] = (YY[t,mc]-YY[t-1,mc]) / YY[t-1,mc] if t!=0 else 0
                AA[t,mc] = sum(A) # aggregated net worth
                BB[t,mc] = sum(B) # aggregated debt
                LEV[t,mc] = BB[t,mc] / AA[t,mc] # aggregated leverage (aggregated debt divided by aggregated equity)

            #print(np.mean(YY_growth[:,mc])) # average growth rates around 0 (since no research and development)

            if self.filters:
                
                """
                applying the filters to each mc simulated time series
                """
                filters = Filters(GDP=YY[:,mc], inflation=None, unemployment=None)
                components = filters.HP_filter(empirical = False)    
                # print(components)      

            if self.plots:
                
                """
                Plotting the aggregate report variables
                """
                # log GDP figure
                plt.clf()
                plt.plot(np.log(YY[:,mc]), label = "GDP mc=%s" %mc) # GDP of 1st MC round
                plt.ylabel("Log output")
                plt.savefig("plots/toymodel/GDP_mc%s.png" %mc)

                # GDP growth rate figure
                plt.clf()
                plt.plot(YY_growth[1:self.Time,mc], label = "GDP_growth mc=%s" %mc) # GDP growth of 1st MC round
                plt.xlabel("Time")
                plt.ylabel("Output growth rate")
                plt.ylim(-0.1, 0.1)
                plt.savefig("plots/toymodel/GDP_growth_mc%s.png" %mc)

                if self.filters:
                    # GDP components
                    plt.clf()
                    plt.figure("GDP Components") 
                    plt.plot(components[mc][:0], label = "cycle") # GDP of 1st MC round
                    plt.xlabel("Time")
                    plt.ylabel("Log output")
                    plt.savefig("plots/toymodel/??_mc%s.png" %mc)
                
                # plt.show() # show all figures
        
        return YY
        # what if I use filter (what to return.. ??)



    def simulation_toyplus(self, theta):

        """
        In the +_version of the toymodel similar to the BAM_+ version a sinmple growth process is included, i.e. ...

        model parameters:
        
        gamma = investment accelerator
        pbar = constant price part 
        delta = capital depreceation rate
        rbar = constant interest rate part
        """
        gamma = theta[0]
        pbar = theta[1]
        delta = theta[2]
        rbar = theta[3] 

        """
        Aggregate report variables: dimension = Time x MC (1 time series for each MC run)
        """
        YY = np.zeros((self.Time,self.MC)) # aggregate production (GDP)
        YY_growth = np.zeros((self.Time,self.MC)) # GDP growth rate
        AA = np.zeros((self.Time,self.MC)) # aggregate net worth
        BB = np.zeros((self.Time,self.MC)) # aggregate debt
        LEV = np.zeros((self.Time,self.MC)) # leverage 

        YY_cycle = np.zeros((self.Time,self.MC))
        YY_trend = np.zeros((self.Time,self.MC))
        
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
            Z = 2*np.random.uniform(low=0,high=1, size = self.Ni) + pbar # initial profits are scaled random uniform distributed numbers
            R = np.ones(self.Ni) # interest rates
            
            phi = 0.1

            for t in range(self.Time):

                """1) Each firm determines investment level"""
                I = gamma * Z # investment of current round are profits of last round times the investment accelerator
                for f in range(self.Ni):
                    I[f] = 0 if I[f] < 0 else I[f] # since investments cannot be negative, they are set to 0 in case profits were negative last round
                
                """2) capital accumulation"""
                K = (1-delta)*K + I # capital of each firm in this round is capital of last round minus depreceation plus investments in this round

                """3) Production"""
                Y = phi * K # productivity of the capital determines production (constant share each round)
                
                # include GROWTH HERE !!

                """4) Firms determine Debt"""
                B = K - A # debt depends on capital accumulation and net worth of the firm 
                for f in range(self.Ni):
                    B[f] = 0 if B[f] < 0 else B[f] # firms with negative debt are not considered, hence their debt is set to 0 in case debt is negative (firms only self financed)

                """5) Prices"""
                P = 2 * np.random.uniform(low=0,high=1, size = self.Ni) + pbar # stochastic price changes each period 

                """6) Interest rates"""
                R = rbar + rbar*(B/A)**rbar # firms specific interest rate: base interest rate (set by CB) + adjusted risk premium (i.e. leverage = loans/equity)

                """7) Profits"""
                Z = (P * Y) - (R * K) # profit is price * quantity (assuming supply = demand, i.e. everything sold) - interest paid on kapital 

                """8) Net worth"""
                A = A + Z # new net worth is past net worth + profits of this round

                """9) Entry-exit"""
                for f in range(self.Ni):
                    # if net worth of f firm goes negative, firm goes bankrupt => have a 1:1 replacement with the new firm 
                    if A[f] < 0: 
                        Z[f] = 0 # profits of new firm are 0
                        K[f] = 1 # capital of new firm is 1
                        A[f] = 1 # net worth of new firm is also 1 

                """Compute and store aggregated report variables"""
                YY[t,mc] = sum(Y) # aggregate production 
                YY_growth[t,mc] = (YY[t,mc]-YY[t-1,mc]) / YY[t-1,mc] if t!=0 else 0
                AA[t,mc] = sum(A) # aggregated net worth
                BB[t,mc] = sum(B) # aggregated debt
                LEV[t,mc] = BB[t,mc] / AA[t,mc] # aggregated leverage (aggregated debt divided by aggregated equity)

            #print(np.mean(YY_growth[:,mc])) # average growth rates around 0

        """Plotting logarithms of aggregate report variables"""
        plt.figure("GDP") # log GDP figure
        plt.plot(np.log(YY[:,0]), label = "GDP mc=1") # GDP of 1st MC round
        plt.plot(np.log(YY[:,1]), label = "GDP mc=2") # GDP of 2nd MC round
        plt.plot(np.log(YY[:,2]), label = "GDP mc=3") # GDP of 3rd MC round
        plt.xlabel("Time")
        plt.ylabel("Log output")
        plt.legend()

        plt.figure("GDP growth") # GDP growth rate figure
        plt.plot(YY_growth[1:self.Time,0], label = "GDP_growth mc=1") # GDP growth of 1st MC round
        #plt.plot(YY_growth[1:self.Time,1], label = "GDP_growth mc=2") # GDP growth of 2nd MC round
        #plt.plot(YY_growth[1:self.Time,2], label = "GDP_growth mc=3") # GDP growth of 3rd MC round
        plt.xlabel("Time")
        plt.ylabel("Output growth rate")
        plt.ylim(-0.1, 0.1)
        plt.legend()
        
        # hence need to include some growth in order to get around 2% growth 
        # growth similar to BAM MODEL !!??!!
        
        plt.show() # show all figures
        
        
        return YY, AA, BB, LEV
            

    



    


        

    