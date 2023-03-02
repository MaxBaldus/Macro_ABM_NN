import numpy as np
import matplotlib.pyplot as plt
from estimation.data_prep import Filters

"""
Explaining model in short words...
"""

class Toymodel:
    
    """
    Initializing the class and constructer objects (init class) by assigning arguments passed to attributes of the class. 
    These attributes from the init class can then be used throughout all methods of the particular class.
    """ 
    
    def __init__(self, Time: int, Ni: int, MC: int, gamma: int, pbar: int, delta:int, rbar: int, plots:bool, filters:bool):
        
        """
        simulation hyperparameters
        """
        self.Time = Time # number of simulation runs
        self.Ni = Ni # number of firms
        self.MC = MC # MC runs
        
        self.plots = plots # plot parameter decides whether plots are produced
        self.filters = filters # decide whether ts filters are applied
        
        """
        model parameters
        """
        self.gamma = gamma # investment accelerator
        self.pbar = pbar # constant price part 
        self.delta = delta # capital depreceation rate
        self.rbar = rbar # constant interest rate part

        """
        Aggregate report variables: dimension = Time x MC (1 time series for each MC run)
        """
        self.YY = np.zeros((self.Time,self.MC)) # aggregate production (GDP)
        self.YY_growth = np.zeros((self.Time,self.MC)) # GDP growth rate
        self.AA = np.zeros((self.Time,self.MC)) # aggregate net worth
        self.BB = np.zeros((self.Time,self.MC)) # aggregate debt
        self.LEV = np.zeros((self.Time,self.MC)) # leverage 

        self.YY_cycle = np.zeros((self.Time,self.MC))
        self.YY_trend = np.zeros((self.Time,self.MC))
    
    """ 
    The simulation of the toy base model is run with the simulation method
    """
    def simulation_toy(self):
        
        for mc in range(self.MC):
            
            """
            By setting a different seed for each mc run a different sequence of random numbers is sampled during each simulation.
            """
            np.random.seed(mc) 

            print("MC run %s" %mc)

            """
            Initializing the individual report variables. Data is overwritten with each time period t (i.e. no tensors used here).
            """ 
            A = np.ones(self.Ni) # net worth
            K = np.ones(self.Ni) # capital
            B = np.zeros(self.Ni) # debt
            I = np.zeros(self.Ni) # investment
            P = np.zeros(self.Ni) # prices (stochastic)
            Y = np.zeros(self.Ni) # production
            Z = 2*np.random.uniform(low=0,high=1, size = self.Ni) + self.pbar # initial profits are scaled random uniform distributed numbers
            R = np.ones(self.Ni) # interest rates
            
            phi = 0.1 # capital productivity

            for t in range(self.Time):

                """
                1) Each firm determines investment level
                """
                I = self.gamma * Z # investment of current round are profits of last round times the investment accelerator
                for f in range(self.Ni):
                    I[f] = 0 if I[f] < 0 else I[f] # since investments cannot be negative, they are set to 0 in case profits were negative last round
                
                """
                2) capital accumulation
                """
                K = (1-self.delta)*K + I # capital of each firm in this round is capital of last round minus depreceation plus investments in this round

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
                P = 2 * np.random.uniform(low=0,high=1, size = self.Ni) + self.pbar # stochastic price changes each period 

                """
                6) Interest rates
                """
                R = self.rbar + self.rbar*(B/A)**self.rbar # firms specific interest rate: base interest rate (set by CB) + adjusted risk premium (i.e. leverage = loans/equity)

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
                self.YY[t,mc] = sum(Y) # aggregate production 
                self.YY_growth[t,mc] = (self.YY[t,mc]-self.YY[t-1,mc]) / self.YY[t-1,mc] if t!=0 else 0
                self.AA[t,mc] = sum(A) # aggregated net worth
                self.BB[t,mc] = sum(B) # aggregated debt
                self.LEV[t,mc] = self.BB[t,mc] / self.AA[t,mc] # aggregated leverage (aggregated debt divided by aggregated equity)

            #print(np.mean(self.YY_growth[:,mc])) # average growth rates around 0 (since no research and development)

            if self.filters:
                
                """
                applying the filters to each mc simulated time series
                """
                filters = Filters(GDP=self.YY[:,mc], inflation=None, unemployment=None)
                components = filters.HP_filter(empirical = False)    
                # print(components)      

            if self.plots:
                
                """
                Plotting logarithms of aggregate report variables
                """
                plt.figure("GDP") # log GDP figure
                plt.plot(np.log(self.YY[:,mc]), label = "GDP mc=%s" %mc) # GDP of 1st MC round
                plt.ylabel("Log output")
                plt.savefig("plots/toymodel/GDP_mc%s.png" %mc)


                plt.figure("GDP growth") # GDP growth rate figure
                plt.plot(self.YY_growth[1:self.Time,mc], label = "GDP_growth mc=%s" %mc) # GDP growth of 1st MC round
                plt.xlabel("Time")
                plt.ylabel("Output growth rate")
                plt.ylim(-0.1, 0.1)
                plt.savefig("plots/toymodel/GDP_growth_mc%s.png" %mc)

                if self.filters:
                    plt.figure("GDP Components") # log GDP figure
                    plt.plot(components[mc][:0], label = "cycle") # GDP of 1st MC round
                    plt.xlabel("Time")
                    plt.ylabel("Log output")
                    plt.savefig("plots/toymodel/??_mc%s.png" %mc)
                
                # plt.show() # show all figures
        
        
        return self.YY


    """
    In the +_version of the toymodel similar to the BAM_+ version a sinmple growth process is included, i.e. ...
    """
    def simulation_toyplus(self):
        
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
            Z = 2*np.random.uniform(low=0,high=1, size = self.Ni) + self.pbar # initial profits are scaled random uniform distributed numbers
            R = np.ones(self.Ni) # interest rates
            
            phi = 0.1

            for t in range(self.Time):

                """1) Each firm determines investment level"""
                I = self.gamma * Z # investment of current round are profits of last round times the investment accelerator
                for f in range(self.Ni):
                    I[f] = 0 if I[f] < 0 else I[f] # since investments cannot be negative, they are set to 0 in case profits were negative last round
                
                """2) capital accumulation"""
                K = (1-self.delta)*K + I # capital of each firm in this round is capital of last round minus depreceation plus investments in this round

                """3) Production"""
                Y = phi * K # productivity of the capital determines production (constant share each round)
                
                # include GROWTH HERE !!

                """4) Firms determine Debt"""
                B = K - A # debt depends on capital accumulation and net worth of the firm 
                for f in range(self.Ni):
                    B[f] = 0 if B[f] < 0 else B[f] # firms with negative debt are not considered, hence their debt is set to 0 in case debt is negative (firms only self financed)

                """5) Prices"""
                P = 2 * np.random.uniform(low=0,high=1, size = self.Ni) + self.pbar # stochastic price changes each period 

                """6) Interest rates"""
                R = self.rbar + self.rbar*(B/A)**self.rbar # firms specific interest rate: base interest rate (set by CB) + adjusted risk premium (i.e. leverage = loans/equity)

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
                self.YY[t,mc] = sum(Y) # aggregate production 
                self.YY_growth[t,mc] = (self.YY[t,mc]-self.YY[t-1,mc]) / self.YY[t-1,mc] if t!=0 else 0
                self.AA[t,mc] = sum(A) # aggregated net worth
                self.BB[t,mc] = sum(B) # aggregated debt
                self.LEV[t,mc] = self.BB[t,mc] / self.AA[t,mc] # aggregated leverage (aggregated debt divided by aggregated equity)

            #print(np.mean(self.YY_growth[:,mc])) # average growth rates around 0

        """Plotting logarithms of aggregate report variables"""
        plt.figure("GDP") # log GDP figure
        plt.plot(np.log(self.YY[:,0]), label = "GDP mc=1") # GDP of 1st MC round
        plt.plot(np.log(self.YY[:,1]), label = "GDP mc=2") # GDP of 2nd MC round
        plt.plot(np.log(self.YY[:,2]), label = "GDP mc=3") # GDP of 3rd MC round
        plt.xlabel("Time")
        plt.ylabel("Log output")
        plt.legend()

        plt.figure("GDP growth") # GDP growth rate figure
        plt.plot(self.YY_growth[1:self.Time,0], label = "GDP_growth mc=1") # GDP growth of 1st MC round
        #plt.plot(YY_growth[1:self.Time,1], label = "GDP_growth mc=2") # GDP growth of 2nd MC round
        #plt.plot(YY_growth[1:self.Time,2], label = "GDP_growth mc=3") # GDP growth of 3rd MC round
        plt.xlabel("Time")
        plt.ylabel("Output growth rate")
        plt.ylim(-0.1, 0.1)
        plt.legend()
        
        # hence need to include some growth in order to get around 2% growth 
        # growth similar to BAM MODEL !!??!!
        
        plt.show() # show all figures
        
        
        return self.YY, self.AA, self.BB, self.LEV
            

    



    


        

    