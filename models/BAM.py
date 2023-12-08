"""
BAM model from Delli Gatti 2011. 
@author: maxbaldus
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import csv 
import math
from tqdm import tqdm

from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression



"""
Overall class structure: The class BAM_base contains the BAM model from Delli Gatti 2011. 
The inputs are the the simulation parameters and some integer parameter values calibrated in advance.  
"""

class BAM_mc:

    def __init__(self, T:int, MC:int, plots:bool, csv:bool, empirical:bool, path_empirical,
                 Nh:int, Nf:int, Nb:int):
                
        """
        General model parameter
        """
        self.MC = MC # Monte Carlo replications
        self.T = T # simulation periods
        self.Nh = Nh # number of HH
        self.Nf = Nf # number of firms - Fc in main_base
        self.Nb = Nb # number of banks

        self.plots = plots # plot parameter decides whether plots are produced
        self.csv = csv # decides whether csv file is written or not
        self.empirical = empirical # plotting when using estimated parameter values
        self.path_empirical = path_empirical # path to save empirical values
        
        """
        Parameters set by modeller / calibrated
        """        
        self.Z = 2 # Number of trials in the goods market (number of firms HH's randomly chooses to buy goods from)
        self.M = 4 # Number of trials in the labor market (number of firms one HH applies to)
        self.H = 2 # Number of trials in the credit market (Number of banks a firm selects randomly to search for credit)
        
        self.beta = 4 # MPC parameter that determines shape of decline of consumption propensity when consumption increases 
        self.theta = 8 # Duration of individual contract
        self.r_bar = 0.01 # Base interest rate set by central bank (exogenous in this model)
        self.delta = 0.2 # Dividend payments parameter (fraction of net profits firm have to pay to the shareholders)
        self.capital_requirement_coef = 0.1 # capital requirement coefficients uniform across banks (nu)
        
  
    def simulation(self, theta):

        """
        The simulation of the toy base model is run with the simulation method
        
        Input is a 1-d numpy array with the following model parameters:

        H_eta = maximum growth rate of prices (upper bound of price updates) 
        H_rho = maximum growth rate of quantities (upper bound of wage updates)
        H_phi = Maximum amount of banks' costs (upper bound of interest rate shock)
        h_xi = Maximum growth rate of wages (upper bound of random wage shock)
        """
        
        H_eta = theta[0] 
        H_rho = theta[1] 
        H_phi = theta[2] 
        h_xi = theta[3] 


        """
        Aggregate report variables: dimension = Time x MC (1 time series for each MC run)
        """
        unemp_rate = np.zeros((self.T,self.MC)) # unemployment rate U
        unemp_growth_rate = np.zeros((self.T,self.MC)) # unemployment growth rate U^tilde
        
        P_lvl = np.zeros((self.T,self.MC)) # price level P^lvl
        inflation = np.zeros((self.T,self.MC)) # price inflation pi

        wage_lvl = np.zeros((self.T,self.MC)) # wage level W^lvl
        real_wage_lvl = np.zeros((self.T,self.MC)) # real wage
        wage_inflation = np.zeros((self.T,self.MC)) # wage inflation (not in %) W^pi
        avg_prod = np.zeros((self.T,self.MC)) # average productivity of labor alpha^L
        product_to_real_wage = np.zeros((self.T,self.MC)) # productivity/real wage ratio Alpha^L/R
        
        production = np.zeros((self.T,self.MC)) # nominal gdp (quantities produced) Y
        output_growth_rate = np.zeros((self.T,self.MC)) # output growth rate 
    
        aver_savings = np.zeros((self.T,self.MC)) # average HH income S^lvl
        vac_rate = np.zeros((self.T,self.MC)) # vacancy rate approximated by number of job openings and the labour force at the beginning of a period vac^lvl
        bankrupt_firms_total = np.zeros((self.T,self.MC)) # number of bankrupt firms in current round
        
        print("")
        print('--------------------------------------')
        print("Simulating BAM model %s times" %self.MC)
        #for mc in tqdm(range(self.MC)):
        for mc in range(self.MC):

            # set new seed each MC run to ensure different random numbers are sampled each simulation
            np.random.seed(mc)

            if self.csv:
                # initialize csv file and add header
                header = ['t', 'Ld', 'L', 'u' ,'Sum C/sum(p)', 'Sum NW', '#bankrupt', 'Sum Profit', 'f_cr' , 'p_min', 'p_max', 'Sum Qp', 'sum HH Savings', 'Sum Qd', 
                        'Sum Qr', 'Sum Qs', 'sum(C)', 'wage lvl', 'E', 'Cr'] 
                with open('simulated_data.csv', 'w', encoding = 'UTF8') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

        
            #################################################################################################
            # INDIVIDUAL REPORT VARIABLES
            #################################################################################################

            """
            Initialize all individual report variables used during one simulation.
            Structure: a vector (usually either a list or a numpy array) with elements for each respective agent,
            which is either a single entry or also a list (s.t. each agent has individual list, yielding nested lists).
            Dimension: 1xNf, 1xNh, 1xNb 
            In order to save memory space, the vector are overwritten with each period t, s.t. individual report data 
            is not saved.  
            """

            # HOUSEHOLDS
            S = np.zeros(self.Nh) # savings
            C = np.zeros(self.Nh) # (actual) consumption
            C_d = np.zeros(self.Nh) # desired / demanded consumption
            MPC = np.zeros(self.Nh) # marginal propensity to consume (declines with wealth)

            w = np.zeros(self.Nh) # wage of each HH
            w_last_round = np.zeros(self.Nh) # wage of round before 
            div = np.zeros(self.Nh) # dividend payment to HH from profit of firm (new each round)
            
            is_employed = np.zeros(self.Nh, bool) # True if HH has job: e.g. array([False, False, False, False, False])
            d_employed = np.zeros(self.Nh) # counts the days a HH is employed
            prev_firm_id = np.zeros(self.Nh, int) # firm id HH was employed last

            expired = np.zeros(self.Nh, bool) # flag in case contract of a HH expired, it is set to True 
            goods_visited = np.zeros(self.Nh)# one list for each HH in which firm id's are stored that sold product to HH in round before
            
            # FIRMS
            Qd = np.zeros(self.Nf) # Desired qty production : Y^D_it = D_ie^e
            Qp = np.zeros(self.Nf) # Actual Qty produced: Y_it
            Qs = np.zeros(self.Nf) # Qty sold
            Qr = np.zeros(self.Nf) # Qty Remaining
            
            rho = np.zeros(self.Nf) # intiaize Qty update parameter
            eta = np.zeros(self.Nf) # Price update random number to be sampled later
            
            p = np.zeros(self.Nf) # price
            
            Ld = np.zeros(self.Nf) # Desired Labor to be employed
            Lhat = np.zeros(self.Nf) # Labor whose contracts are expiring
            L = np.zeros(self.Nf)  # actual Labor Employed
            Lo = np.zeros(self.Nf) # actual workforce (L of last round - amount of just expired contracts)
            vac = np.zeros(self.Nf) # Vacancy
            w_emp = [[] for _ in range(self.Nf)] # Ids of workers/household employed - each firm has a list with worker ids w^emp
            W = np.zeros(self.Nf) # aggregated Wage bills (wage payments to each Worker * employed workers)
            time_stamp_wage_adjustment = np.arange(3,self.T,4)
            Wp = np.zeros(self.Nf) # Wage payments of firm this round
           
            P = np.zeros(self.Nf) # net Profits
            NW = np.ones(self.Nf)  # initial Net worth
            Rev = np.zeros(self.Nf) # Revenues of a firm (gross profits)
            RE = np.zeros(self.Nf) # Retained Earnings (net profits - divididens)
            divs = np.zeros(self.Nf) # Dividend payments to the households

            B = np.zeros(self.Nf) # amount of desired credit (total)
            lev = np.zeros(self.Nf) # invdivudal leverage (debt ratio)
            Qp_last = np.zeros(self.Nf) # variable to save quantity produced in round before 
            bankrupt_flag = np.zeros(self.Nf) # set entry to 1 in case firm went bankrupt 

            # BANKS
            E = np.zeros(self.Nb) # equity base 
            phi = np.zeros(self.Nb)
            Cr = np.zeros(self.Nb) # amount of credit supplied / "Credit vacancies"
            Cr_rem = np.zeros(self.Nb) # credit remaining 
            
            Bad_debt = np.zeros(self.Nb)

            """
            Initialization:
            Fill or initialize some individual report variables with random numbers when t = 0
            """
            # HOUSEHOLDS
            S = np.around(np.random.uniform(low=1,high=1,size=self.Nh), decimals = 4) # initial (random) income of each HH  
            MPC = np.around(np.random.uniform(low=0.6,high=0.9,size=self.Nh), decimals = 4) # initial random marginal propensity to consume of each HH 

            # FIRMS
            NW = np.around(np.random.uniform(low=5 ,high=5, size=self.Nf), decimals = 4) # initial net worth of the firms 
            p = np.around(np.random.uniform(low=1.4, high=1.4,size=self.Nf), decimals = 4) # initial prices
            Ld_zero = 5 # 4.5 # s.t. in first round sum(vac) = 400 and therefore u = 1 - (L/Nh) = 0.2 
            Wp = np.around(np.random.uniform(low= 1,high=1,size=self.Nf), decimals = 4) # initial wages
            alpha = np.random.uniform(low=0.5, high=0.5, size = self.Nf) # Productivity alpha stays constant here, since no R&D in this version
            # Delli Gatti & Grazzini 2020 set alpha to 0.5 in their model (constant)
            w_min_zero = 0.95
            
            # BANKS
            E = np.random.uniform(low = 5, high = 5, size = self.Nb)
        
            """
            Simulation:
            """
            for t in range(self.T):
                #print("time period equals %s" %t)
                #print("")
               
                """ 
                1) 
                Firms decide on how much output to be produced and thus how much labor required (desired) for the production amount.
                Accordingly price or quantity (cannot be changed simultaneously) is updated along with firms expected demand 
                (adaptively based on firm's past experience).
                The firm's quantity remaining from last round and the deveation of the individual price the firm set last round compared to the
                average price of last round determines how quantity or prices are adjusted in this round.
                There is no storage or inventory, s.t. in case quantity remaining from last round (Qr > 0), it is not carried over.
                """

                #print("1) Firms adjust and determine their prices and desired quantities and set their labour demand")
                #print("")
                # Sample random quantity and price shocks for each firm, upper bound depending on respective parameter values (shocks change each period)
                eta = np.random.uniform(low=0,high=H_eta, size = self.Nf) # sample value for the (possible) price update (growth rate) for the current firm 
                rho = np.random.uniform(low=0,high=H_rho, size = self.Nf) # sample growth rate of quantity of firm i
                for i in range(self.Nf):
                    if t == 0: # set labor demand and price of each firm in t = 0 
                        Ld[i] = Ld_zero
                    # otherwise: each firm decides output and price depending on last round s.t. it can actually determine its labor demand for this round
                    else: # if t > 0, price and quantity demanded (Y^D_it = D_ie^e) are to be adjusted 
                        prev_avg_p = P_lvl[t-1,mc] # extract (current) average previous price 
                        p_d = p[i] - prev_avg_p  # price difference btw. individual firm price of last round and average price of last round
                        # decideProduction:
                        if p_d >= 0: # in case firm charged higher price than average price (price difference positive)
                            if Qr[i] > 0: # and firm had positive inventories, firm should reduce price to sell more and leave quantity unchanged
                                # a)
                                p[i] = np.around(max(p[i]*(1-eta[i]), prev_avg_p), decimals=4) # max of previous average price or previous firm price  * 1-eta (price reduction) 
                                #  P_lvl[t-4,mc] if t >= 3 else P_lvl[0,mc]
                                Qd[i] = np.around(Qp[i]) #if Qs[i] > 0 else Qs_last_round # use average of quantity demanded last round in case Qd were non positive last round
                            else: # firm sold all products, i.e. had negative inventories: firm should increase quantity since it expects higher demand
                                # d)
                                p[i] = np.around(p[i],decimals=2) # price remains
                                Qd[i] = np.around(Qp[i]*(1+rho[i]),decimals=4) #if Qs[i] > 0 else Qs_last_round
                        else: # if price difference negative, firm charged higher (individual) price than the average price 
                            if Qr[i] > 0: # if previous inventories of firm > 0 (i.e. did not sell all products): firm expects lower demand
                                # c)
                                p[i] = np.around(p[i],decimals=2)
                                Qd[i] = np.around(Qp[i]*(1-rho[i]),decimals=4) #if Qs[i] > 0 else Qs_last_round
                            else: # excess demand (zero or negative inventories): price is increased
                                # b)
                                p[i] = np.around(max(p[i]*(1+eta[i]), prev_avg_p ),decimals=4)
                                Qd[i] = np.around(Qp[i],decimals=4) #if Qs[i] > 0 else Qs_last_round
                        # setRequiredLabor: alpha is constant in the base version
                        Ld[i] = np.around(Qd[i] / alpha[i], decimals = 4) # labor demand of current round
                

                """ 
                2) 
                A fully decentralized labor market opens. Firms post vacancies with offered wages. 
                Workers approach subset of firms (randomly selected, size determined by M) acoording to the wage offer. 
                """
                
                #print("2) Labor Market opens")
                #print("")

                f_empl = [] # initialize list of Firms whose ids are saved in case they have open vacancies
                c_ids = np.arange(1, self.Nh + 1, 1, dtype=int) # array with consumer ids

                """
                Minimum wage is is revised upwards every 4 periods to catch up with inflation
                """
                if t in range(3):
                    w_min = w_min_zero
                elif t in time_stamp_wage_adjustment:
                    w_min = w_min *(1 + inflation[t-1,mc])

                """
                Firms determine Vacancies. 
                After 8 rounds the contract of a worker expires. In that case the worker will apply to the firm first she workerd at before. 
                """
                for i in range(self.Nf):

                    HH_employed = w_emp[i] # getEmployedHousehold: slice the list with worker id's (all workers) for firm i 
                    n = 0 # counter for the number of expired contracts of each firm i 
                    for j in (HH_employed): # j takes on HH id employed within current firm i 
                        if d_employed[j-1] > self.theta: # if contract expired (i.e. days employed is greater than 8) 
                            
                            n = n + 1 # add number of expired contracts

                            expired[j-1] = True # set entry to true when contract expired
                            is_employed[j-1] = False # HH will be searching for job in upcoming labor market
                            d_employed[j-1] = 0 # 0 days employed from now on
                            w_emp[i].remove(j) # delete the current HH id in the current firm array 
                            prev_firm_id[j-1] = i # add firm id to list of expired contract s.t. HH directly adds firm to her list for applications
                                                
                    Lhat[i] = n # number of just expired contracts
                    Lo[i] =  L[i] - Lhat[i]  # actual workforce
                    
                    # calcVacancies of firm i
                    vac[i] = np.around(max((Ld[i] - Lo[i]) ,0), decimals = 8)
                    if vac[i] > 0: 
                        f_empl.append(i+1) # if firm i has vacancies, then firm id i is saved to list 

                    """
                    Firms set their wages. The minimum wage is periodically revised upward every four time steps (quarters) in order to catch up with inflation.
                    In the first 4 rounds (up to t=4) the initial price level is used as the min. wage. Then the Price level of the period before is used respectively 
                    for every four periods. 
                    E.g. in t = 3 (fourth period), the wage level of t = 2 (third period) is used for 4 consective periods. In case t = 7, then the wage level of t=6
                    is used for 4 periods, and so on.
                    """
                    xi = np.random.uniform(low=0,high=h_xi) # sample xi (uniformly distributed random wage shock) for firm i
 
                    if vac[i] > 0: # firm increases wage in case it has currently open vacancies:
                            Wp[i] = np.around(max(w_min, Wp[i]*(1+xi)),decimals=4) # Wage of firm i is set: either minimum wage or wage of period before revised upwards 
                            #is_hiring[i] = True 
                    else:
                        Wp[i] = np.around(max(w_min, Wp[i]),decimals=2) # wage is either min. wage or wage of period before 
                        #is_hiring[i] = False        

                """
                Search and Match: After firms made conctractual terms (wages and vacancies) public, search and match on the labour market begins.  
                Unemployed Households visit M random firms and apply only to the firm out of their (randomly) chosen subsample that offers the highest wage. 
                The firm hires the worker in case it has to still open vacancies. Since contracts are closed sequentially, there are two types of coordination 
                failures that can occur: 
                First the number of unemployed workers searching for a job might be unqual to vacancies. Second firms with high wages experience excess of demand, 
                other firms with low wages may not be able to fill all vacancies, because HH's don't know whether the vacancy at 'high-wage firms' are already filled. 
                Because they can apply only to one firm per round (firm with highest) wage, they may remain unemployment although they are willing to work.
                """ 
                if len(f_empl) > 0: # if there are firms that have open vacancies, search and match begins
                    vac_to_update = vac.copy() # initialize new numpy array in order to update vacancies of firms during the hiring process

                    """
                    Unemployed Households visit M random firms and try to sign a contract with the firm offering the highest wage out of the chosen subsample.
                    In case the contract just expired and the same firm is also hiring, HH tries to apply to firm she worked before first. 
                    If the worker was fired or lost her job due to bankruptcy, she visits M firms (instead of M-1 as in the former case), but it can happen 
                    that worker also applies again at the firm she worked at before.
                    A worker can be fired (later) only in case the internal and external funds are not enough to pay for the desired wage bill.
                    """
                    c_ids =  [j for j in c_ids if is_employed[int(j-1)] == False]  # get Household id's who are unemployed 
                    hired = 0 # initialize counter for hired HH's

                    np.random.shuffle(c_ids) # randomly selected household starts to apply (sequentially)
                    for j in c_ids:
                        
                        f_empl_current = f_empl.copy() # copy the list with firm ids which are employing in this round
                        appl_firm = [] # list with firm ids the HH j will apply to 
                        prev_emp = int(prev_firm_id[j-1]) # getPrevEmployer, i.e. the id of firm HH j were employed before (prev_emp is 0 as long as HH never fired or contract expired)
                        M = self.M # number of firms HH visits
                        if expired[j-1] == True and prev_emp in f_empl:
                            appl_firm.append(prev_emp) # if contract expired, HH directly chooses prev_emp
                            M = self.M - 1 
                            f_empl_current.remove(prev_emp) # remove previous employer from firms that emply s.t. firm cannot be chosen twice from the same HH
                        
                        # HH chooses firms and extracts their wages
                        if len(f_empl_current) > M:
                            chosen_firms = list(np.random.choice(f_empl_current, M, replace = False)) # random chosen M firms the HH considers
                            appl_firm.extend(chosen_firms) # add firm ids to the list where HH applies
                        else:
                            chosen_firms = f_empl_current
                            appl_firm.extend(f_empl_current)

                        wages_chosen_firms = [] # initialize list with the wages of the chosen firms 
                        for ii in appl_firm: 
                                wages_chosen_firms.append(np.around(Wp[ii-1], decimals = 4)) # extract the wages of the firms where HH applied 

                        """
                        HH tries to sign conctract with the firm that offers highest wage, in case the firm still has open vacancies, 
                        otherwise she tries the firm with the 2nd highest wage, and so on until either HH is either hired or no more more
                        firms to apply to
                        """
                        while len(appl_firm) > 0:
                            
                            w_max = max(wages_chosen_firms)
                            f_max_id = appl_firm[wages_chosen_firms.index(w_max)] # id of the firm that offered highest wage

                            current_vac = vac_to_update[f_max_id - 1] # extract current number of vacancies of firm that offers the highest wage
                            
                            if current_vac > 0:
                                vac_to_update[f_max_id - 1] = current_vac - 1 # update vacancies of firm that just hired HH j
                                
                                # update labor market variables
                                is_employed[j-1] = True # updateEmploymentStatus
                                w[j-1] = np.around(w_max,decimals=2) # save wage HH l is earning
                                hired = hired + 1 # counter for #HH increases by 1
                                w_emp[f_max_id - 1].append(j) # employHousehold: save HH id to list of firm that is employing

                                break

                            else:
                                wages_chosen_firms.remove(w_max)
                                appl_firm.remove(f_max_id)

                        # labor market closes in case no more open vacancies (i.e. no more firms employing) 
                        #if sum(vac) == 0:
                        #    break"              
    
                #else:
                    #print("No Firm with open vacancies")
                    #print("")

                # current labour employed
                for f in range(self.Nf):

                    L[f] = len(w_emp[f])
                
                #print("Labor Market CLOSED!!!! with %d NEW household hired!!" %(hired))
                #print("Labor Market CLOSED!!!! with %d HH's currently employed (L)" %(np.sum(L)))
                

                """
                3)
                Credit market: If there is a financing gap, i.e. the desired wage bill W is larger than the current Net worth, firms go to credit market. 
                They randomly choose H number of  banks and 'apply' for loans, starting with bank offering the lowest interest rate. 
                Banks sort borrowers application according to financial conditions (NW) and satisfy credit demand until exhaustion of credit supply.
                Interest rate is calc acc. to markup on an baseline rate (set exogenous by 'CB'). 
                After credit market closed, if firms are still short in net worth, they fire workers.
                """
                # Initialize list to store values again in each round
                banks = [[] for _ in range(self.Nf)] # list of banks from where the firm got the loans (nested list for each firm)
                Bi = [[] for _ in range(self.Nf)] # Amount of credit taken by banks
                r_f = [[] for _ in range(self.Nf)] # rate of Interest on credit by banks for which match occured (only self.r bei ihm) 
                
                r = [[] for _ in range(self.Nb)] # list of interest rate bank determines for each firm 
                
                f_cr = [] # initialize list with id's of firms which need credit in the following
                hh_layed_off = [] # initialize list to store HH id's that might be fired in case total credit supply not sufficient
                
                """
                Firms compute total wage bills and compute their credit demand as well as their individual leverage they have to report to the banks 
                in order to receive a loan. 
                """
                for f in range(self.Nf):
                   
                    W[f] = np.around(Wp[f] * L[f], decimals=4)  # (actual) total wage bills = wages * labour employed

                    if W[f] > NW[f]: # if wage bills f has to pay is larger than the net worth 
                        B[f] = max(np.around(W[f] - NW[f], decimals=4), 0) # amount of credit needed
                        f_cr.append(f + 1) # save id of firm that needs credit

                        # leverage (debt ratio)
                        lev[f] = np.around(B[f] / NW[f], decimals=4)
                
                """
                Credit market opens if there are any firms at all that need credit. 
                """
                if len(f_cr) > 0:
                    #print("3) Credit market opens")
                    #print("")
                    
                    """
                    Banks decides on its total amount of credit and computes the interest rates. 
                    The interest rate is composed of the exogenously given policy rate (by CB for example),
                    a shock representing idiosyncratic operating costs and the current financial fragility of each firm.
                    Here square root of the individual firm leverage is used to evaluate the soundness. 
                    """
                    b_cred = []
                    for b in range(self.Nb):
                        
                        # total credit supply of each bank
                        Cr[b] = np.around(E[b] / self.capital_requirement_coef, decimals=4) # amount of credit that can be supplied is an multiple of the equity base: E

                        # interest rates for each firm are computed (also for firms that don't apply for credit)
                        phi[b] = np.random.uniform(low=0,high=H_phi) # idiosyncratic interest rate shock
                        for f in range(self.Nf):
                            current_r = self.r_bar*(1 + (phi[b] * np.sqrt(lev[f]))) # current interest rate 
                            r[b].append(current_r) # append r to list with interest rates for each firm 

                        # firms set their remaining credit for this round
                        Cr_rem[b] = Cr[b]

                        b_cred.append(b + 1) # attach bank id to list if bank provides credit

                    """
                    Search and Match (Credit):  
                    """
                    np.random.shuffle(f_cr) # randomly selected firms starts to apply (sequentially)
                    for f in f_cr:

                        appl_bank = [] # list with bank ids the firms f will apply to 
                        chosen_banks = list(np.random.choice(b_cred, self.H, replace = False)) # firm randomly chooses H banks
                        appl_bank.extend(chosen_banks) # add chosen banks to list of banks firm will apply for credit

                        # extract the specific interest rate for fim f by each bank
                        interest_chosen_banks = [] # initialize list with the rates of the chosen banks 
                        for ii in appl_bank: 
                                r_current = r[ii-1][f-1] # interest rate of the current firm (first slice list of all r of bank ii, then r for firm actually applying)
                                interest_chosen_banks.append(r_current) # extract the interest rateS of the chosen banks
                        
                        # firms receive credit from chosen banks, starting with the bank offering the lowest rate
                        while len(appl_bank) > 0:
                            r_min = min(interest_chosen_banks)
                            b_min_id = appl_bank[interest_chosen_banks.index(r_min)] # id of the bank that offered lowest rate

                            Credit_remaining = Cr_rem[b_min_id -1] # extract remaining credit supply of currently chosen bank
                            B_current = B[f - 1] # oustanding credit demand of current firm f

                            # firm either receives entire credit demand at once or goes to next bank, as long as banks supply credit
                            if Credit_remaining > 0:
                                if Credit_remaining >= B_current:
                                    cr_supplied = B_current # supplied credit by bank b_min_id is the entire demand
                                                                        
                                    # update information on firm side
                                    B[f-1] = B_current - cr_supplied # update remaining credit demand
                                    banks[f-1].append(b_min_id) # append bank id that supplied lowest credit 
                                    Bi[f-1].append(cr_supplied) # append received credit
                                    r_f[f-1].append(r_min) # interest rate charged by bank b_min_id

                                    # update information on bank side
                                    Cr_rem[b_min_id -1] = np.around(Credit_remaining - cr_supplied, decimals=4) # credit remaining of bank b
                                    #firms_loaned[b_min_id -1].append(f) # save firm id the bank with lowest interest loaned money to 
                                    #Cr_s[b_min_id-1].append(cr_supplied) # save credit amount given out to firm by bank (id) with lowest interest rate 
                                    #r_s[b_min_id-1].append(r_min) # save interest rate charged to firm by bank with lowest interest rate

                                    break

                                else:
                                    cr_supplied = Credit_remaining # supplied credit by bank b_min_id is the entire remaining credit remaining
                                    B_current = np.around(B_current - cr_supplied, decimals=4) # remainig credit demand by firm 
                                    

                                    # update information on firm side
                                    B[f-1] = B_current # update remaining credit demand
                                    banks[f-1].append(b_min_id) # append bank id that supplied lowest credit 
                                    Bi[f-1].append(cr_supplied) # append received credit
                                    r_f[f-1].append(r_min) # interest rate charged by bank b_min_id

                                    # update information on bank side
                                    Cr_rem[b_min_id -1] = np.around(Credit_remaining - cr_supplied, decimals=4) # credit remaining of bank b
                                    #firms_loaned[b_min_id -1].append(f) # save firm id the bank with lowest interest loaned money to 
                                    #Cr_s[b_min_id-1].append(cr_supplied) # save credit amount given out to firm by bank (id) with lowest interest rate 
                                    #r_s[b_min_id-1].append(r_min) # save interest rate charged to firm by bank with lowest interest rate

                                    appl_bank.remove(b_min_id) # delete bank id from list
                                    interest_chosen_banks.remove(r_min) # delete interest rate of bank that supplied some B
                                    
                            else:
                                appl_bank.remove(b_min_id) # delete bank id from list
                                interest_chosen_banks.remove(r_min) # delete interest rate of bank that supplied some B
                    
                        """
                        Firm checks whether external finances received are enough now to cover the wage bills (W).
                        If not, firm starts to fire random selected HH's until internal and external resources are enough to cover wage bills.
                        """
                        capital = np.around(NW[f-1] + sum(Bi[f-1]), decimals = 4)
                        wage_bill = W[f-1] # extract wage bill 
                        
                        if wage_bill > capital: # if wage bills f has to pay is still larger than the net worth + received credit
                            
                            # firm updates its wage bill and fires randomly chosen HH's
                            diff = np.around(W[f-1] - capital, decimals=4) # firm computes amount not enough to cover wage bill
                            
                            # rounded difference btw. wage bill and capital (int. and ext.) to next higher integer is number of HH's fired
                            L_fire = math.ceil(diff / Wp[f-1]) 

                            # W[f-1] = wage_bill - L_fire*Wp[f-1]# update wage bill by subtracting individual wages

                            h_lo = np.random.choice(w_emp[f-1], L_fire, replace = False) # choose random worker(s) which are fired
                            for h in h_lo:

                                hh_layed_off.append(h)

                                # update employments on firms side
                                w_emp[f-1].remove(h) # update list with HH ids working at firm f
                                            
                                # update employment status on HH's side
                                prev_firm_id[h-1] = 0 # set previous firm id to 0, since HH no more preferential attachment, since it was sacked
                                #firm_id[h-1] = 0   # setEmployedFirmId: HH not employed anymore, therefore id is 0
                                is_employed[h-1] = False # set employment status to false      
                                w[h-1] = 0 # no wage payments in this round iff layed off
                                #firm_id[j-1] = 0 # HH not employed anymore             
                                
                            L[f-1] = len(w_emp[f-1]) # update labour employed at firm f
                            W[f-1] = np.around(Wp[f-1] * L[f-1], decimals=4)  # update (total wage bills = wages * labour employed

                #else: 
                    #print("No credit requirement in the economy")
                    #print("")            
                
                #print("Credit Market closed. %s HH's were layed off." %len(hh_layed_off))
                #print("")    


                """
                4)
                Production takes one unit of time regardless of the scale of prod/firms (constant return to scale, labor as the only input).  
                Since Firms either received enough credit or layed some HH's off if credit received was not enough, wage payments can all be paid,
                using internal (net worth) and external (credit) funds.
                The actual Wage payments to the workers are done in 7) after credit is paid back. Since the wage payments are transacted after the goods market closes 
                and any loans granted on the preceding credit market have to be paid back at the end of $t$, 
                firms are forced to make enough revenue (gross profit) to be able to cover Wp, Bi 
                and potential dividend payments to the households (in case the company remains with positive net worth in the end).
                Productivity alpha is 0.5 for each firm and remains constant throughout the entire simulation (hence no aggregate Output growth in the long run).
                In the BAM_plus version there is also technical innovation included s.t. alpha increases over time (through R&D in 6). 
                """
                #print("4) Production")
                #print("")            

                f_produced = [] # intialize list to store firm id if firm produced this round 
                
                for f in range(self.Nf):

                    #NW[f] = NW[f] - W[f] # wage payment are subtracted from the net worth 
                    
                    if t > 0:
                        Qp_last[f] = Qp[f] # save the quantity produced in the last round 

                    # Firms producing their products: 
                    Qp[f] = np.around(alpha[f] * L[f], decimals = 4) # productivity * labor employed = quantity produced in this round
                    
                    # append firm id if firm actually produced something
                    if Qp[f] > 0:
                        f_produced.append(f+1) # append firm id if firm produced anything 
                    
                    Qs[f] = 0 # resetQtySold
                    Qr[f] = Qp[f] # Initialize the remaining quantity by subtracting the quantity sold (currently 0, since goods market did not open yet)  

                #print("Wage payments and Production done!!!")
                #print("")

                """
                5) 
                After production is completed, the goods market opens. Again, as in the labour- and credit market, search and match algorithm applies:
                Firm post their offer price. Consumers contact subset of randomly chosen firm acc to price and satisfy their demand by using the MPC fraction
                of their income of last round (t-1) and their savings. The marginal propensity to consume is determined right before. 
                Goods with excess supply can't be stored in an inventory and they are disposed with no cost (no warehouse). 
                Savings of HH's is measured in goods of the firm charging the lowest price in the respective subset of randomly chosen firms Z. 
                """ 
                savg = np.mean(S) # average savings level of last round (for MPC)
                c_list = [1+f for f in range(self.Nh)] # initialize list of consumers
                np.random.shuffle(c_list) # randomly selected household starts to apply (sequentially)
                
                for c in c_list:
                    
                    """
                    For t = 0, the wage of this round instead of the wage of last round and the initially drawn MPC are used.
                    Entire consumption of each HH is set to 0.
                    """
                    C[c-1] = 0 # initialze overall consumption

                    # marginal propensity to consume c
                    MPC[c-1] = np.around(1/(1 + (np.tanh((S[c-1])/savg))**self.beta),decimals=4) if t > 0 else MPC[c-1] 
                    
                    # determine spending of HH in this round
                    w_fraction = np.around(w_last_round[c-1]*MPC[c-1], decimals=8) if t > 0 else np.around(w[c-1]*MPC[c-1], decimals=8) 
                    S_fraction = np.around(S[c-1]*MPC[c-1], decimals=8) 
                    spending = np.around(w_fraction + S_fraction , decimals = 8)
                    C_d[c-1] = spending # desired consumption 

                    # update current savings 
                    S[c-1] = np.around(S[c-1] - S_fraction, decimals=8)

                    prev_firm = int(goods_visited[c-1]) # id of largest firm with lowest price HH visited in last round
                    f_produced_current = f_produced.copy() # firm ids that actually produced
                    
                    appl_firms = [] # list with firm ids the HH c will apply to 

                    # compute number of firms HH c will visit this round
                    if bankrupt_flag[prev_firm - 1] == 0 and prev_firm in f_produced and t > 0:
                        Z = self.Z - 1  # Z reduces by 1 due to preferential attachmend in case firm from last round did not went bankrupt 
                        appl_firms.append(prev_firm)
                        f_produced_current.remove(prev_firm) # avoid HH sampling the same firm twice 
                    else:
                        Z = self.Z
                    
                    if len(f_produced_current) > Z:
                        random_firms = list(np.random.choice(f_produced_current, Z, replace = False)) # HH randomly chooses Z firms
                        appl_firms.extend(random_firms) # add chosen banks to list of banks firm will apply for credit
                    else:
                        random_firms = f_produced_current
                        appl_firms.extend(random_firms)


                    # extract the specific prices of each firm
                    prices_chosen_firms = []  # initialize list with the prices of the chosen firms 
                    for ii in appl_firms: 
                            p_current = p[ii-1] # price of the current firm (similar: first slice list of all r of bank ii, then r for firm actually applying)
                            prices_chosen_firms.append(p_current) # extract price (interest rates of the chosen banks)

                    if len(prices_chosen_firms) > 0:
                        goods_visited[c-1] = int(appl_firms[prices_chosen_firms.index(min(prices_chosen_firms))]) # save firm id out of firms she visits with lowest price
                    #else:
                    #    goods_visited[c-1] = appl_firms

                    # HH start to pay firms, starting with lowest price
                    while len(appl_firms) > 0:
                        p_min = min(prices_chosen_firms) # minimum price
                        f_min_id = appl_firms[prices_chosen_firms.index(p_min)] # id of the firm that offers the lowest price

                        consumption_remaining = C_d[c-1] / p_min # extract remaining demand of HH c relative to price of current firm
                        Y_current = Qr[f_min_id - 1] # remaining quantity of chosen firm
                        #Y_current_p = Y_current * p_min # quantity in terms of price of chosen firm

                        # HH starts to consume sequentially, until either chosen firms out of stock or entire demand satisfied
                        if Y_current > 0:
                            if Y_current >= consumption_remaining:
                                consumption = consumption_remaining # entire demand is satisfied
                                consumption_p = consumption*p_min
                                                                    
                                # update information on HH side
                                C_d[c-1] = C_d[c-1] - consumption_p # updated remaining consumption is 0
                                C[c-1] = C[c-1] + consumption_p # new consumption level in terms of 'overall prices'

                                # update information on firms side
                                Qs[f_min_id -1] = np.around(Qs[f_min_id -1] + consumption , decimals=4) # actual quantity sold 
                                Qr[f_min_id- 1] = np.round(Qr[f_min_id-1] - consumption, decimals = 4)

                                appl_firms.remove(f_min_id) # delete firm id from list
                                prices_chosen_firms.remove(p_min) # delete current min price

                                break

                            else:
                                consumption = Y_current # demand is the entire remaining supply of firm 
                                consumption_p = consumption*p_min # consumption in terms of overall price level 

                                # update information on HH side
                                C_d[c-1] = np.around(C_d[c-1] - consumption_p, decimals=4) # updated remaining consumption in terms of 'overall price level'
                                C[c-1] = np.around(C[c-1] + consumption_p, decimals= 4) # new consumption level in terms of 'overall prices'

                                # update information on firms side
                                Qs[f_min_id -1] = Qs[f_min_id -1] + consumption # actual quantity sold 
                                Qr[f_min_id- 1] = Qr[f_min_id-1] - consumption # quantity remaining
                                
                                appl_firms.remove(f_min_id) # delete firm id from list
                                prices_chosen_firms.remove(p_min) # delete current min price
                                
                        else:
                            appl_firms.remove(f_min_id) # delete firm id from list
                            prices_chosen_firms.remove(p_min) # delete current min price

                    """
                    Consumers saves wage of last round for the consumption in the next round and updates her savings in terms of produced goods (Y) of firm with 
                    lowest price out of her subsample.
                    """
                    w_last_round[c-1] = w[c-1] # wage of this round becomes wage of last round
                    C_d[c-1] = np.around(C_d[c-1], decimals=8) # round remaining desired consumption 
                    S[c-1] = np.around(S[c-1] + C_d[c-1] + (w[c-1] - w_fraction), decimals=8) # update savings with desired consumption that was not satisfied

                #print("Consumption Goods market closed!")           
                #print("")

                """
                6) 
                Firms collect revenues and calc their gross profits. 
                If gross profits are high enough, they pay principal and interest to bank. Subtracting the debt commitments from gross profits yields 
                the net profits.
                Positive net profits are then used to pay dividends to owners of the firm (here no investments to increase productivity in next round). It
                is assumed that each HH has the same constant share in each firm. 
                In the BAM_plus version, R&D is carried out before the dividend payments in order to increase firm's productivity. 
                If gross profits are not high enough, firms record an oustanding debt amount and try to pay it back by using their net worth (7)

                7) 
                Firms compute Retained Earnings, i.e. their Earnings after interest payments and dividend payments (zero investment here).
                They are added to the current net worth of each firm which are carried forward to the next period. 
                HH's update their income after the dividend payments and Banks update their equity after principal and interest payments (if any).
                If a firm could not pay back entire credit with its revenue, the firm pays back as much as she can and (hence zero profit) and then 
                uses its NW (if any) to pay back the rest of the outstanding amount.
                If the current (available) net worth is not sufficient to cover for the remaining oustanding amount(s), 
                the firm will exit the market in 8).
                The bank(s) which have open accounts (i.e. did not receive entire credit and interest payments) record non performing loan(s) or bad debt,
                ??? which amounts to the share of credit firm had at bank i times its remaining oustanding amount (not net-worth). 
                The banks again receive the open payments sequentially starting with the one bank firm received credit from first (in case firm had credit 
                from more than one bank).
                """
                #print("Firms calculating Revenues, net profits (settling debt and paying dividends) and retained earnings")
                #print("")
                for f in range(self.Nf):
                    
                    Rev[f] = np.around(Qs[f]*p[f], decimals = 4)

                    # finally firms subtract entire wage payments
                    # NW[f] = np.around(NW[f] - W[f], decimals=8)

                    """
                    Firms paying back principal and interest from the revenue of this round, in case they went to the credit market this round (and if firm made 
                    enough gross Profits).
                    If firm made not enough revenue (gross profit) this round, the rest of the oustanding amount will be subtracted from its net worth in 7). 
                    RE are set to 0 and remaining outstanding amounts (negative profits) are saved.
                    """
                    banks_f = banks[f].copy() # getBanks: id of bank(s) that supplied credit to firm f
                    credit = Bi[f].copy() # getCredits: amount of credit firm f took from each bank 
                    rates = r_f[f].copy() # getRates: interest rate(s) charged to firm i 

                    if len(banks_f) > 0:
                        loans_to_be_paid = np.around(np.sum( np.array(credit)*(1 + np.array(rates))),decimals=4)  # compute current total amount of all loans

                        if Rev[f] >= loans_to_be_paid:
                            
                            # firm pays back entire principal and interest to each bank 
                            for i in range(len(banks_f)):
                                
                                E[banks_f[i] -1] = np.around(E[banks_f[i] -1] + (credit[i]*rates[i]), decimals=4) 

                            P[f] = np.around(Rev[f] - loans_to_be_paid ,decimals=8) # net profits
                            divs[f] = np.around(P[f]*self.delta , decimals = 8) # dividends are profits * delta (share), if firm f has positive profits
                            NW[f] = np.around(NW[f] + (1-self.delta)*P[f], decimals = 8) # udpate firms NW
                            
                        else:

                            """
                            If firm cannot pay back all loans, i.e her gross revenue is smaller than total amount of loans firm has to pay back: 
                            The firm starts to pay back using her entire revenue and parts of its net worth, starting with the first bank she got credit from. 
                            """
                            amount_left = Rev[f] # initialize the entire amount of revenue firm has left to pay back firms
                            
                            banks_f_current = banks_f.copy() # initialize new bank list

                            for i in range(len(banks_f)):

                                current_loan_to_be_paid = np.around((credit[i]*(rates[i]+1)), decimals = 8) # amount firm has to pay back to current bank i

                                if amount_left >= current_loan_to_be_paid: 
                                    
                                    # If amount left to pay back loan of current bank i greater than loan & interest charged: firm pays back entire amount to current bank i
                                    E[banks_f[i] -1] = np.around(E[banks_f[i] -1] + (credit[i]*rates[i]), decimals=8) 
                                    
                                    amount_left = np.around(amount_left - current_loan_to_be_paid, decimals=4) # update new amount left and check whether next bank can be paid back , decimals=2) 
                                    banks_f_current.remove(banks_f[i]) # remove bank id from list 
                                
                                else:
                                    
                                    # if amount left not enough to pay back entire credit to current bank, then the payment is just the amount left and payment loop stops
                                    E[banks_f[i] -1] = np.around(E[banks_f[i] -1] + (rates[i]*amount_left), decimals=8) # update equtity with the rest of the payment

                                    oustanding_amount = current_loan_to_be_paid - amount_left

                                    # firm checks whether it can use its current NW to pay back rest of oustanding amount, otherwise bank reports bad debt
                                    if NW[f] > oustanding_amount:
                                        
                                        E[banks_f[i] -1] = np.around(E[banks_f[i] -1] + (rates[i]*oustanding_amount), decimals=8) # update equtity with the rest of the payment
                                        banks_f_current.remove(banks_f[i])

                                    else:
                                        
                                        # If NW is not enough, then bank registers bad debt and firm goes bankrupt
                                        share = np.around(credit[i] / sum(credit), decimals=8) # share of credit amount bank had 
                                        Bad_debt = np.around(share * NW[f], decimals=8)

                                        E[banks_f[i] -1] = np.around(E[banks_f[i] -1] - Bad_debt, decimals=8) # update equtity
                                        
                                        banks_f_current.remove(banks_f[i]) # remove bank that registered bad debt from list
                                        
                                        # iterate through rest of the banks and s.t. more banks registering bad debt
                                        while len(banks_f_current) > 0:
                                            i = i + 1

                                            share = np.around(credit[i] / sum(credit)) # share of credit amount bank had 
                                            Bad_debt = share * NW[f]

                                            E[banks_f[i] -1] = np.around(E[banks_f[i] -1] - Bad_debt, decimals=8) # update equtity

                                            banks_f_current.remove(banks_f[i]) # remove bank that registered bad debt from list

                                    # firm update profits and net worth and exit market in 8)
                                    P[f] = np.around(Rev[f] - amount_left ,decimals=8) # net profit
                                    NW[f] = 0 # update net worth

                                    break

                    else:

                        P[f] = np.around(Rev[f], decimals = 8) # gross profits are net profits in case firm did not take any credit
                        divs[f] = np.around(P[f]*self.delta , decimals = 8) # dividends are profits * delta (share), if firm f has positive profits
                        NW[f] = np.around(NW[f] + (1-self.delta)*P[f], decimals = 8) # udpate firms NW
                    
                    # add retained profits to the NW, then subtract then finally subtract the wage payments
                    NW[f] = NW[f] + P[f] - W[f]

                    
  
                """
                HH's receive dividend payments (assumed that each agent has an equal share in all firms) and update their days of employment.
                """
                # each HH receives an equal share
                for c in range(self.Nh):

                    div[c] = div[c] + np.around(divs[f]/self.Nh ,decimals=4)
                    S[c] = S[c] + div[c] # dividends are added to savings

                    # update employment count
                    if is_employed[c] == True: # getEmploymentStatus
                        d_employed[c] = d_employed[c] + 1 # incrementEdays

                #print("Firms calculated Revenues, net profits (settling debt and paying dividends) and retained earnings. HH's updated employment days.")
                #print("")

                """ 
                Aggregate report variables are computed.
                Each aggregation is saved into a corresponding report matrix (2-d numpy array) with number of rows = T and each column being the current mc round.
                """ 

                """
                Inflation: determine aggregate price level and compute inflation:
                """
                P_lvl[t,mc] = np.mean(p) # average price level 
                inflation[t,mc] = ( P_lvl[t,mc] - P_lvl[t-1,mc] ) / P_lvl[t-1,mc] if t!=0 else 0  

                """
                Output
                """
                production[t,mc] = np.sum(Qp) # nominal GDP = sum of quantity produced by each firm
                output_growth_rate[t,mc] = ( production[t,mc] - production[t-1,mc] ) / production[t-1,mc] if t!=0 else 0

                """
                unemployment rate
                """
                unemp_rate[t,mc] = 1 - (np.sum(L)/self.Nh)
                unemp_growth_rate[t,mc] =  (unemp_rate[t,mc] - unemp_rate[t-1,mc] ) / unemp_rate[t-1,mc] if t!=0 else 0

                """
                Wages
                """
                wage_lvl[t,mc] = np.sum(W)/np.sum(L) # nominal wage level: sum of wage paid * number of employees relative to number of employees
                wage_inflation[t,mc] = (wage_lvl[t,mc]- wage_lvl[t-1,mc]) / np.sum(wage_lvl[t-1,mc]) if t!=0 else 0
                real_wage_lvl[t,mc] = (1-inflation[t,mc])*wage_lvl[t,mc] if t!= 0 else 0 # (1-inflation rate)* wage = real income Wr

                """
                Productivity / real wage ratio
                """
                avg_prod[t,mc] = np.sum(alpha*L)/np.sum(L) # average productivity (only considering workers)
                product_to_real_wage[t,mc] = avg_prod[t,mc] / real_wage_lvl[t,mc] if t!=0 else 0 # productivity relative to real wage lvl 

                """
                Vacancies
                """
                vac_rate[t,mc] = np.round(np.sum(vac) / np.sum(L), decimals = 4) # vacancie rate is number of job openings (i.e. vacancies) relative to the labor force (labor employed currently)

                """
                Saving and consumption
                """
                aver_savings[t,mc] = np.mean(S) # averge HH savings


                """
                8) 
                Firms with positive net worth/equity survive, but otherwise firms/banks with negative (or zero) net worth/equity go bankrupt and are replaced by.
                new firms/banks which enter the market of size smaller than average size of those who got bankrupt (by using a truncated mean).
                """
                # Replacement values, only using values of incumbent firms
                NW_incumbent = [i for i in NW if i > 0]
                Qp_replacement = np.round(stats.trim_mean(Qp, 0.1), decimals = 4) # average produced quantity
                Qr_replacement = np.round(stats.trim_mean(Qr, 0.1), decimals = 4) # average remaining quantity
                p_replacement =  np.round(stats.trim_mean(p, 0.1), decimals = 4) # average price

                E_replacement = np.round(stats.trim_mean(E, 0.1), decimals = 4) # average Equity of new bank
                NW_replacement = np.round(stats.trim_mean(NW_incumbent, 0.1), decimals = 4) # averaged trimmed net worth of current round (used for replacements later)

                L_report= sum(L)
                
                # 1) Firms
                bankrupt_firms = sum(1 for i in NW if i < 0) # count the bankrupt firms and report
                #print("%s Firms went bankrupt" %bankrupt_firms)
                
                bankrupt_firms_total[t,mc] = bankrupt_firms # save amount of bankrupt firms this round 

                # checkForBankrupcy: Firm
                for f in range(self.Nf):
                    
                    if NW[f] <= 0:

                        h_emp = w_emp[f].copy() # get HH's working at firm f
                        bankrupt_flag[f] = 1 # set flag to 1 in case firm went bankrupt

                        # update workers
                        for i in h_emp:

                            w_emp[f].remove(i) # delete worker form employment list of bankrupt firm
                            prev_firm_id[i-1] = 0 # no more preferential attachment, since HH went bankrupt
                            is_employed[i-1] = False # updateEmploymentStatus
                            # firm_id[i-1] = 0 # setEmployedFirmId (HH no contract anymore => hence firm number that employed him set to 0)

                            w[i-1] = 0 # HH receives no more wage
                            d_employed[i-1] = 0 
                            #firm_went_bankrupt[i-1] = 1 # OWN: set flag s.t. worker uses M instead of M - 1 when searching for job again 
                        
                        """
                        New firm is entering: Resetting individual data by using the truncated mean at 10%. Hence lower and upper 5% are not included,
                        when the averages are computed for the new individual report values.
                        """
                        Qp[f] = Qp_replacement
                        Qr[f] = Qr_replacement
                        p[f] =  p_replacement
                        NW[f] = NW_replacement

                        L[f] =  len(w_emp[f]) # new firm enters with 0 labor employed
                        w_emp[f] = [] # workers employed is empty list 
                       
                        # alpha[f] = np.random.uniform(low=0.5,high=0.5) # draw new productivity btw. 5 and 6 
                            

                # 2) Banks
                bankrupt_banks = sum(1 for i in E if i < 0)
                #print("%s Banks went bankrupt" %bankrupt_banks)

                for b in range(self.Nb):

                    if E[b] <= 0:

                        E[b] = E_replacement

                if self.csv:
                    """
                    Open csv file created and add data of the current round
                    """
                    add_row = [t, sum(Ld) / self.Nf, L_report, unemp_rate[t,mc] ,sum(C)/np.mean(p), sum(NW), bankrupt_firms, sum(P), len(f_cr) ,min(p), 
                            max(p), sum(Qp), sum(S), sum(Qd), sum(Qr), sum(Qs), sum(C), wage_lvl[t,mc], sum(E), sum(Cr)]
                    with open('simulated_data.csv', 'a', encoding = 'UTF8') as f:
                        writer = csv.writer(f)
                        writer.writerow(add_row)

                # firm size distribution of t = 500,600,700,800,900,T
                if self.plots:
                    if t in [self.T-499,self.T-399,self.T-299,self.T-199,self.T-99,self.T-1]:
                        
                        plt.clf()
                        plt.hist(L)
                        plt.xlabel("Firm Size distribution by labor L")
                        plt.savefig("plots/cut/firm_size/size_disribution_mc%s_t=%s.png" %(mc,t))
                
                # firm size distribution of t = 500,600,700,800,900,T
                if self.empirical:
                    if t in [self.T-499,self.T-399,self.T-299,self.T-199,self.T-99,self.T-1]:
                        
                        plt.clf()
                        plt.hist(L)
                        plt.xlabel("Firm Size distribution by labor L")
                        plt.savefig(self.path_empirical + "cut/firm_size/size_disribution_mc%s_t=%s.png" %(mc,t))

            """
            Plotting main aggregate report variables if number of MC replications small enough and simulation ran through:
            The plots are saved in folder: 
            """
            # if growth prameter != 0:
                # path = plots/BAM_plus/
            # else:
                # path = plots/BAM/
            
            if self.plots:

                # Log output
                plt.clf()
                plt.plot(np.log(production[:,mc]))
                plt.xlabel("Time")
                plt.ylabel("Log output")
                plt.savefig("plots/full/LogOutput_mc%s.png" %mc)

                # inflation rate
                plt.clf()
                plt.plot(inflation[:,mc])
                plt.xlabel("Time")
                plt.ylabel("Inflation rate")
                plt.savefig("plots/full/Inflation_mc%s.png" %mc)

                # unemployment 
                plt.clf()
                plt.plot(unemp_rate[:,mc])
                plt.xlabel("Time")
                plt.ylabel("unemployment rate")
                plt.savefig("plots/full/unemployment_mc%s.png" %mc)
                
                # productivity/real wage
                plt.clf()
                plt.plot(product_to_real_wage[:,mc])
                plt.xlabel("Time")
                plt.ylabel("Productivity-real wage")
                plt.savefig("plots/full/product_to_real_wage_mc%s.png" %mc)


                # Philips curve
                plt.clf()
                plt.scatter(unemp_rate[:,mc], wage_inflation[:,mc])
                plt.xlabel("Unemployment Rate")
                plt.ylabel("Wage Inflation")
                plt.savefig('plots/full/philips_mc%s.png' %mc)

                # Okuns law
                plt.clf()
                plt.scatter(unemp_growth_rate[:,mc], output_growth_rate[:,mc])
                plt.xlabel("Unemployment growth rate")
                plt.ylabel("Output growth rate")
                plt.savefig('plots/full/Okun_mc%s.png' %mc)

                # Beveridge
                plt.clf()
                plt.scatter(unemp_rate[:,mc], vac_rate[:,mc])
                plt.xlabel("Unemployment rate")
                plt.ylabel("Vacancy rate")
                plt.savefig('plots/full/Beveridge_mc%s.png' %mc)

                # bankrutpcy
                plt.clf()
                plt.plot(bankrupt_firms_total[:,mc])
                plt.xlabel("Time")
                plt.ylabel("bankruptcies")
                plt.savefig("plots/full/bankruptcies%s.png" %mc)


                # mean HH Income
                plt.clf()
                plt.plot(aver_savings[:,mc])
                plt.xlabel("Time")
                plt.ylabel("average HH Income")
                plt.savefig("plots/full/HH_income_mc%s.png" %mc)
                

                """
                Plots only using the last 500 iterations 
                """
                # Log output
                plt.clf()
                plt.plot(np.log(production[self.T -499:,mc]))
                plt.xlabel("Time")
                plt.ylabel("Log output")
                plt.savefig("plots/cut/LogOutput_mc%s.png" %mc)

                # inflation rate
                plt.clf()
                plt.plot(inflation[self.T -499:,mc])
                plt.xlabel("Time")
                plt.ylabel("Inflation rate")
                plt.savefig("plots/cut/Inflation_mc%s.png" %mc)

                # unemployment 
                plt.clf()
                plt.plot(unemp_rate[self.T -499:,mc])
                plt.xlabel("Time")
                plt.ylabel("unemployment rate")
                plt.savefig("plots/cut/unemployment_mc%s.png" %mc)
                
                # productivity/real wage
                plt.clf()
                plt.plot(product_to_real_wage[self.T -499:,mc])
                plt.xlabel("Time")
                plt.ylabel("Productivity-real wage")
                plt.savefig("plots/cut/product_to_real_wage_mc%s.png" %mc)

                # linear regression object for putting regression lines through following scatter plots
                linear_regressor = LinearRegression() 
                
                # Philips curve
                linear_regressor = LinearRegression() 
                reg_phillip = linear_regressor.fit(unemp_rate[self.T -499:,mc].reshape(-1,1), wage_inflation[self.T -499:,mc].reshape(-1,1))
                Y_pred = linear_regressor.predict(unemp_rate[self.T -499:,mc].reshape(-1,1))  # make predictions
                plt.clf()
                plt.scatter(unemp_rate[self.T -499:,mc], wage_inflation[self.T -499:,mc])
                plt.plot(unemp_rate[self.T -499:,mc],Y_pred, color='red')
                plt.xlabel("Unemployment Rate")
                plt.ylabel("Wage Inflation")
                plt.savefig('plots/cut/philips_mc%s.png' %mc)

                # Okuns law
                linear_regressor = LinearRegression() 
                # drop inf values in case there are any
                if np.isfinite(unemp_growth_rate[self.T -499:,mc]).any() == True:
                    mask = np.isfinite(unemp_growth_rate[self.T -499:,mc])
                    reg_okun = linear_regressor.fit(unemp_growth_rate[self.T -499:,mc][mask].reshape(-1,1), output_growth_rate[self.T -499:,mc][mask].reshape(-1,1))
                    Y_pred = linear_regressor.predict(unemp_growth_rate[self.T -499:,mc][mask].reshape(-1,1))  # make predictions
                    plt.clf()
                    plt.scatter(unemp_growth_rate[self.T -499:,mc][mask], output_growth_rate[self.T -499:,mc][mask])
                    plt.plot(unemp_growth_rate[self.T -499:,mc][mask],Y_pred, color='red')
                    plt.xlabel("Unemployment growth rate")
                    plt.ylabel("Output growth rate")
                    plt.savefig('plots/cut/Okun_mc%s.png' %mc)
                else:
                    reg_okun = linear_regressor.fit(unemp_growth_rate[self.T -499:,mc].reshape(-1,1), output_growth_rate[self.T -499:,mc].reshape(-1,1))
                    Y_pred = linear_regressor.predict(unemp_growth_rate[self.T -499:,mc].reshape(-1,1))  # make predictions
                    plt.clf()
                    plt.scatter(unemp_growth_rate[self.T -499:,mc], output_growth_rate[self.T -499:,mc])
                    plt.plot(unemp_growth_rate[self.T -499:,mc],Y_pred, color='red')
                    plt.xlabel("Unemployment growth rate")
                    plt.ylabel("Output growth rate")
                    plt.savefig('plots/cut/Okun_mc%s.png' %mc)

                # Beveridge
                linear_regressor = LinearRegression() 
                reg_beveridge = linear_regressor.fit(unemp_rate[self.T -499:,mc].reshape(-1,1), vac_rate[self.T -499:,mc].reshape(-1,1))
                Y_pred = linear_regressor.predict(unemp_rate[self.T -499:,mc].reshape(-1,1))  # make predictions
                plt.clf()
                plt.scatter(unemp_rate[self.T -499:,mc], vac_rate[self.T -499:,mc])
                plt.plot(unemp_rate[self.T -499:,mc],Y_pred, color='red')
                plt.xlabel("Unemployment rate")
                plt.ylabel("Vacancy rate")
                plt.savefig('plots/cut/Beveridge_mc%s.png' %mc)

                # bankrutpcy
                plt.clf()
                plt.plot(bankrupt_firms_total[self.T -499:,mc])
                plt.xlabel("Time")
                plt.ylabel("bankruptcies")
                plt.savefig("plots/cut/bankruptcies%s.png" %mc)


                # mean HH Income
                plt.clf()
                plt.plot(aver_savings[self.T -499:,mc])
                plt.xlabel("Time")
                plt.ylabel("average HH Income")
                plt.savefig("plots/cut/HH_income_mc%s.png" %mc)
                
                """
                DF test & averages
                """
                # Augmented Dickey-Fuller unit root test on gdp and inflation
                fuller_gdp = adfuller(np.log(production[self.T -499:,mc]))
                fuller_inflation = adfuller(inflation[self.T -499:,mc])
                print(fuller_gdp)
                print(fuller_inflation)
                
                print("Average production %s" %np.mean(np.log(production[self.T -499:,mc])))
                print("Average unemployment %s" %np.mean(unemp_rate[self.T -499:,mc]))
                print("Average inflation %s" %np.mean(inflation[self.T -499:,mc]))
                
                """
                Correlation coefficients & betas
                """
                
                print("Corr phill curve %s" %np.corrcoef(unemp_rate[self.T -499:,mc], wage_inflation[self.T -499:,mc]))
                print("beta phillips %s" %reg_phillip.coef_)
                # drop inf values in case there are any
                if np.isfinite(unemp_growth_rate[self.T -499:,mc]).any() == True:
                    print("Corr okun %s" %np.corrcoef(unemp_growth_rate[self.T -499:,mc][mask], output_growth_rate[self.T -499:,mc][mask]))
                else:
                    print("Corr okun %s" %np.corrcoef(unemp_growth_rate[self.T -499:,mc], output_growth_rate[self.T -499:,mc]))
                print("beta okun %s" %reg_okun.coef_)
                print("Corr beveridge %s" %np.corrcoef(unemp_rate[self.T -499:,mc], vac_rate[self.T -499:,mc]))
                print("beta beveridge %s" %reg_beveridge.coef_)
            
            if self.empirical:
    
                # Log output
                plt.clf()
                plt.plot(np.log(production[:,mc]))
                plt.xlabel("Time")
                plt.ylabel("Log output")
                plt.savefig(self.path_empirical + "full/LogOutput_mc%s.png" %mc)

                # inflation rate
                plt.clf()
                plt.plot(inflation[:,mc])
                plt.xlabel("Time")
                plt.ylabel("Inflation rate")
                plt.savefig(self.path_empirical + "full/Inflation_mc%s.png" %mc)

                # unemployment 
                plt.clf()
                plt.plot(unemp_rate[:,mc])
                plt.xlabel("Time")
                plt.ylabel("unemployment rate")
                plt.savefig(self.path_empirical + "full/unemployment_mc%s.png" %mc)
                
                # productivity/real wage
                plt.clf()
                plt.plot(product_to_real_wage[:,mc])
                plt.xlabel("Time")
                plt.ylabel("Productivity-real wage")
                plt.savefig(self.path_empirical + "full/product_to_real_wage_mc%s.png" %mc)


                # Philips curve
                plt.clf()
                plt.scatter(unemp_rate[:,mc], wage_inflation[:,mc])
                plt.xlabel("Unemployment Rate")
                plt.ylabel("Wage Inflation")
                plt.savefig(self.path_empirical + 'full/philips_mc%s.png' %mc)

                # Okuns law
                plt.clf()
                plt.scatter(unemp_growth_rate[:,mc], output_growth_rate[:,mc])
                plt.xlabel("Unemployment growth rate")
                plt.ylabel("Output growth rate")
                plt.savefig(self.path_empirical + 'full/Okun_mc%s.png' %mc)

                # Beveridge
                plt.clf()
                plt.scatter(unemp_rate[:,mc], vac_rate[:,mc])
                plt.xlabel("Unemployment rate")
                plt.ylabel("Vacancy rate")
                plt.savefig(self.path_empirical + 'full/Beveridge_mc%s.png' %mc)

                # bankrutpcy
                plt.clf()
                plt.plot(bankrupt_firms_total[:,mc])
                plt.xlabel("Time")
                plt.ylabel("bankruptcies")
                plt.savefig(self.path_empirical + "full/bankruptcies%s.png" %mc)


                # mean HH Income
                plt.clf()
                plt.plot(aver_savings[:,mc])
                plt.xlabel("Time")
                plt.ylabel("average HH Income")
                plt.savefig(self.path_empirical + "full/HH_income_mc%s.png" %mc)
                

                """
                Plot only using the last 500 iterations 
                """
                # Log output
                plt.clf()
                plt.plot(np.log(production[self.T -499:,mc]))
                plt.xlabel("Time")
                plt.ylabel("Log output")
                plt.savefig(self.path_empirical + "cut/LogOutput_mc%s.png" %mc)

                # inflation rate
                plt.clf()
                plt.plot(inflation[self.T -499:,mc])
                plt.xlabel("Time")
                plt.ylabel("Inflation rate")
                plt.savefig(self.path_empirical + "cut/Inflation_mc%s.png" %mc)

                # unemployment 
                plt.clf()
                plt.plot(unemp_rate[self.T -499:,mc])
                plt.xlabel("Time")
                plt.ylabel("unemployment rate")
                plt.savefig(self.path_empirical + "cut/unemployment_mc%s.png" %mc)
                
                # productivity/real wage
                plt.clf()
                plt.plot(product_to_real_wage[self.T -499:,mc])
                plt.xlabel("Time")
                plt.ylabel("Productivity-real wage")
                plt.savefig(self.path_empirical + "cut/product_to_real_wage_mc%s.png" %mc)

                # linear regression object for putting regression lines through following scatter plots
                linear_regressor = LinearRegression() 
                
                # Philips curve
                linear_regressor = LinearRegression() 
                reg_phillip = linear_regressor.fit(unemp_rate[self.T -499:,mc].reshape(-1,1), wage_inflation[self.T -499:,mc].reshape(-1,1))
                Y_pred = linear_regressor.predict(unemp_rate[self.T -499:,mc].reshape(-1,1))  # make predictions
                plt.clf()
                plt.scatter(unemp_rate[self.T -499:,mc], wage_inflation[self.T -499:,mc])
                plt.plot(unemp_rate[self.T -499:,mc],Y_pred, color='red')
                plt.xlabel("Unemployment Rate")
                plt.ylabel("Wage Inflation")
                plt.savefig(self.path_empirical + 'cut/philips_mc%s.png' %mc)

                # Okuns law
                linear_regressor = LinearRegression() 
                # drop inf values in case there are any
                if np.isfinite(unemp_growth_rate[self.T -499:,mc]).any() == True:
                    mask = np.isfinite(unemp_growth_rate[self.T -499:,mc])
                    reg_okun = linear_regressor.fit(unemp_growth_rate[self.T -499:,mc][mask].reshape(-1,1), output_growth_rate[self.T -499:,mc][mask].reshape(-1,1))
                    Y_pred = linear_regressor.predict(unemp_growth_rate[self.T -499:,mc][mask].reshape(-1,1))  # make predictions
                    plt.clf()
                    plt.scatter(unemp_growth_rate[self.T -499:,mc][mask], output_growth_rate[self.T -499:,mc][mask])
                    plt.plot(unemp_growth_rate[self.T -499:,mc][mask],Y_pred, color='red')
                    plt.xlabel("Unemployment growth rate")
                    plt.ylabel("Output growth rate")
                    plt.savefig(self.path_empirical + 'cut/Okun_mc%s.png' %mc)
                else:
                    reg_okun = linear_regressor.fit(unemp_growth_rate[self.T -499:,mc].reshape(-1,1), output_growth_rate[self.T -499:,mc].reshape(-1,1))
                    Y_pred = linear_regressor.predict(unemp_growth_rate[self.T -499:,mc].reshape(-1,1))  # make predictions
                    plt.clf()
                    plt.scatter(unemp_growth_rate[self.T -499:,mc], output_growth_rate[self.T -499:,mc])
                    plt.plot(unemp_growth_rate[self.T -499:,mc],Y_pred, color='red')
                    plt.xlabel("Unemployment growth rate")
                    plt.ylabel("Output growth rate")
                    plt.savefig(self.path_empirical + 'cut/Okun_mc%s.png' %mc)

                # Beveridge
                linear_regressor = LinearRegression() 
                reg_beveridge = linear_regressor.fit(unemp_rate[self.T -499:,mc].reshape(-1,1), vac_rate[self.T -499:,mc].reshape(-1,1))
                Y_pred = linear_regressor.predict(unemp_rate[self.T -499:,mc].reshape(-1,1))  # make predictions
                plt.clf()
                plt.scatter(unemp_rate[self.T -499:,mc], vac_rate[self.T -499:,mc])
                plt.plot(unemp_rate[self.T -499:,mc],Y_pred, color='red')
                plt.xlabel("Unemployment rate")
                plt.ylabel("Vacancy rate")
                plt.savefig(self.path_empirical + 'cut/Beveridge_mc%s.png' %mc)

                # bankrutpcy
                plt.clf()
                plt.plot(bankrupt_firms_total[self.T -499:,mc])
                plt.xlabel("Time")
                plt.ylabel("bankruptcies")
                plt.savefig(self.path_empirical + "cut/bankruptcies%s.png" %mc)


                # mean HH Income
                plt.clf()
                plt.plot(aver_savings[self.T -499:,mc])
                plt.xlabel("Time")
                plt.ylabel("average HH Income")
                plt.savefig(self.path_empirical + "cut/HH_income_mc%s.png" %mc)
                
                """
                DF test & averages
                """
                # Augmented Dickey-Fuller unit root test on gdp and inflation
                fuller_gdp = adfuller(np.log(production[self.T -499:,mc]))
                fuller_inflation = adfuller(inflation[self.T -499:,mc])
                print(fuller_gdp)
                print(fuller_inflation)
                
                print("Average production %s" %np.mean(np.log(production[self.T -499:,mc])))
                print("Average unemployment %s" %np.mean(unemp_rate[self.T -499:,mc]))
                print("Average inflation %s" %np.mean(inflation[self.T -499:,mc]))
                
                """
                Correlation coefficients & betas
                """
                
                print("Corr phill curve %s" %np.corrcoef(unemp_rate[self.T -499:,mc], wage_inflation[self.T -499:,mc]))
                print("beta phillips %s" %reg_phillip.coef_)
                # drop inf values in case there are any
                if np.isfinite(unemp_growth_rate[self.T -499:,mc]).any() == True:
                    print("Corr okun %s" %np.corrcoef(unemp_growth_rate[self.T -499:,mc][mask], output_growth_rate[self.T -499:,mc][mask]))
                else:
                    print("Corr okun %s" %np.corrcoef(unemp_growth_rate[self.T -499:,mc], output_growth_rate[self.T -499:,mc]))
                print("beta okun %s" %reg_okun.coef_)
                print("Corr beveridge %s" %np.corrcoef(unemp_rate[self.T -499:,mc], vac_rate[self.T -499:,mc]))
                print("beta beveridge %s" %reg_beveridge.coef_)    
                
                
                
                
                
            
        return production