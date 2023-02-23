#### labor market altlernatives

"""
                    jobs are distributed equally => s.t. no firm with more vacancy than other firm 
                    i.e. firms that are full deleted from hiring list 
                     
                    c_ids =  [j for j in c_ids if is_employed[int(j-1)] == False]  # get Household id's who are unemployed 
                    hired = 0 # initialize counter for hired HH's

                    np.random.shuffle(c_ids) # randomly selected household starts to apply (sequentially)
                    for j in c_ids:
                        
                        appl_firm = [] # list with firm ids the HH j applies to 
                        prev_emp = int(prev_firm_id[j-1]) # getPrevEmployer, i.e. the id of firm HH j were employed before (prev_emp is 0 as long as HH never fired or contract expired)
                        M = self.M # number of firms HH visits
                        if expired[j-1] == True and prev_emp in f_empl:
                            appl_firm.append(prev_emp) # if contract expired, HH directly chooses prev_emp
                            M = self.M - 1 
                            f_empl.remove(prev_emp)
                        
                        # HH chooses firms and extracts their wages
                        if len(f_empl) > M:
                            chosen_firms = list(np.random.choice(f_empl, M, replace = False)) # random chosen M firms the HH considers
                            appl_firm.extend(chosen_firms) # add firm ids to the list where HH applies
                        else:
                            chosen_firms = f_empl
                            appl_firm.extend(f_empl)

                        wages_chosen_firms = [] # initialize list with the wages of the chosen firms 
                        for ii in appl_firm: 
                                wages_chosen_firms.append(np.around(Wp[ii-1], decimals = 2)) # extract the wages of the firms where HH applied 
                        
                        # HH signs conctract with the firm that offers highest wage
                        w_max = max(wages_chosen_firms)
                        f_max_id = chosen_firms[wages_chosen_firms.index(w_max)] # id of the firm that offered highest wage
                        
                        # update labor market variables
                        is_employed[j-1] = True # updateEmploymentStatus
                        firm_id[j-1] = f_max_id # save the firm id where HH is employed (add one since Python starts counting at 0)
                        w[j-1] = np.around(w_max,decimals=2) # save wage HH l is earning
                        hired = hired + 1 # counter for #HH increases by 1
                        w_emp[f_max_id - 1].append(j) # employHousehold: save HH id to list of firm that is employing
                        L[f_max_id - 1] = len(w_emp[f_max_id - 1]) # updateTotalEmployees: update number of HH employed 
                        firm_went_bankrupt[j-1] = 0 # reset flag for employed worker in case he became unemployed because his previous firm went bankrupt

                        # firm stops hiring in case no more open vacancies
                        current_vac = vac[f_max_id - 1]
                        vac[f_max_id - 1] = current_vac - 1 # update vacancies of firm that just hired HH j
                        if vac[f_max_id - 1] == 0:
                            f_empl.remove(f_max_id)
                        
                        # labor market closes in case no more open vacancies or no more firms employing
                        if len(f_empl) == 0 or sum(vac) == 0:
                            break"""



import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import csv 
import math

"""
Overall class structure: The class BAM_base contains the BAM model from Delli Gatti 2011. 
The inputs are the values of the simulation parameters and specific model parameters. The output of the simulation function is none, 
only simulates the entire model according to the parameter settings and produces plots.
The function BAM_estimate also simulates the model, but actually returns specific time series (e.g. gdp as numpy array) 
without producing any plots. The actual code of the model is identical though. 
"""

class BAM_base:

    def __init__(self, T:int, MC:int, plots:bool, 
                 Nh:int, Nf:int, Nb:int, 
                 H_eta:float, H_rho:float, H_phi:float, h_xi:float):
        
        #################################################################################################
        # Parameters
        #################################################################################################
        
        """
        General model parameter
        """
        self.MC = MC # Monte Carlo replications
        self.plots = plots # plot parameter decides whether plots are produced
        self.T = T # simulation periods
        self.Nh = Nh # number of HH
        self.Nf = Nf # number of firms - Fc in main_base
        self.Nb = Nb # number of banks
        
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
        self.capital_requirement_coef = 0.1 # capital requirement coefficients uniform across banks 

        """
        Parameter to be estimated
        """
        self.H_eta = H_eta # Maximum growth rate of prices (max value of price update parameter) - h_eta in main_ase
        self.H_rho = H_rho # Maximum growth rate of quantities (Wage increase parameter)
        self.H_phi = H_phi # Maximum amount of banks’ costs
        self.h_xi = h_xi # Maximum growth rate of wages (Wage increase parameter)
        
        """
        ???
        self.c_P = c_P # Propensity to consume of poorest people
        self.c_R = c_R # Propensity to consume of richest people
        """

        #################################################################################################
        # AGGREGATE REPORT VARIABLES
        #################################################################################################

        """
        Here the aggregate report variables are initialised, which are filled with values throughout the simulations.
        """
        self.unemp_rate = np.zeros((self.T,self.MC)) # unemployment rate
        self.unemp_growth_rate = np.zeros((self.T,self.MC)) # unemployment growth rate 
        
        self.P_lvl = np.zeros((self.T,self.MC)) # price level
        self.inflation = np.zeros((self.T,self.MC)) # price inflation

        self.wage_lvl = np.zeros((self.T,self.MC)) # wage level
        self.real_wage_lvl = np.zeros((self.T,self.MC)) # real wage
        self.wage_inflation = np.zeros((self.T,self.MC)) # wage inflation (not in %)
        self.avg_prod = np.zeros((self.T,self.MC)) # average productivity of labor 
        self.product_to_real_wage = np.zeros((self.T,self.MC)) # productivity/real wage ratio
        
        
        self.production = np.zeros((self.T,self.MC)) # nominal gdp (quantities produced)
        self.output_growth_rate = np.zeros((self.T,self.MC)) # output growth rate 
    
        self.aver_savings = np.zeros((self.T,self.MC)) # average HH income 
        
        self.vac_rate = np.zeros((self.T,self.MC)) # vacancy rate approximated by number of job openings and the labour force at the beginning of a period 

        self.bankrupt_firms_total = np.zeros((self.T,self.MC)) # number of bankrupt firms in current round

    
    
    def simulation(self):

        for mc in range(self.MC):
            print("MC run number: %s" %mc)
            print("")
            
            # set new seed each MC run to ensure different random numbers are sampled each simulation
            np.random.seed(mc)

            # initialize csv file and add header
            header = ['t', 'Ld', 'L', 'u' ,'Sum C', 'Sum NW', '#bankrupt', 'Sum Profit', 'p_min', 'p_max', 'Sum Qp', 'sum HH Savings', 'Sum Qd'] 
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
            # Y = np.zeros(self.Nh) # income of HH
            S = np.zeros(self.Nh) # savings
            C = np.zeros(self.Nh) # (actual) consumption
            C_d = np.zeros(self.Nh) # desired / demanded consumption
            MPC = np.zeros(self.Nh) # marginal propensity to consume (declines with wealth)

            w = np.zeros(self.Nh) # wage of each HH
            w_last_round = np.zeros(self.Nh) # wage of round before 
            unemp_benefit = np.zeros(self.Nh) # uneployment benefits in case HH has no employment
            div = np.zeros(self.Nh) # dividend payment to HH from profit of firm (new each round)
            w_flag = np.zeros(self.Nh) # flag in case HH received wage in current t
            div_flag = np.zeros(self.Nh) # flag in case HH received dividend payment in current t

            is_employed = np.zeros(self.Nh, bool) # True if HH has job: e.g. array([False, False, False, False, False])
            d_employed = np.zeros(self.Nh) # counts the days a HH is employed
            d_unemployed = np.zeros(self.Nh) # counts the days a HH is unemployed
            #firm_id = np.zeros(self.Nh) # firm_id where HH is employed
            prev_firm_id = np.zeros(self.Nh, int) # firm id HH was employed last
            firms_applied =  [[] for _ in range(self.Nh)] # list of firm ids HH applied to 
            job_offered_from_firm = [[] for _ in range(self.Nh)] # list of an offered job from firm i for each HH

            #firm_went_bankrupt = np.zeros(self.Nh) # flag (value of 1) in case the firm HH was employed just went bankrupt 
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
            vac = np.zeros(self.Nf, int) # Vacancy
            w_emp = [[] for _ in range(self.Nf)] # Ids of workers/household employed - each firm has a list with worker ids
            W = np.zeros(self.Nf) # aggregated Wage bills (wage payments to each Worker * employed workers)
            w_min_slice = np.repeat(np.arange(1,self.T,4), 4)  # array with values from 3 up to T-3 appearing 4 times in a row for slicing the minimum wage s.t. min. wage is revised every 4 periods
            time_stamp_wage_adjustment = np.arange(3,self.T,4)
            Wp = np.zeros(self.Nf) # Wage payments of firm this round
            job_applicants = [[] for _ in range(self.Nf)] # Ids of Job applicants (HH that applied for job) -> Liste für each firm => array with [] entries for each i
           
            is_hiring = np.zeros(self.Nf, bool) # Flag to be activated when firm enters labor market to hire 

            P = np.zeros(self.Nf) # net Profits
            NW = np.ones(self.Nf)  # initial Net worth
            Rev = np.zeros(self.Nf) # Revenues of a firm (gross profits)
            RE = np.zeros(self.Nf) # Retained Earnings (net profits - divididens)
            divs = np.zeros(self.Nf) # Dividend payments to the households

            B = np.zeros(self.Nf) # amount of desired credit (total)
            lev = np.zeros(self.Nf) # invdivudal leverage (debt ratio)
            loan_paid = np.zeros(self.Nf)
            credit_appl = [[] for _ in range(self.Nf)] # list with bank ids each frim chooses randomly to apply for credit
            #loans_to_be_paid = np.zeros(self.Nf) # outstanding debt + interest rate firm has to pay back to bank 

            outstanding = [[] for _ in range(self.Nf)] # under 6) if firm cannot pay back entire loan(s), amount outstanding for each bank (not paid back) is saved here
            outstanding_flag = np.zeros(self.Nf) # under 6), set to 1 if firm cannot pay back entire amount of loans
            outstanding_to_bank = [[] for _ in range(self.Nf)] # save the bank ids each firm has outstanding amount to 
            outstanding_r = [[] for _ in range(self.Nf)] # interest rate charged by bank where firm has outstanding credit

            Qp_last = np.zeros(self.Nf) # variable to save quantity produced in round before 
            bankrupt_flag = np.zeros(self.Nf) # set entry to 1 in case firm went bankrupt 

            # BANKS
            E = np.zeros(self.Nb) # equity base 
            phi = np.zeros(self.Nb)
            tCr = np.zeros(self.Nb) # initial amount of Credit of all banks, filled with random numbers for t = 0
            Cr = np.zeros(self.Nb) # amount of credit supplied / "Credit vacancies"
            Cr_a = np.zeros(self.Nb) # actual amount of credit supplied (after..)
            Cr_rem = np.zeros(self.Nb) # credit remaining 
            # Cr_s = [[] for _ in range(self.Nb)] # amount of credit supplied 
            # r_s = [[] for _ in range(self.Nb)] # interest rate supplied 
            
            firms_applied_b = [[] for _ in range(self.Nb)] 
            # firms_loaned = [[] for _ in range(self.Nb)] 
            Cr_p = np.zeros(self.Nb)

            Bad_debt = np.zeros(self.Nb)

            """
            Initialization:
            Fill or initialize some individual report variables with random numbers when t = 0
            """
            # HOUSEHOLDS
            S = np.around(np.random.uniform(low=10,high=10,size=self.Nh), decimals = 8) # initial (random) income of each HH  
            # S_initial = 10 => sum(C_d) 3179 ; S_intiial = 2 => sum(Qr) = 25
            MPC = np.around(np.random.uniform(low=0.6,high=0.9,size=self.Nh), decimals = 8) # initial random marginal propensity to consume of each HH 

            # FIRMS
            NW = np.around(np.random.uniform(low=20 ,high=20,size=self.Nf), decimals = 8) # initial net worth of the firms 
            # assuming: B / NW = 0.4 => 0.4 * NW should be loaned (circa) 
            # in t = 0 ->  NW = 20 => few firms are around 0.2, 0.15 , other are higher up to 1 (depending on number of vacancies)
            # B often the same value if NW in t=0 the same for all firms => changed NW initial NW a little bit
            p = np.around(np.random.uniform(low=7.5 ,high=8,size=self.Nf), decimals = 8) # initial prices


            #Ld_zero = np.around(np.random.uniform(low=12 ,high=12), decimals = 2) # initial labour demand per firm 
            Ld_zero = 16.5 # 4.5 # s.t. in first round sum(vac) = 400 and therefore u = 1 - (L/Nh) = 0.2 
            # with 16.5 => intial vacancies s.t. u = 7% => sum(Qp) = 241
            Wp = np.around(np.random.uniform(low= 2,high=2.1,size=self.Nf), decimals = 8) # initial wages
            alpha = np.random.uniform(low=0.5, high=0.5, size = self.Nf) # Productivity alpha stays constant here, since no R&D in this version
            # Delli Gatti & Grazzini 2020 set alpha to 0.5 in their model (constant)
            # p_zero = np.around(np.random.uniform(low = 2, high = 2), decimals=2) # initial prices of firms 
            p_zero = 2
            w_min_zero = 1.9
            
            # BANKS
            E = np.random.uniform(low = 5, high = 5, size = self.Nb)
            # E / v = 5 / 0.25 = 20 => sum(Cr) = 200 ; sum(B)=206 in t=0 => u goes up 12% 
            # increasing captial requrirment coefficient to 0.1 => only 6 HH's layed off now

            """
            Simulation:
            """
            for t in range(self.T):
                print("time period equals %s" %t)
                print("")
                
                """ 
                1) 
                Firms decide on how much output to be produced and thus how much labor required (desired) for the production amount.
                Accordingly price or quantity (cannot be changed simultaneously) is updated along with firms expected demand 
                (adaptively based on firm's past experience).
                The firm's quantity remaining from last round and the deveation of the individual price the firm set last round compared to the
                average price of last round determines how quantity or prices are adjusted in this round.
                There is no storage or inventory, s.t. in case quantity remaining from last round (Qr > 0), it is not carried over.
                """

                print("1) Firms adjust and determine their prices and desired quantities and set their labour demand")
                print("")

                # Sample random quantity and price shocks for each firm, upper bound depending on respective parameter values (shocks change each period)
                eta = np.random.uniform(low=0,high=self.H_eta, size = self.Nf) # sample value for the (possible) price update (growth rate) for the current firm 
                rho = np.random.uniform(low=0,high=self.H_rho, size = self.Nf) # sample growth rate of quantity of firm i
                for i in range(self.Nf):
                    if t == 0: # set labor demand and price of each firm in t = 0 
                        Ld[i] = Ld_zero
                        
                        """elif bankrupt_flag[i] == 1: 
                        # in case firm went bankrupt in round before, initial labor demand of new firm is average labor demand of last round
                        Ld[i] = np.around(np.mean(Ld), decimals=2)
                        # Ld[i] = L_zero # ?? 
                        """

                    # otherwise: each firm decides output and price depending on last round s.t. it can actually determine its labor demand for this round
                    else: # if t > 0, price and quantity demanded (Y^D_it = D_ie^e) are to be adjusted 
                        prev_avg_p = self.P_lvl[t-1,mc] # extract (current) average previous price 
                        p_d = p[i] - prev_avg_p  # price difference btw. individual firm price of last round and average price of last round
                        # decideProduction:
                        if p_d >= 0: # in case firm charged higher price than average price (price difference positive)
                            if Qr[i] > 0: # and firm had positive inventories, firm should reduce price to sell more and leave quantity unchanged
                                # a)
                                p[i] = np.around(max(p[i]*(1-eta[i]), prev_avg_p ), decimals=8) # max of previous average price or previous firm price  * 1-eta (price reduction) 
                                Qd[i] = np.around(Qp[i]) #if Qs[i] > 0 else Qs_last_round # use average of quantity demanded last round in case Qd were non positive last round
                            else: # firm sold all products, i.e. had negative inventories: firm should increase quantity since it expects higher demand
                                # d)
                                p[i] = np.around(p[i],decimals=2) # price remains
                                Qd[i] = np.around(Qp[i]*(1+rho[i]),decimals=8) #if Qs[i] > 0 else Qs_last_round
                        else: # if price difference negative, firm charged higher (individual) price than the average price 
                            if Qr[i] > 0: # if previous inventories of firm > 0 (i.e. did not sell all products): firm expects lower demand
                                # c)
                                p[i] = np.around(p[i],decimals=2)
                                Qd[i] = np.around(Qp[i]*(1-rho[i]),decimals=8) #if Qs[i] > 0 else Qs_last_round
                            else: # excess demand (zero or negative inventories): price is increased
                                # b)
                                p[i] = np.around(max(p[i]*(1+eta[i]), prev_avg_p),decimals=8)
                                Qd[i] = np.around(Qp[i],decimals=8) #if Qs[i] > 0 else Qs_last_round
                        # setRequiredLabor: alpha is constant in the base version
                        Ld[i] = np.around(Qd[i] / alpha[i], decimals = 8) # labor demand of current round
                

                """ 
                2) 
                A fully decentralized labor market opens. Firms post vacancies with offered wages. 
                Workers approach subset of firms (randomly selected, size determined by M) acoording to the wage offer. 
                """
                
                print("2) Labor Market opens")
                print("")

                f_empl = [] # initialize list of Firms whose ids are saved in case they have open vacancies
                c_ids = np.arange(1, self.Nh + 1, 1, dtype=int) # array with consumer ids

                """
                Minimum wage is is revised upwards every 4 periods to catch up with inflation
                """
                if t in range(3):
                    w_min = w_min_zero
                elif t in time_stamp_wage_adjustment:
                    w_min = w_min *(1 + self.inflation[t-1,mc])
                # w_min = w_min_zero if t in range(3) else self.wage_lvl[w_min_slice[t-3],mc] # minimum wage is the average price level 4 rounds before
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
                            
                            """
                            
                            
                            firm_id[j-1] = 0 # setEmployedFirmId: 0 now, because no employed firm
                            w[j-1] = 0 # setWage: HH receives no wage now """
                    
                    Lhat[i] = n # number of just expired contracts
                    Lo[i] =  L[i] - Lhat[i]  # actual workforce
                    
                    # calcVacancies of firm i
                    vac[i] = math.ceil(max((Ld[i] - Lo[i]) ,0)) # vacancies are always rounded to the next higher and cannot be negative
                    if vac[i] > 0: 
                        f_empl.append(i+1) # if firm i has vacancies, then firm id i is saved to list 

                    """
                    Firms set their wages. The minimum wage is periodically revised upward every four time steps (quarters) in order to catch up with inflation.
                    In the first 4 rounds (up to t=4) the initial price level is used as the min. wage. Then the Price level of the period before is used respectively 
                    for every four periods. 
                    E.g. in t = 3 (fourth period), the wage level of t = 2 (third period) is used for 4 consective periods. In case t = 7, then the wage level of t=6
                    is used for 4 periods, and so on.
                    """
                    xi = np.random.uniform(low=0,high=self.h_xi) # sample xi (uniformly distributed random wage shock) for firm i
 
                    if vac[i] > 0: # firm increases wage in case it has currently open vacancies:
                            Wp[i] = np.around(max(w_min, Wp[i]*(1+xi)),decimals=8) # Wage of firm i is set: either minimum wage or wage of period before revised upwards 
                            is_hiring[i] = True 
                    else:
                        Wp[i] = np.around(max(w_min, Wp[i]),decimals=2) # wage is either min. wage or wage of period before 
                        is_hiring[i] = False        

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
                        
                        f_empl_current = f_empl.copy()
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
                                wages_chosen_firms.append(np.around(Wp[ii-1], decimals = 8)) # extract the wages of the firms where HH applied 
                        
                        # HH signs conctract with the firm that offers highest wage, in case the firm still has open vacancies
                        w_max = max(wages_chosen_firms)
                        f_max_id = appl_firm[wages_chosen_firms.index(w_max)] # id of the firm that offered highest wage

                        current_vac = vac[f_max_id - 1] # extract current number of vacancies of firm that offers the highest wage
                        if current_vac > 0:
                            vac[f_max_id - 1] = current_vac - 1 # update vacancies of firm that just hired HH j
                            
                            # update labor market variables
                            is_employed[j-1] = True # updateEmploymentStatus
                            #firm_id[j-1] = f_max_id # save the firm id where HH is employed (add one since Python starts counting at 0)
                            w[j-1] = np.around(w_max,decimals=2) # save wage HH l is earning
                            hired = hired + 1 # counter for #HH increases by 1
                            w_emp[f_max_id - 1].append(j) # employHousehold: save HH id to list of firm that is employing
                            L[f_max_id - 1] = len(w_emp[f_max_id - 1]) # updateTotalEmployees: update number of HH employed 
                            #firm_went_bankrupt[j-1] = 0 # reset flag for employed worker in case she became unemployed because his previous firm went bankrupt

                        # labor market closes in case no more open vacancies (i.e. no more firms employing) 
                        if sum(vac) == 0:
                            break
                    
                    """
                    labor market issues:
                    - min_wage: use wage_level or P_lvl ??
                    - is_hiring needed ? 
                    - vac = int(..) => because then ??
                    - with coordination failure: Ld_zero = 16.5 (hence each firm 16 vacancies) s.t. sum(L) =  46 (u = 8)
                    => 2.4782608695652173 , too high ? but only in t = 0 .. 
                    """                
    
                else:
                    print("No Firm with open vacancies")
                    print("")
                
                print("Labor Market CLOSED!!!! with %d NEW household hired!!" %(hired))
                print("Labor Market CLOSED!!!! with %d HH's currently employed (L)" %(np.sum(L)))
                

                """
                3)
                Credit market: If there is a financing gap, i.e. the desired wage bill W is larger than the current Net worth, firms go to credit market. 
                They randomly choose H number of  banks and 'apply' for loans, starting with bank offering the lowest interest rate. 
                Banks sort borrowers application according to financial conditions (NW) and satisfy credit demand until exhaustion of credit supply.
                Interest rate is calc acc. to markup on an baseline rate (set exogenous by 'CB'). 
                After credit market closed, if firms are still short in net worth, they fire workers.
                """
                # Initialize list to store values
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
                    W[f] = np.around(Wp[f] * L[f], decimals=8)  # (actual) total wage bills = wages * labour employed

                    if W[f] > NW[f]: # if wage bills f has to pay is larger than the net worth 
                        B[f] = max(np.around(W[f] - NW[f], decimals=8), 0) # amount of credit needed
                        f_cr.append(f + 1) # save id of firm that needs credit

                        # leverage (debt ratio)
                        lev[f] = np.around(B[f] / NW[f], decimals=8)
                
                """
                Credit market opens if there are any firms at all that need credit. 
                """
                if len(f_cr) > 0:
                    print("3) Credit market opens")
                    print("")
                    
                    """
                    Banks decides on its total amount of credit and computes the interest rates. 
                    The interest rate is composed of the exogenously given policy rate (by CB for example),
                    a shock representing idiosyncratic operating costs and the current financial fragility of each firm.
                    Here square root of the individual firm leverage is used to evaluate the soundness. 
                    """
                    b_cred = []
                    for b in range(self.Nb):
                        
                        # total credit supply of each bank
                        Cr[b] = np.around(E[b] / self.capital_requirement_coef, decimals=8) # amount of credit that can be supplied is an multiple of the equity base: E

                        # interest rates for each firm are computed (also for firms that don't apply for credit)
                        phi[b] = np.random.uniform(low=0,high=self.H_phi) # idiosyncratic interest rate shock
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
                                    Cr_rem[b_min_id -1] = np.around(Credit_remaining - cr_supplied, decimals=8) # credit remaining of bank b
                                    #firms_loaned[b_min_id -1].append(f) # save firm id the bank with lowest interest loaned money to 
                                    #Cr_s[b_min_id-1].append(cr_supplied) # save credit amount given out to firm by bank (id) with lowest interest rate 
                                    #r_s[b_min_id-1].append(r_min) # save interest rate charged to firm by bank with lowest interest rate

                                    break

                                else:
                                    cr_supplied = Credit_remaining # supplied credit by bank b_min_id is the entire remaining credit remaining
                                    B_current = np.around(B_current - cr_supplied, decimals=8) # remainig credit demand by firm 
                                    

                                    # update information on firm side
                                    B[f-1] = B_current # update remaining credit demand
                                    banks[f-1].append(b_min_id) # append bank id that supplied lowest credit 
                                    Bi[f-1].append(cr_supplied) # append received credit
                                    r_f[f-1].append(r_min) # interest rate charged by bank b_min_id

                                    # update information on bank side
                                    Cr_rem[b_min_id -1] = np.around(Credit_remaining - cr_supplied, decimals=8) # credit remaining of bank b
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
                        capital = np.around(NW[f-1] + sum(Bi[f-1]), decimals = 8)
                        wage_bill = W[f-1] # extract wage bill 
                        
                        if wage_bill > capital: # if wage bills f has to pay is still larger than the net worth + received credit
                            
                            # firm updates its wage bill and fires randomly chosen HH's
                            diff = np.around(W[f-1] - capital, decimals=8) # firm computes amount not enough to cover wage bill
                            L_fire = math.ceil(diff / Wp[f-1]) # rounded difference btw. wage bill and capital (int. and ext.) to next higher integer is number of HH's fired
                            # W[f-1] = wage_bill - L_fire*Wp[f-1]# update wage bill by subtracting individual wages

                            h_lo = np.random.choice(w_emp[f-1], L_fire, replace = False) # choose random worker(s) which are fired
                            for h in h_lo:

                                hh_layed_off.append(h)

                                # update employments on firms side
                                w_emp[f-1].remove(h) # update list with HH ids working at firm f
                                L[f-1] = len(w_emp[f-1]) # update labour employed at firm f
                                W[f-1] = np.around(Wp[f-1] * L[f-1], decimals=8)  # update (total wage bills = wages * labour employed

            
                                # update employment status on HH's side
                                prev_firm_id[h-1] = 0 # set previous firm id to 0, since HH no more preferential attachment, since it was sacked
                                #firm_id[h-1] = 0   # setEmployedFirmId: HH not employed anymore, therefore id is 0
                                is_employed[h-1] = False # set employment status to false      
                                w[h-1] = 0 # no wage payments in this round iff layed off
                                #firm_id[j-1] = 0 # HH not employed anymore             
                                
                    """
                    Issues:
                    - interest rate differences really small, since mark up so small !! 
                    """                    
                else: 
                    print("No credit requirement in the economy")
                    print("")            
                
                print("Credit Market closed. %s HH's were layed off." %len(hh_layed_off))
                print("")    


                """
                4)
                Production takes one unit of time regardless of the scale of prod/firms (constant return to scale, labor as the only input).  
                Since Firms either received enough credit or layed some HH's off if credit received was not enough, wage payments can all be paid,
                using internal (net worth) and external (credit) funds.
                The actual Wage payments have to the workers have to be made now in order to start production, which might become negative in case the 
                firm had to borrow money in the credit market. 
                Firms then have to make enough revenue (gross profit) in the following goods market s.t. it has enough income to cover the wage payments
                (negative net worth) and be able to pay back bank and potentially dividends (i.e. remaining with positive net worth in the end).
                Productivity alpha is 0.5 for each firm and remains constant throughout the entire simulation (hence no aggregate Output growth in the long run).
                In the BAM_plus version there is also technical innovation included s.t. alpha increases over time (through R&D in 6). 
                """
                print("4) Production")
                print("")            

                f_produced = [] # intialize list to store firm id if firm produced this round 

                for f in range(self.Nf):

                    NW[f] = NW[f] - W[f]
                    
                    if t > 0:
                        Qp_last[f] = Qp[f] # save the quantity produced in the last round 

                    # Firms producing their products: 
                    Qp[f] = np.around(alpha[f] * L[f], decimals = 8) # productivity * labor employed = quantity produced in this round
                    
                    # append firm id if firm actually produced something
                    if Qp[f] > 0:
                        f_produced.append(f+1) # append firm id if firm produced anything 
                    
                    Qs[f] = 0 # resetQtySold
                    Qr[f] = Qp[f] # Initialize the remaining quantity by subtracting the quantity sold (currently 0, since goods market did not open yet)
                """
                - subtracting W from NW ??
                """    

                print("Wage payments and Production done!!!")
                print("")

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
                    """
                    # marginal propensity to consume c
                    MPC[c-1] = np.around(1/(1 + (np.tanh((S[c-1])/savg))**self.beta),decimals=8) if t > 0 else MPC[c-1] 
                    # determine spending of HH in this round
                    w_fraction = np.around(w_last_round[c-1]*MPC[c-1], decimals=8) if t > 0 else np.around(w[c-1]*MPC[c-1], decimals=8) 
                    S_fraction = np.around(S[c-1]*MPC[c-1], decimals=8) 
                    spending = w_fraction + S_fraction 
                    C_d[c-1] = spending # desired consumption 

                    # update current savings
                    S[c-1] = S[c-1] - S_fraction

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
                    
                    
                    random_firms = list(np.random.choice(f_produced_current, Z, replace = False)) # HH randomly chooses Z firms
                    appl_firms.extend(random_firms) # add chosen banks to list of banks firm will apply for credit

                    # extract the specific prices of each firm
                    prices_chosen_firms = []  # initialize list with the prices of the chosen firms 
                    for ii in appl_firms: 
                            p_current = p[ii-1] # interest rate of the current firm (first slice list of all r of bank ii, then r for firm actually applying)
                            prices_chosen_firms.append(p_current) # extract the interest rateS of the chosen banks
                
                    goods_visited[c-1] = int(appl_firms[prices_chosen_firms.index(min(prices_chosen_firms))]) # save firm id out of firms she visits with lowest price

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
                                Qs[f_min_id -1] = np.around(Qs[f_min_id -1] + consumption , decimals=8) # actual quantity sold 
                                Qr[f_min_id- 1] = np.round(Qr[f_min_id-1] - consumption, decimals = 8)

                                appl_firms.remove(f_min_id) # delete firm id from list
                                prices_chosen_firms.remove(p_min) # delete current min price

                                break

                            else:
                                consumption = Y_current # demand is the entire remaining supply of firm 
                                consumption_p = consumption*p_min # consumption in terms of overall price level 

                                # update information on HH side
                                C_d[c-1] = np.around(C_d[c-1] - consumption_p, decimals=8) # updated remaining consumption in terms of 'overall price level'
                                C[c-1] = np.around(C[c-1] + consumption_p, decimals= 8) # new consumption level in terms of 'overall prices'

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
                    w_last_round[c-1] = w[c-1] if t > 0 else w[c-1] - w_fraction # wage of this round becomes wage of last round
                    C_d[c-1] = np.around(C_d[c-1], decimals=8) # round remaining desired consumption 
                    S[c-1] = S[c-1] + C_d[c-1] # update savings with desired consumption that was not satisfied

                print("Consumption Goods market closed!")           
                print("")

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
                print("Firms calculating Revenues, net profits (settling debt and paying dividends) and retained earnings")
                print("")
                for f in range(self.Nf):
                    Rev[f] = np.around(Qs[f]*p[f], decimals = 8)

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
                        loans_to_be_paid = np.around(np.sum( np.array(credit)*(1 + np.array(rates))),decimals=8)  # compute current total amount of all loans

                        if Rev[f] >= loans_to_be_paid:
                            
                            # firm pays back entire principal and interest to each bank 
                            for i in range(len(banks_f)):
                                
                                E[banks_f[i] -1] = np.around(E[banks_f[i] -1] + (credit[i]*rates[i]), decimals=8) 

                            P[f] = np.around(Rev[f] - loans_to_be_paid ,decimals=8) # net profits
                            divs[f] = np.around(P[f]*self.delta , decimals = 8) # dividends are profits * delta (share), if firm f has positive profits
                            NW[f] = np.around(NW[f] + (1-self.delta)*P[f], decimals = 8) # udpate firms NW
                            
                        else:

                            """
                            If firm cannot pay back all loans, i.e her gross revenue is smaller than total amount of loans firm has to pay back: 
                            The firm starts to pay back using her entire revenue, starting with the first bank she got credit from. 
                            """
                            amount_left = Rev[f] # initialize the entire amount of revenue firm has left to pay back firms
                            
                            banks_f_current = banks_f.copy() # initialize new bank list

                            for i in range(len(banks_f)):

                                current_loan_to_be_paid = np.around((credit[i]*(rates[i]+1)), decimals = 8) # amount firm has to pay back to current bank i

                                if amount_left >= current_loan_to_be_paid: 
                                    
                                    # If amount left to pay back loan of current bank i greater than loan & interest charged: firm pays back entire amount to current bank i
                                    E[banks_f[i] -1] = np.around(E[banks_f[i] -1] + (credit[i]*rates[i]), decimals=8) 
                                    
                                    amount_left = np.around(amount_left - current_loan_to_be_paid, decimals=8) # update new amount left and check whether next bank can be paid back , decimals=2) 
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
                                        P[f] = np.around(Rev[f] - loans_to_be_paid ,decimals=8) # net profit
                                        NW[f] = np.around(NW[f] - oustanding_amount, decimals=8) # update net worth

                                        break

                    else:

                        P[f] = np.around(Rev[f]) # gross profits are net profits in case firm did not take any credit
                        divs[f] = np.around(P[f]*self.delta , decimals = 8) # dividends are profits * delta (share), if firm f has positive profits
                        NW[f] = np.around(NW[f] + (1-self.delta)*P[f], decimals = 8) # udpate firms NW
                      
                """
                HH's receive dividend payments (assumed that each agent has an equal share in all firms) and update their days of employment.
                """
                # each HH receives an equal share
                for c in range(self.Nh):

                    div[c] = div[c] + np.around(divs[f]/self.Nh ,decimals=8)
                    S[c] = S[c] + div[c] # dividends are added to savings

                    # update employment count
                    if is_employed[c] == True: # getEmploymentStatus
                        d_employed[c] = d_employed[c] + 1 # incrementEdays

                print("Firms calculated Revenues, net profits (settling debt and paying dividends) and retained earnings. HH's updated employment days.")
                print("")

                """ 
                Aggregate report variables are computed.
                Each aggregation is saved into a corresponding report matrix (2-d numpy array) with number of rows = T and each column being the current mc round.
                """ 

                """
                Inflation: determine aggregate price level and compute inflation:
                """
                self.P_lvl[t,mc] = np.mean(p) # average price level 
                self.inflation[t,mc] = ( self.P_lvl[t,mc] - self.P_lvl[t-1,mc] ) / self.P_lvl[t-1,mc] if t!=0 else 0  

                """
                Output
                """
                self.production[t,mc] = np.sum(Qp) # nominal GDP = sum of quantity produced by each firm
                self.output_growth_rate[t,mc] = ( self.production[t,mc] - self.production[t-1,mc] ) / self.production[t-1,mc] if t!=0 else 0

                """
                unemployment rate
                """
                self.unemp_rate[t,mc] = 1 - (np.sum(L)/self.Nh)
                self.unemp_growth_rate[t,mc] =  ( self.unemp_rate[t,mc] - self.unemp_rate[t-1,mc] ) / self.unemp_rate[t-1,mc] if t!=0 else 0

                """
                Wages
                """
                self.wage_lvl[t,mc] = np.sum(W)/np.sum(L) # nominal wage level: sum of wage paid * number of employees relative to number of employees
                self.real_wage_lvl[t,mc] = (1-self.inflation[t,mc])*self.wage_lvl[t,mc] if t!= 0 else 0 # (1-inflation rate)* wage = real income

                """
                Productivity / real wage ratio
                """
                self.avg_prod[t,mc] = np.sum(alpha*L)/np.sum(L) # average productivity (only considering workers)
                self.product_to_real_wage[t,mc] = self.avg_prod[t,mc] / self.real_wage_lvl[t,mc] if t!=0 else 0 # productivity relative to real wage lvl 

                """
                Vacancies
                """
                self.vac_rate[t,mc] = np.round(np.sum(vac) / np.sum(L), decimals = 8) # vacancie rate is number of job openings (i.e. vacancies) relative to the labor force (labor employed currently)

                """
                Saving and consumption
                """
                self.aver_savings[t,mc] = np.mean(S) # averge HH savings


                """
                8) 
                Firms with positive net worth/equity survive, but otherwise firms/banks with negative (or zero) net worth/equity go bankrupt and are replaced by.
                new firms/banks which enter the market of size smaller than average size of those who got bankrupt (by using a truncated mean).
                """
                
                # 1) Firms
                bankrupt_firms = sum(1 for i in NW if i < 0) # count the bankrupt firms and report
                print("%s Firms went bankrupt" %bankrupt_firms)
                
                self.bankrupt_firms_total[t,mc] = bankrupt_firms # save amount of bankrupt firms this round 

                # checkForBankrupcy: Firm
                for f in range(self.Nf):
                    
                    if NW[f] <= 0:

                        h_emp = w_emp[f]# get HH's working at firm f
                        bankrupt_flag[f] = 1 # set flag to 1 in case firm went bankrupt

                        # update workers
                        for i in h_emp:
                            
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
                        Qp[f] = np.round(stats.trim_mean(Qp, 0.1), decimals = 8) # average produced quantity
                        Qr[f] = np.round(stats.trim_mean(Qr, 0.1), decimals = 8) # average remaining quantity
                        p[f] =  np.round(stats.trim_mean(p, 0.1), decimals = 8) # average price
                        #Ld[f] = 0 # labor demand again determined
                        L[f] =  0 # new firm enters with 0 labor employed
                        w_emp[f] = [] # workers employed is empty list 
                        NW[f] = np.mean(NW) # intial Net worth of new firm
                        alpha[f] = np.random.uniform(low=0.5,high=0.5) # draw new productivity btw. 5 and 6 
                            

                # 2) Banks
                bankrupt_banks = sum(1 for i in E if i < 0)
                print("%s Banks went bankrupt" %bankrupt_banks)

                for b in range(self.Nb):

                    if E[b] <= 0:

                        E[b] = np.round(stats.trim_mean(E, 0.1), decimals = 8) # initial Equity of new bank

               
                """
                Open csv file created and add data of the current round
                """

                add_row = [t, sum(Ld) / self.Nf, sum(L), 1-(sum(L)/self.Nh) ,sum(C), sum(NW), bankrupt_firms, sum(P), min(p), max(p), sum(Qp), sum(S), sum(Qd)]
                with open('simulated_data.csv', 'a', encoding = 'UTF8') as f:
                    writer = csv.writer(f)
                    writer.writerow(add_row)

            
                """ 
                TO DO:

                Consumption market
                -  # NO UNEMPLYOMENT PAYMENT ??!!
                - Qr set to 0 as soon everything is sold -> in the formulas !!!!
                - if bankrupt, remove goods_purchased.. 
                - firing worker if w_pay not enough
                
                Qs[f] = 0 # resetQtySold
                Qr[f] = Qp[f] - Qs[f] # setQtyRemaining: Initialize the remaining quantity by subtracting the quantity sold (currently 0, since goods market did not open yet)
                """

                """
                General:
                - no HH income Y
                - Y should be Qp, Q_des = Y_d and W_pay = W (book notation)
                - when is wage bill subtracted ??!!
                - in the end: check which "updating lists of each market" are really needed in the end ..  (go through everything step by step for t = 1 (&t=2))
                - initial entry: Ld = average ??!!
                - resetting all key lists in beginning of each market ??
                """

            """
            Plotting main aggregate report variables if number of MC replications small enough and simulation ran through:
            The plots are saved in folder: 
            """
            if self.plots and self.MC <= 2:

                # Log output
                plt.clf()
                plt.plot(np.log(self.production[:,mc]))
                plt.xlabel("Time")
                plt.ylabel("Log output")
                plt.savefig("plots/LogOutput_mc%s.png" %mc)

                # Log output ZOOM
                plt.clf()
                plt.plot(np.log(self.production[:,mc]))
                plt.xlabel("Time")
                plt.ylabel("Log output")
                plt.ylim(4.65, 5.5)
                plt.savefig("plots/LogOutput_mc%s_ZOOM.png" %mc)

                # inflation rate
                plt.clf()
                plt.plot(self.inflation[:,mc])
                plt.xlabel("Time")
                plt.ylabel("Inflation rate")
                plt.savefig("plots/Inflation_mc%s.png" %mc)

                # unemployment 
                plt.clf()
                plt.plot(self.unemp_rate[:,mc])
                plt.xlabel("Time")
                plt.ylabel("unemployment rate")
                plt.savefig("plots/unemployment_mc%s.png" %mc)
                
                # productivity/real wage
                plt.clf()
                plt.plot(self.product_to_real_wage[:,mc])
                plt.xlabel("Time")
                plt.ylabel("Productivity-real wage")
                plt.savefig("plots/product_to_real_wage_mc%s.png" %mc)


                # Philips curve
                plt.clf()
                plt.scatter(self.unemp_rate[:,mc], self.wage_inflation[:,mc])
                plt.xlabel("Unemployment Rate")
                plt.ylabel("Wage Inflation")
                plt.savefig('plots/philips_mc%s.png' %mc)

                # Okuns law
                plt.clf()
                plt.scatter(self.unemp_growth_rate[:,mc], self.output_growth_rate[:,mc])
                plt.xlabel("Unemployment growth rate")
                plt.ylabel("Output growth rate")
                plt.savefig('plots/Okun_mc%s.png' %mc)

                # Beveridge
                plt.clf()
                plt.scatter(self.unemp_rate[:,mc], self.vac_rate[:,mc])
                plt.xlabel("Unemployment rate")
                plt.ylabel("Vacancy rate")
                plt.savefig('plots/Beveridge_mc%s.png' %mc)

                # firm size distribution

                # bankrutpcy
                plt.clf()
                plt.plot(self.bankrupt_firms_total[:,mc])
                plt.xlabel("Time")
                plt.ylabel("bankruptcies")
                plt.savefig("plots/bankruptcies%s.png" %mc)


                # mean HH Income
                plt.clf()
                plt.plot(self.aver_savings[:,mc])
                plt.xlabel("Time")
                plt.ylabel("average HH Income")
                plt.savefig("plots/HH_income_mc%s.png" %mc)

                # maybe HH stop consuming at one point ??!!


                """Plots only using the last 500 iterations and zooming in s.t. fluctuations are better depicted"""
                """# Log output
                plt.clf()
                plt.plot(np.log(self.production[500:,mc]))
                plt.xlabel("Time")
                plt.ylabel("Log output")
                plt.ylim(7.8, 7.95)
                plt.savefig("plots/LogOutput_mc%s_zoom.png" %mc)"""
            

            print("Done")
