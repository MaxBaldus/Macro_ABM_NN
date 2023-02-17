import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import csv 

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
        self.capital_requirement_coef = 0.11 # capital requirement coefficients uniform across banks 

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
        self.product_to_real_wage = np.zeros((self.T,self.MC)) # productivity/real wage ratio
        
        
        self.production = np.zeros((self.T,self.MC)) # nominal gdp (quantities produced)
        self.output_growth_rate = np.zeros((self.T,self.MC)) # output growth rate 
    
        self.aver_income = np.zeros((self.T,self.MC)) # average HH income 
        
        self.vac_rate = np.zeros((self.T,self.MC)) # vacancy rate approximated by number of job openings and the labour force at the beginning of a period 

    
    
    def simulation(self):

        for mc in range(self.MC):
            print("MC run number: %s" %mc)
            print("")
            
            # set new seed each MC run to ensure different random numbers are sampled each simulation
            np.random.seed(mc)

            # initialize csv file and add header
            header = ['t', 'Ld', 'L', 'u' ,'Sum C', '#C', 'Sum NW', '#bankrupt', 'Sum Profit', 'p_min', 'p_max', 'Sum Qp', 'sum HH Income', 'Sum Qd'] 
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
            Y = np.zeros(self.Nh) # income of HH
            S = np.zeros(self.Nh) # savings
            C = np.zeros(self.Nh) # (actual) consumption
            C_d = np.zeros(self.Nh) # desired / demanded consumption
            MPC = np.zeros(self.Nh) # marginal propensity to consume (declines with wealth)

            w = np.zeros(self.Nh) # wage of each HH
            unemp_benefit = np.zeros(self.Nh) # uneployment benefits in case HH has no employment
            div = np.zeros(self.Nh) # dividend payment to HH from profit of firm (new each round)
            w_flag = np.zeros(self.Nh) # flag in case HH received wage in current t
            div_flag = np.zeros(self.Nh) # flag in case HH received dividend payment in current t

            is_employed = np.zeros(self.Nh, bool) # True if HH has job: e.g. array([False, False, False, False, False])
            d_employed = np.zeros(self.Nh) # counts the days a HH is employed
            d_unemployed = np.zeros(self.Nh) # counts the days a HH is unemployed
            firm_id = np.zeros(self.Nh) # firm_id where HH is employed
            prev_firm_id = np.zeros(self.Nh, int) # firm id HH was employed last
            firms_applied =  [[] for _ in range(self.Nh)] # list of firm ids HH applied to 
            job_offered_from_firm = [[] for _ in range(self.Nh)] # list of an offered job from firm i for each HH

            firm_went_bankrupt = np.zeros(self.Nh) # flag (value of 1) in case the firm HH was employed just went bankrupt 
            expired = np.zeros(self.Nh, bool) # flag in case contract of a HH expired, it is set to True 
            goods_purchased = [[] for _ in range(self.Nh)] # one list for each HH in which firm id's are stored that sold product to HH in round before
            
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
            L = np.zeros(self.Nf, int)  # actual Labor Employed
            vac = np.zeros(self.Nf, int) # Vacancy
            w_emp = [[] for _ in range(self.Nf)] # Ids of workers/household employed - each firm has a list with worker ids
            Wb_a = np.zeros(self.Nf) # aggregated Wage bills (wage payments to each Worker * employed workers)
            w_min_slice = np.repeat(np.arange(3,self.T,4), 4)  # array with values from 3 up to T-3 appearing 4 times in a row for slicing the minimum wage s.t. min. wage is revised every 4 periods
            Wp = np.zeros(self.Nf) # Wage payments of firm this round
            job_applicants = [[] for _ in range(self.Nf)] # Ids of Job applicants (HH that applied for job) -> Liste für each firm => array with [] entries for each i
           
            is_hiring = np.zeros(self.Nf, bool) # Flag to be activated when firm enters labor market to hire 

            P = np.zeros(self.Nf) # Profits
            NW = np.ones(self.Nf)  # initial Net worth
            Rev = np.zeros(self.Nf) # Revenues of a firm
            RE = np.zeros(self.Nf) # Retained Earnings
            divs = np.zeros(self.Nf) # Dividend payments to the households

            B = np.zeros(self.Nf) # amount of credit (total)
            banks = [[] for _ in range(self.Nf)] # list of banks from where the firm got the loans -> LISTE FOR EACH FIRM (nested)
            Bi = [[] for _ in range(self.Nf)] # Amount of credit taken by banks
            r_f = [[] for _ in range(self.Nf)] # rate of Interest on credit by banks for which match occured (only self.r bei ihm) 
            loan_paid = np.zeros(self.Nf)
            credit_appl = [[] for _ in range(self.Nf)] # list with bank ids each frim chooses randomly to apply for credit
            loan_to_be_paid = np.zeros(self.Nf) # outstanding debt + interest rate firm has to pay back to bank 

            outstanding = [[] for _ in range(self.Nf)] # under 6) if firm cannot pay back entire loan(s), amount outstanding for each bank (not paid back) is saved here
            outstanding_flag = np.zeros(self.Nf) # under 6), set to 1 if firm cannot pay back entire amount of loans
            outstanding_to_bank = [[] for _ in range(self.Nf)] # save the bank ids each firm has outstanding amount to 

            Qp_last = np.zeros(self.Nf) # variable to save quantity produced in round before 
            bankrupt_flag = np.zeros(self.Nf) # set entry to 1 in case firm went bankrupt 

            # BANKS
            E = np.zeros(self.Nb) # equity base 
            phi = np.zeros(self.Nb)
            tCr = np.zeros(self.Nb) # initial amount of Credit of all banks, filled with random numbers for t = 0
            Cr = np.zeros(self.Nb) # amount of credit supplied / "Credit vacancies"
            Cr_a = np.zeros(self.Nb) # actual amount of credit supplied (after..)
            Cr_rem = np.zeros(self.Nb) # credit remaining 
            Cr_d = [[] for _ in range(self.Nb)]
            r_d = [[] for _ in range(self.Nb)]
            r = [[] for _ in range(self.Nb)]
            firms_applied_b = [[] for _ in range(self.Nb)] 
            firms_loaned = [[] for _ in range(self.Nb)] 
            Cr_p = np.zeros(self.Nb)

            Bad_debt = np.zeros(self.Nb)

            """
            Initialization:
            Fill or initialize some individual report variables with random numbers when t = 0
            """
            # HOUSEHOLDS
            Y = np.around(np.random.uniform(low=10,high=10,size=self.Nh), decimals = 2) # initial (random) income of each HH  
            MPC = np.around(np.random.uniform(low=0.6,high=0.9,size=self.Nh), decimals = 2) # initial random marginal propensity to consume of each HH 

            # FIRMS
            NW = np.around(np.random.uniform(low=15 ,high=15,size=self.Nf), decimals = 2) # initial net worth of the firms 
            #Ld_zero = np.around(np.random.uniform(low=12 ,high=12), decimals = 2) # initial labour demand per firm 
            Ld_zero = 4.5 # s.t. in first round sum(vac) = 400 and therefore u = 1 - (L/Nh) = 0.2 
            # intial vacancies s.t. 
            Wp = np.around(np.random.uniform(low=1.9 ,high=2.1,size=self.Nf), decimals = 2) # initial wages
            alpha = np.random.uniform(low=0.5, high=0.5, size = self.Nf) # Productivity alpha stays constant here, since no R&D in this version
            # Delli Gatti & Grazzini 2020 set alpha to 0.5 in their model (constant)
            # p_zero = np.around(np.random.uniform(low = 2, high = 2), decimals=2) # initial prices of firms 
            p_zero = 2
            # BANKS
            # ??? 
            Cr = np.random.uniform(low = 3000, high = 3000, size = self.Nb) # initial credit supply in first round (amount of credit each bank can give out)
            E = np.random.uniform(low = 5000, high = 5000, size = self.Nb)

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
                    # alpha is constant in the base version, i.e. no growth
                    if t == 0: # set labor demand and price of each firm in t = 0 
                        p[i] = p_zero
                        Ld[i] = Ld_zero
                    elif bankrupt_flag[i] == 1: 
                        # in case firm went bankrupt in round before, initial labor demand of new firm is average labor demand of last round
                        Ld[i] = np.around(np.mean(Ld), decimals=0) 
                    # otherwise: each firm decides output and price depending on last round s.t. it can actually determine its labor demand for this round
                    else: # if t > 0, price and quantity demanded (Y^D_it = D_ie^e) are to be adjusted 
                        prev_avg_p = self.P_lvl[t-1,mc] # extract (current) average previous price 
                        p_d = p[i] - prev_avg_p  # price difference btw. individual firm price of last round and average price of last round
                        # decideProduction:
                        if p_d >= 0: # in case firm charged higher price than average price (price difference positive)
                            if Qr[i] > 0: # and firm had positive inventories, firm should reduce price to sell more and leave quantity unchanged
                                # a)
                                p[i] = np.around(max(p[i]*(1-eta[i]), prev_avg_p ), decimals=2) # max of previous average price or previous firm price  * 1-eta (price reduction) 
                                Qd[i] = np.around(Qp[i]) #if Qs[i] > 0 else Qs_last_round # use average of quantity demanded last round in case Qd were non positive last round
                            else: # firm sold all products, i.e. had negative inventories: firm should increase quantity since it expects higher demand
                                # d)
                                p[i] = np.around(p[i],decimals=2) # price remains
                                Qd[i] = np.around(Qp[i]*(1+rho[i]),decimals=2) #if Qs[i] > 0 else Qs_last_round
                        else: # if price difference negative, firm charged higher (individual) price than the average price 
                            if Qr[i] > 0: # if previous inventories of firm > 0 (i.e. did not sell all products): firm expects lower demand
                                # c)
                                p[i] = np.around(p[i],decimals=2)
                                Qd[i] = np.around(Qp[i]*(1-rho[i]),decimals=2) #if Qs[i] > 0 else Qs_last_round
                            else: # excess demand (zero or negative inventories): price is increased
                                # b)
                                p[i] = np.around(max(p[i]*(1+eta[i]), prev_avg_p),decimals=2)
                                Qd[i] = np.around(Qp[i],decimals=2) #if Qs[i] > 0 else Qs_last_round
                        # setRequiredLabor
                        Ld[i] = np.around(Qd[i] / alpha[i], decimals = 2) # labor demand of current round
                

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
                Firms determine Vacancies. 
                After 8 rounds the contract of a worker expires. In that case the worker will apply to the firm first she workerd at before. 
                """
                for i in range(self.Nf):

                    HH_employed = w_emp[i] # getEmployedHousehold: slice the list with worker id's (all workers) for firm i 
                    n = 0 # counter for the number of expired contracts of each firm i 
                    for j in (HH_employed): # j takes on HH id employed within current firm i 
                        if d_employed[j-1] > self.theta: # if contract expired (i.e. days employed is greater than 8) 
                            w_emp[i].remove(j) # removeHousehold: delete the current HH id in the current firm array 
                            prev_firm_id[j-1] = i # updatePrevEmployer: since HH is fired from firm i 
                            is_employed[j-1] = False # updateEmploymentStatus
                            firm_id[j-1] = 0 # setEmployedFirmId: 0 now, because no employed firm
                            w[j-1] = 0 # setWage: HH receives no wage now
                            d_employed[j-1] = 0 # 0 days employed from now on
                            n = n + 1 # add number of expired contracts
                            expired[j-1] = True # set entry to true when contract expired
                    
                    Lhat[i] = Lhat[i] + n # updateNumberOfExpiredContracts: add number of fired workers n and update fired workers (in this round) 
                    L[i] = len(w_emp[i]) # updateTotalEmployees: workforce or length of list with workers id, i.e. the number of wokers signed at firm i
                    vac[i] = int(max(np.around((Ld[i] - L[i] + Lhat[i]), decimals = 0) ,0)) # calcVacancies of firm i
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
                    w_min = np.around(p_zero, decimals = 2) if t in range(3) else self.P_lvl[w_min_slice[t-3],mc] # minimum wage is the average price level 4 rounds before 
                    if vac[i] > 0: # firm increases wage in case it has currently open vacancies:
                            Wp[i] = np.around(max(w_min, Wp[i]*(1+xi)),decimals=2) # Wage of firm i is set: either minimum wage or wage of period before revised upwards 
                            is_hiring[i] = True 
                    else:
                        Wp[i] = np.around(max(w_min, Wp[i]),decimals=2) # wage is either min. wage or wage of period before 
                        is_hiring[i] = False        

                """
                SearchandMatch: After firms made conctractual terms (wages and vacancies) public, search and match on the labour market begins.  
                There are two types of coordination failures that can occur: First the number of unemployed workers searching for a job is unqual to vacancies.
                Second firms with high wages experience excess of demand, other firms with low wages may not be able to fill all vacancies, in case
                there are more open vacancies than HH's searching for a job.
                """ 
                if len(f_empl) > 0: # if there are firms that have open vacancies, search and match begins
                # lieber raus, da es immer open vacancies geben sollte !!??    

                    """
                    A: Unemployed Households visit M random firms and sign contract with firm offering the highest wage out of the chosen subsample.
                    In case the contract just expired and same firm is also hiring, HH tries to apply to firm she worked before first. 
                    In case the worker was fired or lost her job due to bankruptcy, she send out M applications (instead of M-1 as in the former case). 
                    But it can happen that worker also applies again at the firm she worked at before
                    A worker can be fired (later) only in case the internal and external funds are not enough to pay for the desired wage bill.
                    """
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
                            break
    
                else:
                    print("No Firm with open vacancies")
                    print("")
                
                print("Labor Market CLOSED!!!! with %d NEW household hired!!" %(hired))
                print("Labor Market CLOSED!!!! with %d HH's currently employed (L)" %(np.sum(L)))
                print("hier weiter")






                    















                """ 
                labor market:
                - min_wage: use wage_level or P_lvl ??
                - is_hiring needed ? 
                - vac = int(..) => because then ??

                Consumption market
                - Qr set to 0 as soon everything is sold -> in the formulas !!!!
                - firing worker if w_pay not enough"""