import numpy as np

"""# main.py rein kopieren und aufbereiten ...
    #	erst initializieren => dann füllen (jeweils für HH, Firms und Banks) => checken mit NWA .. 
    #	orientiren an version_2_2 model 
    
    # alle individual variables mit self. ??
    	# an base orientieren.. ???s"""

class BAM_base:


    MC = 2

    def __init__(self, MC:int, parameters:dict, plots:bool):
        
        self.MC = MC # Monte Carlo replications
        self.plots = plots # plot parameter decides whether plots are produced

        #################################################################################################
        # Parameters
        #################################################################################################
        
        self.Nh = parameters['Nh'] # number of HH
        self.Nf = parameters['Nf'] # number of firms - Fc in main_base
        self.Nb = parameters['Nb'] # number of banks
        self.T = parameters['T'] # simulation periods
        """Parameter to be estimated??"""
        self.Z = parameters['Z'] # Number of trials in the goods market (number of firms HH's randomly chooses to buy goods from)
        self.M = parameters['M'] # Number of trials in the labor market (number of firms one HH applies to)
        self.H = parameters['H'] # Number of trials in the credit market (Number of banks a firm selects randomly to search for credit)
        self.H_eta = parameters['H_eta'] # Maximum growth rate of prices (max value of price update parameter) - h_eta in main_ase
        self.H_rho = parameters['H_rho'] # Maximum growth rate of quantities (Wage increase parameter)
        self.H_phi = parameters['H_phi'] # Maximum amount of banks’ costs
        self.h_xi = parameters['h_xi'] # Maximum growth rate of wages (Wage increase parameter)
        
        """These two parameters are actually not needed, since MPC is computed in each round for each HH, always depending on the current savings respectively???"""
        self.c_P = parameters['c_P'] # Propensity to consume of poorest people
        self.c_R = parameters['c_R'] # Propensity to consume of richest people

        "Parameters set by modeller"
        self.beta = 4 # MPC parameter that determines shape of decline of consumption propensity when consumption increases 
        self.theta = 8 # Duration of individual contract
        self.r_bar = 0.04 # Base interest rate set by central bank (exogenous in this model)
        self.capital_requirement_coef = 0.11 # capital requirement coefficients uniform across banks 
        self.delta = 0.5 # Dividend payments parameter (fraction of net profits firm have to pay to the shareholders)

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

    def simulation(self):

        for mc in range(self.MC):
            print("MC run number: %s" %mc)
            
            # set new seed each MC run
            np.random.seed(mc)
            
            #################################################################################################
            # INDIVIDUAL REPORT VARIABLES
            #################################################################################################

            """Initialize all individual report variables used in one simulation:
            structure: a vector with elements, one element for each Agent (1xNf, 1xNh, 1xself.Nb) """

            # HOUSEHOLDS
            id_H = np.zeros(self.Nh, int) # HH id -> nur id in Household.py
            Y = np.zeros(self.Nh) # income of HH
            S = np.zeros(self.Nh) # savings
            C = np.zeros(self.Nh) # (actual) consumption
            C_d = np.zeros(self.Nh) # desired / demanded consumption
            MPC = np.zeros(self.Nh) # marginal propensity to consume (declines with wealth)
            
            w = np.zeros(self.Nh) 
            w_r = np.zeros(self.Nh)
            unemp_benefit = np.zeros(self.Nh)
            div = np.zeros(self.Nh)
            w_flag = np.zeros(self.Nh)
            div_flag = np.zeros(self.Nh)

            is_employed = np.zeros(self.Nh, bool) # e.g. H = 5 => array([False, False, False, False, False])
            d_employed = np.zeros(self.Nh)
            d_unemployed = np.zeros(self.Nh)
            firm_id = np.zeros(self.Nh) # firm_id where HH is employed
            prev_firm_id = np.zeros(self.Nh) # each HH only emplyed at 1 firm => 1d array reicht ..  
            firms_applied =  [[] for _ in range(self.Nh)] # list of firm ids HH applied to for each j?
            job_offered_from_firm = [[] for _ in range(self.Nh)] # list for each HH that gets an offered job from firm i
            prev_F = np.zeros(self.Nh)

            firm_went_bankrupt = np.zeros(self.Nh)

            # FIRMS
            id_F = np.zeros(self.Nf, int) # Firm id -> nur id in CFirm.py
            Qd = np.zeros(self.Nf) # Desired qty production : Y^D_it = D_ie^e
            Qp = np.zeros(self.Nf) # Actual Qty produced: Y_it
            Qs = np.zeros(self.Nf) # Qty sold
            Qr = Qp - Qs # Qty Remaining
            eta = np.random.uniform(low=0,high=self.H_eta, size = self.Nf) # Price update parameter
            p = np.zeros(self.Nf) # price
            pho = np.random.uniform(low=0,high=self.H_rho, size = self.Nf) # Qty update parameter
            bankrupcy_period = np.zeros(self.Nf)

            alpha = np.random.uniform(low=5,high=6, size = self.Nf) # Labor Productivity
            Ld = np.zeros(self.Nf) # Desired Labor to be employed
            Lhat = np.zeros(self.Nf) # Labor whose contracts are expiring
            L = np.zeros(self.Nf)  # actual Labor Employed
            Wb_d = np.zeros(self.Nf) # desired Wage bills
            Wb_a = np.zeros(self.Nf) # aggregated Wage bills (wage payments to each Worker * employed workers)
            Wp = np.zeros(self.Nf) # Wage level
            vac = np.zeros(self.Nf, int) # Vacancy
            W_pay = np.zeros(self.Nf) # Wage updated when paying
            job_applicants = [[] for _ in range(self.Nf)] # Ids of Job applicants (HH that applied for job) -> Liste für each firm => array with [] entries for each i
            job_offered_to_HH = [[] for _ in range(self.Nf)] # Ids of selected candidates (HH that applied for job and which may be picked by firm i)
            ast_wage_t = np.zeros(self.Nf) # Previous wage offered
            w_emp = [[] for _ in range(self.Nf)] # Ids of workers/household employed - each firm has a list with worker ids
            is_hiring = np.zeros(self.Nf, bool) # Flag to be activated when firm enters labor market to hire 

            P = np.zeros(self.Nf) # Profits
            p_low = np.zeros(self.Nf) # Min price below which a firm cannot charge
            NW = np.ones(self.Nf)  # initial Net worth
            Rev = np.zeros(self.Nf) # Revenues of a firm
            Total_Rev = np.zeros(self.Nf)
            RE = np.zeros(self.Nf) # Retained Earnings
            divs = np.zeros(self.Nf) # Dividend payments to the households

            B = np.zeros(self.Nf) # amount of credit (total)
            banks = [[] for _ in range(self.Nf)] # list of banks from where the firm got the loans -> LISTE FOR EACH FIRM (nested)
            Bi = [[] for _ in range(self.Nf)] # Amount of credit taken by banks
            r_f = [[] for _ in range(self.Nf)] # rate of Interest on credit by banks !! only self.r bei ihm 
            loan_paid = np.zeros(self.Nf)
            pay_loan = False # Flag to be activated when firm takes credit
            credit_appl = [[] for _ in range(self.Nf)] # list with bank ids each frim chooses randomly to apply for credit
            loan_to_be_paid = np.zeros(self.Nf) 

            Qty = [] # Quantity ?? - ???????????
            
            # OWN
            w_min_slice = np.repeat(np.arange(3,self.T,4), 4) # array with values from 2 up to 998 appearing 4 times for slicing the minimum wage

            outstanding = [[] for _ in range(self.Nf)] # under 6) if firm cannot pay back entire loan(s), amount outstanding for each bank (not paid back) is saved here
            outstanding_flag = np.zeros(self.Nf) # under 6), set to 1 if firm cannot pay back entire amount of loans
            outstanding_to_bank = [[] for _ in range(self.Nf)] # save the bank ids each firm has outstanding amount to 

            # BANKS
            id_B = np.zeros(self.Nb, int) # (own) Bank id -> nur id in Bank.py
            E = np.zeros(self.Nb) # equity base 
            phi = np.zeros(self.Nb)
            tCr = np.zeros(self.Nb) # initial amount of Credit of all banks, filled with random numbers for t = 0
            Cr = np.zeros(self.Nb) # amount of credit supplied / "Credit vacancies"
            Cr_a = np.zeros(self.Nb) # actual amount of credit supplied (after..)
            Cr_rem = np.zeros(self.Nb) # credit remaining 
            Cr_d = [[] for _ in range(self.Nb)]
            r_d = [[] for _ in range(self.Nb)]
            r = [[] for _ in range(self.Nb)]
            firms_applied_b = [[] for _ in range(self.Nb)] # ACHTUNG: firm_appl in bank class
            firms_loaned = [[] for _ in range(self.Nb)] 
            Cr_p = np.zeros(self.Nb)

            # OWN
            Bad_debt = np.zeros(self.Nb)


            """3-D Tensors to store individual data after a simulation: 
            i.e. one matrix for each round t with dimensions #agents x #variables (each agent is a row with entry on variable)"""
            c_data = np.zeros((self.T+2, self.Nh, 14))  # T+2 Hx14 matrices
            f_data = np.zeros((self.T+2, self.Nf, 19))  # saving individual data of each firm 
            b_data = np.zeros((self.T+2, self.Nb, 20))  # saving individual bank data    
            

            """Fill some individual report variables with random number when t = 0: """
            
            # HOUSEHOLDS
            Y0 = np.random.uniform(low=50,high=100,size=self.Nh) # initial (random) income of each HH  
            for h in range(self.Nh): # for each of the 500 HH
                    MPC[h] = np.random.uniform(low=0.6,high=0.9) # initial MPC for each HH (then recomputed in each loop)
                    id_H[h] = h + 1 # each HH gets number which is the id
                    Y[h] = Y0[h]  # fill initial income of each HH
                    # update Data
                    c_data[:,h,0] = h+1 # in all matrices id in first column
                    c_data[0,h,1] = Y[h] # initial income in t = 0
                    c_data[0,h,2] = MPC[h]*Y[h] # initial (actual) consumption in t = 0
                    c_data[0,h,4] = MPC[h] # 
                    c_data[0,h,5] = MPC[h]*Y[h] # initial desired consumption (same as C in t = 0)
            
            # FIRMS
            # DEBUGGING BZGL. hbyf etc. => erstmal als ones machen => dann füllen ??
            NWa = np.mean(Y0) # actual initial net worth is mean of initial HH's income
            hbyf = self.Nh // self.Nf # initial labour demand Ld in t = 0: ratio of HH and firms (household by firm)
            aa = (NWa *0.6) # minimum (required) wage is 60% of initial mean income of the HH's 
            # NW = np.ones(self.Nf) * (NWa*200) # Net-Worth of a firm -> do I need seperat NWa for each firm ??
            # NW = NWa*200   # initial Net worth per firm !!!
            # NW = np.ones(self.Nf) * (NWa*200)  # WARUMS SCALING mit 200 !! ???
            # wenn * 200 => Wb_a >> NW in t = 0 => credit market never opens
            NW = np.ones(self.Nf) * (NWa)  # initial net worth of each firm 
            for f in range(self.Nf):
                id_F[f] = f + 1 # set id 
                f_data[:,f,0] = f+1 # start with 1 for id
                Qty.append(Qd[f])

            # BANKS
            NB0 = np.random.uniform(low = 3000, high = 6000, size = self.Nb) # 
            aNB0 = np.mean(NB0)
            factor = self.Nh*9
            tCr = (NB0 / aNB0)*10*factor # amount of credit to be supplied if equity is 0 -> KANN ICH SELBER MACHEN über initial equity ???!!
            # WOHER KOMMEN DIE ZAHLEN?
            for b in range(self.Nb):
                b_data[:,b,0] = b + 1
            # E = tCr*0.11 # OWN: initial equitiy of each bank -> sample für E s.t. in range 


            """ List of data matrices:
            1) HH
            column 0 = id
            column 1 = income Y 
            column 2 = consumption C
            column 3 = savings S
            column 4 = MPC
            column 5 = desired consumption C_d
            column 6 = wage w
            column 7 = unemployment benefit unemp_benefit
            column 8 = firm where HH was employed previously prev_firm_id
            column 9 = firm where HH is employed currently firm_id
            column 10 = 1 if HH had a job in simulation round, 0 of not is_employed
            column 11 = previous consumption firm (id) of HH prev_F (included in Code ????)
            column 12 = dividends paid to HH div (always the same amount for each HH??) 
            column 13 = number of days a HH was employed (incremented each round) d_employed
            column 14 = number of days a HH was not employed (incremented each round) d_unemployed
            
            2) Firms
            column 1 = desired Quantity Qd
            column 2 = quantity produced Qp
            column 3 = quantity sold Qs
            column 4 = quantity remaining Qr
            column 5 = price p 
            column 6 = productivity of each firm alpha 
            column 7 = labor demanded/required Ld 
            column 8 = number of expired contracts 
            column 9 = number of employees L 
            column 10 = wage paid W_pay: wage firm offered each worker
            column 11 = desired wage bill Wb_d
            column 12 = actual wage bill Wb_a: entire wage (wage*workers) firm had to pay
            column 13 = revenues Rev 
            column 14 = loan firm has to pay to bank(s) loan_to_be_paid
            column 15 = Profits P 
            column 16 = dividends each firm has to pay divs
            column 17 = retainend Earnings RE
            column 18 = net worth NW
            column 19 = vancies vac  

            3) Banks
            column 1 = Equity E   
            column 2 = Credit supply Cr  
            column 3 = actual Credit Supply Cr_a  
            column 4 = remaining credit Cr_rem  
            column 5 = ratio:  credit paid out * credit rate relative to credit paid out(by bank b to each firm that used credit) 
            column 7 = total amount loans paid back to bank b Cr_p
            """

            """Simulation"""
            
            for t in range(self.T):
                print("time period equals %s" %t)

                """ 1) 
                Firms decide on how much output to be produced and thus how much labor required (desired) for the production amount.
                Accordingly price or quantity is updated along with firms expected demand (adaptively based on firm's past experience)."""

                # HOW TO BREAK: t = 0 durchlaufen lassen und ab t = 1 dann break rein !!! 

                for i in range(self.Nf):
                    # range starts with 0
                    id_F[i] = i + 1 # OWN: set id_F (eig gar nicht nötig, da running index dafür läuft) ; start with 1, 
                    # setRequiredLabor <- adjust price and Y^D_it = D_ie^e
                    if t == 0:
                        Ld[i] = hbyf # labor demanded is HH per firm for each firm in first period => hence each firm same demand in first period 
                    # otherwise: each firm decides output and price => s.t. it can determine labour demanded
                    else: # t > 0 -> need price to set Qd <=> RequiredLabor
                        """CHECKEN IN t = 1"""
                        prev_avg_p = self.P_lvl[t-1][mc] # extract (current) average previous price 
                        # decideProduction 
                        p_d = f_data[t-1, i, 5] - prev_avg_p  # demanded price = prev_FPrice - prev_APrice
                        # p_d decides about how to set price, including inventories:
                        if p_d >= 0:
                            if f_data[t-1, i, 4] > 0: # if prev_inv > 0 => firm should reduce price to sell more
                                p[i] = np.around(max(f_data[t-1, i, 5]*(1-eta[i]), prev_avg_p ), decimals=2) # either previous firm price * eta or previous average price 
                                Qd[i] = np.around(f_data[t-1, i, 3]) if f_data[t-1, i, 3] > 0 else max(1.01*f_data[t-1, i, 1], 10)
                            else: # firm should increase quantity - firm expect higher demand
                                p[i] = np.around(f_data[t-1, i, 5],decimals=2)
                                Qd[i] = np.around(f_data[t-1, i, 3]*(1+pho[i]),decimals=2) if f_data[t-1, i, 3] > 0 else max(1.01*f_data[t-1, i, 1],10)
                        else: # if p_d < 0 -> change quantity, depending on inventories
                            if f_data[t-1, i, 4] > 0: # if prev_inv > 0
                                #Reduce quantity - firm expect lower demand
                                p[i] = np.around(f_data[t-1, i, 5],decimals=2)
                                Qd[i] = np.around(f_data[t-1, i, 3]*(1-pho[i]),decimals=2) if f_data[t-1, i, 3] > 0 else max(1.01*f_data[t-1, i, 1],10)
                            else:
                                #Increase only price
                                p[i] = np.around(max(f_data[t-1, i, 5]*(1+eta[i]), prev_avg_p),decimals=2)
                                Qd[i] = np.around(f_data[t-1, i, 3],decimals=2) if f_data[t-1, i, 3] > 0 else max(1.01*f_data[t-1, i, 1],10)
                        # setRequiredLabor
                        Ld[i] = Qd[i] // alpha[i] # otherwise: labour demanded = quantity demanded / productivity

                """ 2) 
                A fully decentralized labor market opens. Firms post vacancies with offered wages. 
                Workers approach subset of firms (randomly selected, size determined by M) acoording to the wage offer. 
                Labor contract expires after finite period "theta" and worker whose contract has expired applies to most recent firm first. 
                ????Firm pays wage bills to start."""
                
                print("Labor Market opens")
                hired = 0
                # c_ids = np.array(range(self.Nh))  # c_d[:,0]  - e.g. H = 5 => array([1, 2, 3, 4, 5])
                c_ids = c_data[t,:,0].astype(int) # slice the consumer id's
                f_empl = [] # List of Firms who are going to post vacancies (id of firm that will post v)

                """Firms determine Vacancies and set Wages"""
                for i in range(self.Nf):
                    """Firms updating vacancies in the following by keeping or firing workers in case contract expired:
                    i.e. rounds of employment is greater than 8"""
                    if L[i] >= 0: # L kann nicht negativ sein => immer aktiviert!!
                        c_f = w_emp[i] # getEmployedHousehold: slice the list with worker id's (all workers) for firm i 
                        n = 0 # counter for the number of expired contracts of each firm i 
                        rv_array = np.random.binomial(1,0.5,size=len(c_f)) # firm firing hired workers or not: 0.5 chance 
                        # 1x a "coin-flip" (1x bernoulli RV) for each worker emplyoed (i.e. size determines length of output)
                        # e.g. s = np.random.binomial(1, 0.5, 10) yields array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])
                        for j in (c_f): # j takes on HH id employed within current firm i 
                            j = int(j-1) # python starts slicing with 1
                            if d_employed[j] > self.theta: # if contract expired (i.e. greater 8)
                                rv = rv_array[c_f.index(j)] # check if current worker j is fired or not (0 or 1 entry)
                                # if rv = 0, then worker is not fired and gets new contract instead, hence not searching in the following for new employment 
                                if rv == 1: # if worker is fired: later M - 1 since implied that HH already tried to regain job (50% chance)
                                    w_emp[i].remove(j) # removeHousehold  -> delete the current HH id in the current firm array 
                                    prev_firm_id[j] = i # updatePrevEmployer -> since HH is fired from firm i 
                                    is_employed[j] = False # updateEmploymentStatus
                                    firm_id[j] = 0 # setEmployedFirmId -> 0 now, because no employed firm
                                    w[j] = 0 # setWage -> no wage now
                                    d_employed[j] = 0 # 0 days employed from now on
                                    n = n + 1 # add expired contract
                        Lhat[i] = Lhat[i] + n # updateNumberOfExpiredContracts: add number of fired workers n 
                        L[i] = len(w_emp[i]) # updateTotalEmployees -> length of list with workers id, i.e. the number of wokers signed at firm i
                        """calcVacancies of firm i"""
                        vac[i] = int((Ld[i] - L[i] + Lhat[i])*1) # labour demanded (depending on price) - labour employed + Labor whose contracts are expiring
                        if vac[i] > 0: 
                            f_empl.append(i+1) # if firm i has vacancies => then save firm_id (i) to list 
                    """setWage"""
                    xi = np.random.uniform(low=0,high=self.h_xi) # compute xi (uniformly distributed random wage shock) for firm i
                    # OWN: set the minimum wage: w_min is periodically revised upward every four time steps (quarters) in order to catch up with inflation
                    # in the first 4 rounds (up to t=4): initial value aa is used as min. wage
                    # then the wage_lvl of the period before is used respectively for every four periods
                    # e.g. if t = 3 (fourth period): then the wage level of t = 2 (third period) is used for 4 consective periods: if t = 7, then the wage level of t=6, and so on
                    w_min = aa if t in range(3) else self.wage_lvl[w_min_slice[t-3]][mc] # minimum wage
                    # LATER: USE *0.4 instead the average wage_lvl (minimum wage too high ??) ?????
                    """if f_data[t-1,i,3] == 0: # if firm has no quantitiy remaining (from last round, i.e. did not sell all products)
                        xi = 2*xi # use 2*xi, since firm wants to offer higher wage and increase chance for match later if no more products (or just entered)
                        # w_min = aa[i] # minimum (required) wage of firm i (60% of mean income of HH's): BEI IHM, da sich sein min_wage immer increased via np.random.uniform(low=1.01,high=1.10
                        if vac[i] > 0: # if firm i has currently open vacancies:
                            Wp[i] = np.around(max(w_min, min(w_min*(np.random.uniform(low=1.01,high=1.10)),Wp[i]*(1+xi/2))), decimals=2) # wage
                            # since he always uses same wage (i.e. aa) => he increases the wage every time randomly by a little bit s.t. wage spiral is fullfilled
                            is_hiring[i] = True 
                        Wp[i] = np.around(max(w_min, Wp[i]),decimals=2) # wage if firm i currently no open vacancies
                        is_hiring[i] = False  
                    # firm i uses xi (not 2*xi), if firm had quantitiy remaining (no need to increase chances of getting workers by using higher weight)
                    # DAS IST VON IHM !!"""
                    # w_min = aa[i] # minimum (required) wage of firm i (INITIAL!!??): Bei ihm 
                    if vac[i] > 0: # if firm has currently open vacancies:
                        Wp[i] = np.around(max(w_min, Wp[i]*(1+xi)),decimals=2) # wage if firm i has currently open vacancies 
                        is_hiring[i] = True 
                    else:
                        Wp[i] = np.around(max(w_min, Wp[i]),decimals=2) # wage if firm i currently no open vacancies
                        is_hiring[i] = False  
                    """Finished: Firms posted Vacancies and Wage offers. 
                    In t = 0 wage is the same for each firm (i.e. initial wage aa), because max(aa, 0) (Wp[i] = 0) """
                
                """SearchandMatch: 
                There must be at least 1 entry in the list with firm id's that have open vacancies: otherwise no searchandmatch.""" 
                if len(f_empl) > 0: # if there are firms that offer labour / have open vacancies (i.e. if any firm id's in f_empl list), i.e. L^d > L 
                    # A: Unemployed Households applying for the vacancies
                    c_ids =  [j for j in c_ids if is_employed[int(j-1)] == False]  # get Household id who are unemployed (i.e. is_employed is False)
                    print("%d Households are participating in the labor market (A:)" %len(c_ids))
                    #  FOR COMPARISON
                    np.random.seed(1)
                    np.random.shuffle(c_ids) # randomly selected household starts to apply (sequentially)
                    for j in c_ids:
                        j = int(j)
                        appl_firm = None # store firm id's applied to by HH j
                        prev_emp = int(prev_firm_id[j-1]) #getPrevEmployer, i.e. the id of firm HH j were employed before 
                        f_empl_c = [x for x in f_empl if x != prev_emp] # gather firm ids that have open vacancies (neglect firm if HH was just fired)
                        M = self.M if t == 0 or firm_went_bankrupt[j-1] == 1 else self.M - 1 # number of firms HH applies to
                        # in t = 0 or if firm the HH worked before went bankrput: HH cannot apply to firms where they worked before: in this case M = 4
                        # otherwise HH already tried to apply to firm be worked before: either rv = 0 or rv = 1
                        if len(f_empl_c) > M: # if there are more than M-1 (3) firms with open vacancies
                            appl_firm = np.random.choice(f_empl_c, self.M-1, replace = False) # choose some random firm ids to apply to 
                        else: # otherwise: HH applies to the only firms hiring (with open vacancies)
                            appl_firm = f_empl_c # save firm id's HH applies to
                        firms_applied[j-1].append(appl_firm) # attach list with firm id's HH j will apply to (attach to its entry in list)
                        #print("firms applied:", firms_applied[j-1], "by HH", j-1)
                        # Houshold j applies for the vacancies
                        for a in appl_firm: # a takes on the firm id that HH applies to 
                            job_applicants[a-1].append(j) # add firm id to the list of firm id's the HH applies to 
                        #    print("Job applicants:", job_applicants[a-1])
                                   
                    # B: Firms offer Jobs to randomly selected applicants (HH who applied)
                    for f in f_empl:
                        vac_firm = vac[f-1] # calculated vacancies of firm f computed before
                        applicants = job_applicants[f-1]# getJobApplicants (list of all HH's (id) applying to current firm i that employes)
                        n_applicants = len(applicants) # number of applicants
                        if vac_firm >= n_applicants:
                            # job shortage: if more vacancies than number of applicants => firm i accepts all applicants
                            job_offered_to_HH[f-1] = applicants # HH's that offered job to firm i
                            # updateHouseholdJobOffers (offered jobs by firm (id) to HH)
                            for a in applicants:
                                job_offered_from_firm[a-1].append(f) # save the firm id that offers job to HH which is currently applying 
                        else:
                            # more HH applying for jobs than firms offer vacancies => shuffle random number of applicants (HH id's), according to #vacancies of firm f
                            applicants = np.random.choice(applicants, vac_firm, replace=False) 
                            job_offered_to_HH[f-1] = list(applicants) # HH's id to which a job is offered by firm f (firm -> HH)
                            # updateHouseholdJobOffers (offered jobs by firm (id) to HH)
                            for a in applicants:
                                job_offered_from_firm[a-1].append(f) # save the firm id that offers job to HH which is currently applying (firm -> HH)
                        #print("job offered to HH:", job_offered_to_HH[f-1], "by firm", f)
                    
                    # C: Household accepts if job offered is of highest wage
                    for l in c_ids:
                        # l = int(l-1) # individual report variables start at 0 => reduce running index by 1, s.t. not going out of bounds
                        # getFirmsApplied: extract ids of firms HH j applied to (numpy arrays inside list)
                        # e.g. have [array[9,17,7]] => need to slice with [0] to get the actual list in the following, avoid error when slicing with [0] if entry is empty
                        f_applied_ids = firms_applied[l-1][0] if len(firms_applied[l-1]) > 0 else firms_applied[l-1]
                        f_applied_ids = [x-1 for x in f_applied_ids] # slice firm id out of numpy array (x-1, since python starts counting at 0)
                        f_job_offers = job_offered_from_firm[l-1]  # getJobOffers: extract firm ids (or id) that offered job to HH j
                        f_job_offers = [x-1 for x in f_job_offers]
                        f_e = np.zeros(len(f_applied_ids)) # initialize vector of wages of the firms where HH l applied (use length of array entries inside list)
                        offer_wage = [] # initialize wage of offering firm that is searching
                        print(l,f_applied_ids, f_job_offers)  
                        if len(f_applied_ids) != 0: # if HH applied to some firms (id's) (i.e. if there is array inside f_applied_ids)
                            ind = 0 # counter
                            for ii in f_applied_ids: 
                                f_e[ind] = Wp[ii] # extract the wages of the firms where HH applied 
                                ind += 1 # increase index
                            for of in f_job_offers: # extract the wages of the offering firms
                                offer_wage.append(np.around(Wp[of] ,decimals=2)) 
                            w_max = max(f_e) # max. wage of firms the HH l applied to 
                            # w_max = max(offer_wage) if t == 0 and len(offer_wage) > 0 else max(f_e) 
                            # HH extracts the max. wage of the firm she applied to: If this wage matches the wages the HH was offered, it is the highest
                            # wage and the HH directly accepts the job (w_max in offer_wage).
                            # Otherwise the HH chooses the the firm offering highest wage, also if "preffered firm with highest wage HH applied to" is not employing her (no match)  
                            if w_max in offer_wage: 
                                # Settlement if firm with highest wage HH applied to also offered job to HH. Always the case in t = 0, since all firms offer equal weight
                                # update report variables:
                                f_max_id = f_job_offers[offer_wage.index(w_max)] # save firm id for which match occured (i.e. firm that offered highest wage)
                                is_employed[l-1] = True # updateEmploymentStatus
                                firm_id[l-1] = f_max_id + 1 # save the firm id where HH is employed (add one since Python starts counting at 0)
                                w[l-1] = np.around(w_max,decimals=2) # save wage HH l is earning
                                hired = hired + 1 # counter for #HH increases by 1
                                w_emp[f_max_id].append(l) # employHousehold: save HH id to list of firm that is employing
                                L[f_max_id] = len(w_emp[f_max_id]) # updateTotalEmployees: update number of HH employed 
                                firm_went_bankrupt[l-1] = 0 # reset flag for employed worker in case he became unemployed because his previous firm went bankrupt
                                # print("Household no: %d has got employed in Firm no: %d at wage %f"% (l, f_max_id+1, w_max))
                            # elif t > 0 and len(offer_wage) > 0: # Settlement if not in first round and if list of job offering firms is not empty
                            elif len(offer_wage) > 0: # Settlement if firm with max. wage HH applied to not offering a job, but other firm(s) do
                                mm = max(np.array(offer_wage)) # extract max. offered wage
                                # Settlememt: HH applied to firms and extracts max. wage of the firms she has job offer 
                                # hence if there are job offers to HH => she accepts job with highest wage
                                # if mm > self.wage_lvl[t-1][mc]: # e.g.  * 0.4 => smaller minimum wage ??!! -> wenn davor bei w_min = ...
                                if mm > w_min: # check again whether max. offered wage above current minimum wage (eigentlich not needed) 
                                    # if maximum offered wage is larger than (average) wage_level of before => HH l accepts the job, otherwise HH remains unemployed    
                                    # Update report variables:
                                    f_max_id = f_job_offers[offer_wage.index(mm)] # save firm id for which match occured (i.e. firm that offered highest wage)
                                    is_employed[l-1] = True # updateEmploymentStatus
                                    firm_id[l-1] = f_max_id + 1 # save the firm id where HH is employed (add one since Python starts counting at 0)
                                    w[l-1] = np.around(mm,decimals=2) # save wage HH l is earning
                                    hired = hired + 1 # counter for #HH increases by 1
                                    w_emp[f_max_id].append(l) # employHousehold: save HH id to list of firm that is employing
                                    L[f_max_id] = len(w_emp[f_max_id]) # updateTotalEmployees: update number of HH employed 
                                    firm_went_bankrupt[l-1] = 0 # reset flag for employed worker in case she became unemployed because her previous firm went bankrupt
                                # print("Household no: %d has got employed in Firm no: %d at wage %f"% (l, f_max_id+1, mm))
                else:
                    print("No Firm with open vacancies")
                print("")   
                print("Labor Market CLOSED!!!! with %d household hired!!" %(hired))
                print("")

                """Firms set the wage bills"""
                for f in range(self.Nf):
                    Wb_a[f] = Wp[f] * L[f]  # (actual) wage bills = wages * labour employed
                print("Firms calculated Wage bills!!")
                print("")

                """3)
                Credit market: If there is a financing gap, firms go to credit market, i.e. after firms payed loans and net-worth is negative. 
                They go to random chosen banks (depending on parameter H) to get loans, starting to apply to the bank having lowest interest rate. 
                Banks sort borrowers application according to financial conditions and satisfy credit demand until exhaustion of credit supply.
                Interest rate is calc acc. to markup on an baseline rate (set exogenous by "CB"). 
                After credit market closed, if firms are still short in net worth, they fire workers or dont accept them."""
            
                
                # parameter H (number of banks a firm asks for credit, here H = 2)
                b_ids = np.array(range(self.Nb)) + 1 # bank identities (+1 since range starts with 0)
                f_ids = np.array(range(self.Nf)) + 1 # firm identities
                f_cr = [] # initialize list with id's of firms which need credit in the following
                np.random.shuffle(f_ids) # 
                print("")
                print("Credit Market OPENS!!!!") # Firms apply for Credit (randomly, hence shuffled)
                print("")

                """Firms check if they need credit & randomly pick banks they apply for credit if so"""
                for f in f_ids:
                    f = int(f)
                    # loanRequired: firm f determines whether it needs credit or not
                    if Wb_a[f-1] > NW[f-1]: # if wage bills f has to pay > NW = 200*NWa = 200*np.mean(Y0) (larger than Net worth)
                        # firm will apply for credit
                        B[f-1] = Wb_a[f-1] - NW[f-1] # amount of credit needed
                        f_cr.append(f) # save firm id which needs credit
                        b_apply = np.random.choice(b_ids, self.H, replace=False) # choose random bank id firm f will apply for credit
                        credit_appl[f-1].append(b_apply) # setCreditAppl: save bank ids in list firm f (f-1) choose to apply to 
                        for bb in b_apply:
                            firms_applied_b[bb-1].append(f) # updateFirmsAppl: save firm id in list that applied to a bank 
                
                """searchAndMatchCredit"""
                # A: Banks decides on the interest rates, if there are some firms that need credit / loan
                if len(f_cr) > 0:
                    print("%d Firms need Credit: Banks set their interest rate:" %len(f_cr))
                    np.random.shuffle(b_ids)
                    for b in b_ids:
                        b = int(b-1)
                        # setCreditSupply - BRAUCHE ich nicht wenn ich equitiy initialise ??!!
                        if E[b] == 0: # if bank has no equity
                            Cr[b] = tCr[b] # then credit supplied is tCr
                        else:
                            Cr[b] = E[b] / self.capital_requirement_coef # amount of credit that can be supplied is an multiple of the equity base: E
                            # v is the capital requirement coefficient  => 1/v = maximum allowable leverage  
                        # Cr_a[b] = Cr_a[b] + 0 # setActualCreditSupplied  ???? 
                        Cr_rem[b] = Cr[b] - Cr_a[b] # setRemainingCredit - remaining credit does not change ?!
                        print("Bank %d has a credit supply of %f" %(b+1, Cr[b]))
                        f_appl = firms_applied_b[b] # getFirmsAppl: slice out firms applied to current bank b (b-1)
                        for f_a in f_appl:
                            if f_a is not None:
                                # getLeverageRatio: compute leverage of the firms applying for a loan 
                                if NW[f_a -1] <= 0: # in case firm has negative net worth: lv is random number 
                                    lv = np.random.uniform(low=1.5,high=2.0)
                                else:
                                    lv = B[f_a -1] / NW[f_a -1] # FK / EK 
                                # calcIntRate
                                phi[b] = np.random.uniform(low=0,high=self.H_phi) # calcPhi    
                                rt = self.r_bar*(1 + phi[b]*np.sqrt(lv)) # current interest rate
                                r[b].append(rt) # updateIntRates (append interest rates charged by bank b (list in list) to firm f_a)

                    # B: Firms choose the bank having lowest int. rate and going to next bank(s) if their credit req are not satisfied
                    for f_c in f_cr:
                        f_c = int(f_c) # current firm id 
                        r_fc = [] # initialize list to store interest rates of the banks firm f_c applied to (r bei ihm, da oob)
                        b_a_np = credit_appl[f_c-1][0] if len(credit_appl) > 0 else [] # getCreditAppl: bank ids that firm f_c applied to
                        # arrays inside list entry => slice out array if firm f_c applied for credit, otherwise empty list []; if list is empty: loop is not activated
                        for b in b_a_np:
                            b =int(b)
                            f_b = firms_applied_b[b-1] # getFirmsAppl (list of firms that applied to bank b_a)
                            r_b = r[b-1] # getIntRates (list of interest rates bank_b charges to all firms applied to bank b)
                            r_fc.append(r_b[f_b.index(f_c)]) # append interest rate offered to firm f_c by bank b 
                        r_fc_np = np.array(r_fc) # convert charged interest rates to numpy array
                        """Actual payment process in the following:"""
                        Cr_req = B[f_c-1] # getLoanAmount: credit demanded by firm f_c (f_c - 1 when slicing)
                        cr = 0 # initialize actual obtained credit / credit loaned by bank   
                        rem = Cr_req - cr # initialize remaining required credit of firm 
                        for bk in b_a_np:
                            i = np.argmin(r_fc_np) # extract place of the bank in the list with the lowest interest rate 
                            C_s = 0 # initialze credit supplied by the bank
                            C_r = Cr_rem[int(b_a_np[i])-1] # get ramining credit ampunt of bank (id) that supplied the lowest interest rate
                            if C_r >= rem: # if remaining amount of credit >= remaining required credit by firm f_c
                                C_s = C_s + rem # supplied credit by bank with lowest interest rate 
                                cr = cr + rem # actually obtainend credit amount (loaned amount) for firm f_c by bank with lowest interest rate
                                rem = Cr_req - cr # new remaining required credit by firm f_c is required credit amount - cr supplied by bank 
                                C_r = C_r - C_s  # new remaining credit amount by bank after subtracting the supplied credit amount
                                # addBanksAndRates: save the computations on the firm side
                                banks[f_c-1].append(int(b_a_np[i])) # append bank id that supplied lowest credit 
                                Bi[f_c-1].append(cr) # append updated received credit
                                r_f[f_c-1].append(r_fc_np[i]) # interest rate charged (r_np bei ihm) 
                                # addFirmsRatesCredits:  save the computations on the bank side
                                firms_loaned[int(b_a_np[i])-1].append(f_c) # save firm id the bank with lowest interest loaned money to 
                                Cr_d[int(b_a_np[i])-1].append(cr) # save credit amount given out to firm by bank (id) with lowest interest rate 
                                r_d[int(b_a_np[i])-1].append(r_fc_np[i]) # save interest rate charged to firm by bank with lowest interest rate
                            else: # if credit amount not sufficient: i.e. smaller than required amount by firm 
                                cr  = cr + C_r
                                rem = Cr_req - cr
                                C_s = C_s + C_r 
                                C_r = 0
                                # addBanksAndRates: save the computations on the firm side
                                banks[f_c-1].append(int(b_a_np[i])) 
                                Bi[f_c-1].append(cr) 
                                r_f[f_c-1].append(r_fc_np[i]) 
                                # addFirmsRatesCredits:  save the computations on the bank side
                                firms_loaned[int(b_a_np[i])-1].append(f_c) 
                                Cr_d[int(b_a_np[i])-1].append(cr) 
                                r_d[int(b_a_np[i])-1].append(r_fc_np[i]) 
                            #setActualCreditSupplied
                            Cr_a_current = Cr_a[int(b_a_np[i])-1] # extract the amount current bank has given out so far
                            Cr_a[int(b_a_np[i])-1] = Cr_a_current + C_s # increase amount by the credit supplied above 
                            # setRemainingCredit
                            Cr_rem[int(b_a_np[i])-1] = Cr[int(b_a_np[i])-1] - Cr_a[int(b_a_np[i])-1] 
                            # reset list with bank id firm applied to and list with sliced lowest interest rates
                            b_a_np = np.delete(b_a_np, i)
                            r_fc_np = np.delete(r_fc_np, i)
                            if rem == 0: # break loop if no more credit left to supply 
                                break
                            if len(b_a_np) == 0: # or break loop if no more banks left to get credit from 
                                break
                        # setLoanToBePaid: total amount of loan(s) (hence sum, if multiple banks lend out money): 
                        # loan = amount of credit payment(s) received + interest(s) firm has to pay
                        loan_to_be_paid[f_c-1] = np.around(np.sum( (np.array(Bi[f_c-1]))*(1 + np.array(r_f[f_c-1])) ),decimals=2)
                        print('loan_to_be_paid', loan_to_be_paid[f_c-1])
                else: 
                    print("No credit requirement in the economy")
                print("")            
                print("Credit Market CLOSED!!!!")
                print("")

                """If internal and external finances are not enough to pay for wage bills of HH: some random HH are fired
                before production starts"""
                # A: updateProductionDecisions
                hh_layed_off = [] # initialize list to store HH id's that will be fired in the follwing
                print("")
                print("Firms updating hiring decision.....")
                for f in range(self.Nf): # go through through all firms
                    if B[f] > 0:# if 
                        unsatisfied_cr =  B[f] - sum(Bi[f]) # amount of credit needed minus total amount (sum) of credit received by bank(s) 
                        # number of worker(s) that have to be fired is integer Lb s.t. Lb <= ratio of unsatisfied credit and wage payments
                        # hence one worker is fired if unsatisfied credit amount is a multiple of wage payments that have to be made before production can start
                        Lb = int(np.floor(unsatisfied_cr / Wp[f])) 
                        if Lb > 0:
                            h = w_emp[f]  # getEmployedHousehold: get employed HH's working at firm f
                            if len(h) >= Lb: # if firm has more employees than number of worker(s) firm has to fire
                                h_lo = np.random.choice(h, Lb, replace = False) # choose random worker(s) which are fired
                                h = [x for x in h if x not in h_lo] # update employed HH's (drop fired HH)
                            else: # if firm has as many workers as it has to fire
                                h_lo = h # worker(s) (HH id) that are fired
                                h = [] # firm has no more employed HH's
                            # updateEmployedHousehold
                            w_emp[f] = h # save new list of employed HH's at firm f
                            L[f] = len(w_emp[f]) # update labour employed
                            # save HH ids that got fired
                            for i in h_lo:
                                hh_layed_off.append(i)
                print("Firms succesfully updated hiring decisions")
                print("")
                # B: updateHouseholdEmpStatus
                for h in hh_layed_off: # set the new employment status of all fired HH's
                    prev_firm_id[h-1] = firm_id[h-1] # updatePrevEmployer: save firm id which fired HH and is now firm HH worked previously
                    firm_id[h-1] = 0   # setEmployedFirmId: HH not employed anymore, therefore id is 0
                    is_employed[h-1] = False # set employment status to false

                """4)
                Production takes one unit of time regardless of the scale of prod/firms.
                But before production can start wage payments of the firms have to be made to the HH's.
                Wage payments are directly subtracted from Net worth: 
                Since Firms either received enough credit or layed some HH's off if credit received was not enough, wage payments can all be paid
                although Net worth of firm becomes negative for the moment (in case firm went to a bank because NW not high enough). 
                Firm has to make enough revenue (gross profit) in the following goods market s.t. it can pay back bank (and dividends) and also has
                enough income to cover the wage payments (i.e. having positive net worth again). """

                print("Firms pay wages")
                # wagePayments
                # min_w = self.wage_lvl[t-1][mc] # wage level of period before is minimum wage (0 in t = 0)
                # brauche ich nicht: da bereits w_min oben bestimmt: min_w = w_min
                min_w = w_min # extract the current minimum wage of each simulation round 
                mw = 0 # initialize overall maximum wage that was paid in this round
                for f in range(self.Nf):
                    emp = w_emp[f] # employed HH's (list) at firm f
                    for e in emp:
                        w_flag[e-1] = 1 # setWageFlag: if HH has job (is in list) gets entry one
                        mw = max(mw, w[e-1]) # maximum wage paid updated in case wage payments to current worker e higher 
                    # wagePaid
                    W_pay[f] = Wp[f] # save wage payments firm has to make to each HH employed (wage payment per worker)
                    # WAGES FROM NW Abziehen: aber, nicht genug NW, deshalb auch Credit?? ??!!
                    # => erstmal negatives NW => dann mit revenue wieder NUR wieder positive: muss hoch genug sein, damit credit&zins zurückgezahlt werden kann
                    # & dividenden & dann noch plus.. 
                    # ERST NW - Wb_a EINBAUEN WENN SELBST GEBAUTES REV < CREDIT FERTIG KONTROLLIERT, ANSONTEN FIRMEN VLLT. ERSTMAL GENUG REV. 

                for c in range(self.Nh):
                    # getEmploymentStatus
                    if is_employed[c] == True:
                        # setUnempBenefits
                        unemp_benefit[c] = 0 # if HH c is employed, then she receives no unemployment benefits
                    else:
                        # if HH not employed: receives unemployment payment unequal to 0
                        if t == 0:
                            min_w = 0.5 * mw # minimum wage is half of maximum wage paid in the first round t = 0
                        unemp_benefit[c] = np.around(0.5 * min_w, decimals=2) # unemployment benefits are half of the current min. wage
                    # Sine HH's are either paid or receive unemployment payment before the goods market opens, they update their income accordingly 
                    Y[c] = np.around(Y[c] + w[c] + unemp_benefit[c], decimals = 2) # if wage = 0, then unemployment benefits are added or vice versa

                print("Firms producing....!")
                # doProduction
                for f in range(self.Nf):
                    # Qd[f] = np.around(Qd[f],decimals=2) #setDesiredQty  - needed??? : NO
                    Qs[f] = 0 # resetQtySold
                    # setActualQty: Firm compute 
                    alpha[f] =  np.around(np.random.uniform(low=5,high=6),decimals=2) # setProductivity: labor prductivity of each firm
                    # productivity is btw. 5 and 6 for each firm and remains around this level throughout the entire simulation (hence no aggregate Output growth) 
                    qp = alpha[f] * L[f] # productivity * labor employed = quantity produced 
                    Qp[f] = np.around(qp, decimals=2) # save the quantity produced rounded
                    Qr[f] = Qp[f] - Qs[f] # # setQtyRemaining: Initialize the remaining quantity by subtracting the quantity sold (currently 0, since goods market did not open yet)
                    # Setting initial prices in t = 0, otherwise set before in 1) 
                    if t == 0 and Qp[f] != 0:
                        p[f] = 1.5 * Wb_a[f] / Qp[f] # intial price are aggregated wage payments relative to quantity produced, times 1.5
                print("Wage payments and Production done!!!")
                print("")
                
                """5) 
                After production is completed, the goods market opens. Again, as in the labour- and credit market, search and match algorithm applies:
                Firm post their offer price. Consumers contact subset of randomly chosen firm acc to price and satisfy their demand.
                Goods with excess supply can't be stored in an inventory and they are disposed with no cost (no warehouse). """ 
                
                c_ids = np.array(range(self.Nh)) + 1 # consumer ids starting with 1 (not 0)
                f_ids = np.array(range(self.Nf)) + 1 # firm ids starting with 1 (not 0)
                savg = self.S_avg[t-1,mc] # slice average saving of last round (0 in t = 0)
                f_id = list(f_ids) # convert ids to list
                for f in f_ids:
                    if f is not None:
                        if Qp[f-1] <= 0: # remove firm id from list if quantity produced = 0
                            f_id.remove(f)
                print("")
                print("Consumption Goods market OPENS!!!")
                print("")
                np.random.shuffle(c_ids) # match and search: starts with first HH randomly
                f_out_of_stock = [] # initialize list for saving firm id's if out of stock (i.e. everything sold at certain point)
                
                """SearchandMatch on the goods market"""
                # A: Each consumer enters market sequentially and determines its demand and random number of firms she constacts in this round
                for ck in c_ids:
                    c = int(ck - 1) # current selected HH 
                    C_d_c = 0 # initialize desired demand of the current selected HH c (set to 0 again for each new consumer entering the market)
                    # Determine the desired consumption in this round
                    if t == 0:
                        C_d[c] = MPC[c]*Y[c]  # inital consumption demand of each HH = initial marginal product of consumption * weight updated intial income
                        C_d_c = C_d[c] # save initial desired consumption of current HH if t = 0
                    else:
                        MPC[c] = np.around(1/(1 + (np.tanh(S[c]/savg))**self.beta),decimals=2) # marginal propensity to consume
                        C_d[c] = np.around((Y[c])*MPC[c], decimals=2) # setDesiredCons
                        C_d_c = C_d[c] # getDesiredConsn
                        # ABER: im Buch ändert sich die MPC in jeder Periode.. NICHT?? (c_jt) - geben Parameter an
                    
                    # HH c chooses random firm to (potentially) buy products from
                    if len(f_id) > 0: # if there are any firms producing
                        if len(f_id) >= self.Z: # & if there are enough firms to be matched
                            select_f = np.random.choice(f_id, self.Z, replace = False) # select random list of firms to go to and buy products from 
                        else:
                            select_f = f_id # no random firms to choose because number of firms which produced = Z
                    else:
                        print("There was no production in the economy, hence HH cannot choose firms to buy consumption goods!!!")

                    # B: HH checks the prices of chosen firms & purchases if stock of chosen firms still greater 0
                    cs = 0 # initialize amount of consumption / purchased amount later from firm with lowest price 
                    c_rem = C_d_c - cs  # initializing current remaining consumption demand of HH c
                    # prev_cs =  0 # initialize ??? - not needed ??
                    # update selected firm list (delete choosen firm id) in case firm is out of stock (because preceeding HH's bought too much before)
                    select_f = [x for x in select_f if x not in f_out_of_stock] # updated selected firms (if previous selected firms out of stock by chance)
                    
                    # Get prices of choosen firm(s):
                    if len(select_f) > 0:
                        prices = [] 
                        for f in select_f: 
                            prices.append(p[f-1])
                        select_f_np = np.array(select_f) # numpy array of selected firms
                        prices = np.array(prices) # numpy array
                    
                        # HH purchases from selected firms, going through all firms and beginning with firm that offers smallest price, until either demand satisfied or all firms in list out of stock
                        for i in range(len(select_f_np)):
                            i = np.argmin(prices) # index position of prices of the firm that offers minimum price
                            pmin = np.min(prices) # minimum price
                            fi = select_f[i] # get firm id of selected firms (Z) that offers the minimum (lowest) price
                            Qrf = Qr[fi-1] # extract quantity remaining of firm that offers lowest price
                            print("firm %s"%fi, "current supply (nominal) %s ;"%Qrf, "HH %s"%ck, "rem cons %s"%c_rem)
                            if Qrf > 0.001: # if remaining quantity of firm with minimum price is slightly positive: 
                                # Supply of current firm: 
                                Qr_f = Qrf*pmin # real supply of firm fi is the current supply / stock * price of current firm
                                Qs_f = 0 # initialize amount of supplied quantity of firm fi to consumer c (after purchase)
                                # Purchase:
                                # Firms fi remaining quantity can satisfy remaining demand of HH c (c_rem): Fi satifies entire demand of consumer c
                                if Qr_f >= c_rem:
                                    cs = cs + c_rem # demanded amount / purchase / consumption of HH c
                                    Qs_f = c_rem # update quantity supplied by firm fi to HH c # Qs_f = Qs_f + c_rem
                                    c_rem = C_d_c - cs # remaining demand of HH = 0: desired Consumption of HH equals actual purchase / consumption 
                                    Qr_f = Qr_f - Qs_f  # subtract supplied quantity from remaining quantity to update remaining qunatity of firm fi
                                # Firm fi cannot fullfill entire demand of HH c:
                                else:
                                    cs = cs + Qr_f # consumption/ purchased amount by HH c is the left of the stock of firm fi 
                                    c_rem = C_d_c - cs # remaining demand
                                    Qs_f = Qr_f # supplied amount of firm fi # Qs_f = Qs_f + Qr_f
                                    Qr_f = 0 # no more remaining quantity of firm i
                                # prev_cs = cs - prev_cs # demanded amount of previous HH (c to c-1) when c takes on next random HH id NOT NEEDED??
                                # Update report variables:
                                Qs[fi-1] = Qs[fi-1] + np.around(Qs_f / pmin ,decimals=2) # setQtySold (real): add overall supplied quantity of firm fi
                                # subtract produced quantity by supplied (sold to c) qunatity and update remaining quantity
                                Qr[fi-1] = Qp[fi-1] - Qs[fi-1]  # setQtyRemaining
                                prices = np.delete(prices, i) # delete the price of the firm that sold to c out of price list, since either Firm sold everything or demand is satisfied
                                select_f = np.delete(select_f, i) # delete firm that sold to c form list of chosen firms (Z)
                            # append current firm id (fi) to list of firms without stock if stock went to 0
                            if Qr[fi-1] < 0.001:
                                f_out_of_stock.append(fi)
                                print("Firm %d out of stock!!!!!!" %(fi))
                                print("")
                            # c continues to purchase from (2nd and 3rd) firm until remaining demand = 0
                            if c_rem == 0:
                                break
                    
                    # setActualCons & updateSavings
                    C[c] = np.around(cs, decimals=2) # save the overall consumption of HH c
                    S[c] = np.around(Y[c] - C[c], decimals=2) # new savings are Income - Consumption of HH c
                print("Consumption Goods market CLOSED!!!")           
                print("")
                # Some HH don't consume at all => hence they have large amounts of savings in the following
                # checking MPC: S[c] / average => what range are values in (s.t. beta = 4 makes sense )
                # np.mean(S) = 60.04 in t = 0 => S[0]=22.53 / np.mean(S) = 0.37 => alpha = 0.98 
                # S[1] =  120.51 / np.mean(S) = 2.0070148090413094 => 0.54
        

                """6) 
                Firms collect revenue and calc gross profits. 
                If gross profits are high enough, they pay principal and interest to bank. 
                If net profits is positive, they pay dividends to owner of the firm (here no investments to increase productivity in next round). """
                
                # NOCHMAL DURCHGEHEN: Änderungen aufschreiben in brainstorm .. 

                # computeRevenues
                print("Firms calculating Revenues......")
                print("")
                for f in range(self.Nf):
                    Rev[f] = np.around(Qs[f]*p[f], decimals = 2)
                    Total_Rev[f] = Total_Rev[f] + Rev[f] # calcTotalRevenues: add revenue of this round from firm f to total revenue (of firm f)
                    Total_Rev[f] = Total_Rev[f] - Wb_a[f] # DAS DANN RAUS: schon vom net worth abgezogen in 4)
                    # subtract wage payments from revenue (since wage payments already happened to do production)
                    # FALSCH: Wage payments from NET worth abgezogen: mit oben einbauen !!!
                    # AUCH FALSCH: wage payments 
                print("Revenues Calculated!!!")
                print("")

                # settleDebts: Firms paying back principal and interest from the revenue of this round, if possible.
                # If firm made not enough revenue (gross profit) this round, the rest of the oustanding amount will be subtracted from its net worth in 7) 
                # i.e. firms accounts negative retained earnings (RE)
                # Then firms compute net Profit by subtracting wage payments and bank payments (if any) from their revenue (gross Profit) in this round.
                print("Firms settling Debts by paying back banks and determine their net profit")
                print("")
                for f in range(self.Nf):
                    banks_f = banks[f]# getBanks: id of bank(s) that supplied credit to firm f
                    credit = Bi[f] # getCredits: amount of credit firm f took from each bank 
                    rates = r_f[f] # getRates: interest rate(s) charged to firm i  (r bei ihm -> changed to r_f when initializing under firms!!!!)
                    # If total revenue is greater than the loan firm has to pay back: firm pays back all loan(s)
                    if Total_Rev[f] >= loan_to_be_paid[f]:
                        loan_paid[f] = np.around(loan_to_be_paid[f], decimals=2) #payLoan: loan to be paid by firm f (firm side)
                        for i in range(len(banks_f)):
                            Cr_p[int(banks_f[i])-1] = Cr_p[int(banks_f[i])-1] + (credit[i]*(rates[i]+1))  # setLoanPaid: Credit paid before (from e.g. other firm) + principal and interest (bank side)
                        # calcProfits: compute net profits by subtracting loan payments (credit and interest), wages already subtracted
                        P[f] = np.around(Total_Rev[f] - np.sum( (np.array(Bi[f]))*(np.array(r_f[f])) ) ,decimals=2) # net profits
                    else:
                        # if firm cannot pay back all loans, i.e. gross revenue smaller than total amount of loans firm has to pay back:
                        print("Firm %s cannot pay back all loans"%f)
                        loan_paid[f] = Total_Rev[f] # firm uses its entire revenue of this round to pay back as many loans as possible
                        amount_left = loan_paid[f] # save the amount firm has to pay back loans
                        for i in range(len(banks_f)):
                            # firm starts to pay back loans, starting with first bank: 
                            payback = (credit[i]*(rates[i]+1)) # amount firm has to pay back to current bank i
                            if amount_left >= payback: # If amount left to pay back loans > than loan & interest: firm pays back entire amount to current bank i
                                Cr_p[int(banks_f[i])-1] = Cr_p[int(banks_f[i])-1] + payback  # setLoanPaid
                                amount_left = amount_left - payback # update new amount left and check whether next bank can be paid back 
                            else:
                                # if amount left not enough to pay back entire credit to current bank, then the payment is just the amount left and payment loop stops
                                Cr_p[int(banks_f[i])-1] = Cr_p[int(banks_f[i])-1] + amount_left # setLoanPaid
                                outstanding[f].append(payback - amount_left) # compute outstanding amount firm f could no pay back to current bank i
                                outstanding_flag[f] = 1 # set a flag if firm has outstanding credit
                                outstanding_to_bank[f].append(banks_f[i])
                                while i < (len(banks_f)-1):
                                    # append the outstanding amounts of the other banks firm f could not pay back at all (in case firm borrowed from more than 1 firm)
                                    i = i + 1
                                    payback = (credit[i]*(rates[i]+1)) # amount firm has to pay back to each bank
                                    outstanding[f].append(payback)
                                    outstanding_to_bank[f].append(banks_f[i])
                                P[f] = 0 # net profits are zero, since entire revenue of this round is used to pay back the bank 
                                # The outstanding amount(s) will be subtracted from Net worth in 7), if firm has enough NW 
                                break
                # if no credit requirement in the economy, then no loans (principal) to be paid back, etc.: Hence lists are empty or entries remain 0
                print("Debts settled")
                print("")

                # payDividends:
                # If net profit is positive, firms determine their dividend payments (no investment here)
                n_div = 0 # initialize counter for case no dividend payments happen 
                for f in range(self.Nf):
                    divs_f = 0 # initialize current dividend of firm f (bei ihm auch divs)
                    Total_Rev[f] = 0 # set total revenue of current firm to 0: i.e. reset gross profits of this round 
                    if P[f] > 0:
                        # setDividends: If firm has positive net profit, it pays dividends 
                        divs[f] = np.around(P[f]*self.delta , decimals = 2) # dividends are profits * 0.5, if firm f has positive profits
                        divs_f = divs[f]
                    else:
                        n_div = n_div + 1
                        # dividends paid remain zero if firm f no positive profit: divs[f] = 0
                    # HH's receive the dividends
                    if divs_f > 0:
                        for c in range(self.Nh):
                            div[c] = div[c] + np.around(divs_f/self.Nh ,decimals=2)
                            # each HH gets share of profits (implied assumption that each HH has same share in each firm)
                            div_flag[c] = 1 # ?? if one firms has positive profits => then each HH automatically receives payment()
                print("Out of %d Firms, %d reported profits this period"%(self.Nf,self.Nf-n_div))

                """7) 
                Firms compute Retained Earnings, i.e. their Earnings after interest payments and dividend payments (zero investment here).
                They are added to the current net worth of each firm which are carried forward to next period. 
                HH's update their income after the dividend payments and Banks update their equity after principal and interest payments.

                If a firm could not pay back entire credit with its revenue, 
                FIRMs receive PAYMENTS sequentially !!
                
                ?? IST DAS MIT DRIN: If net worth (equity) is negative, firm (bank) exits the market in 8). 
                For a bankrupt firm a non performing loan have to be registered????
                ?? If firm could not pay back entire loan => then now subtracting from Net worth ??!!
                
                # -> hier: def calcRetainedEarnings(self) ; def getRetainedEarnings(self): ; getNetWorth(self) etc.
                # -> bei Ihm ALLES mit in 8 unter updateData in stats!!
                # hier: ERST COMPUTATIONS, Dann SAVING ("mein updateData")"""

               
                # WAGE PAYMENTS FROM NET WORTH ABZIEHEN bei FIRMEN?! - > macht er in 7) und 8.. : bei mir direkt oben !!
                # CHECK the OUTSTANDING FLAG => subtract missing payments from Net worth in case outstanding_flag = true .. 
                # if outstanding_flag == 1:
                    # subtract outstanding amount from Net worth .. 
                # Dividend payments after profits are realized
                
                # 1) Firms
                for f in range(self.Nf):
                    # if firm did not make enough profit to pay back entire amount of credit, NW becomes even more negative (since firm already paid workers)
                    if outstanding_flag[f] == 1:
                        credit = Bi[f] # getCredits: amount of credit firm f took from each bank 
                        outstanding_banks = outstanding_to_bank[f] # slice the bank id which firm could not pay back entirely (or only amounts)
                        outstanding_amounts = outstanding[f] # slice the outstanding amounts firm has 
                        for i in range(len(outstanding_banks)):
                            # again banks are paid back entirely 
                            current_outstanding = outstanding_amounts[i] # get the outstanding amount of the first firm
                            if NW[f] - Wb_a[f] > current_outstanding: 
                                # if firm has enough NW remaining such that it can pay back the rest of oustanding credit using net worth, firm does so
                                NW[f] = NW[f]- Wb_a[f] - current_outstanding
                                Cr_p[int(outstanding_banks[i])-1] = Cr_p[int(outstanding_banks[i])-1] + current_outstanding
                            else:
                                # if firm cannot pay back banks anymore using its net worth (wage payments subtracted): 
                                # Firm pays back everything she can with remaining Net worth 
                                Cr_p[int(outstanding_banks[i])-1] = Cr_p[int(outstanding_banks[i])-1] + NW[f]
                                # Bank computes bad debt: amount of net worth * share of externally financed net worth
                                share =  NW[f] / credit[i] 
                                Bad_debt[int(outstanding_banks[i])-1] = share * NW[f]
                                while i < (len(outstanding_banks)-1):
                                    # compute bad debt of other banks that also did not get any last payment from firm's net worth
                                    i = i + 1
                                    current_outstanding = outstanding_amounts[i] # get oustanding amount of the next bank
                                    share =  NW[f] / credit[i]  # compute the share that will be recorded as bad debt
                                    Bad_debt[int(outstanding_banks[i])-1] = share * NW[f]
                                    break
                            NW[f] = NW[f] - sum(outstanding[f]) # compute new net worth: since it is negative, firm will go bankrupt and exit in 8) 
                    # WANN SinD DIE WAGE PAYMENTS ???
                    # Eigentlich haben HH schon lohn bekommen => muss vom Net worth schon abgezogen sein .. ??!!
                    else:
                        # updateNetWorth
                        RE[f] = np.around(P[f] - div[f]) # calcRetainedEarnings - Earnings are profits - dividends paid to HH's 
                        # - BANK PAYMENTS???!!!!!
                        NW[f] = NW[f] + RE[f] 

                    # ???? 
                    # setMinPrice -> remains 0 so far !!!
                    if Qs[f] != 0: # if firm f sold products
                        p_low[f] = 0 # loan_to_be_paid[f] / Qs[f] 
                    else:
                        p_low[f] = 0 

                    # calcVacRate
                    vac[f] = vac[f] - L[f] # update vacancies (subtract current amount of labour employed at firm f)
                
                # 2) HH
                # (last) "computations"
                for c in range(self.Nh):
                    # update employment count
                    if is_employed[c] == True: # getEmploymentStatus
                        d_employed[c] = d_employed[c] + 1 # incrementEdays
                    else:
                        d_unemployed[c] = d_unemployed[c] + 1
                    # update income
                    Y[c] = np.around(S[c] + div[c] ) # Income is sum of savings, wage, unemplyoment benefits (if no wage) and dividends
                    # wenn wage schon paid before goods market: w[c] + unemp_benefit[c] hier RAUS !!!
                    # ODER SIND DIE WAGE PAYMENTS DOCH HIER ??!! + WB_a[c] + unemp_benefit[c]  ?? 
                
                # 3) Banks
                # (last) "computations"
                for b in range(self.Nb):
                    # updateEquity
                    E[b] = E[b] + Cr_p[b] - Bad_debt[b] # equity before + total amount of loans paid back 
                    # MINUS BAD DEBT fehlt / muss mit rein !! !!

                
                
            
                
                
                # im goods market: Propensity to consume of poorest people and richest people mit einbauen.. (ab t = 1?.
                # ABER: im Buch werden ändert sich die MPC in jeder Periode.. (c_jt)


                """Store individual data of all agents:
                Here all individual report variables gathered through the simulation are saved in an numpy 3-D array, i.e.
                indivdual data of each round is saved in a T+2x"DatatoStore" matrix for each simulation round.
                This can be switched off in case not needed (e.g. for calibration) to save memory space, which is done in the 'estimate' method below. 
                There overall memory is saved by not saving indidivudal data, but only data on aggregate report variables. """

                n_data = 0 # initialize running index
                # 1) HH
                for c in range(self.Nh):
                    # getActualCons
                    c_data[t, c, 2] = C[c] # save consumption of HH c
                    if C[c] > 0:
                        n_data = n_data + 1 # increase running index if consumption was positive in this round
                    # getMPC
                    if t != 0:
                        c_data[t, c, 4] = MPC[c]
                    c_data[t, c, 3] = S[c] # getSavings
                    c_data[t, c, 5] = C_d[c] # getDesiredCons
                    c_data[t, c, 6] = w[c] # getWage  
                    c_data[t, c, 7] = unemp_benefit[c] # getUnempBenefits  Code for the reservation Wages
                    c_data[t, c, 8] = prev_firm_id[c] # getPrevEmployer
                    c_data[t, c, 9] = firm_id[c] # getEmployedFirmId()
                    c_data[t, c, 10] = 1 if is_employed[c] == True else 0 # getEmploymentStatus
                    c_data[t, c, 11] = prev_F[c] # getPrevCFirm
                    c_data[t, c, 12] = div[c] # getDividends
                    c_data[t, c, 13] = d_employed[c] #getDaysEmployed
                    c_data[t, c, 14] = d_unemployed[c] # getDaysUnemployed
                    c_data[t+1, c, 1] = Y[c] # getIncome - income is generated randomly in first perio 
                print("Out of total %d Household, %d did consume"%(self.Nh,n_data))

                #2) Firms
                for f in range(self.Nf):
                    f_data[t, f, 1] = Qd[f] # getDesiredQty (0 in first round ??)
                    f_data[t, f ,2] = Qp[f] # getActualQty
                    f_data[t, f, 3] = Qs[f] # getQtySold
                    f_data[t, f, 4] = Qr[f] # getQtyRemaining
                    f_data[t, f, 5] = p[f] # getPrice
                    f_data[t, f, 6] = alpha[f] # getProductivity()
                    f_data[t, f, 7] = L[f] # getRequiredLabor
                    f_data[t, f, 8] = Lhat[f] # getNumberOfExpiredContract
                    f_data[t, f, 9] = L[f] # getTotalEmployees
                    f_data[t, f, 10] = W_pay[f] # getWage
                    f_data[t, f, 13] = Rev[f] # getRevenues
                    f_data[t, f, 18] = NW[f] # getNetWorth
                    f_data[t, f, 19] = vac[f] # getVacRate
                    f_data[t, f, 14] = loan_to_be_paid[f] # getLoanToBePaid           
                    f_data[t, f, 11] = Wb_d[f] # getDesiredWageBill
                    f_data[t, f, 12] = Wb_a[f] # getActualWageBill   
                    f_data[t, f, 15] = P[f] # getProfits
                    f_data[t, f, 16] = divs[f] # getDividends
                    f_data[t, f, 17] = RE[f] # getRetainedEarnings
                    #f_data[t, f, 20] = sum(amount_outstanding[f]) # getRetainedEarnings

                # 3) Banks
                for b in range(self.Nb):
                    b_data[t, b, 1] = E[b] # getEquity
                    b_data[t, b, 2] = Cr[b] # getCreditSupply
                    b_data[t, b, 3] = Cr_a[b] # getActualCreditSupplied
                    b_data[t, b, 4] = Cr_rem[b] # getRemainingCredit
                    # ??? where do firms pay back credit ??!!
                    b_data[t, b, 5] = np.sum(np.array(Cr_d[b])*np.array(r_d[b])) / np.sum(np.array(Cr_d[b])) if len(Cr_d[b]) > 0 else 0 
                    # sum of credit disbursed (ausgezahlt) * rates disbused (getRatesDisbursed) / sum of disbursecd credit (only if greater 0, else 0 )
                    b_data[t, b, 7] = Cr_p[b] # getLoanPaid()
                    b_data[t, b, 6] = b_data[t, b, 5] - b_data[t, b, 7] # disbursed credit - loan paid ??
                    # b_data[t, b, 8] = Bad_debt[b]
                

                """ Compute aggregate report variables (calcStatistics)
                # calcStatistics ist FILLING AGGREGATE VARIABLES
                # bei ihm unter 8
                # always include [][mc] beim slicen!! => get value for each timepoint t (and each mc run)
                
                # price level
                plvl = 0  # price * quantity sold 
                for f in range(Fc):
                    plvl = plvl + np.mean(f_data[:t+1, f, 5])*np.sum(f_data[:t+1,f, 3]) # mean of all prices of each firm * Qs of all firms (averaged over t) and then accumulated for each firm
                    # t+1 s.t. all vectors for each firm of all periods before are also used 
                plvl = plvl / np.sum(f_data[:t+1, :, 3]) # divide by overall quantity sold 
                P_lvl[t][mc] = plvl

                S_avg[t][mc] = np.mean(c_data[t, :, 3]) # average savings (mean of savings of all consumers in t)
                unemp_rate[t][mc] = 1 - (np.sum(c_data[t, :, 10])/Nh) # unemployment rate ( 1 - employment rate: sum of 1 if HH has job / number of HH)
                print("total produced:", np.sum(f_data[t, :,2]*f_data[t,:,5]), "total consumed:", np.sum(c_data[t, :,2]))
                avg_prod[t][mc] = np.sum(f_data[t,:,6]*f_data[t, :,9])/np.sum(f_data[t, :,9]) # sum of productivity * #empployees relative to # employees (for each t)

                inflation[t][mc] = P_lvl[t][mc]-P_lvl[t-1][mc] if t!=0 else 0 # / P_lvl[t-1][mc] missing ?? 
                inflation[t][mc] = inflation[t][mc]*100 

                wage_lvl[t][mc] = np.sum(f_data[t,:,10]*f_data[t, :,9])/np.sum(f_data[t, :,9]) # sum of wage paid * number of employees relative to number of employees
                wage_inflation[t][mc] = (wage_lvl[t][mc]-wage_lvl[t-1][mc])/np.sum(wage_lvl[t-1][mc]) if t!=0 else 0
                wage_inflation[t][mc] = wage_inflation[t][mc]*100 

                production[t][mc] = np.sum(f_data[t, :,2]) # sum of quantity produced of firms
                production_by_value[t][mc] = np.sum(f_data[t, :,2]*f_data[t, :,5]) # quantity produced * price (sum)
                consumption[t][mc] = np.sum(c_data[:,2]) # sum of consumption
                demand[t][mc] = np.sum(c_data[t, :,5]) # sum of desired consumption (total demand)
                hh_income[t][mc] = np.sum(c_data[t, :,1]) # sum of individual income
                hh_savings[t][mc] = np.sum(c_data[t, :3]) #  sum of invididual savings 
                hh_consumption[t][mc] = np.sum(c_data[t, :,2]) # hh_consumption = consumption ??

                print("total demand:", demand[t][mc], "avg Price and Wage lvl:", P_lvl[t][mc], wage_lvl[t][mc])

                # own
                consumption_rate[t][mc] = np.sum(c_data[t, :,2]) / Nh """




                """8) 
                Firms with positive net worth/equity survive, but otherwise firms/banks go bankrupt.
                New firms/banks enter the market of size smaller than average size of those who got bankrupt
                # checkForBankrupcy
                # ??? WHAT ABOUT FAILED BANKS ?? & does '4 times in a row' mean.. probability of failing
                # or does firm fail directly as soon as negative NW ?? !!"""
                # 8) New firms/banks enter the market of size smaller than average size ofthose who got bankrupt
                # checkForBankrupcy
                for f in range(Fc):
                    if NW[f] < 0:
                        # updateBankrupcyPeriod
                        bankrupcy_period[f] = bankrupcy_period[f] + 1
                    else:
                        bankrupcy_period[f] = 0 # resetBankrupcyPeriod - not needed because not updated ???
                        # if net worth is positive once, then period set to 0 again
                    # if firm had negative net worth 4 consecutive times
                    if bankrupcy_period[f] > 4: 
                        fid = f_data[t,f,0] # id of current firm
                        h_emp = w_emp[f]# getEmployedHousehold : HH's working at firm f
                        print("Firm %d has gone BANKRUPT!!!"%(fid))
                        # HH (ids in list) working at current firm need to update
                        for i in h_emp:
                            prev_firm_id[i-1] = fid # updatePrevEmployer
                            is_employed[i-1] = False # updateEmploymentStatus
                            firm_id[i-1] = 0 # setEmployedFirmId (HH no contract anymore => hence firm number that employed him set to 0)
                            # setUnempBenefits
                            unemp_benefit[i-1] = np.around(0.8 * w[i-1] ,decimals=2) # 80% of wage is the unemployment payment HH receives now (not 0.5, since HH worked in period before)
                            w[i-1] = 0 # HH receives no more wage
                            d_employed[i-1] = 0
                            firm_went_bankrupt[i-1] = 1 # OWN: set flag s.t. worker uses M instead of M - 1 when searching for job again 
                        # reset all individual report variables for firm f 
                        f_data[t+1,fid-1,:] = np.zeros((1,1,20)) 
                        # initialize new values of firm f
                        f_data[t+1, fid-1, 0] = fid # id

                #################################################################################################
                # Reset some variables 
                # WHICH VARIABLES TO RESET IN EACH t?? -> check the classes
                # resetTotalRevenue -> total revenue resetted in 7) div payments
                
                # 1) HH
                firms_applied = [[] for _ in range(Nh)]
                job_offers = [[] for _ in range(Nh)]
                w_flag = np.zeros(Nh)
                div_flag = np.zeros(Nh)
                div = np.zeros(Nh)

                # 2) Firms
                job_applicants = [[] for _ in range(Fc)]
                job_offered_to_HH = [[] for _ in range(Fc)]
                credit_appl = [[] for _ in range(Fc)]
                Lhat = np.zeros(Fc)
                loan_paid = np.zeros(Fc)
                Ld = np.zeros(Fc)
                banks = [[] for _ in range(Fc)]
                Bi = [[] for _ in range(Fc)]
                r = [[] for _ in range(Fc)]
                Rev = np.zeros(Fc)
                divs = np.zeros(Fc)
                loan_to_be_paid = np.zeros(Fc)
                P = np.zeros(Fc)
                RE = np.zeros(Fc)
                B = np.zeros(Fc)
                Wb_d = np.zeros(Fc)
                Wb_a = np.zeros(Fc)
                vac = np.zeros(Fc)

                outstanding = [[] for _ in range(self.Nf)] # under 6) if firm cannot pay back entire loan(s), amount outstanding for each bank (not paid back) is saved here
                outstanding_flag = np.zeros(self.Nf) # under 6), set to 1 if firm cannot pay back entire amount of loans
                outstanding_to_bank = [[] for _ in range(self.Nf)] # save the bank ids each firm has outstanding amount to 

                # 3) Banks
                Cr_p = np.zeros(Nb)
                firms_appl = [[] for _ in range(Nb)]
                firms_loaned = [[] for _ in range(Nb)]
                Cr_a = np.zeros(Nb)
                Cr_rem = np.zeros(Nb)

                Bad_debt = np.zeros(self.Nb)



            """Plotting main aggregate report variables if number of MC replications small enough:
            The plots are saved in folder: """
            if self.plots and self.MC <= 2:
                """plotting
                - analog zu ihm
                - erstmal nur die Plots, die auch im Buch sind bzw. die ich vergleichen kann"""
                









            
    
    
    
    




class BAM_base_estimate:
    
    """same code as above, but here less invidual data is stored along the way to save memory (fast computation, no plots etc.)
    and output is such that it can be used later """

    def __init__(self, MC:int, parameters:dict):
        
        self.MC = MC # Monte Carlo replications



            

    



    


        

    
