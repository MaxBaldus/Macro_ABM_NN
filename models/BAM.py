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
        """Parameter to be estimated"""
        self.Nh = parameters['Nh'] # number of HH
        self.Nf = parameters['Nf'] # number of firms - Fc in main_base
        self.Nb = parameters['Nb'] # number of banks
        self.T = parameters['T'] # simulation periods
        self.Z = parameters['Z'] # Number of trials in the goods market (number of firms HH's randomly chooses to buy goods from)
        self.M = parameters['M'] # Number of trials in the labor market (number of firms one HH applies to)
        self.H = parameters['H'] # Number of trials in the credit market (Number of banks a firm selects randomly to search for credit)
        self.H_eta = parameters['H_eta'] # Maximum growth rate of prices (max value of price update parameter) - h_eta in main_ase
        self.H_rho = parameters['H_rho'] # Maximum growth rate of quantities (Wage increase parameter)
        self.H_phi = parameters['H_phi'] # Maximum amount of banks’ costs
        self.h_zeta = parameters['h_zeta'] # Maximum growth rate of wages (Wage increase parameter)
        
        """!!Noch nicht mit drin!!"""
        self.c_P = parameters['c_P'] # Propensity to consume of poorest people
        self.c_R = parameters['c_R'] # Propensity to consume of richest people

        "Parameters set by modeller"
        self.beta = 4 # ???
        self.theta = 8 # Duration of individual contract
        self.r_bar = 0.4 # Base interest rate set by central bank (absent in this model)

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
            job_offers = [[] for _ in range(self.Nh)] # list for each HH that gets an offered job from firm i
            prev_F = np.zeros(self.Nh)

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
            Wb_d = np.zeros(self.Nf) # Desired Wage bills
            Wb_a = np.zeros(self.Nf) # Actual Wage bills
            Wp = np.zeros(self.Nf) # Wage level
            vac = np.zeros(self.Nf, int) # Vacancy
            W_pay = np.zeros(self.Nf) # Wage updated when paying
            job_applicants = [[] for _ in range(self.Nf)] # Ids of Job applicants (HH that applied for job) -> Liste für each firm => array with [] entries for each i
            job_offered = [[] for _ in range(self.Nf)] # Ids of selected candidates (HH that applied for job and which may be picked by firm i)
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

            # BANKS
            id_B = np.zeros(self.Nb, int) # (own) Bank id -> nur id in Bank.py
            E = np.zeros(self.Nb) # equity base 
            phi = np.zeros(self.Nb)
            tCr = np.zeros(self.Nb) # ???
            Cr = np.zeros(self.Nb) # amount of credit supplied / "Credit vacancies"
            Cr_a = np.zeros(self.Nb) # actual amount of credit supplied?
            Cr_rem = np.zeros(self.Nb) # credit remaining 
            Cr_d = [[] for _ in range(self.Nb)]
            r_d = [[] for _ in range(self.Nb)]
            r = [[] for _ in range(self.Nb)]
            firms_applied_b = [[] for _ in range(self.Nb)] # ACHTUNG: firm_appl in bank class
            firms_loaned = [[] for _ in range(self.Nb)] 
            Cr_p = np.zeros(self.Nb)


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
                    C_d[h] = MPC[h]*Y[h]  # inital consumption demand of each HH (marginal product of consumption * income)
                    # update Data
                    c_data[:,h,0] = h+1 # in all matrices id in first column
                    c_data[0,h,1] = Y[h] # initial income in t = 0
                    c_data[0,h,2] = MPC[h]*Y[h] # initial (actual) consumption in t = 0
                    c_data[0,h,4] = MPC[h] # 
                    c_data[0,h,5] = MPC[h]*Y[h] # initial desired consumption (same as C in t = 0)
            
            # FIRMS
            # DEBUGGING BZGL. hbyf etc. => erstmal als ones machen => dann füllen ??
            NWa = np.mean(Y0) # # ratio of HH and firms -> only needed for t == 0 respecting labour demand: labour demand Ld same for each firm in t = 0
            hbyf = self.Nh // self.Nf # ratio of HH and firms -> only needed for t == 0 respecting labour demand  NO VECTOR ! (household by firm)
            aa = np.ones(self.Nf) * (NWa *0.6) # minimum (required) wage -> given exogneous - same for each firm ??
            # NW = np.ones(self.Nf) * (NWa*200) # Net-Worth of a firm -> do I need seperat NWa for each firm ??
            # NW = NWa*200   # initial Net worth per firm !!!
            # NW = np.ones(self.Nf) * (NWa*200)  # WARUMS SCALING ???
            NW = np.ones(self.Nf) * (NWa)  # WARUMS SCALING ???
            for f in range(self.Nf):
                id_F[f] = f + 1 # set id 
                f_data[:,f,0] = f+1 # start with 1 for id
                Qty.append(Qd[f])

            # BANKS
            NB0 = np.random.uniform(low = 3000, high = 6000, size = self.Nb)
            aNB0 = np.mean(NB0)
            factor = self.Nb*9
            tCr = (NB0 / aNB0)*10*factor # ???
            for b in range(self.Nb):
                b_data[:,b,0] = b + 1


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
            column 10 = wage paid W_pay 
            column 11 = desired wage bill Wb_d
            column 12 = actual wage bill Wb_a 
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

                for i in range(self.Nf):
                    # range starts with 0
                    id_F[i] = i + 1 # OWN: set id_F (eig gar nicht nötig, da running index dafür läuft) ; start with 1, 
                    # setRequiredLabor <- adjust price and Y^D_it = D_ie^e
                    if t == 0:
                        Ld[i] = hbyf # labor demanded is HH per firm for each firm in first period => hence each firm same demand in first period 
                    # otherwise: each firm decides output and price => s.t. it can determine labour demanded
                    else: # t > 0 -> need price to set Qd <=> RequiredLabor
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
                # hired = 0 # for each firm ?  
                hired = np.zeros(self.Nf)
                # c_ids = np.array(range(self.Nh))  # c_d[:,0]  - e.g. H = 5 => array([1, 2, 3, 4, 5])
                c_ids = c_data[t,:,0].astype(int) # slice the consumer id's
                f_empl = [] # List of Firms who are going to post vacancies (id of firm that will post v)

                for i in range(self.Nf):
                    """Firms determine Vacancies and set Wages"""
                    # getTotalEmployees 
                    n = 0 # counter for the number of expired contracts of each firm i 
                    if L[i] >= 0: # if firm i has actual Labor Employed > 0 (not newly entered the market)
                        c_f = w_emp[i] # getEmployedHousehold - slice the list with worker id's (all workers) for firm i  : 
                        # getVacancies
                        rv_array = np.random.binomial(1,0.5,size=len(c_f)) # firm firing worker or not, 0.5 chance 
                        # 1x "coin-flip" (1x bernoulli RV) , size determines length of output
                        # e.g. s = np.random.binomial(1, 0.5, 10) yields array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])
                        for j in (c_f): # j takes on HH id employed within current firm i 
                            j = int(j-1) # python starts slicing with 1
                            if d_employed[j] > self.theta: # if contract expired (i.e. greater 8)
                                rv = rv_array[c_f.index(j)] # check if current worker j is fired or not (0 or 1 entry)
                                if rv == 1: # if worker is fired
                                    w_emp[i].remove(j) # removeHousehold  -> delete the current HH id in the current firm array 
                                    prev_firm_id[j] = i # updatePrevEmployer -> since HH is fired from firm i 
                                    is_employed[j] = False # updateEmploymentStatus
                                    firm_id[j] = 0 # setEmployedFirmId -> 0 now, because no employed firm
                                    w[j] = 0 # setWage -> no wage now
                                    d_employed[j] = 0 # 0 days employed from now on
                                    n = n + 1 # add expired contract
                    Lhat[i] = Lhat[i] + n # updateNumberOfExpiredContracts (add n if L[i] >= 0)
                    L[i] = len(w_emp[i]) # updateTotalEmployees -> length of list with workers id = number of wokers signed at firm i (!!!not at t-1????)
                    # calcVacancies -> of firm i
                    vac[i] = int((Ld[i] - L[i] + Lhat[i])*1) # labour demanded (depending on price) - labour employed + Labor whose contracts are expiring
                    if vac[i] > 0: 
                        f_empl.append(i+1) # if firm i has vacancies => then save firm_id (i) to list 
                    # setWage
                    zeta = np.random.uniform(low=0,high=self.h_zeta) # compute zeta for each firm
                    if f_data[t-1,i,3] == 0: # if firm did not produce last round ?????
                        zeta = 2*zeta # use 2*zeta   ????
                        w_min = aa[i] # minimum (required) wage of firm i  (INITIAL!!??) 
                        if vac[i] > 0: # if firm i has currently open vacancies:
                            Wp[i] = np.around(max(w_min, min(w_min*(np.random.uniform(low=1.01,high=1.10)),Wp[i]*(1+zeta/2))),decimals=2) # wage
                            is_hiring[i] = True 
                        Wp[i] = np.around(max(w_min, Wp[i]),decimals=2) # wage if firm i currently no open vacancies
                        is_hiring[i] = False  
                    # firm i uses zeta (not 2*zeta), if it had employees in previous round:  f_data[t-1,i,3] >= 0
                    w_min = aa[i] # minimum (required) wage of firm i (INITIAL!!??) 
                    if vac[i] > 0: # if firm has currently open vacancies:
                        Wp[i] = np.around(max(w_min, min(w_min*(np.random.uniform(low=1.01,high=1.10)),Wp[i]*(1+zeta/2))),decimals=2) # wage
                        is_hiring[i] = True 
                    Wp[i] = np.around(max(w_min, Wp[i]),decimals=2) # wage if firm i currently no open vacancies
                    is_hiring[i] = False  
                    """Finished: Firms posted Vacancies and Wage offers"""
                
                """SearchandMatch: 
                There must be at least 1 entry in the list with firm id's that have open vacancies: otherwise no searchandmatch """ 
                if len(f_empl) > 0: # if there are firms that offer labour: open vacancies (i.e. any firm id's in f_empl list), i.e. L^d > L 
                    
                    # A: Unemployed Households applying for the vacancies
                    c_ids =  [j for j in c_ids if is_employed[int(j-1)] == False]  # get Household id who are unemployed (i.e. is_employed is False)
                    print("%d Households are participating in the labor market (A:)" %len(c_ids))
                    np.random.shuffle(c_ids) # randomly selected household starts to apply (sequentially)
                    for j in c_ids:
                        j = int(j-1)
                        appl = None # To store firms (id) applied by this household
                        prev_emp = int(prev_firm_id[j]) #getPrevEmployer <- id of firm 
                        # Households always apply to previous employer FIRST (if) where the contract has expired
                        f_empl_c = [x for x in f_empl if x != prev_emp] # gather firm ids that have open vacancies (neglect firm if HH was just fired)
                        if len(f_empl) > self.M - 1: # if there are more than M-1 (3) firms with open vacancies
                            appl = list(np.random.choice(f_empl_c, self.M-1, replace = False)) # choose some random firm ids to apply to 
                            np.append(appl, prev_emp) # append also firm that just fired HH (last entry)
                        else: # otherwise: HH applies to the only firms hiring (with open vacancies)
                            appl = f_empl_c # save H-1 firm id's HH applies to
                            np.append(appl, prev_emp)
                        firms_applied[j].append(appl) # attach list with firm id's HH j will apply to (attach to its entry in list)
                        # Houshold j applies for the vacancies
                        for a in appl: # a takes on the firm id that HH applies to 
                            a = a - 1  # range of job_applicants starts with 0, but f_empl_c goes from 1 to 10 
                            job_applicants[a].append(j) # add HH id to the list of firm id's the HH applies to 
                                   
                    # B: Firms offer Jobs to randomly selected applicants (HH who applied)
                    for i in f_empl:
                        i = int(i - 1) 
                        # use the cacluated vacancies from before: vac   
                        applicants = job_applicants[i]# getJobApplicants (list of all HH's (id) applying to current firm i that employes)
                        n_applicants = len(applicants) # number of applicants
                        if vac[i] >= n_applicants:
                            # job shortage: if more vacancies than number of applicants => firm i accepts all applicants
                            job_offered[i] = applicants # HH's that offered job to firm i
                            # updateHouseholdJobOffers (offered jobs by firm (id) to HH)
                            for a in applicants:
                                # job_offers starts at 0 => use a - 1
                                job_offers[a-1].append(i) # save the firm id that offers job to HH which is currently applying 
                        else:
                            # more HH applying for jobs than firms offer vacancies => shuffle random number of applicants (HH id), according to vacancies of firm i
                            applicants = np.random.choice(applicants, int(vac[i]), replace=False) 
                            job_offered[i] = applicants # HH's id that offered job to firm i (HH -> firm)
                            # updateHouseholdJobOffers (offered jobs by firm (id) to HH)
                            for a in applicants:
                                job_offers[a-1].append(i) # save the firm id that offers job to HH which is currently applying 

                    # C: Household accepts if job offered is of highest wage
                    for l in c_ids:
                        l = int(l-1) # individual report variables start at 0 => reduce running index by 1, s.t. not going out of bounds
                        # getFirmsApplied: extract ids of firms HH j applied to (numpy array inside list)
                        # e.g. have [[9,17,7]] => need to slice with [0] to get the actual list in the following, avoid error if list is empty with [0]
                        f_applied_ids = firms_applied[l][0] if len(firms_applied[j]) > 0 else firms_applied[l]
                        f_job_offers = job_offers[l]  # getJobOffers:extract firm ids (or id) that offered job to HH j
                        f_e = np.empty(len(f_applied_ids)) # initialize vector of wages of the firms where HH l applied (use length of array entries inside list)
                        offer_wage = [] # initialize wage of offering firm that is searching
                        if len(f_applied_ids) != 0: # if HH applied to some firms (id's) (i.e. if there is array inside f_applied_ids)
                            ind = 0 # counter
                            for ii in f_applied_ids: 
                                f_e[ind] = Wp[ii-1] # extract the wages of the firms where HH applied (where HH applied)
                                ind += 1 # increase index
                                # HIER WEITER !!!!
                            for of in f_job_offers: # extract the wages the offering firm set before, for each firm that is searching
                                offer_wage.append(np.around(Wp[of-1] ,decimals=2))
                            w_max = max(f_e) # max. wage of firm the HH l applied to 
                            if w_max in offer_wage: 
                                # HIRED <- if maximum wage of firm HH applied to is in list of wages the searching firm pays => HH l is hired 
                                # update report variables
                                f_max_id = f_job_offers[offer_wage.index(w_max)] # save firm id for which match occured 
                                is_employed[l] = True
                                firm_id[l] = f_max_id + 1 # save the firm id where HH is employed (add one since Python starts counting at 0)
                                w[l] = np.around(w_max,decimals=2)
                                hired = hired + 1
                                w_emp[f_max_id].append(l) # save HH to list of firm that is employing
                                L[f_max_id] = len(w_emp[f_max_id])
                                # print("Household no: %d has got employed in Firm no: %d at wage %f"% (l, f_max_id+1, w_max))
                            elif t > 0 and len(offer_wage) > 0: # not in first round and if wage_offer list of offering firm not empty
                                mm = max(np.array(offer_wage)) # extract max. offered wage
                                if mm > self.wage_lvl[t-1][mc]: 
                                    # if maximum offered wage is larger than (average) wage_level of before => HH l accepts the job
                                    # (otherwise HH reamins unemployed)    
                                    # Update report variables
                                    f_max_id = f_job_offers[offer_wage.index(mm)] # save firm id for which match occured (i.e. that paid highest wage)
                                    is_employed[l] = True # change employment status
                                    firm_id[l] = f_max_id + 1 # save the firm id where HH is employed (add one since Python starts counting at 0)
                                    w[l] = np.around(w_max,decimals=2)
                                    hired = hired + 1
                                    w_emp[f_max_id].append(l) # save HH to list of firm that is employing
                                    L[f_max_id] = len(w_emp[f_max_id])
                                # print("Household no: %d has got employed in Firm no: %d at wage %f"% (l, f_max_id+1, w_max))
                else:
                    print("No Firm with open vacancies")
                print("")   
                print("Labor Market CLOSED!!!! with %d household hired!!"%(hired))
                print("")







                """Store individual data of all agents:
                Here all individual report variables gathered through the simulation are saved in an numpy 3-D array, i.e.
                indivdual data of each round is saved in a T+2x"DatatoStore" matrix for each simulation round.
                This can be switched off in case not needed (e.g. for calibration) to save memory space, which is done in the 'estimate' method below. 
                There memory is saved by not saving indidivudal data, but only aggregate report variables. """

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
                    f_data[t, f, 11] = Wb_d[f] # getDesiredWageBill()
                    f_data[t, f, 12] = Wb_a[f] # getActualWageBill()     
                    f_data[t, f, 15] = P[f] # getProfits()
                    f_data[t, f, 16] = divs[f] # getDividends()
                    f_data[t, f, 17] = RE[f] # getRetainedEarnings()

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



            """Plot main aggregate report variables and save them in FOLDER"""
            if self.plots and self.MC <= 2:
                "plot"
                









            
    
    
    
    




class BAM_base_estimate:
    
    """same code as above, but here less invidual data is stored along the way to save memory (fast computation, no plots etc.)
    and output is such that it can be used later """

    def __init__(self, MC:int, parameters:dict):
        
        self.MC = MC # Monte Carlo replications



            

    



    


        

    
