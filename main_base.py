import numpy as np

MC = 2 # number of MC simulations

#################################################################################################
# WORLD (hyper parameters)
#################################################################################################
Nh = 50 # mumber of Households - Nh 500 - umbenannt von H
Fc = 10 # number of consumption firms - Nf 100 
Nb = 5 # number of banks - Nb - umbenannt von B
T = 15 # simulation periods 100

#################################################################################################
# PARAMETERS
#################################################################################################
# Households
Z = 3 # number of trials in the goods market (number of firms HH randomly chooses to buy goods from) Z = 2 in book
M = 4 # number of firms HH apply to 
beta = 4

# Firms 
H = 2 # Number of banks a firm select randomly to search for credit 
h_eta = 0.1 # max value of price update parameter
h_pho = 0.1 # max value of qty update parameter
r_bar = 0.4 # Base interest rate set by central bank(absent in this model)
theta = 8 # Duration of contract
h_zeta = 0.01 # Wage increase parameter
delta = 0.5 # Dividend payments parameter

# Banks
h_phi = 0.1
mu = 0.11
r_bar = 0.04
###############################
# ...


#################################################################################################
# AGGREGATE REPORT VARIABLES
#################################################################################################
unemp_rate = np.zeros((T+2,MC)) # unemployment rate
S_avg = np.zeros((T+2,MC)) # ??
P_lvl = np.zeros((T+2,MC)) # price level
avg_prod = np.zeros((T+2,MC)) # average production level
inflation = np.zeros((T+2,MC)) # price inflation
wage_lvl = np.zeros((T+2,MC)) # wage level
wage_inflation = np.zeros((T+2,MC)) # wage inflation
production = np.zeros((T+2,MC)) # YY
production_by_value = np.zeros((T+2,MC)) # YY ??
consumption = np.zeros((T+2,MC)) # HH extra consumption ??
demand = np.zeros((T+2,MC)) # Demand
hh_income = np.zeros((T+2,MC)) # HH income
hh_savings = np.zeros((T+2,MC)) # HH savings
hh_consumption = np.zeros((T+2,MC)) # HH consumption ??

# own
consumption_rate = np.zeros((T+2,MC))

# KEINE MC mit DRIN: Ziel = TxMC (2-d array)
''' np.zeros((10,2))
array([[0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.]])'''

for mc in range(MC):
    print("MC run number: %s" %mc)

    # seed 
    np.random.seed(mc)
    
    #################################################################################################
    # INDIVIDUAL REPORT VARIABLES - Initialize & fill 
    #################################################################################################
    # HOUSEHOLDS
    # __init__  : "arrays"
    # 1xH (each column is for 1 HH -> is overwritten in every time step: otherwise TxH)
    id_H = np.zeros(Nh, int) # HH id -> nur id in Household.py
    Y = np.zeros(Nh) # fill with Y0 to initialize
    S = np.zeros(Nh)
    C = np.zeros(Nh)
    C_d = np.zeros(Nh) # compute in loop
    MPC = np.zeros(Nh)
    
    w = np.zeros(Nh)
    w_r = np.zeros(Nh)
    unemp_benefit = np.zeros(Nh)
    div = np.zeros(Nh)
    w_flag = np.zeros(Nh)
    div_flag = np.zeros(Nh)

    is_employed = np.zeros(Nh, bool) # e.g. H = 5 => array([False, False, False, False, False])
    d_employed = np.zeros(Nh)
    d_unemployed = np.zeros(Nh)
    firm_id = np.zeros(Nh) # firm_id where HH is employed
    prev_firm_id = np.zeros(Nh) # each HH only emplyed at 1 firm => 1d array reicht ..  
    firms_applied =  [[] for _ in range(Nh)] # list of firm ids HH applied to for each j?
    job_offers = [[] for _ in range(Nh)] # list for each HH that gets an offered job from firm i
    prev_F = np.zeros(Nh)
    
    # DATA to be stored??  (consumer data)
    c_data = np.zeros((T+2, Nh, 20))  # T+2 Hx20 matrices
    
    Y0 = np.random.uniform(low=50,high=100,size=Nh) # initial HH Income ??

    for h in range(Nh): # for each of the 500 HH
            MPC[h] = np.random.uniform(low=0.6,high=0.9) # initial MPC for each HH (then recomputed in each loop)
            # fill initial __init__ attributes that contain random numbers 
            id_H[h] = h + 1
            Y[h] = Y0[h]  # Y always recomputed in each loop 
            C_d[h] = MPC[h]*Y0[h]  # inital consumption demand of each HH
            # update Data ??
            c_data[:,h,0] = h+1 # in alle matrices id in first matrix: same for all t-rows
            c_data[0,h,1] = Y0[h] # Y0 value into 1.row, column of current HH, 2nd array
            c_data[0,h,2] = MPC[h]*Y0[h]
            c_data[0,h,4] = MPC[h]
            c_data[0,h,5] = MPC[h]*Y0[h]
            
    # FIRMS
    # __init__  : "arrays"
    # DEBUGGING BZGL. hbyf etc. => erstmal als ones machen => dann füllen ??
    id_F = np.zeros(Fc, int) # Firm id -> nur id in CFirm.py
    NWa = np.mean(Y0) # initial net worth value for each firm 
    # hbyf= np.ones(Fc) * (Nh // Fc) # each initial hbyf of firm is Floor Division - rounded to the next smallest whole number (same value) - only number needed!
    hbyf = Nh // Fc # ratio of HH and firms -> only needed for t == 0 respecting labour demand  NO VECTOR ! (household by firm)
    aa = np.ones(Fc) * (NWa *0.6) # minimum (required) wage -> given exogneous - same for each firm ??
    # aa = NWa * 0.6
    Qd = np.zeros(Fc) # Desired qty production : Y^D_it = D_ie^e
    Qp = np.zeros(Fc) # Actual Qty produced: Y_it
    Qs = np.zeros(Fc) # Qty sold
    Qr = Qp - Qs # Qty Remaining
    eta = np.random.uniform(low=0,high=h_eta, size = Fc) # Price update parameter
    p = np.zeros(Fc) # price
    pho = np.random.uniform(low=0,high=h_pho, size = Fc) # Qty update parameter
    bankrupcy_period = np.zeros(Fc)

    alpha = np.random.uniform(low=5,high=6, size = Fc) # Labor Productivity
    Ld = np.zeros(Fc) # Desired Labor to be employed
    Lhat = np.zeros(Fc) # Labor whose contracts are expiring
    L = np.zeros(Fc)  # actual Labor Employed
    Wb_d = np.zeros(Fc) # Desired Wage bills
    Wb_a = np.zeros(Fc) # Actual Wage bills
    Wp = np.zeros(Fc) # Wage level
    vac = np.zeros(Fc, int) # Vacancy
    W_pay = np.zeros(Fc) # Wage updated when paying
    job_applicants = [[] for _ in range(Fc)] # Ids of Job applicants (HH that applied for job) -> Liste für each firm => array with [] entries for each i
    job_offered = [[] for _ in range(Fc)] # Ids of selected candidates (HH that applied for job and which are picked now by firm i)
    ast_wage_t = np.zeros(Fc) # Previous wage offered
    w_emp = [[] for _ in range(Fc)] # Ids of workers/household employed - each firm has a list with worker ids
    is_hiring = np.zeros(Fc, bool) # Flag to be activated when firm enters labor market to hire 

    P = np.zeros(Fc) # Profits
    p_low = np.zeros(Fc) # Min price below which a firm cannot charge
    # NW = np.ones(Fc) * (NWa*200) # Net-Worth of a firm -> do I need seperat NWa for each firm ??
    # NW = NWa*200   # initial Net worth per firm !!!
    # NW = np.ones(Fc) * (NWa*200)  # WARUMS SCALING ???
    NW = np.ones(Fc) * (NWa)  # WARUMS SCALING ???
    Rev = np.zeros(Fc) # Revenues of a firm
    Total_Rev = np.zeros(Fc)
    RE = np.zeros(Fc) # Retained Earnings
    divs = np.zeros(Fc) # Dividend payments to the households

    B = np.zeros(Fc) # amount of credit (total)
    banks = [[] for _ in range(Fc)] # list of banks from where the firm got the loans -> LISTE FOR EACH FIRM (nested)
    Bi = [[] for _ in range(Fc)] # Amount of credit taken by banks
    r_f = [[] for _ in range(Fc)] # rate of Interest on credit by banks !! only self.r bei ihm 
    loan_paid = np.zeros(Fc)
    pay_loan = False # Flag to be activated when firm takes credit
    credit_appl = [[] for _ in range(Fc)] # list with bank ids each frim chooses randomly to apply for credit
    loan_to_be_paid = np.zeros(Fc) 

    Qty = [] # Quantity ?? 
    # DATA 
    f_data = np.zeros((T+2, Fc, 20))  # saving individual data of each firm 
    # column 0 = id
    # column 5 = prev_FPrice
    # cokumn 4 = prev_Inv 

    for f in range(Fc):
            id_F[f] = f + 1 # set id 
            f_data[:,f,0] = f+1 # start with 1
            Qty.append(Qd[f])
    
    # Banks
    # DATA 
    b_data = np.zeros((T+2, Nb, 20))  # saving individual bank data
    # column 0 = id

    # from world:
    NB0 = np.random.uniform(low = 3000, high = 6000, size = Nb)
    aNB0 = np.mean(NB0)
    factor = Nb*9
    for b in range(Nb):
        b_data[:,b,0] = b + 1

    # __init__  : "arrays"
    id_B = np.zeros(Nb, int) # (own) Bank id -> nur id in Bank.py
    E = np.zeros(Nb) # equity base 
    phi = np.zeros(Nb)
    tCr = (NB0 / aNB0)*10*factor # ???
    Cr = np.zeros(Nb) # amount of credit supplied / "Credit vacancies"
    Cr_a = np.zeros(Nb) # actual amount of credit supplied?
    Cr_rem = np.zeros(Nb) # credit remaining 
    Cr_d = [[] for _ in range(Nb)]
    r_d = [[] for _ in range(Nb)]
    r = [[] for _ in range(Nb)]
    firms_applied_b = [[] for _ in range(Nb)] # ACHTUNG: firm_appl in bank class
    firms_loaned = [[] for _ in range(Nb)] 
    Cr_p = np.zeros(Nb)
    

#>>> for i in range(5):
#...     print(i)
#... 
#0
#1
#2
#3
#4
# 5 excluded!!!

    for t in range(T):
        # variables filled with random numbers for t = 1
        # HH:  Y, MPC, C_d
        # Firms: eta, rho, alpha 
        print("time period equals %s" %t)
        #################################################################################################
        # 1) 
        # Firms decide output to be produced and thus labor required (desired), accordingly price or quantity is 
        # updated along with firms expected demand
        # -> mit in beginning of 2

        print("Labor Market opens")
        hired = 0 # for each firm ?  hired = np.zeros(Fc)
        c_ids = np.array(range(Nh))  # c_d[:,0]  - e.g. H = 5 => array([1, 2, 3, 4, 5])
        f_empl = [] # List of Firms who are going to post vacancies (id of firm that will post v)
        
        
        for i in range(Fc):
            # range starts with 0
            id_F[i] = i + 1 # OWN: set id_F (eig gar nicht nötig, da running index dafür läuft) ; start with 1, 
            # setRequiredLabor <- adjust price and Y^D_it = D_ie^e
            if t == 0:
                Ld[i] = hbyf # labor demanded is HH per firm for each firm in first period => hence each firm same demand in first period 
            # otherwise: each firm decides output and price => s.t. it can determine labour demanded
            else: # t > 0 -> need price to set Qd <=> RequiredLabor
                prev_avg_p = P_lvl[t-1][mc] # extract (current) average previous price 
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
            
            # 2) A fully decentr labor market opens. Firms post vacancies with wages. Workers approach subset of 
            # firms (randomly selected) acoording to wage offer. Laborr contract expires after finite period "theta"
            # and worker whose contract has expired applies to most recent firm first. Firm pays wage bills to start.

            # Firms determine Vacancies and set Wages
            
            # getTotalEmployees <- update
            n = 0 # counter for the number of expired contracts of each firm i 
            if L[i] >= 0: # if firm i has actual Labor Employed > 0 (not newly entered the market)
                # getEmployedHousehold
                c_f = w_emp[i] # slice the list with worker id's (all workers) for firm i  : 
                # getVacancies
                rv_array = np.random.binomial(1,0.5,size=len(c_f)) # firm firing worker or not, 0.5 chance 
                # 1x "coin-flip" (1x bernoulli RV) , size determines length of output
                # e.g. s = np.random.binomial(1, 0.5, 10) yields array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])
                for j in (c_f): # j takes on HH id employed within current firm i 
                    if d_employed[j] > theta: # if contract expired (i.e. greater 8)
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
            zeta = np.random.uniform(low=0,high=h_zeta) # compute zeta for each firm
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
            # Finished: Firms posted Vacancies and Wage offers
        
        # SearchandMatch 
        if len(f_empl) > 0: # if there are firms that offer labour: open vacancies (i.e. any firm id's in f_empl list), i.e. L^d > L 
        # -> otherwise no "searchandmatch"
            # A: Unemployed Households applying for the vacancies
            c_ids =  [j for j in c_ids if is_employed[int(j)] == False]  # get Household id who are unemployed (i.e. is_employed is False)
            print("%d Households are participating in the labor market (A:)" %len(c_ids))
            np.random.shuffle(c_ids) # randomly selected household starts to apply (sequentially)
            for j in c_ids:
                j = int(j)
                appl = None # To store firms (id) applied by this household
                prev_emp = int(prev_firm_id[j]) #getPrevEmployer <- id of firm 
                # Households always apply to previous employer FIRST (if) where the contract has expired
                f_empl_c = [x for x in f_empl if x != prev_emp] # gather firm ids that have open vacancies (neglect firm if HH was just fired)
                if len(f_empl) > M - 1: # if there are more than M-1(3) firms with open vacancies
                    appl = np.random.choice(f_empl_c, M-1, replace = False) # choose some random firm ids to apply to 
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
            print("B:")
            for i in f_empl:
                i = int(i - 1) # f_empl goes from 1 to 10 (job_applicants starts at 0)
                # use the cacluated vacancies from before: vac   
                applicants = job_applicants[i]# getJobApplicants (list of all HH (id) applying to current firm i)
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
                        job_offers[a].append(i) # save the firm id that offers job to HH which is currently applying 
            
            # C: Household accepts if job offered is of highest wage
            print("C:")
            for l in c_ids:
                l = int(l-1) # INDIVIDUAL REPORT VARIABLES start at 0 => reduce running index by 1, s.t. not going out of bounds
                f_applied_ids = firms_applied[l] # getFirmsApplied <- extract ids of firms HH j applied to (numpy array inside list)
                #f_applied_ids = [x-1 for x in firms_applied[l]] 
                # hence have [array([9,17,7])] => need to slice with [0] to get the actual array in the following!!
                f_job_offers = job_offers[l] # getJobOffers <- extract firm ids (or id) that offered job to HH j
                #f_job_offers = [x-1 for x in job_offers[l]] 
                # f_e = np.zeros(f_applied_ids[0].size) # initialize the wages of the firms where HH applied (use length of array entries inside list)
                f_e = np.empty(len(f_applied_ids))
                # use [0] to enter numpy array in the list
                # NEED TO RESET F_APPLIED_IDS? -> oder werden die irgendwo abgespeichert?
                offer_wage = [] # wage of offering firm that is searching
                if len(f_applied_ids) != 0: # if HH applied to some firms (id's) (i.e. if there is array inside f_applied_ids)
                    ind = 0 # counter
                    for ii in f_applied_ids[0]: # extract the wages of the firms where HH applied (where HH applied)
                        # hence have [array([9,17,7])] => need to slice with [0] to get the actual array elements
                        # f_e = Wp[ii-1] # slice Wp starting with 0
                        f_e[ind] = Wp[ii-1]
                        # ind += 1 
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
                        print("Household no: %d has got employed in Firm no: %d at wage %f"% (l, f_max_id+1, w_max))
                    elif t > 0 and len(offer_wage) > 0: # not in first round and if wage_offer list of offering firm not empty
                        mm = max(np.array(offer_wage)) # extract max. offered wage
                        if mm > wage_lvl[t-1][mc]: 
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
                        print("Household no: %d has got employed in Firm no: %d at wage %f"% (l, f_max_id+1, w_max))
        else:
            print("No Firm with open vacancies")
        print("")   
        print("Labor Market CLOSED!!!! with %d household hired!!"%(hired))
        print("")
        # setActualWageBill
        for f in range(Fc):
            Wb_a[f] = Wp[f] * L[f]  # wage bills = wages * labour employed
        print("Firms calculated Wage bills!!")

        #################################################################################################
        # 3) If there is a financing gap, firms go to credit market. 
        # They go to random chosen bank to get loans starting with one having lowest interest rate. 
        # Banks sort borrowers application according financial condition and satisfy untill exhaustion of credit supply.
        # Interest rate is calc acc to markup on an baseline rate.
        # After credit market closed, if firms are short, they fire workers or dont accept them.
        
        # searchAndMatchCredit
        # parameter H (number of banks a firm asks for credit, here H = 2)
        b_ids = np.array(range(Nb)) + 1 # bank identities
        f_ids = np.array(range(Fc)) + 1 # firm identities
        f_cr = [] # initialize list with id's of firms which want credit
        np.random.shuffle(f_ids)
        print("")
        print("Credit Market OPENS!!!!") # Firms apply for Credit (randomly, hence shuffled)
        print("")

        # Firms check if they need credit & randomly pick banks they apply for credit if so
        for f in f_ids:
            f = int(f)
            # loanRequired -> firm determines whether it needs credit or not
            # use f-1, since python starts counting at 0
            if Wb_a[f-1] > NW[f-1]: # if wage bills f has to pay > NW = 200*NWa = 200*np.mean(Y0) (larger than Net worth)
                # firm will apply for credit
                B[f] = Wb_a[f-1] - NW # amount of credit needed
                f_cr.append(f-1) # save firm id which needs credit
                b_apply = np.random.choice(b_ids, H, replace=False) # choose random bank id firm will apply for credit
                credit_appl[f-1].append(b_apply) # setCreditAppl - save bank ids in list firm will apply to 
                firms_applied_b[f] # updateFirmsAppl - save firm id in list that applied to a bank 
            else:
                B[f-1] = 0 
        
        # Banks decides on the interest rates, if Firm needs credits
        if len(f_cr) > 0:
            np.random.shuffle(b_ids)
            for b in b_ids:
                b = int(b)
                # setCreditSupply
                if E[b] == 0: # if bank has no equity
                    Cr[b] = tCr[b] # then credit supplied is tCr
                else:
                    Cr[b] = E[b] / 0.11 # amount of credit supplied as an multiple of the equity base: 
                    # v is the capital requirement coefficient  => 1/v = maximum allowable leverage  
                Cr_a[b] = Cr_a[b] + 0 # setActualCreditSupplied 
                Cr_rem[b] = Cr[b] - Cr_a[b] # setRemainingCredit
                print("Bank %d has a credit supply of %f" %(b, Cr[b]))
                # getFirmsAppl
                for f_a in firms_applied_b:
                    if f_a is not None:
                        # getLeverageRatio
                        if NW[f_a] <= 0: # in case firm has negative net worth
                            lv = np.random.uniform(low=1.5,high=2.0)
                        else:
                            lv = B[f_a] / NW[f_a] # FK / EK 
                        # calcIntRate
                        phi[b] = np.random.uniform(low=0,high=h_phi) # calcPhi    
                        rt = r_bar*(1 + phi[b]*np.sqrt(lv)) # current interest rate
                        r[b].append(rt) # updateIntRates (append interest rate charged by bank b to firm f_a)

            # Firms chooses the bank having lowest int rate and goes to second bank if its credit req are not satisfied
            for f_c in f_cr:
                f_c = int(f_c) # save current firm id 
                r_fc = [] # initialize list to store interest rates of the banks firm f_c applied to (r bei ihm, da oob)
                b_a =  credit_appl[f_c-1] # getCreditAppl (bank ids that firm f_c applied to)    
                b_a_np = np.array(b_a) # convert list with bank ids that f_c applied to to numpy array
                for b in b_a:
                    b =int(b)
                    f_b = firms_applied_b[b-1] # getFirmsAppl (list of firms that applied to bank b_a)
                    r_b = r[b-1] # getIntRates (list of interest rates bank_b charges to all firms applied to bank b)
                    r_fc.append(r_b[f_b.index(f_c)]) # append interest rate offered to firm f_c
                r_fc_np = np.array(r_fc) # convert to numpy array
                Cr_req = B[f_c] # getLoanAmount
                cr = 0 # initialize new remaining credit 
                rem = Cr_req - cr # initialize remaining credit
                for bk in b_a_np:
                    i = np.argmin(r_fc_np) # extract lowest interest rate of banks firm f_c applied to
                    C_s = 0 # initialze ???
                    C_r = Cr_rem[int(b_a[i])-1] # get ramining credit of bank (id) that supplied the lowest interest rate
                    if C_r >= rem:
                        C_s = C_s + rem # credit supply left
                        cr = cr + rem # new remaining credit
                        rem = Cr_req - cr # remaining credit
                        C_r = C_r - C_s   
                        # addBanksAndRates
                        banks[f_c-1].append(int(b_a[i])) # append bank id that supplied lowest credit 
                        Bi[f_c-1].append(cr) # append new remaining credit
                        r_f[f_c-1].append(r_fc_np[i]) # interest on credit charged (r_np bei ihm) 
                        # addFirmsRatesCredits
                        firms_loaned[int(b_a[i])-1].append(f_c) # save firm id bank with lowest interest loaned money to 
                        Cr_d[int(b_a[i])-1].append(cr) # save credit demanded of bank (id) with lowest interest rate 
                        r_d[int(b_a[i])-1].append(r_fc_np[i]) # save interest rate charged to firm by bank with lowest interest rate
                    else:
                        cr  = cr + C_r
                        rem = Cr_req - cr
                        C_s = C_s + C_r 
                        C_r = 0
                        # addBanksAndRates
                        banks[f_c-1].append(int(b_a[i])) # append bank id that supplied lowest credit 
                        Bi[f_c-1].append(cr) # append new remaining credit
                        r_f[f_c-1].append(r_fc_np[i]) # interest on credit charged (r_np bei ihm) 
                        # addFirmsRatesCredits
                        firms_loaned[int(b_a[i])-1].append(f_c) # save firm id bank with lowest interest loaned money to 
                        Cr_d[int(b_a[i])-1].append(cr) # save credit demanded of bank (id) with lowest interest rate 
                        r_d[int(b_a[i])-1].append(r_fc_np[i]) # save interest rate charged to firm by bank with lowest interest rate
                    #setActualCreditSupplied
                    Cr_a_current = Cr_a[int(b_a[i])-1]
                    Cr_a[int(b_a[i])-1] = Cr_a_current + C_s
                    # setRemainingCredit
                    Cr_rem[int(b_a[i])-1] = Cr[int(b_a[i])-1] - Cr_a[int(b_a[i])-1] 
                    # reset list with bank id firm applied to and 
                    b_a = np.delete(b_a, i)
                    r_fc_np = np.delete(r_fc_np, i)
                    if rem == 0: # break loop if no more credit left to supply 
                        break
                    if len(b_a) == 0: # no more firms applying to any banks 
                        break
                # setLoanToBePaid (loan = amount of credit + interest firm has to pay)
                loan_to_be_paid[f_c-1] = np.around(np.sum((np.array(Bi[f_c-1]))*(1 + np.array(r_f))),decimals=2)
                # print(loan_to_be_paid)

        else: 
            print("No credit requirement in the economy")
        print("")            
        print("Credit Market CLOSED!!!!")
        print("")

        # updateProductionDecisions
        hh_layed_off = []
        print("")
        print("Firms updating hiring decision.....")
        for f in range(Fc):
            if B[f] > 0:# if loan amount of firm > 0 (if firm got a loan)
               unsatisfied_cr =  B[f] - sum(Bi[f]) # total amount of credit - amount of credit taken by banks ??
               Lb = int(np.floor(unsatisfied_cr / Wp[f])) 
               if Lb > 0:
                   h = w_emp[f]  # getEmployedHousehold (employed HH working at firm f)
                   if len(h) >= Lb: # if more employees than firm can pay => firm has to fire some worker(s)
                       h_lo = np.random.choice(h, Lb, replace = False)
                       h = [x for x in h if x not in h_lo] # update emplyed HH (drop fired HH)
                   else: # if firm can pay all employed workers
                       h_lo = h 
                       h = [] 
                   # updateEmployedHousehold
                   w_emp[f] = h# ??? does python change list with new list (replace all entries in list) (works: tested on shell)
                   L[f] = len(w_emp[f]) # update labour emplpyed
                   # save HH (id) that got fired
                   for i in h_lo:
                       hh_layed_off.append(i)
        print("Firms succesfully updated hiring decisions")
        print("")
        print("")
        # updateHouseholdEmpStatus
        for h in hh_layed_off:
            prev_firm_id[h-1] = firm_id[h-1] # updatePrevEmployer (save firm id which fired HH h)
            firm_id[h-1] = 0   # setEmployedFirmId (HH not employed anymore)
            is_employed[h-1] = False # set employment status to false

        #################################################################################################
        # 4) Production takes one unit of time regardless of the scale of prod/firms
        # doProduction
        print("")
        print("Firms producing....!")
        for f in range(Fc):
            Qd[f] = np.around(Qd[f],decimals=2) #setDesiredQty  - needed???
            Qs[f] = 0 # resetQtySold
            # setActualQty
            alpha[f] =  np.around(np.random.uniform(low=5,high=6),decimals=2) # setProductivity (labor prductivity of each firm)
            qp = alpha[f] * L[f] # productivity * labor employed = quantity produced 
            Qp[f] = np.around(qp, decimals=2) # quantity produced rounded
            # setQtyRemaining
            Qr[f] = Qp[f] - Qs[f]
            if t == 0 and Qp[f] != 0:
                p[f] = 1.5 * Wb_a[f] / Qp[f] # some initial prices reamin 0, if no production! => hence in CMarket => some firm
        print("Production DONE!!!")
        print("")

        #################################################################################################
        # 5) After production is completed, goods market opens. 
        # Firm post their offer price, 
        # consumers contact subset of randomly chosen firm acc to price and satisfy their demand 
        # goods with excess supply can't be stored in an inventory and they are disposed with no cost. 
        
        # searchAndMatchCGoods
        # parameter Z 
        c_ids = np.array(range(Nh)) + 1 # consumer ids starting with 1 (not 0)
        f_ids = np.array(range(Fc)) + 1 # firm ids starting with 1 (not 0)
        savg = S_avg[t-1,mc] # can slice in first round when t = 0 -> yes, because T+2 initialized ??
        f_id = list(f_ids) # convert ids to list
        for f in f_ids:
            if f is not None:
                # remove firm id from list if quantity produced = 0
                if Qp[f-1] <= 0: # start slicing with first element (python starts at 0)
                    f_id.remove(f)
        print("")
        print("Consumption Goods market OPENS!!!")
        print("")
        np.random.shuffle(c_ids) 
        f_out_of_stock = [] # initialize list

        for c in c_ids:
            # for each consumer / HH c sequentially (potentially first come first serve when buying products)
            c = c -1 # need to start slicing with 0, c_ids starts with 1
            C_d_c = 0 # initialize desired demand of current HH - bei ihm C_d
            # getDesiredConsn
            if t == 0:
                C_d_c = C_d[c] # save desired consumption of current HH
            else:
                MPC[c] = np.around(1/(1 + (np.tanh(S[c]/savg))**beta),decimals=2) # calcMPC
                C_d[c] = np.around((Y[c])*MPC[c], decimals=2) # setDesiredCons
                C_d_c = C_d[c] # getDesiredConsn
            
            # HH c chooses random firm to (potentially) buy products from
            if len(f_id) > 0:
                # if there are any firms producing
                if len(f_id) >= Z:
                    # if there are enough firms to be matched
                    select_f = np.random.choice(f_id, Z, replace = False) # select random list of firms to go to to buy products
                else:
                    select_f = f_id # no random firms to choose because number of firms which produced = Z
            else:
                print("There was no production in the economy, hence HH cannot choose firms to buy consumption goods!!!")

            # initialize
            cs = 0
            c_rem = C_d_c - cs  # remaining consumption demand
            prev_cs =  0 
            # update selected firm list (delete choosen firm id) in case firm is out of stock (because preceeding HH's bought too much before)
            select_f = [x for x in select_f if x not in f_out_of_stock] # rausgenommen: and f_array[x-1] is not None
            
            # get prices of choosen firm(s)
            if len(select_f) > 0:
                prices = [] 
                for f in select_f: 
                    prices.append(p[f-1])

                # convert to numpy arrays
                select_f_np = np.array(select_f) # numpy array of selected firms
                prices = np.array(prices)
            
                # Purchase
                for i in range(len(select_f_np)):
                    i = np.argmin(prices) # firm id that offers minimum price
                    pmin = np.min(prices) # minimum price
                    fi = select_f[i] # get firm id (index) of selected firms (Z) that offers the minimum (lowest) price
                    Qrf = Qr[fi-1] # extract quantity remaining of firm that offers lowest price
                    if Qrf > 0.001: # if remaining quantity of firm with minimum price is slightly positive
                        Qr_f = Qrf*pmin # update remaining quantiy by multiplying with price of current firm
                        Qs_f = 0 # initialize supplied quantity of firm fi
                        # if firms remaining quantity can satisfy remaining demand of HH c (c_rem) 
                        if Qr_f >= c_rem:
                            cs = cs + c_rem # demanded amount (purchase) of HH c
                            Qs_f = Qs_f + c_rem # update current supply of firm fi
                            c_rem = C_d_c - cs # remaining demand of HH 
                            Qr_f = Qr_f - Qs_f  # subtract supplied quantity from remaining quantity to update remaining qunatity of firm fi
                        # firm fi cannot fullfill entire demand of HH c
                        else:
                            cs = cs + Qr_f # demanded (purchased amount) by HH c
                            c_rem = C_d_c - cs # remaining demand
                            Qs_f = Qs_f + Qr_f # supplied amount of firm fi
                            Qr_f = 0 # no more remaining quantity of firm i
                        prev_cs = cs - prev_cs # demanded amount of previous HH (c to c-1) when c takes on next random HH id
                        # add overall supplied quantity of firm fi
                        Qs[fi-1] = Qs[fi-1] + np.around(Qs_f / pmin ,decimals=2) # setQtySold (real)
                        # subtract produced quantity by supplied (sold to c) qunatity and update remaining quantity
                        Qr[fi-1] = Qp[fi-1] - Qs[fi-1]  # setQtyRemaining
                        prices = np.delete(prices, i) # delete the price of the firm that sold to c
                        select_f = np.delete(select_f, i) # delete firm that sold to c form list of chosen firms (Z)
                    # append current firm id (fi) to list of firms without stock if stock went to 0
                    if Qr[fi-1] < 0.001:
                        f_out_of_stock.append(fi)
                    # c continues to purchase from (2nd and 3rd) firm until remaining demand = 0
                    if c_rem == 0:
                        break
            
            # setActualCons & updateSavings
            C[c] = np.around(cs, decimals=2) # save the overall consumption of HH c
            S[c] = np.around(Y[c] - C[c], decimals=2) # new savings are Income - Consumption of HH c
        
        print("Consumption Goods market CLOSED!!!")           
        print("")
        
        #################################################################################################
        # 6) Firms collect revenue and calc gross profits. 
        # If gross profits are high enough, they pay principal and interest to bank. 
        # If net profits is +ve, they pay dividends to owner of the firm 
        # (bei ihm noch unter searchAndMatchCGoods)

        # computeRevenues
        print("Firms calculating Revenues......")
        for f in range(Fc):
            Rev[f] = np.around(Qs[f]*p[f], decimals = 2)
            # calcTotalRevenues
            Total_Rev[f] = Total_Rev[f] + Rev[f] # add revenue of this round from firm f to total revenue (of firm f)
        print("")
        print("Revenues Calculated!!!")
        print("")
        
        # wagePayments  
        # -> laut Buch paid when production starts!! ? (in 2: labour market??)
        # bei ihm unter 7
        min_w = wage_lvl[t-1][mc] # wage level of period before is minimum wage
        mw = 0 # initialize maximum wage
        for f in range(Fc):
            emp = w_emp[f] # emplyed HH's (list) at firm f
            for e in emp:
                w_flag[e-1] = 1 # setWageFlag  - HH in list with 1 receives wage
                mw = max(mw, w[e]) # maximum wage paid 
            # wagePaid
            W_pay[f] = Wp[f] # save wage payments
        for c in range(Nh):
            # getEmploymentStatus
            if is_employed[c] == True:
                # setUnempBenefits
                unemp_benefit[c] = 0 # if HH c is employed => then she receives no unemp_benefits
            else:
                if t == 0:
                    min_w = 0.5 * mw # minimum wage is half of maximum wage in first round (otherwise minimum wage would be 0)
                unemp_benefit[c] = np.around(0.5 * min_w, decimals=2)
            
        #settleDebts
        # bei ihm unter 7
        print("Firms settling DEBTS......")
        for f in range(Fc):
            # if no credit requirement in the economy => then no loans (principal) to be paid back, etc. => lists are empty or entries remain 0
            loan_paid[f] = np.around(loan_to_be_paid[f], decimals=2) #payLoan - loan to be paid by firm f
            banks_f = banks[f]# getBanks (auch banks bei ihm) - id of bank(s) that supplied credit to firm f
            credit = Bi[f] # getCredits - amount of credit firm f took from each bank 
            rates = r_f[f] # getRates - interest rate(s) charged to firm i  (r bei ihm -> changed to r_f when initializing under firms!!!!)
            for i in range(len(banks_f)):
                Cr_p[i] = Cr_p[i] + (credit[i]*(rates[i]+1))  # setLoanPaid - Credit paid before (from e.g. other firm) + principal and interest 
        print("DEBTS settled!!!!!!!")
        print("")

        # payDividends
        # bei ihm unter 7
        n_div = 0 # initialize counter for no dividends paid 
        for f in range(Fc):
            divs_f = 0 # initialize current dividend of firm f (bei ihm auch divs)
            # calcProfits
            P[f] = np.around( Total_Rev[f] - Wb_a[f] - np.sum( (np.array(Bi[f]))*(np.array(r_f[f])) ) ,decimals=2) # r bei ihm!
            # profits are Total_Rev of this round - wage bill of this round - interest on credit(s) taken by firm f
            Total_Rev[f] = 0 # resetTotalRevenue    
            if P[f] > 0:
                # setDividends
                divs[f] = np.around(P[f]*delta , decimals = 2) # dividends are profits * 0.5, if firm f has positive profits
                divs_f = divs[f]
            else:
                nn_div = n_div + 1
                # dividends paid remain zero if firm f no positive profit: divs[f] = 0
            # HH receive dividend
            if divs_f > 0:
                for c in range(Nh):
                    div[c] = div[c] + np.around(divs_f/Nh ,decimals=2)
                    # each HH gets share of profits (implied assumption that each HH has same share in each firm.. ???)
                    div_flag[c] = 1 # ?? if one firms has positive profits => then each HH automatically receives payment()
        print("Out of %d Firms, %d reported profits this period"%(Fc,Fc-n_div))

         
        #################################################################################################
        # 7) Earnings after interest payments and dividends are RETAINED EARNINGS,
        #  which are carried forword to next period to increase net worth. 
        # Firms with positive net worth survive => otherwise firms/banks go bankrupt.
        
        # -> hier: def calcRetainedEarnings(self) ; def getRetainedEarnings(self): ; getNetWorth(self) etc.
        # -> bei Ihm ALLES mit in 8 unter updateData in stats!!
        # hier: ERST COMPUTATIONS, Dann SAVING ("mein updateData")
        
        # 1) Firms
        for f in range(Fc):
            # updateNetWorth
            RE[f] = np.around( P[f] - div[f] ) # calcRetainedEarnings - Earnings are profits - dividends paid to HH's 
            # - BANK PAYMENTS???!!!!!
            NW[f] = NW[f] + RE[f] 

            # setMinPrice -> remains 0 so far !!!
            if Qs[f] != 0: # if firm f sold products
                p_low[f] = 0 # loan_to_be_paid[f] / Qs[f] 
            else:
                p_low[f] = 0 

            # calcVacRate
            vac[f] = vac[f] - L[f] # update vacancies (subtract labour employed at firm f)

        
        # 2) HH
        # (last) "computations"
        for c in range(Nh):
            # update employment count
            if is_employed[c] == True: # getEmploymentStatus
                d_employed[c] = d_employed[c] + 1 # incrementEdays
            else:
                d_unemployed[c] = d_unemployed[c] + 1
            # update income
            Y[c] = np.around(S[c] + w[c] + unemp_benefit[c] + div[c] ) # Income is sum of savings, wage, unemplyoment benefits (if no wage) and dividends
        
        # 3) Banks
        # (last) "computations"
        for b in range(Nb):
            # updateEquity
            E[b] = E[b] + Cr_p[b]  # equity before + total amount of loans paid back 



        # updateData 
        # -> hier all individual report variables gathered through the simulation are saved in an numpy 3-D array
        # hence indivdual data of each round is saved => T+2x"DatatoStore" matrix for each simulation round
        # can be switched off in case not needed (e.g. for calibration) to save memory space
        # in neuem file dann if ReportIndData = True: save everything, else: only aggregated data is saved
        # save memory space when calibrating (estimating model through mc simulations)

        n_data = 0 # initialize running index
        # 1) HH
        for c in range(Nh):
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
        print("Out of total %d Household, %d did consume"%(Nh,n_data))

        #2) Firms
        for f in range(Fc):
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
        for b in range(Nb):
            b_data[t, b, 1] = E[b] # getEquity
            b_data[t, b, 2] = Cr[b] # getCreditSupply
            b_data[t, b, 3] = Cr_a[b] # getActualCreditSupplied
            b_data[t, b, 4] = Cr_rem[b] # getRemainingCredit
            # ??? where do firms pay back credit ??!!
            b_data[t, b, 5] = np.sum(np.array(Cr_d[b])*np.array(r_d[b])) / np.sum(np.array(Cr_d[b])) if len(Cr_d[b]) > 0 else 0 
            # sum of credit disbursed (ausgezahlt) * rates disbused (getRatesDisbursed) / sum of disbursecd credit (only if greater 0, else 0 )
            b_data[t, b, 7] = Cr_p[b] # getLoanPaid()
            b_data[t, b, 6] = b_data[t, b, 5] - b_data[t, b, 7] # disbursed credit - loan paid ??

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
        consumption_rate[t][mc] = np.sum(c_data[t, :,2]) / Nh

        #################################################################################################
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
                    unemp_benefit[i-1] = np.around(0.8 * w[i-1] ,decimals=2) # 80% of wage is unemployment payments
                    w[i-1] = 0 # HH receives no more wage
                    d_employed[i-1] = 0
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
        job_offered = [[] for _ in range(Fc)]
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

        # 3) Banks
        Cr_p = np.zeros(Nb)
        firms_appl = [[] for _ in range(Nb)]
        firms_loaned = [[] for _ in range(Nb)]
        Cr_a = np.zeros(Nb)
        Cr_rem = np.zeros(Nb)


        v=input("Press Enter to continue...")
        print("blub")

    print("")
    print("end of simulation round")
    
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
    # Initialitaton:
            c_data[:,h,0] = h+1 # in alle matrices id in first matrix: same for all t-rows
            c_data[0,h,1] = Y0[h] # Y0 value into 1.row, column of current HH, 2nd array
            c_data[0,h,2] = MPC[h]*Y0[h]
            c_data[0,h,4] = MPC[h]
            c_data[0,h,5] = MPC[h]*Y0[h]
    
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
print("unemp_rate", unemp_rate)
print("consumption rate", consumption_rate)