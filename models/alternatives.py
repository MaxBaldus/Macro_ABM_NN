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
