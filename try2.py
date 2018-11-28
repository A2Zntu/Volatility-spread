# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:14:50 2018

@author: user_2
"""

def missing_vs(day_token, one_day_vs, cym):
    s = 0
    if not day_token == 0: 
        day_begin = dim_loc[day_token - 1] 
    else:  # If it is the first day, then special case
        day_begin = 0 
        
    tomorrow = dim_loc[day_token] 
    day_begin_loc = hid_loc.index(day_begin) 
    tomorrow_loc = hid_loc.index(tomorrow) 
    list_temp_day = []
    list_j = list(range(14))
    
    for i in range(day_begin_loc, tomorrow_loc):
        try:
            if df['TRADE_TIME'][hid_loc[i]] <= time_period['Tail'][list_j[i-day_begin_loc]] and df['TRADE_TIME'][hid_loc[i]] >= time_period['Head'][list_j[i-day_begin_loc]]:
                list_temp_day.append(1)
        
            else:
                list_temp_day.append(0)
                day_begin_loc = day_begin_loc -1
                
    
        except IndexError: #If the day exists too many missing period, we exclude.
            x = np.zeros(14 - len(one_day_vs), dtype=float)
            x.fill(np.nan)
            one_day_vs = list(x)
            s = 1 #it is only a switch
    
    if len(list_temp_day) == 14 and s != 1:    
        problem_loc = list_temp_day.index(0) # in this case, we assume there is only one problem in a day
        one_day_vs.insert(problem_loc, np.nan)

    elif len(list_temp_day) != 14 and s != 1:
        one_day_vs.extend(repeat(np.nan, 14-len(list_temp_day)))
        list_temp_day.extend(repeat(0, 14-len(list_temp_day)))

        
    return one_day_vs