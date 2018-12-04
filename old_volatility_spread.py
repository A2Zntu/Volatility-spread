# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 23:39:57 2018

@author: Evan
"""

import mysql.connector
from mysql.connector.constants import ClientFlag
import pandas as pd
import numpy as np
from datetime import date, time, datetime
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as iv
from math import exp, sqrt, log, fabs
import matplotlib.pyplot as plt
from itertools import repeat

#%% load the data from Mysql, ZCB
config = {
    'user': 'admin',
    'password': 'Ntunew123',
    'host': '140.112.111.161'
}
#140.112.111.161

cnx = mysql.connector.connect(**config)
cursor = cnx.cursor(buffered=True)

list_year_and_month = []
start_year = 2004
end_year = 2017
start_month = 1
end_month = 12

for i in range(start_year, end_year + 1):
    for j in range(start_month, end_month + 1):
        if j < 10:
            list_year_and_month.append(str(i) + '0' + str(j))
        else:
            list_year_and_month.append(str(i) +str(j))
            
#Xero coupon bond           
zcb = pd.read_csv("E:/Spyder/ZCB.csv")
zcb_list = list(zcb['date'][:])

#There are 14 time-period in one day
time_period =  pd.read_csv("E:/Spyder/period_trade_1.csv", header = None)
time_period.columns = ['Head', 'Tail', 'Fake_middle', 'Middle']

# List the 14 time period for column names 
x_axis_hour = [] #generate x axis
for s in time_period['Middle']:
    if s < 100000: #Avoid the disorder in ploting the x-axis
        if not int((s % 10000)/100) == 0:
            x_axis_hour.append('0' + str(int(s/10000)) +':' + str(int((s % 10000)/100)))
        else:
            x_axis_hour.append('0' + str(int(s/10000)) +':00')
    else:
        if not int((s % 10000)/100) == 0:
            x_axis_hour.append(str(int(s/10000)) +':' + str(int((s % 10000)/100)))
        else:
            x_axis_hour.append(str(int(s/10000)) +':00')
x_axis_hour = np.array(x_axis_hour)

#%% read the data

def sql_df(cym):
    sql = "SELECT * FROM evan.mdr_trade" + list_year_and_month[cym]
    cursor.execute(sql)
    results = cursor.fetchall() 
    results = list(results)       
    
    df = pd.DataFrame(results, columns = ["SERIES_SEQ_NBR", "TRADE_DATE", 
                                          "TRADE_TIME",  
                                          "EXPIRATION_DATE", "PUT_CALL_CODE", 
                                          "EXERCISE_PRICE", "TRADE_PRICE", 
                                          "TRADE_SIZE",  
                                          "UNDERLYING_INSTRUMENT_PRICE"])
    
    df = df.sort_values(['TRADE_DATE','TRADE_TIME', 'EXPIRATION_DATE', 'EXERCISE_PRICE', 'PUT_CALL_CODE'], ascending=[True,True,True,True,True])
    df = df.reset_index(drop=True)
    return df

#%% time_gap 
def time_delta(start_time, end_time):
    s = start_time
    e = end_time
    date_time0 = date(year = int(s/10000), month = int((s % 10000)/100), day = int(s % 100))
    date_time1 = date(year = int(e/10000), month = int((e % 10000)/100), day = int(e % 100))
    date_time2 = date_time1 - date_time0
    delta_t = date_time2.days
    return delta_t

def seconds_delta(start_time, end_time):
    s = start_time
    e = end_time
    minute_time0 = time(hour = int(s/10000), minute = int((s % 10000)/100), second = int(s % 100))
    minute_time1 = time(hour = int(e/10000), minute = int((e % 10000)/100), second = int(e % 100))
    minute_time2 = abs(datetime.combine(date.min, minute_time0) - datetime.combine(date.min, minute_time1))
    sec_t = minute_time2.seconds
    return sec_t

def dummy_hour(the_time):
    dummy = 0
    for i in range(len(time_period)):
        if the_time >= time_period['Head'][i] and the_time <= time_period['Tail'][i]:
            dummy = i
    return dummy

#%% ZCB

def zcb_rate(trade_time, expire_time):
    try:
        init_day = zcb_list.index(trade_time)
    except ValueError:
        try:
            init_day = zcb_list.index(trade_time - 1)
        except ValueError:
            init_day = zcb_list.index(trade_time + 1)
    
    j = True
    while j == True:
        if zcb['days'][init_day] <  time_delta(trade_time, expire_time):
            if zcb['days'][init_day + 1] < time_delta(trade_time, expire_time):
                init_day = init_day + 1
            else:
                numerator_gap = zcb['days'][init_day + 1] - zcb['days'][init_day]
                denominator_gap = time_delta(trade_time, expire_time) - zcb['days'][init_day]
                goal_rate = ((denominator_gap/numerator_gap)*(zcb['rate'][init_day + 1] - zcb['rate'][init_day]) + zcb['rate'][init_day])*0.01
                
                j = False
        else:
            goal_rate = zcb['rate'][init_day]*0.01
            j = False
            
    return goal_rate

#%% define information aggregation method

def MK_disc(S, K, t): # Maturity and strike discount 
    try:
        m = float(K/S - 1) # moneyness
        M = max(1, t/30.0) #days to month; at least one-month
        w = exp(-(m**2)/2 - (M - 1)**2)
    except ZeroDivisionError:
        M = max(1, t/30.0)
        w = exp(- (M - 1)**2)
    return w

def MD_disc(t):
    M = max(1, t/30)
    w = exp(- (M - 1)**2)
    return w

def KD_disc(S, K, t):
    m = float(K/S - 1) 
    w = exp(-(m**2)/2)
    return w
    
                  
#%% record the volume and put-call pair

def call_put_pair(start, end):
    pair_list = []
    pair_dic = {}
    df["Dist"] = np.nan
    for i in range(start, end):
        #calaulate the time-distance for each tick
        dist = seconds_delta(df['TRADE_TIME'][i], time_period['Middle'][hour_period[i]])
            
        df.loc[i,'Dist'] = dist
        td = time_delta(df["TRADE_DATE"][i], df["EXPIRATION_DATE"][i]) #calulate the delta T
    
        dict_key = "K:"+ str(df["EXERCISE_PRICE"][i])+"/"+"T:" +str(td)
#-------------------------------------------------------------------------
# --> current volume (without adjusted)
#        cur_vol = df['TRADE_SIZE'][i] 
#------------------------------------------------------------------------- 
# --> current volume (with adjusted)
        S = float(df['UNDERLYING_INSTRUMENT_PRICE'][i])
        K = float(df['EXERCISE_PRICE'][i])
        t = time_delta(df['TRADE_DATE'][i], df['EXPIRATION_DATE'][i])
        cur_vol = df['TRADE_SIZE'][i]*MK_disc(S, K, t)
#-------------------------------------------------------------------------
        if dict_key not in pair_dic:

            pair_dic[dict_key] = len(pair_dic)
            if df["PUT_CALL_CODE"][i] == "C":
                pair_list.append([df.iloc[i], [], cur_vol])
            if df["PUT_CALL_CODE"][i] == "P":
                pair_list.append([[], df.iloc[i], cur_vol])
                
        else:
            if df['PUT_CALL_CODE'][i] == "C":
                if len(pair_list[pair_dic[dict_key]][0]) < 1:
                    pair_list[pair_dic[dict_key]][0] = df.iloc[i]
                    pair_list[pair_dic[dict_key]][2] = pair_list[pair_dic[dict_key]][2] + cur_vol
                    
                elif pair_list[pair_dic[dict_key]][0]['Dist'] > df['Dist'][i]:
                    pair_list[pair_dic[dict_key]][0] = df.iloc[i]
                    pair_list[pair_dic[dict_key]][2] = pair_list[pair_dic[dict_key]][2] + cur_vol
                else:
                    pair_list[pair_dic[dict_key]][2] = pair_list[pair_dic[dict_key]][2] + cur_vol
                
            if df["PUT_CALL_CODE"][i] == "P":
                if len(pair_list[pair_dic[dict_key]][1]) < 1:
                    pair_list[pair_dic[dict_key]][1] = df.iloc[i]
                    pair_list[pair_dic[dict_key]][2] = pair_list[pair_dic[dict_key]][2] + cur_vol
                    
                elif pair_list[pair_dic[dict_key]][1]['Dist'] > df['Dist'][i]:
                    pair_list[pair_dic[dict_key]][1] = df.iloc[i]
                    pair_list[pair_dic[dict_key]][2] = pair_list[pair_dic[dict_key]][2] + cur_vol
                else:
                    pair_list[pair_dic[dict_key]][2] = pair_list[pair_dic[dict_key]][2] + cur_vol
    
    return pair_list
#%% aggrgate vs information
aggre_vs_inform = []    
def record_vs(vs_information):
    aggre_vs_inform.append(vs_information)
    return aggre_vs_inform
    
#%% volatility spread per hour
def volatility_spread_hour(start, end):
    volatility_spread = []
    spread_volume = []
    pair_list = call_put_pair(start, end)
    for i in range(len(pair_list)):
        if len(pair_list[i][0]) > 1 and len(pair_list[i][1]) > 1:
            Sc = float(pair_list[i][0]['UNDERLYING_INSTRUMENT_PRICE'])
            Sp = float(pair_list[i][1]['UNDERLYING_INSTRUMENT_PRICE'])
            Sa = (Sc+Sp)/2.0
            r = zcb_rate(pair_list[i][0]['TRADE_DATE'], pair_list[i][0]['EXPIRATION_DATE'])
                
            K = float(pair_list[i][0]['EXERCISE_PRICE'])
            t = time_delta(pair_list[i][0]['TRADE_DATE'], pair_list[i][0]['EXPIRATION_DATE'])/365
            q = 0
            call_price = float(pair_list[i][0]['TRADE_PRICE'])
            put_price = float(pair_list[i][1]['TRADE_PRICE'])
            vol = pair_list[i][2]
            timediff = seconds_delta(pair_list[i][0]['TRADE_TIME'], pair_list[i][1]['TRADE_TIME'])
            dummy = dummy_hour(pair_list[i][0]['TRADE_DATE'])
            PV_K = K*exp(-r*t)
            intrinsic_c = fabs(max(Sc - PV_K, 0.0))
            intrinsic_p = fabs(max(PV_K - Sp, 0.0))
            
            if Sa != 0:
                moneyness = float(K/Sa - 1) #As for put, it's the level of ITM; for call, OTM.             
            else:
                moneyness = 0
                
            if t < 10/365: #eliminate the option with expiration less than 30 days
                volatility_spread.append(0.0)
                spread_volume.append(0.0)
                
            elif call_price < intrinsic_c:
                volatility_spread.append(0.0)
                spread_volume.append(0.0)
    
            elif call_price >= Sc or put_price >= PV_K:
                volatility_spread.append(0.0)
                spread_volume.append(0.0)
                
            elif put_price < intrinsic_p:
                volatility_spread.append(0.0)
                spread_volume.append(0.0)                
            
            elif Sc == 0 or Sp == 0: #typo error
                volatility_spread.append(0.0)
                spread_volume.append(0.0) 
                
            else:
                call_iv = iv(price = call_price, 
                             flag = 'c', 
                             S = Sc, 
                             K = K, 
                             t = t, 
                             r = r,
                             q = q)
                
                put_iv = iv(price = put_price, 
                             flag = 'p', 
                             S = Sp, 
                             K = K, 
                             t = t, 
                             r = r,
                             q = q)
                
                vol_spread = call_iv - put_iv
                volatility_spread.append(vol_spread)
                spread_volume.append(vol)
                record_vs([vol_spread, timediff, moneyness, t, dummy])
                
        else:
            volatility_spread.append(0.0)
            spread_volume.append(0.0)
            
    try: 
        weights_aggr = [i/sum(spread_volume) for i in spread_volume]
        hour_vs = np.average(volatility_spread, weights = weights_aggr)
        
    except ZeroDivisionError:
        hour_vs = np.nan

    return hour_vs

    
#%% Deal with the v.s. missing due to time missing

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
            one_day_vs.reverse()
            one_day_vs.extend(repeat(np.nan, 14-len(one_day_vs)))
            one_day_vs.reverse()
            s = 1 #it is only a switch
    
    if len(list_temp_day) == 14 and s != 1:    
        problem_loc = list_temp_day.index(0) # in this case, we assume there is only one problem in a day
        one_day_vs.insert(problem_loc, np.nan)

    elif len(list_temp_day) != 14 and s != 1:
        one_day_vs.extend(repeat(np.nan, 14-len(list_temp_day)))
        list_temp_day.extend(repeat(0, 14-len(list_temp_day)))

        
    return one_day_vs

#Black friday CBOE only works to 12:30
black_friday = pd.read_csv("E:/Spyder/blackfriday.csv")
black_friday = list(black_friday['blackfriday'])

def black_fri_vs(one_day_vs):
    list_len = len(one_day_vs)
    while list_len < 14:
        one_day_vs.append(np.nan)
        list_len = len(one_day_vs)
    return one_day_vs


#%% Draw 2 kinds of plot 
def plot_vs_by_halfhr(df_one_period_vs, start_period, end_period):
    mean_one_month_vs = df_one_period_vs.mean()
    plt.style.use('ggplot')
    
    plt.figure(figsize=(15,6))
    plt.plot(x_axis_hour, mean_one_month_vs)
    plt.title(list_year_and_month[start_period] +'~' + list_year_and_month[end_period-1] + "_By half-hour")
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Spread Volatility', fontsize=14)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
def plot_vs_by_day(df_one_period_vs, start_period, end_period):
    mean_one_month_vs_1 = df_one_period_vs.mean(axis=1)
    x_axis_day = [str(i)[-2:] for i in dim_list] #[-2:] keep last 2 digits
    plt.figure(figsize=(15,6))
    plt.plot(x_axis_day, mean_one_month_vs_1)
    plt.title(list_year_and_month[start_period] +'~' + list_year_and_month[end_period-1] + "_By day")
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Spread Volatility', fontsize=14)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

#%% organize the time code
def hid_dim_loc(cym):
    print("Now, we are in: %s"%list_year_and_month[cym])
    global df, hour_period
    df = sql_df(cym)
    current_date = df["TRADE_DATE"][0]
    tpl = 0 #time period location
    day_period = []
    hour_period = []
    dim_list = [df["TRADE_DATE"][0]]
    dim_loc = [] #days in one month 
    hid_loc = [0] #hours in one day
    rc = len(df)
    for i in range(rc): 
        if df["TRADE_DATE"][i] == current_date:
    
            if df["TRADE_TIME"][i] >= time_period['Head'][tpl] and df["TRADE_TIME"][i] <= time_period['Tail'][tpl]:
                hour_period.append(tpl)
            else: 
                tpl = tpl + 1
                hour_period.append(tpl)
                hid_loc.append(i)
                
    
            
        elif df["TRADE_DATE"][i] != current_date:
            dim_list.append(df["TRADE_DATE"][i])
            dim_loc.append(i)
            current_date = df["TRADE_DATE"][i]
            tpl = 0
            hid_loc.append(i)
            
            if df["TRADE_TIME"][i] >= time_period['Head'][tpl] and df["TRADE_TIME"][i] <= time_period['Tail'][tpl]:
                hour_period.append(tpl)
            else: 
                tpl = tpl + 1
                hour_period.append(tpl)
    
    # attach the final index of df           
    hid_loc.append(rc)
    dim_loc.append(rc)

    return hid_loc, dim_loc, dim_list


#%% Main Code
    
def one_period_vs(start_period, end_period):
    global hid_loc, dim_loc, dim_list
    df_one_period_vs = pd.DataFrame()
    
    for cym in range(start_period, end_period):
        hid_loc, dim_loc, dim_list = hid_dim_loc(cym)
    
        one_day_vs = []
        one_month_vs = []
        for i in range(len(hid_loc)):
    
            if i != len(hid_loc)-1:
                if hid_loc[i] not in dim_loc: # 08:30~14:30
                    start = hid_loc[i]
                    end = hid_loc[i+1]
                    hour_vs = volatility_spread_hour(start, end)
                    one_day_vs.append(hour_vs)
                else:  #15:00
                    if not len(one_day_vs) == 14:
    
                        if not df["TRADE_DATE"][start] in black_friday:
                            print(dim_loc.index(hid_loc[i]))
                            print("== MOM! I am in the Area.==")
                            one_day_vs = missing_vs(dim_loc.index(hid_loc[i]), one_day_vs, cym)
                            
                        else:
                            one_day_vs = black_fri_vs(one_day_vs)
                    start = hid_loc[i]
                    end = hid_loc[i+1]
                    one_month_vs.append(one_day_vs)
                    one_day_vs = []
                    one_day_vs.append(volatility_spread_hour(start, end))
    
            else:
                
                one_month_vs.append(one_day_vs)
        # transfer 2-D lists to dataframe
        # combine the one month_vs into one year vs 
        df_one_month_vs = pd.DataFrame(one_month_vs)
        df_one_month_vs.index = dim_list
        df_one_period_vs = df_one_period_vs.append(df_one_month_vs)
        
    df_one_period_vs.columns = x_axis_hour
    return df_one_period_vs
#%%
total_period = len(list_year_and_month)

yearloc = []
for i in range(total_period+1):
    if i%12 == 0:
        yearloc.append(i)                                      

endyearloc = len(yearloc)

#df_overall_vs = pd.DataFrame()
for i in range(14, endyearloc):
    period_start = yearloc[i-1] #Begin in 1
    period_end = yearloc[i]
    df_year_vs = one_period_vs(period_start, period_end)
    plot_vs_by_halfhr(df_year_vs, period_start, period_end)
    df_overall_vs = df_overall_vs.append(df_year_vs)
    print("I finish a year!!")
#%%
#period_start = yearloc[10-1] 
#period_end = yearloc[10]
#df_year_vs = one_period_vs(period_start, period_end)   
#plot_vs_by_halfhr(df_year_vs, period_start, period_end)

#%%

df_overall_vs = df_IVS
period_start = 0
#df_overall_vs.to_csv("E:/Spyder/vs_10days_withinfo.csv")
plot_vs_by_halfhr(df_overall_vs, period_start, period_end)

