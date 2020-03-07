# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 22:19:48 2018

@author: Evan
"""
import pymysql
import pandas as pd
import numpy as np
from datetime import date, time, datetime
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as iv
from math import exp, sqrt, log, fabs
import matplotlib.pyplot as plt
from itertools import repeat
import os
import tqdm

#%%
config = {
    'user': 'root',
    'password': 'XXXXXXXXX',
    'host': 'localhost'
}
#140.112.111.161

cnx = pymysql.connect(**config)
cursor = cnx.cursor()

work_dir = os.getcwd()
Path_default_readcsv = os.path.join(work_dir, 'Read_csv')


black_friday = pd.read_csv(Path_default_readcsv + "/blackfriday.csv")
black_friday = list(black_friday['blackfriday'])

#%% load the data from Mysql, ZCB
def load_prepared_data(first_year, end_year):
    'Load ZCB data and produce the list of year and month'
    list_year_and_month = []
    start_year = first_year
    end_year = end_year
    start_month = 1
    end_month = 12
    
    for i in range(start_year, end_year + 1):
        for j in range(start_month, end_month + 1):
            if j < 10:
                list_year_and_month.append(str(i) + '0' + str(j))
            else:
                list_year_and_month.append(str(i) +str(j))
                
    total_period = len(list_year_and_month)
    
    yearloc = []
    for i in range(total_period+1):
        if i%12 == 0:
            yearloc.append(i)                                      
    
    endyearloc = len(yearloc)
                
    #Zero coupon bond           
    zcb = pd.read_csv(Path_default_readcsv + "\ZCB.csv")
    zcb_list = list(zcb['date'][:])
    
    #There are 14 time-period in one day
    time_period =  pd.read_csv(Path_default_readcsv + "\period_Quote.csv", header = None)
    time_period.columns = ['Head', 'Tail', 'Middle']
    
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
    return  zcb, zcb_list, x_axis_hour, list_year_and_month, time_period, endyearloc

#%% 
def sql_df(cym):
    'read the data from SQL'
    SQL_database = 'quote' + str(list_year_and_month[cym])[:4] #year
    
    sql = "SELECT * FROM " + SQL_database + ".mdr_trade" + list_year_and_month[cym]
    cursor.execute(sql)
    results = cursor.fetchall() 
    results = list(results)       
    
    df = pd.DataFrame(results, columns = ["SERIES_SEQ_NBR", "TRADE_DATE", 
                                          "TRADE_TIME",  
                                          "EXPIRATION_DATE", "PUT_CALL_CODE", 
                                          "EXERCISE_PRICE", 
                                          "BID_PRICE", "BID_SIZE", 
                                          "ASK_PRICE", "ASK_SIZE",
                                          "UNDERLYING_INSTRUMENT_PRICE", 
                                          "PERIOD_TIME"])
    
    df = df.sort_values(['TRADE_DATE','TRADE_TIME', 'EXPIRATION_DATE', 'EXERCISE_PRICE', 'PUT_CALL_CODE'], ascending=[True,True,True,True,True])
    df = df.reset_index(drop=True)

    return df

#%% time_gap 
def time_delta(start_time, end_time):
    'Compute the distance between two TRADE_DATEs'
    s = start_time
    e = end_time
    date_time0 = date(year = int(s/10000), month = int((s % 10000)/100), day = int(s % 100))
    date_time1 = date(year = int(e/10000), month = int((e % 10000)/100), day = int(e % 100))
    date_time2 = date_time1 - date_time0
    delta_t = date_time2.days
    return delta_t

def seconds_delta(start_time, end_time):
    'Compute the distance between two TRADE_TIMEs'
    s = start_time
    e = end_time
    minute_time0 = time(hour = int(s/10000), minute = int((s % 10000)/100), second = int(s % 100))
    minute_time1 = time(hour = int(e/10000), minute = int((e % 10000)/100), second = int(e % 100))
    minute_time2 = abs(datetime.combine(date.min, minute_time0) - datetime.combine(date.min, minute_time1))
    sec_t = minute_time2.seconds
    return sec_t

def dummy_hour(the_time):
    'Identify the_time in which period'
    dummy = 0
    for i in range(len(time_period)):
        if the_time == time_period['Middle'][i]:
            dummy = i
    return dummy

#%% ZCB

def zcb_rate(trade_time, expire_time):
    'Call the ZCB Rate'
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
    'Maturity and Moneyness Discount'
    try:
        m = float(K/S - 1) # moneyness
        M = max(1, t/30.0) #days to month; at least one-month
        w = exp(-(m**2)/2 - (M - 1)**2)
    except ZeroDivisionError:
        M = max(1, t/30.0)
        w = exp(- (M - 1)**2)
    return w

def MD_disc(t):
    'Maturity Discount'
    M = max(1, t/30)
    w = exp(- (M - 1)**2)
    return w

def KD_disc(S, K, t):
    'Moneyness Discount'
    m = float(K/S - 1) 
    w = exp(-(m**2)/2)
    return w
    
                  
#%% record the volume and put-call pair

def call_put_pair(start, end, volume_size_method = "MK_disc"):
    'Pair the Put Call Option'
    pair_list = []
    pair_dic = {}
    # Record the Number of each option
    num_put_call = 0
    num_call = 0
    num_put = 0
    
    
    for i in range(start, end):
        num_put_call = num_put_call + 1

        td = time_delta(df["TRADE_DATE"][i], df["EXPIRATION_DATE"][i]) #calulate the delta T
        dict_key = "K:"+ str(df["EXERCISE_PRICE"][i])+"/"+"T:" +str(td)
        
        S = float(df['UNDERLYING_INSTRUMENT_PRICE'][i])
        K = float(df['EXERCISE_PRICE'][i])
        t = time_delta(df['TRADE_DATE'][i], df['EXPIRATION_DATE'][i])
        Bid = float(df['BID_PRICE'][i])
        Ask = float(df['ASK_PRICE'][i])
        # Different mothod for vol
        if volume_size_method  == "MK_disc":
            cur_vol = MK_disc(S, K, t)
        elif volume_size_method  == "MD_disc":
            cur_vol = MD_disc(S, K, t)
        elif volume_size_method  == "KD_disc":
            cur_vol = KD_disc(S, K, t)
        elif volume_size_method  == "Trade_size":
            cur_vol = df['TRADE_SIZE'][i] 

#=============================================================================
#   Transform the Dict into Pair            
#=============================================================================
        if dict_key not in pair_dic:

            pair_dic[dict_key] = len(pair_dic)
            if df["PUT_CALL_CODE"][i] == "C":
                pair_list.append([df.iloc[i], [], cur_vol, Bid, Ask, [], []])
                num_call = num_call + 1
            if df["PUT_CALL_CODE"][i] == "P":
                pair_list.append([[], df.iloc[i], cur_vol, [], [], Bid, Ask])
                num_put = num_put + 1
                
        else:
            if df['PUT_CALL_CODE'][i] == "C":
                num_call = num_call + 1
                if len(pair_list[pair_dic[dict_key]][0]) < 1:
                    pair_list[pair_dic[dict_key]][0] = df.iloc[i]
                    pair_list[pair_dic[dict_key]][2] = cur_vol
                    pair_list[pair_dic[dict_key]][3] = Bid
                    pair_list[pair_dic[dict_key]][4] = Ask
                    
                elif pair_list[pair_dic[dict_key]][0]['BID_PRICE'] < df['BID_PRICE'][i]:#choose higher bid price
                    pair_list[pair_dic[dict_key]][3] = Bid
                    
                elif pair_list[pair_dic[dict_key]][0]['ASK_PRICE'] > df['ASK_PRICE'][i]: #choose lower ask price
                    pair_list[pair_dic[dict_key]][4] = Ask
                
            if df["PUT_CALL_CODE"][i] == "P":
                num_put = num_put + 1
                if len(pair_list[pair_dic[dict_key]][1]) < 1:
                    pair_list[pair_dic[dict_key]][1] = df.iloc[i]
                    pair_list[pair_dic[dict_key]][2] = cur_vol
                    pair_list[pair_dic[dict_key]][5] = Bid
                    pair_list[pair_dic[dict_key]][6] = Ask
                    
                elif pair_list[pair_dic[dict_key]][1]['BID_PRICE'] < df['BID_PRICE'][i]:#choose higher bid price
                    pair_list[pair_dic[dict_key]][5] = Bid
                    
                elif pair_list[pair_dic[dict_key]][1]['ASK_PRICE'] > df['ASK_PRICE'][i]: #choose lower ask price
                    pair_list[pair_dic[dict_key]][6] = Ask    
    
    # In this period, how long did it take to gather the data
    try:
        spendtime = seconds_delta(df["TRADE_TIME"][start], df["TRADE_TIME"][end-1])
    except KeyError:
        spendtime = 0
    
    return pair_list, num_put_call, num_call, num_put, spendtime


#%% aggrgate vs information and dispriptive statisitic
aggre_vs_inform = []    
def record_vs(vs_information):
    aggre_vs_inform.append(vs_information)
    return aggre_vs_inform

    
#%% volatility spread per hour
# vs wighted by MK discount
def volatility_spread_hour(start, end, print_out_info = False):
    'Filter and Construct the Valid Pair Option'
    volatility_spread = []
    spread_volume = []
    pair_list, numpc, numc, nump, spendtime = call_put_pair(start, end)
    # for instantiating the failure rates
    totalPairAmount = 0
    maturityRuleVio = 0
    callRuleVio = 0
    putRuleVio = 0
    noMatch = 0
    
    for i in range(len(pair_list)):
        totalPairAmount +=  1
        if len(pair_list[i][0]) > 1 and len(pair_list[i][1]) > 1:
            S = min(float(pair_list[i][0]['UNDERLYING_INSTRUMENT_PRICE']),
                    float(pair_list[i][1]['UNDERLYING_INSTRUMENT_PRICE'])) #choose small S for intrinsic_c
            r = zcb_rate(pair_list[i][0]['TRADE_DATE'], pair_list[i][0]['EXPIRATION_DATE'])
                
            K = float(pair_list[i][0]['EXERCISE_PRICE'])
            t = time_delta(pair_list[i][0]['TRADE_DATE'], pair_list[i][0]['EXPIRATION_DATE'])/365
            q = 0
            call_price = float((pair_list[i][3] + pair_list[i][4])/2)
            put_price = float((pair_list[i][5] + pair_list[i][6])/2)
            volume = MK_disc(S, K, t)
            
            timediff = seconds_delta(pair_list[i][0]['TRADE_TIME'], pair_list[i][1]['TRADE_TIME'])
            dummy = pair_list[i][0]['PERIOD_TIME']
            PV_K = K*exp(-r*t)

            intrinsic_c = fabs(max(S - PV_K, 0.0))
            intrinsic_p = fabs(max(PV_K - S, 0.0))
            
            if S != 0:
                moneyness = float(K/S - 1) #As for put, it's the level of ITM; for call, OTM.             
            else:
                moneyness = 0
                
            if t < 10/365: #eliminate the option with expiration less than 10 days
                maturityRuleVio += 1
                volatility_spread.append(0.0)
                spread_volume.append(0.0)
                
            elif call_price < intrinsic_c or call_price >= S:
                callRuleVio +=  1
                volatility_spread.append(0.0)
                spread_volume.append(0.0)
    
            elif put_price < intrinsic_p or put_price >= PV_K:
                putRuleVio += 1
                volatility_spread.append(0.0)
                spread_volume.append(0.0)
                               
            
            elif S == 0 : #typo error
                volatility_spread.append(0.0)
                spread_volume.append(0.0) 
                
            else:
                call_iv = iv(price = call_price, 
                             flag = 'c', 
                             S = S, 
                             K = K, 
                             t = t, 
                             r = r,
                             q = q)
                
                put_iv = iv(price = put_price, 
                             flag = 'p', 
                             S = S, 
                             K = K, 
                             t = t, 
                             r = r,
                             q = q)
                
                vol_spread = call_iv - put_iv
                volatility_spread.append(vol_spread)
                spread_volume.append(volume)
                record_vs([vol_spread, timediff, moneyness, t, dummy])
                
                
                
        else:
            noMatch = noMatch + 1
            volatility_spread.append(0.0)
            spread_volume.append(0.0)
    
    if totalPairAmount != 0:
        rate1 = noMatch/totalPairAmount #due to no match 
        no_match(rate1)
        
        rate2 = callRuleVio/totalPairAmount #due to call lower bound
        call_lbnd(rate2)
        
        rate3 = maturityRuleVio/totalPairAmount
        lessthan10days(rate3)
    try: 
        weights_aggr = [i/sum(spread_volume) for i in spread_volume]
        hour_vs = np.average(volatility_spread, weights = weights_aggr)
        
    except ZeroDivisionError:
        hour_vs = np.nan
    
    show_pl(pair_list)
    if print_out_info == True:
        print("總PAIR:%i"%totalPairAmount)
        print("Match不到%i"%noMatch)
        print("到期日小於十天%i"%maturityRuleVio)
        print("Call價格不合%i"%callRuleVio)
        print("PUT價格不合%i"%putRuleVio)
        print("==========")
    
    return hour_vs, numpc, numc, nump, spendtime

no_match_rate = []
def no_match(rate):
    no_match_rate.append(rate)

call_lbnd_rate = []
def call_lbnd(rate):
    call_lbnd_rate.append(rate)
    
less_than_10days = []
def lessthan10days(rate):
    less_than_10days.append(rate)

pair_list = []
def show_pl(pl):
    pair_list.append(pl)
    
#%% 

def missing_vs(day_token, one_day_vs, cym):
    'Deal with the v.s. missing due to time missing'
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
    title_name = list_year_and_month[start_period] +'~' + list_year_and_month[end_period-1] 
    plt.savefig(os.path.join(work_dir, 'Graph_Quote', title_name))
    
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
    print("\n")
    print("Loading data of {} ...".format(list_year_and_month[cym]))
    global df, hour_period
    tpl = 0
    df = sql_df(cym)
    current_date = df["TRADE_DATE"][0]
    tpl = 0 #time period location
    hour_period = [] #record every tick belongs to which hour
    dim_list = [df["TRADE_DATE"][0]]
    dim_loc = [] #record every different day in df
    hid_loc = [0] #record every differnet hour in df
    rc = len(df) #whole rows
    df_quant(rc)

    for i in range(rc): 
        if df["TRADE_DATE"][i] == current_date:
    
            if df["TRADE_TIME"][i] >= time_period['Head'][tpl] and df["TRADE_TIME"][i] <= time_period['Tail'][tpl]:
                hour_period.append(tpl)

            else: 
                hid_loc.append(i)
                while tpl <= 14:
                    tpl = tpl + 1
                    if df["TRADE_TIME"][i] >= time_period['Head'][tpl] and df["TRADE_TIME"][i] <= time_period['Tail'][tpl]:
                        hour_period.append(tpl)
                        break
                    else:
                        pass
    
            
        elif df["TRADE_DATE"][i] != current_date:
            dim_list.append(df["TRADE_DATE"][i])
            dim_loc.append(i)
            current_date = df["TRADE_DATE"][i]
            tpl = 0
            hid_loc.append(i)
            
            if df["TRADE_TIME"][i] >= time_period['Head'][tpl] and df["TRADE_TIME"][i] <= time_period['Tail'][tpl]:
                hour_period.append(tpl)
            else: 
                while tpl <= 14:
                    tpl = tpl + 1
                    if df["TRADE_TIME"][i] >= time_period['Head'][tpl] and df["TRADE_TIME"][i] <= time_period['Tail'][tpl]:
                        hour_period.append(tpl)
                        break
                    else:
                        pass
    
    # attach the final index of df           
    hid_loc.append(rc)
    dim_loc.append(rc)

    return hid_loc, dim_loc, dim_list


#%% Main Code
    
def one_period_vs(start_period, end_period):
    'Period IVS, Amount of Put and Call, Seconds spended'
    global hid_loc, dim_loc, dim_list, df
    df_one_period_vs = pd.DataFrame() #volatility Spread
    df_one_period_npc = pd.DataFrame() #number of puts and calls
    df_one_period_nc = pd.DataFrame() #number of calls 
    df_one_period_np = pd.DataFrame() #number of puts
    df_one_period_ts = pd.DataFrame() #time spread
    for cym in range(start_period, end_period):
        hid_loc, dim_loc, dim_list = hid_dim_loc(cym)
    
        one_day_vs = []
        one_month_vs = []
        one_day_npc = []
        one_month_npc = []
        one_day_nc = []
        one_month_nc = []
        one_day_np = []
        one_month_np = []
        one_day_ts = []
        one_month_ts = []        
        for i in range(len(hid_loc)):
            
            if i != len(hid_loc)-1:
                if hid_loc[i] not in dim_loc: # 08:30~14:30
                    start = hid_loc[i]
                    end = hid_loc[i+1]
                    hour_vs, numpc, numc, nump, spendtime = volatility_spread_hour(start, end)

                    one_day_vs.append(hour_vs)
                    one_day_npc.append(numpc)
                    one_day_nc.append(numc)
                    one_day_np.append(nump)
                    one_day_ts.append(spendtime)
                else:  #15:00
                    if not len(one_day_vs) == 14:

                        if not df["TRADE_DATE"][start] in black_friday:
#                            print(dim_loc.index(hid_loc[i]))
#                            print("== The length of IVS is not 14==")
                            one_day_vs = missing_vs(dim_loc.index(hid_loc[i]), one_day_vs, cym)

                            
                        else:
                            one_day_vs = black_fri_vs(one_day_vs)
                    start = hid_loc[i]
                    end = hid_loc[i+1]
                    one_month_vs.append(one_day_vs)
                    one_month_npc.append(one_day_npc)
                    one_month_nc.append(one_day_nc)
                    one_month_np.append(one_day_np)
                    one_month_ts.append(one_day_ts)
                    
                    hour_vs, numpc, numc, nump, spendtime = volatility_spread_hour(start, end)

                    one_day_vs = []
                    one_day_vs.append(hour_vs)
                    one_day_npc = []
                    one_day_npc.append(numpc)
                    one_day_nc = []
                    one_day_nc.append(numc)
                    one_day_np = []
                    one_day_np.append(nump)
                    one_day_ts = []
                    one_day_ts.append(spendtime)
    
            else:
                
                one_month_vs.append(one_day_vs)
                one_month_npc.append(one_day_npc)
                one_month_nc.append(one_day_nc)
                one_month_np.append(one_day_np)
                one_month_ts.append(one_day_ts)
        # transfer 2-D lists to dataframe
        # combine the one month_vs into one year vs 
        df_one_month_vs = pd.DataFrame(one_month_vs)
        df_one_month_vs.index = dim_list
        df_one_period_vs = df_one_period_vs.append(df_one_month_vs)
        
        df_one_month_npc = pd.DataFrame(one_month_npc)
        df_one_month_npc.index = dim_list
        df_one_period_npc = df_one_period_npc.append(df_one_month_npc)

        df_one_month_nc = pd.DataFrame(one_month_nc)
        df_one_month_nc.index = dim_list
        df_one_period_nc = df_one_period_nc.append(df_one_month_nc)

        df_one_month_np = pd.DataFrame(one_month_np)
        df_one_month_np.index = dim_list
        df_one_period_np = df_one_period_np.append(df_one_month_np)
    
        df_one_month_ts = pd.DataFrame(one_month_ts)
        df_one_month_ts.index = dim_list
        df_one_period_ts = df_one_period_ts.append(df_one_month_ts)

        
    df_one_period_vs.columns = x_axis_hour
    df_one_period_npc.columns = x_axis_hour
    df_one_period_np.columns = x_axis_hour
    df_one_period_nc.columns = x_axis_hour
    df_one_period_ts.columns = x_axis_hour
    
    return df_one_period_vs, df_one_period_npc, df_one_period_nc, df_one_period_np, df_one_period_ts

#%% calculate the df and pair_option quantity

sum__rc = 0
def df_quant(rc):
    global sum__rc
    sum__rc = sum__rc + rc
    return sum__rc  

#%% save data 
def saving_file(df, saving_name = "df", results_path = os.path.join(work_dir, "Output_Result")):
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    results_path = os.path.join(results_path, datetime.today().strftime('%Y%m%d')) 
    print ("writing file to {}".format(results_path))

    if os.path.exists(results_path):
        results_path = results_path + '_' + datetime.datetime.today().strftime('_%H%M%S')
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
        
    if df:
        df.to_csv(os.path.join(results_path, saving_name))
    else:
        print("Saving Files Failure!")
    


#%% Generate the intraday SPX500 price

def half_hour_SPX(loc):
    
    S = float(df['UNDERLYING_INSTRUMENT_PRICE'][loc])
    return S

def SPX_price(start_period, end_period):
    global hid_loc, dim_loc, dim_list
    df_intraday_SPX = pd.DataFrame()
    
    for cym in range(start_period, end_period):
        hid_loc, dim_loc, dim_list = hid_dim_loc(cym)
        
        one_day_price = []
        one_month_price = []
        for i in range(len(hid_loc)):
    
            if i != len(hid_loc)-1:
                if hid_loc[i] not in dim_loc: # 08:30~14:30
                    loc = hid_loc[i]
                    half_hour_price = half_hour_SPX(loc)
                    one_day_price.append(half_hour_price)
                    
                else:  #15:00
                    loc = hid_loc[i]
                    one_month_price.append(one_day_price)
                    one_day_price = []
                    one_day_price.append(half_hour_SPX(loc))
    
            else:
                
                one_month_price.append(one_day_price)
        # transfer 2-D lists to dataframe
        # combine the one month_vs into one year vs 
        df_one_month_price = pd.DataFrame(one_month_price)
        df_one_month_price.index = dim_list
        df_intraday_SPX = df_intraday_SPX.append(df_one_month_price)
        
    df_intraday_SPX.columns = x_axis_hour
    return df_intraday_SPX


#%% Run the IVS
       
if __name__ ==  '__main__':
    '''
    Year and Start code: (2007, 1), (2008, 12), (2009, 24), (2010, 36)
    (2011, 48), (2012, 60), (2013, 72), (2014, 84), (2015, 96), (2016, 108), 
    (2017, 120)
    ''' 
    
    zcb, zcb_list, x_axis_hour, list_year_and_month, time_period, endyearloc = load_prepared_data(2007, 2017)
    df_overall_vs = pd.DataFrame()
    df_overall_npc = pd.DataFrame()
    df_overall_nc = pd.DataFrame()
    df_overall_np = pd.DataFrame()
    df_overall_ts = pd.DataFrame()
    for i in tqdm.tqdm(range(0, 11), desc= 'IVS'):
        period_start = i*12
        period_end = (i+1)*12
   

        df_year_vs, df_year_npc, df_year_nc, df_year_np, df_year_ts = one_period_vs(period_start, period_end)
        plot_vs_by_halfhr(df_year_vs, period_start, period_end)
        df_overall_vs = df_overall_vs.append(df_year_vs)
        df_overall_npc = df_overall_npc.append(df_year_npc)
        df_overall_nc = df_overall_nc.append(df_year_nc)
        df_overall_np = df_overall_np.append(df_year_np)
        df_overall_ts = df_overall_ts.append(df_year_ts)
        
        print("\nI finish a year!!")

   
    df_aggre_vs_info = pd.DataFrame(aggre_vs_inform)
    info_aggre_columns_name = ['IVS', 'timediff', 'Moneyness', 'Maturity', 'Dummy']
    df_aggre_vs_info.columns = info_aggre_columns_name
    #saving file
    saving_file(df_overall_vs, saving_name = 'df_overall_vs')
    saving_file(df_overall_npc, saving_name = 'df_overall_npc')
    saving_file(df_overall_nc, saving_name = 'df_overall_nc')
    saving_file(df_overall_np, saving_name = 'df_overall_np')
    saving_file(df_overall_ts, saving_name = 'df_overall_ts')
    saving_file(df_aggre_vs_info, saving_name = 'df_aggre_vs_info')
    cnx.close()


