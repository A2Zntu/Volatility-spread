# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 23:39:57 2018

@author: Evan
"""

import MySQLdb
import pandas as pd
import numpy as np
from datetime import date
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as iv
from math import exp, sqrt, log, fabs
import matplotlib.pyplot as plt

#%%
db = MySQLdb.connect(host="localhost",
    user="root", passwd="Ntunew123", db="evan")
cursor = db.cursor()

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

cym = 12 #certain year and month
sql = "SELECT * FROM evan.mdr_trade" + list_year_and_month[cym]
cursor.execute(sql)
rc = cursor.rowcount # 取得資料總筆數
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

zcb = pd.read_csv("E:/Spyder/ZCB.csv")
zcb_list = list(zcb['date'][:])


#start-time, end_time, mid_time 

time_period =  pd.read_csv("E:/Spyder/period_trade_1.csv", header = None)
time_period.columns = ['Head', 'Tail', 'Fake_middle', 'Middle']

#There are 14 time-period in one day

#%% Given time period location 
current_date = df["TRADE_DATE"][0]
tpl = 0 #time period location
day_period = []
hour_period = []
dim_list = [df["TRADE_DATE"][0]]
dim_loc = [] #days in one month 
hid_loc = [0] #hours in one day

for i in range(rc): 
    if df["TRADE_DATE"][i] == current_date:

        if df["TRADE_TIME"][i] >= time_period[tpl][0] and df["TRADE_TIME"][i] <= time_period[tpl][1]:
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
        
        if df["TRADE_TIME"][i] >= time_period[tpl][0] and df["TRADE_TIME"][i] <= time_period[tpl][1]:
            hour_period.append(tpl)
        else: 
            tpl = tpl + 1
            hour_period.append(tpl)

# attach the final index of df           
hid_loc.append(rc)
dim_loc.append(rc)


#%% time_gap 
def time_delta(start_time, end_time):
    s = start_time
    e = end_time
    date_time0 = date(year = int(s/10000), month = int((s % 10000)/100), day = int(s % 100))
    date_time1 = date(year = int(e/10000), month = int((e % 10000)/100), day = int(e % 100))
    date_time2 = date_time1 - date_time0
    delta_t = date_time2.days
    return delta_t

#%% ZCB

def zcb_rate(trade_time, expire_time):
    init_day = zcb_list.index(trade_time)
    
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
    m = float(K/S - 1) # moneyness
    M = max(1, t/30) #days to month
    w = exp(-(m**2)/2 - (M - 1)**2)
    return w

def MD_disc(t):
    M = max(1, t/30)
    w = exp(- (M - 1)**2)
    return w

def KD_disc(t):
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
        if df['TRADE_TIME'][i] < time_period
        dist = abs(df['TRADE_TIME'][i] - time_period[hour_period[i]][2])
        if dist > 4000:
            dist = abs(df['TRADE_TIME'][i] - time_period[hour_period[i]][3])
            
        df.loc[i,'Dist'] = dist
        td = time_delta(df["TRADE_DATE"][i], df["EXPIRATION_DATE"][i]) #calulate the delta T
    
        dict_key = "K:"+ str(df["EXERCISE_PRICE"][i])+"/"+"T:" +str(td)
        cur_vol = df['TRADE_SIZE'][i] #current volume

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
#%% volatility spread per hour
def volatility_spread_hour(start, end):
    volitility_spread = []
    spread_volume = []
    pair_list = call_put_pair(start, end)
    for i in range(len(pair_list)):

        if len(pair_list[i][0]) > 1 and len(pair_list[i][1]) > 1:
            Sc = float(pair_list[i][0]['UNDERLYING_INSTRUMENT_PRICE'])
            Sp = float(pair_list[i][1]['UNDERLYING_INSTRUMENT_PRICE'])
            r = zcb_rate(pair_list[i][0]['TRADE_DATE'], pair_list[i][0]['EXPIRATION_DATE'])
            K = float(pair_list[i][0]['EXERCISE_PRICE'])
            t = time_delta(pair_list[i][0]['TRADE_DATE'], pair_list[i][0]['EXPIRATION_DATE'])/365
            q = 0
            call_price = float(pair_list[i][0]['TRADE_PRICE'])
            put_price = float(pair_list[i][1]['TRADE_PRICE'])
            vol = pair_list[i][2] 
            F = Sc*exp(r*t)
            
            intrinsic_c = fabs(max(F - K, 0.0))
            intrinsic_p = fabs(max(K - F, 0.0))
            if call_price < intrinsic_c:
                volitility_spread.append(0.0)
                spread_volume.append(0.0)
    
            elif call_price >= F or put_price >= K:
                volitility_spread.append(0.0)
                spread_volume.append(0.0)
                
            elif put_price < intrinsic_p:
                volitility_spread.append(0.0)
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
                volitility_spread.append(vol_spread)
                spread_volume.append(vol)
                
        else:
            volitility_spread.append(0.0)
            spread_volume.append(0.0)
            
    try:
        weights_aggr = [i/sum(spread_volume) for i in spread_volume]
        hour_vs = np.average(volitility_spread, weights = weights_aggr)
        
    except ZeroDivisionError:
        hour_vs = np.nan

    return hour_vs 
#%%
one_day_vs = []
print("=========================")
print("以下為%s的Volatility Spread " %df["TRADE_DATE"][hid_loc[0]])
print("=========================")
one_month_vs = []
for i in range(len(hid_loc)):
    if i != len(hid_loc)-1:
        if hid_loc[i] not in dim_loc:
            start = hid_loc[i]
            end = hid_loc[i+1]
            one_day_vs.append(volatility_spread_hour(start, end))
        else:  
            start = hid_loc[i]
            end = hid_loc[i+1]
            one_day_vs.append(volatility_spread_hour(start, end))
            print(one_day_vs)
            one_day_vs = []
            print("=========================")
            print("以下為%s的Volatility Spread " %df["TRADE_DATE"][hid_loc[i]])
            print("=========================")
    else:
        print(one_day_vs)
        

           
        
        



