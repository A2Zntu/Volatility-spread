# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 23:39:57 2018

@author: Evan
"""

import MySQLdb
import pandas as pd
import implied_volatility as iv
import numpy as np

db = MySQLdb.connect(host="localhost",
    user="root", passwd="Ntunew123", db="evan")
cursor = db.cursor()

sql = "SELECT * FROM evan.mdr_trade"
cursor.execute(sql)
rc = cursor.rowcount # 取得資料總筆數
results = cursor.fetchall() 
results = list(results)       

df = pd.DataFrame(results, columns = ["RECORD_TYPE_CODE", "TRADE_DATE", 
                                      "TRADE_TIME", "SERIES_SEQ_NBR", 
                                      "MARKET_CONDITION_CODE", "CLASS_SYMBOL",
                                      "EXPIRATION_DATE", "PUT_CALL_CODE", 
                                      "EXERCISE_PRICE", "TRADE_PRICE", 
                                      "TRADE_SIZE", "FILLER_1", "FILLER_2", 
                                      "UNDERLYING_INSTRUMENT_PRICE", 
                                      "UNDERLYING_INSTRUMENT_SYMBOL"])

df = df.drop(["FILLER_1", "FILLER_2", "UNDERLYING_INSTRUMENT_SYMBOL"], axis=1)
df = df.sort_values(['TRADE_DATE','TRADE_TIME', 'EXPIRATION_DATE', 'EXERCISE_PRICE', 'PUT_CALL_CODE'], ascending=[True,True,True,True,True])
df = df.reset_index(drop=True)

zcb = pd.read_csv("E:/Spyder/ZCB.csv")
sigma_test_2 = iv.find_vol(target_value = 2.65, call_put = 'p', S =1216.89, K = 1175, T = 1/12, r = 0.02383996)

#          start-time, end_time, mid_time 
time_period =  [[83000,  83500,  83000, 83000],
                [85730,  90230,  90000, 86000],
                [95730, 100230, 100000, 96000],
                [105730,110230, 110000, 106000],
                [115730,120230, 120000, 116000],
                [125730,130230, 130000, 126000],
                [135730,140230, 140000, 136000],
                [145730,150230, 150000, 146000],
                [152500,153000, 153000, 153000]]

#%% Given time period location 
current_date = df["TRADE_DATE"][0]
tpl = 0 #time period location
day_period = []
hour_period = []
dim_list = [df["TRADE_DATE"][0]] #days in month 

for i in range(rc): 
    if df["TRADE_DATE"][i] == current_date:

        if df["TRADE_TIME"][i] >= time_period[tpl][0] and df["TRADE_TIME"][i] <= time_period[tpl][1]:
            hour_period.append(tpl)
        else: 
            tpl = tpl + 1
            hour_period.append(tpl)

        
    elif df["TRADE_DATE"][i] != current_date:
        dim_list.append(df["TRADE_DATE"][i])
        current_date = df["TRADE_DATE"][i]
        tpl = 0
        
        if df["TRADE_TIME"][i] >= time_period[tpl][0] and df["TRADE_TIME"][i] <= time_period[tpl][1]:
            hour_period.append(tpl)
        else: 
            tpl = tpl + 1
            hour_period.append(tpl)


#%% record the volume and put-call pair

pair_list = []
pair_dic = {}

df["Dist"] = np.nan
for i in range(rc):
    
    dist = abs(df['TRADE_TIME'][i] - time_period[hour_period[i]][2])
    if dist > 4000:
        dist = abs(df['TRADE_TIME'][i] - time_period[hour_period[i]][3])
        
    df.loc[i,'Dist'] = dist
    dict_key = "K:"+ str(df["EXERCISE_PRICE"][i])+"/"+"T:" +str(df["EXPIRATION_DATE"][i])
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




           
        
        



