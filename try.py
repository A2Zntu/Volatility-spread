# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:55:52 2018

@author: user_2
"""


start = 0
end = 1845

pair_list = []
pair_dic = {}
df["Dist"] = np.nan
for i in range(start, end):
    #calaulate the time-distance for each tick
    if df['TRADE_TIME'][i] < time_period['Fake_middle'][hour_period[i]]:
        dist = time_period['Fake_middle'][hour_period[i]] - df['TRADE_TIME'][i]
    else:
        dist = abs(df['TRADE_TIME'][i] - time_period['Middle'][hour_period[i]])
        
    df.loc[i,'Dist'] = dist
    td = time_delta(df["TRADE_DATE"][i], df["EXPIRATION_DATE"][i]) #calulate the delta T

    dict_key = "K:"+ str(df["EXERCISE_PRICE"][i])+"/"+"T:" +str(td)
#-------------------------------------------------------------------------
# --> current volume (without adjusted)
    cur_vol = df['TRADE_SIZE'][i] 
#------------------------------------------------------------------------- 
# --> current volume (with adjusted)
#        S = float(df['UNDERLYING_INSTRUMENT_PRICE'][i])
#        K = float(df['EXERCISE_PRICE'][i])
#        t = time_delta(df['TRADE_DATE'][i], df['EXPIRATION_DATE'][i])
#        cur_vol = df['TRADE_SIZE'][i]*MK_disc(S, K, t)
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
                
                
#%%
volatility_spread = []
spread_volume = []                
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
        Fc = Sc*exp(r*t)
        Fp = Sp*exp(r*t)
        print(i)
        print("=============")
        print("Sc:%f"%Sc)
        print("Sp:%f"%Sp)
        print("r:%f"%r)
        print("K:%f"%K)
        print("t:%f"%t)
        print("call price:%f"%call_price)
        print("put_price:%f"%put_price)
        
        
        intrinsic_c = fabs(max(Fc - K, 0.0))
        intrinsic_p = fabs(max(K - Fp, 0.0))
        temp_vs = 0.0
        temp_vol = 0.0
        if t < 30/365: #eliminate the option with expiration less than 30 days
#                print("Trade date: %s" %pair_list[i][0]['TRADE_DATE'])
#                print("Expiration date: %s" %pair_list[i][0]['EXPIRATION_DATE'])
#                print("Exercise price: %f" % K)
#                print("===========================")
            volatility_spread.append(temp_vs)
            spread_volume.append(temp_vol)
            
        elif call_price < intrinsic_c:
            volatility_spread.append(temp_vs)
            spread_volume.append(temp_vol)

        elif call_price >= Fc or put_price >= K:
            volatility_spread.append(temp_vs)
            spread_volume.append(temp_vol)
            
        elif put_price < intrinsic_p:
            volatility_spread.append(temp_vs)
            spread_volume.append(temp_vol)
            
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
            temp_vs = vol_spread
            temp_vol = vol
            volatility_spread.append(temp_vs)
            spread_volume.append(temp_vol)
        print("temp_vs:%f"%temp_vs)
        print("temp_vol:%f"%temp_vol)
        print("=============")
        
    else:
        volatility_spread.append(0.0)
        spread_volume.append(0.0)
      
try: 
    weights_aggr = [i/sum(spread_volume) for i in spread_volume]
    hour_vs = np.average(volatility_spread, weights = weights_aggr)
    
except ZeroDivisionError:
    hour_vs = np.nan