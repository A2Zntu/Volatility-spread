# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:12:54 2018

@author: Evan
"""

import pandas as pd 
import numpy as np 

df_123 = pd.read_csv("E:/Spyder/test_combine.csv")

num = len(df_123)

df123 = df_123.sort_values(['House Range','Age', 'Sex'], ascending=[True,True,True])
df123 = df123.reset_index(drop=True)

current_house = df123['House Range'][0] 
current_age = df123['Age'][0]

#existloc = []
#for i in range(1, num):
#    if df123['House Range'][i] == current_house:
#        if df123['Age'][i] == current_age:
#            if df123['Sex'][i] == 'G' and df123['Sex'][i-1] != 'G':
#                print("yes I exist in %d, my House Range is %d, and my age is %d" %(i, df123['House Range'][i], df123['Age'][i])) 
#                existloc.append(i)
#            else:
#                pass
#        else:            
#            current_age = df123['Age'][i]
#    else:        
#        current_house = df123['House Range'][i]
#
#goal_HR =  [df123['House Range'][e] for e in existloc]
#goal_Age =  [df123['Age'][e] for e in existloc]


#%% find out the least distance 

pair_list = []
pair_dic = {}

for i in range(num):
    dist = abs(df123['Date'][i] - 83000)
    df123.loc[i,'Dist'] = dist
    dict_key = "House Range:"+ str(df123["House Range"][i])+"/"+"Age:" +str(df123["Age"][i])
    cur_vol = df123['Volume'][i]
    
    if dict_key not in pair_dic:
        pair_dic[dict_key] = len(pair_dic)
        if df123["Sex"][i] == "G":
            pair_list.append([df123.iloc[i], [], cur_vol])
        if df123["Sex"][i] == "B":
            pair_list.append([[], df123.iloc[i], cur_vol])
    else:
        if df123['Sex'][i] == "G":
            if len(pair_list[pair_dic[dict_key]][0]) < 1:
                pair_list[pair_dic[dict_key]][0] = df123.iloc[i]
                pair_list[pair_dic[dict_key]][2] = pair_list[pair_dic[dict_key]][2] + cur_vol
                
            elif pair_list[pair_dic[dict_key]][0]['Dist'] > df123['Dist'][i]:
                pair_list[pair_dic[dict_key]][0] = df123.iloc[i]
                pair_list[pair_dic[dict_key]][2] = pair_list[pair_dic[dict_key]][2] + cur_vol
            else:
                pair_list[pair_dic[dict_key]][2] = pair_list[pair_dic[dict_key]][2] + cur_vol
            
        if df123["Sex"][i] == "B":
            if len(pair_list[pair_dic[dict_key]][1]) < 1:
                pair_list[pair_dic[dict_key]][1] = df123.iloc[i]
                pair_list[pair_dic[dict_key]][2] = pair_list[pair_dic[dict_key]][2] + cur_vol
                
            elif pair_list[pair_dic[dict_key]][1]['Dist'] > df123['Dist'][i]:
                pair_list[pair_dic[dict_key]][1] = df123.iloc[i]
                pair_list[pair_dic[dict_key]][2] = pair_list[pair_dic[dict_key]][2] + cur_vol
            else:
                pair_list[pair_dic[dict_key]][2] = pair_list[pair_dic[dict_key]][2] + cur_vol
