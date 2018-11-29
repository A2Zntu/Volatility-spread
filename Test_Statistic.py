# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 22:08:45 2018

@author: Evan
"""

import pandas as pd
import numpy as np
from datetime import date
from math import exp, sqrt, log, fabs
import matplotlib.pyplot as plt
from itertools import repeat
from scipy.stats import shapiro, norm
from statsmodels.graphics.gofplots import qqplot

#%%
time_period =  pd.read_csv("E:/Spyder/period_trade_1.csv", header = None)
time_period.columns = ['Head', 'Tail', 'Fake_middle', 'Middle']

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

#%% large sample size convert t-test to Z-test 
def twosampleZ(data1, data2):
    X1_bar = np.nanmean(data1)
    X2_bar = np.nanmean(data2)
    std1 = np.nanstd(data1)
    std2 = np.nanstd(data2)
    n1 = len(data1) - sum(np.isnan(data1))
    n2 = len(data1) - sum(np.isnan(data2))
    pooledSE = sqrt(std1**2/n1 + std2**2/n2)
    z = ((X1_bar - X2_bar))/pooledSE
    pval = 2*(1 - norm.cdf(abs(z))) #two-sided z test
    
    return round(z, 4), round(pval, 4)    
            
#%%
df_IVS = pd.read_csv("E:/Spyder/IVS/overall_vs_withadj_10days.csv", index_col = 0)
# Assume all the distributions are not normal. 
list_stat = []
list_p = []
for i, hour in enumerate(x_axis_hour):
    list_stat.append([])
    list_p.append([])
    for other_hour in x_axis_hour: 
        stat, p = twosampleZ(df_IVS[hour], df_IVS[other_hour])
        list_stat[i].append(stat)
        list_p[i].append(p)

df_p = pd.DataFrame(list_p, columns = x_axis_hour)
df_p.index = x_axis_hour
df_stat = pd.DataFrame(list_stat, columns = x_axis_hour)
df_stat.index = x_axis_hour
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

#%%